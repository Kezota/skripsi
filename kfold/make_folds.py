#!/usr/bin/env python3
"""
make_folds.py — Membuat pembagian 5-fold untuk evaluasi K-Fold Student KD v3.

DESAIN FOLD (didokumentasikan sesuai kesepakatan, BUKAN skema 70/15/15):
=========================================================================
Teacher (YOLOv8l) dilatih HANYA dari direktori `train` resmi BDD100K.
Gambar dari direktori `val` dan `test` resmi TIDAK PERNAH dilihat teacher
maupun student baseline — kami sebut "unseen pool" (~27.6k gambar
day+night).

Skema per fold (k=5):
  - TEST  : 1/5 dari unseen pool, DISJOINT antar fold (stratified
            multilabel: flag siang/malam + bin jumlah objek per kelas).
            Karena seluruhnya teacher-unseen, teacher BOLEH tetap beku
            tanpa kebocoran data — aturan "teacher harus retrain per fold"
            terpenuhi lewat desain, bukan lewat retraining.
  - VAL   : 2.000 siang + 2.000 malam, disampel dari unseen pool di luar
            test fold tsb (ukuran sama dengan val pipeline lama).
  - TRAIN : subsample 10k siang + 10k malam dari direktori `train` resmi,
            dengan seed BERBEDA per fold (TRAIN_SEED = 42 + fold).
            Jadi tiap fold melihat kombinasi train ≠ dan test ≠ →
            variasi data yang genuine, tanpa retrain teacher.

Konsekuensi yang diakui: baseline TIDAK ikut di-fold, sehingga klaim
k-fold = "stabilitas KD v3 terhadap variasi pembagian data", bukan
"KD menang vs baseline di tiap fold" (itu tetap dari 3-seed fixed split).

Stratifikasi: MultilabelStratifiedKFold (paket iterative-stratification),
k=5, shuffle=True, random_state=42. Vektor stratifikasi per gambar:
  [flag_night] + one-hot bin jumlah objek per kelas, bin = {0,1,2,3-5,6+}.
Fallback (jika paket tidak tersedia): sklearn StratifiedKFold dengan kunci
gabungan `timeofday + kelas dominan` — ditandai jelas di output.

Cara pakai (di Kaggle, CPU cukup):
  !pip install iterative-stratification
  !python kfold/make_folds.py            # cari cache otomatis di /kaggle/input
  # atau: !python kfold/make_folds.py --cache /path/ke/bdd100k_cache.pkl

Output:
  kfold/folds.json  — {"meta": {...}, "fold_0": {"test": [...], "val": [...],
                       "train_seed": 42}, ...}
                      (train TIDAK disimpan sebagai daftar — cukup seed-nya,
                       loader existing yang men-subsample deterministik)
  stdout            — tabel verifikasi per fold + hasil cek kebocoran.
"""
import argparse
import json
import os
import pickle
import random
import sys
from collections import Counter, defaultdict

VALID_CLASSES = sorted([
    "car", "truck", "bus", "person", "rider",
    "bike", "motor", "traffic light", "traffic sign"
])
K = 5
RANDOM_STATE = 42
VAL_DAY, VAL_NIGHT = 2000, 2000
BINS = [(0, 0), (1, 1), (2, 2), (3, 5), (6, 10**9)]  # 0,1,2,3-5,6+


def find_cache(explicit=None):
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates.append(
        "/kaggle/input/datasets/yosephoktavianus/"
        "bdd100k-cache-from-notebook1-version-19/bdd100k_cache.pkl"
    )
    for base in ("/kaggle/input", "/kaggle/working", "."):
        if os.path.isdir(base):
            for root, _dirs, files in os.walk(base):
                if "bdd100k_cache.pkl" in files:
                    candidates.append(os.path.join(root, "bdd100k_cache.pkl"))
    for c in candidates:
        if c and os.path.exists(c):
            return c
    sys.exit("[ERROR] bdd100k_cache.pkl tidak ditemukan. Pakai --cache <path>.")


def iter_objects(data):
    """Ekstrak daftar kategori objek dari satu json label BDD100K.
    Menangani dua format: {"frames":[{"objects":[...]}]} dan {"labels":[...]}."""
    if "frames" in data:
        for fr in data["frames"]:
            for o in fr.get("objects", []):
                yield o.get("category", "")
    for o in data.get("labels", []):
        yield o.get("category", "")


def class_counts(data):
    c = Counter(cat for cat in iter_objects(data) if cat in VALID_CLASSES)
    return [c.get(cls, 0) for cls in VALID_CLASSES]


def bin_index(n):
    for i, (lo, hi) in enumerate(BINS):
        if lo <= n <= hi:
            return i
    return len(BINS) - 1


def video_id(name):
    # Nama file BDD100K = ID video-nya (subset deteksi = 1 keyframe per video),
    # mis. "b1c66a42-6f7d68ca.json" -> "b1c66a42-6f7d68ca"
    return os.path.splitext(os.path.basename(name))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None, help="path ke bdd100k_cache.pkl")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__) or ".", "folds.json"))
    args = ap.parse_args()

    cache_path = find_cache(args.cache)
    print(f"[INFO] Memuat cache: {cache_path}")
    with open(cache_path, "rb") as f:
        all_data = pickle.load(f)

    # ---------------- Unseen pool: seluruh isi direktori val + test ----------
    pool = []  # (name, split_dir, timeofday, counts)
    for split in ("val", "test"):
        for tod in ("daytime", "night"):
            for jf, data in all_data[split][tod]:
                pool.append({
                    "name": os.path.basename(jf),
                    "dir": split,
                    "tod": tod,
                    "counts": class_counts(data),
                })
    n_day = sum(1 for p in pool if p["tod"] == "daytime")
    n_night = len(pool) - n_day
    print(f"[INFO] Unseen pool (val+test dirs): {len(pool)} gambar "
          f"({n_day} siang, {n_night} malam)")

    # ---------------- Vektor stratifikasi multilabel -------------------------
    # kolom: [is_night] + one-hot bin per kelas (9 kelas x 5 bin = 45) -> 46 kolom
    Y = []
    for p in pool:
        row = [1 if p["tod"] == "night" else 0]
        for n in p["counts"]:
            onehot = [0] * len(BINS)
            onehot[bin_index(n)] = 1
            row.extend(onehot)
        Y.append(row)

    # ---------------- Split k-fold -------------------------------------------
    fallback = False
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
        import numpy as np
        mskf = MultilabelStratifiedKFold(n_splits=K, shuffle=True,
                                         random_state=RANDOM_STATE)
        X = np.zeros((len(pool), 1))
        test_folds = [test_idx for _tr, test_idx in mskf.split(X, np.array(Y))]
        print("[INFO] Stratifikasi: MultilabelStratifiedKFold "
              "(iterative-stratification), k=5, shuffle, seed=42")
    except ImportError:
        # FALLBACK: iterative-stratification tidak terpasang.
        # StratifiedKFold biasa dengan kunci gabungan timeofday + kelas dominan.
        # Kualitas stratifikasi per-kelas lebih kasar; catat di paper bila terpakai.
        fallback = True
        from sklearn.model_selection import StratifiedKFold
        import numpy as np
        keys = []
        for p in pool:
            dom = VALID_CLASSES[max(range(len(VALID_CLASSES)),
                                    key=lambda i: p["counts"][i])] \
                if sum(p["counts"]) > 0 else "empty"
            keys.append(f"{p['tod']}|{dom}")
        skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=RANDOM_STATE)
        test_folds = [test_idx for _tr, test_idx in
                      skf.split(np.zeros(len(pool)), np.array(keys))]
        print("[WARNING] FALLBACK AKTIF: StratifiedKFold biasa "
              "(timeofday + kelas dominan). Install iterative-stratification "
              "untuk stratifikasi multilabel penuh.")

    # ---------------- Susun folds + val per fold ------------------------------
    folds = {"meta": {
        "design": ("test = 1/5 unseen pool (val+test dirs resmi, teacher-unseen, "
                   "disjoint antar fold); val = 2000 siang + 2000 malam dari "
                   "unseen pool di luar test fold; train = subsample 10k+10k "
                   "dari train dir dengan seed 42+fold (via loader existing)"),
        "k": K, "random_state": RANDOM_STATE,
        "stratifier": "fallback-StratifiedKFold" if fallback
                      else "MultilabelStratifiedKFold",
        "classes": VALID_CLASSES,
    }}
    for fi, test_idx in enumerate(test_folds):
        test_idx = set(int(i) for i in test_idx)
        test_names = sorted(pool[i]["name"] for i in test_idx)
        rest = [i for i in range(len(pool)) if i not in test_idx]
        rng = random.Random(1000 + fi)
        rest_day = [i for i in rest if pool[i]["tod"] == "daytime"]
        rest_night = [i for i in rest if pool[i]["tod"] == "night"]
        val_idx = (rng.sample(rest_day, min(VAL_DAY, len(rest_day)))
                   + rng.sample(rest_night, min(VAL_NIGHT, len(rest_night))))
        folds[f"fold_{fi}"] = {
            "test": test_names,
            "val": sorted(pool[i]["name"] for i in val_idx),
            "train_seed": RANDOM_STATE + fi,
        }

    with open(args.out, "w") as f:
        json.dump(folds, f)
    print(f"[INFO] Tersimpan: {args.out}\n")

    # ================= TABEL VERIFIKASI ======================================
    by_name = {p["name"]: p for p in pool}
    print("=" * 100)
    print("TABEL VERIFIKASI PER FOLD (bagian TEST)")
    print("=" * 100)
    header = f"{'fold':<7}{'siang':>7}{'malam':>7}  " + \
             "".join(f"{c[:9]:>10}" for c in VALID_CLASSES)
    print(header)
    for fi in range(K):
        names = folds[f"fold_{fi}"]["test"]
        day = sum(1 for n in names if by_name[n]["tod"] == "daytime")
        night = len(names) - day
        inst = [0] * len(VALID_CLASSES)
        for n in names:
            for ci, v in enumerate(by_name[n]["counts"]):
                inst[ci] += v
        print(f"fold_{fi:<2}{day:>7}{night:>7}  " +
              "".join(f"{v:>10}" for v in inst))
    print()
    print(f"{'fold':<7}{'val siang':>10}{'val malam':>10}{'train_seed':>12}")
    for fi in range(K):
        names = folds[f"fold_{fi}"]["val"]
        day = sum(1 for n in names if by_name[n]["tod"] == "daytime")
        print(f"fold_{fi:<2}{day:>10}{len(names)-day:>10}"
              f"{folds[f'fold_{fi}']['train_seed']:>12}")

    # ================= CEK KEBOCORAN (Langkah 2) =============================
    print("\n" + "=" * 100)
    print("CEK KEBOCORAN")
    print("=" * 100)
    seen = {}
    dup = 0
    for fi in range(K):
        for n in folds[f"fold_{fi}"]["test"]:
            if n in seen:
                dup += 1
                print(f"  [LEAK] {n} ada di fold_{seen[n]} dan fold_{fi}")
            seen[n] = fi
    print(f"[1] Duplikasi nama file antar test fold : "
          f"{'GAGAL, ' + str(dup) + ' duplikat' if dup else 'LOLOS (0 duplikat)'}")

    train_names = set()
    for tod in ("daytime", "night"):
        for jf, _d in all_data["train"][tod]:
            train_names.add(os.path.basename(jf))
    overlap = sum(1 for n in seen if n in train_names)
    print(f"[2] Test fold vs direktori train teacher : "
          f"{'GAGAL, ' + str(overlap) + ' overlap' if overlap else 'LOLOS (0 overlap — semua test teacher-unseen)'}")

    vids = defaultdict(set)
    for fi in range(K):
        for n in folds[f"fold_{fi}"]["test"]:
            vids[video_id(n)].add(fi)
    multi = {v: fs for v, fs in vids.items() if len(fs) > 1}
    if multi:
        print(f"[3] Video ID tersebar antar fold        : GAGAL, {len(multi)} video")
    else:
        print("[3] Video ID tersebar antar fold        : LOLOS — subset deteksi "
              "BDD100K memuat 1 keyframe per video, tiap nama file = 1 video unik, "
              "sehingga tidak ada dua gambar dari video yang sama di fold berbeda.")

    print("\n[SELESAI] Kirim tabel di atas untuk direview sebelum lanjut ke "
          "pembuatan notebook per fold (Langkah 3).")


if __name__ == "__main__":
    main()
