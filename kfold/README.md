# Evaluasi 5-Fold — Student KD v3

Evaluasi Multilabel Stratified K-Fold (k=5) untuk **Student KD v3** (soft confidence
weighting + beta cosine schedule + CWD), tanpa menyentuh pipeline lama —
semua notebook asli tetap utuh, pembagian split lama tetap bisa direproduksi.

## Isi folder

| File | Fungsi |
|---|---|
| `make_folds.ipynb` / `make_folds.py` | Membuat `folds.json` (jalankan **sekali**, satu orang) + tabel verifikasi + cek kebocoran |
| `folds.json` | Definisi 5 fold (test/val per fold + `train_seed`) — di-upload sebagai Kaggle dataset, dipakai SEMUA akun |
| `notebook3-kd-kfold.ipynb` | Training KD v3 per fold (set `FOLD = 0..4` di cell atas) |
| `notebook3-2-resume-kd-kfold.ipynb` | Resume training (limit 12 jam Kaggle) |
| `notebook3-5-eval-student-kd-kfold.ipynb` | Evaluasi final per fold → `student_kd_overall_fold{N}_9class.csv` |
| `aggregate.ipynb` | Rekap kelima CSV → `kfold_summary.csv` + tabel markdown (mean ± SD, termasuk penurunan relatif siang→malam). CSV dicari otomatis di `/kaggle/input` |

## Kenapa Multilabel Stratified, bukan k-fold biasa?

K-fold biasa membagi acak — bisa menghasilkan fold yang timpang: satu fold
kebanyakan gambar malam, fold lain kekurangan kelas langka (`motor` hanya
~1.200 instance di seluruh unseen pool; acak murni bisa membuat satu fold
kebagian sangat sedikit). Karena klaim utama penelitian ini justru tentang
**perbandingan siang vs malam** dan mencakup kelas-kelas langka (VRU: bike,
rider, motor), tiap fold wajib punya proporsi siang/malam dan distribusi kelas
yang setara. Stratifikasi multilabel (vektor: flag siang/malam + bin jumlah
objek per kelas {0, 1, 2, 3–5, 6+}) menjamin itu — terverifikasi: selisih
siang/malam antar fold maksimal 1 gambar, spread instance per kelas ~1–2%.

## Kenapa teacher TIDAK dilatih ulang per fold — dan kenapa itu sah

Aturan umumnya: teacher harus di-retrain per fold, karena kalau teacher pernah
melihat gambar test fold saat trainingnya sendiri, pengetahuan itu bocor ke
student lewat distilasi. **Desain di sini memenuhi kekhawatiran itu lewat
konstruksi, bukan lewat retraining:**

- Teacher hanya pernah dilatih dari **direktori `train` resmi** BDD100K.
- Seluruh test fold diambil **eksklusif dari direktori `val` + `test` resmi**
  (~27.6k gambar) yang tidak pernah dilihat teacher — diverifikasi otomatis
  oleh `make_folds` (cek #2: 0 overlap dengan direktori train teacher).
- Yang bervariasi antar fold: (a) partisi test yang disjoint, dan
  (b) subsample train student dengan seed berbeda (`train_seed = 42 + fold`),
  sehingga tiap fold tetap merupakan kombinasi data train ≠ dan test ≠.

Konsekuensi yang diakui secara jujur: karena baseline tidak ikut di-fold,
klaim k-fold ini = **"stabilitas KD v3 terhadap variasi pembagian data"**
(mean ± SD antar 5 fold), bukan "KD menang vs baseline di tiap fold" —
perbandingan vs baseline tetap bersandar pada bukti 3-seed fixed-split yang
sudah ada.

## Perkiraan waktu training

- Rencana awal (retrain teacher + baseline + KD per fold): 5 fold × 3 model =
  **15 training ≈ 375 GPU-jam** (~3–6 minggu dengan kuota tim).
- Desain ini (hanya KD v3 per fold): **5 training ≈ 120 GPU-jam** —
  dengan 5 akun Kaggle paralel (1 orang = 1 fold), selesai **~2–4 hari
  kalender** (per fold: ~24 jam = 2 sesi × 12 jam + eval ~1 jam).

## Alur kerja per orang (1 fold)

1. Upload `notebook3-kd-kfold.ipynb`, set `FOLD` di cell atas.
   Input: dataset BDD100K + cache + **dataset folds.json** + output notebook1
   (teacher checkpoint). Run sampai kena limit 12 jam → Save Version.
2. Upload `notebook3-2-resume-kd-kfold.ipynb`, set `FOLD` sama.
   Tambah Input: output notebook train langkah 1 (checkpoint dicari otomatis
   via `find_file_in_input`, tidak perlu edit path). Ulangi resume bila perlu
   sampai early-stop / epoch 50.
3. Upload `notebook3-5-eval-student-kd-kfold.ipynb`, set `FOLD` sama.
   Input: BDD100K + cache + folds.json + output run resume terakhir.
   Hasil: `student_kd_overall_fold{N}_9class.csv` + per-class CSV.
4. Setelah kelima fold selesai: upload `aggregate.ipynb`, add kelima output
   notebook eval sebagai Input (CSV dicari otomatis, tidak perlu dikumpulkan
   manual), Run All → `kfold_summary.csv` + tabel markdown siap tempel ke
   skripsi/paper.

## Reproduksibilitas

Semua acak ter-seed: stratifikasi `random_state=42`; val sampler per fold
`Random(1000+fold)`; subsample train `Random(42+fold)`; pembagian lama tetap
direproduksi notebook asli (di luar folder ini) dengan seed 42 aslinya.
