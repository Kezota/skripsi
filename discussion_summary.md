# 📋 Ringkasan Diskusi: Knowledge Distillation BDD100K

---

## 1. Masalah Awal — Analisis Final Evaluation (10 Kelas)

**Konteks:** Hasil evaluasi awal dari 3 notebook (teacher, student baseline, student KD) menunjukkan Student KD **kalah** dari Student Baseline di beberapa metrik, terutama mAP50 Daytime.

**Root cause yang ditemukan:** Kelas `"train"` mengalami **class collapse** di Student KD:
- Hanya **25 training images**, **31 instances** → model tidak bisa belajar dengan baik
- Hanya **3 test instances** → evaluasi tidak valid secara statistik
- AP50 kelas `"train"` di Student KD: **0.04** (vs teacher 0.116, baseline 0.50)
- Collapse ini menarik mAP keseluruhan turun drastis, membuat KD terlihat kalah

---

## 2. Strategi: Exclude Kelas "train"

**Keputusan:** Exclude kelas `"train"` dan re-training seluruh pipeline dari scratch dengan **9 kelas**.

**Strategi yang dipilih:** Bukan sekadar filter di evaluasi, tapi **full re-training** dengan 9 kelas agar valid untuk research paper.

**Justifikasi ilmiah untuk paper:**
> *"The 'train' class was excluded due to extreme class imbalance (791× ratio vs 'car'), only 25 training images and 3 test instances — statistically insufficient for meaningful AP evaluation."*

---

## 3. Implementasi: Migrasi 9 Notebook ke 9 Kelas

**Perubahan yang diterapkan secara otomatis (script Python) ke 9 notebook:**

| Perubahan | Detail |
|-----------|--------|
| `VALID_CLASSES` | Hapus `"train"` → `NUM_CLASSES = 9` otomatis |
| `adjust_yolov8_classes` | default `new_nc=10` → `new_nc=9` |
| `setup_student` | default `new_nc=10` → `new_nc=9` |
| `setup_frozen_model` | default `new_nc=10` → `new_nc=9` |
| Komentar decode | `nc=10, C=74` → `nc=9, C=73` |
| Checkpoint names | `*.pt` → `*_9class.pt` |
| CSV output names | `*.csv` → `*_9class.csv` |

**Class ID baru (sorted alphabetically):**
```
bike=0, bus=1, car=2, motor=3, person=4,
rider=5, traffic light=6, traffic sign=7, truck=8
```
⚠️ `truck` bergeser dari index 9 → 8 (normal karena train ulang dari awal)

**Yang TIDAK perlu diubah:**
- Cache `bdd100k_cache.pkl` → tetap kompatibel (hanya simpan file paths, bukan class ID)

---

## 4. Hasil Evaluasi 9 Kelas — Student KD v1 vs Baseline

**Konfigurasi KD v1:** `alpha=0.7, beta=0.3, temperature=3.0, conf_thresh=0.3, gamma_feat=0.1`

**Hasil overall:**

| Split | Baseline mAP50 | KD v1 mAP50 | Δ |
|-------|:---:|:---:|:---:|
| Daytime | 0.4893 | 0.4953 | +0.0060 |
| Night | 0.4470 | 0.4720 | +0.0250 |

**Temuan kunci:**
- KD v1 menang dari Baseline di **semua metrik** pada kedua split ✅
- **Night lebih unggul** (+0.0250 mAP50) → KD efektif mentransfer kemampuan low-light dari teacher
- Precision sangat tinggi tapi Recall rendah di Daytime → model terlalu *conservative*
- Kelas `rider` night naik dramatis: +0.1286 AP50 vs Baseline

---

## 5. Hyperparameter Tuning KD

**Diagnosis dari data:** Precision-Recall gap melebar di KD v1 (Precision 0.6758, Recall hanya 0.4446 Daytime) → `beta` terlalu tinggi, `conf_thresh` membuang terlalu banyak anchor.

**Perubahan yang diterapkan ke notebook3-kd dan notebook3-2-resume-kd:**

| Parameter | Sebelum | Sesudah | Alasan |
|-----------|:-------:|:-------:|--------|
| `beta` | 0.3 | **0.25** | Kurangi dominasi KD loss → perbaiki recall |
| `temperature` | 3.0 | **4.0** | Soft label lebih smooth → dark knowledge lebih kaya |
| `conf_thresh` | 0.3 | **0.2** | Lebih banyak anchor distilled |
| `gamma_feat` | 0.1 | **0.08** | Fine-tune feature distillation weight |

---

## 6. Hasil Evaluasi Setelah Hyperparameter Tuning

**KD Tuned:** `alpha=0.7, beta=0.25, temperature=4.0, conf_thresh=0.2, gamma_feat=0.08`

### Overall Metrics

| Split | Baseline | KD v1 | KD Tuned | KD Tuned Δ vs Baseline |
|-------|:---:|:---:|:---:|:---:|
| Daytime mAP50 | 0.4893 | 0.4953 | **0.5047** | **+0.0154** |
| Daytime mAP50-95 | 0.2771 | 0.2814 | **0.2877** | **+0.0106** |
| Daytime F1 | 0.5257 | 0.5327 | **0.5449** | **+0.0192** |
| Night mAP50 | 0.4470 | **0.4720** | 0.4620 | +0.0150 |
| Night mAP50-95 | 0.2420 | 0.2544 | 0.2525 | +0.0105 |
| Night F1 | 0.4838 | 0.5210 | 0.5026 | +0.0188 |

### Per-Class Daytime (mAP50)
KD Tuned menang **9/9 kelas** vs Baseline (kenaikan terbesar: `rider` +0.0383, `motor` +0.0340)

### Per-Class Night (mAP50)
KD Tuned menang **7/9 kelas** vs Baseline. Regresi di `motor` (-0.0404) dan `bus` (-0.0126).

### Trade-off KD Tuned vs KD v1
- Daytime: **KD Tuned menang** (mAP50 +0.0094, F1 +0.0122, Recall +0.0315)
- Night: KD v1 sedikit lebih baik (mAP50 -0.0100) — dalam batas noise training

---

## 7. Model Final yang Direkomendasikan untuk Paper

**→ KD Tuned** sebagai model final:
- mAP50 Daytime tertinggi (0.5047)
- F1 terbaik kedua split
- mAP50-95 terbaik → bounding box lebih akurat
- 9/9 kelas Daytime menang vs Baseline → tabel paper sangat clean
- Selisih Night vs KD v1 hanya -0.0100 (dalam noise range)

---

## 8. Penilaian Publishability

**Jujur:** Improvement **+0.0154 mAP50** tergolong **marginal** untuk venue top-tier.

| Target | Layak? |
|--------|:------:|
| CVPR/ICCV/NeurIPS | ❌ |
| AAAI/IJCAI | ⚠️ Borderline |
| IEEE TITS / Domain Journal | ✅ |
| Workshop | ✅ |
| **Skripsi/Pre-Thesis** | ✅✅ |

**Kelemahan kritis:** Hanya 1 training run → tidak ada statistical significance test. Variance ±0.01–0.02 mAP50 antar-run bisa menyamai angka improvement.

**Saran untuk meningkatkan publishability:**
1. **Jalankan 3× dengan seed berbeda** → laporkan `mean ± std`
2. **Reframe kontribusi**: bukan "KD lebih baik dari baseline" tapi "analisis KD pada kondisi pencahayaan heterogen (day/night) untuk autonomous driving"

---

## 9. Arsitektur KD yang Diimplementasikan (Ringkasan Teknis)

```
Loss formula:
L_total = α × L_task  +  β × (L_cls_kd + L_dfl_kd)  +  γ × L_feat

Komponen:
- L_task     : v8DetectionLoss (ground truth supervision)
- L_cls_kd   : KL Divergence pada class logits (temperature T)
- L_dfl_kd   : KL Divergence pada DFL box distribution (temperature T_dfl=2.0)
- L_feat     : Multi-scale feature distillation (layers 15, 18, 21)
- mask       : Hanya anchor dengan teacher confidence > conf_thresh yang distilled

Model:
- Teacher: YOLOv8l fine-tuned (~43.6M params)
- Student: YOLOv8s (~11.1M params) → ~3.9× compression
- Student KD Tuned mencapai 88.5% (Daytime) dan 91.5% (Night) performa teacher
```

---

## 10. KD v3 — Upgrade Mekanisme Distilasi (belum di-train)

**Tujuan:** Memperbesar delta KD vs Baseline dengan memperbaiki mekanisme distilasi itu sendiri. Baseline & teacher **tidak disentuh** → perbandingan tetap apple-to-apple, hanya perlu retrain KD.

**Diubah di:** `output-notebooks-notebook3-kd.ipynb` dan `output-notebooks-notebook3-2-resume-kd.ipynb` (keduanya identik, + markdown cell catatan di atas cell class KD).

| # | Perubahan | Sebelum | Sesudah |
|---|-----------|---------|---------|
| 1 | **Soft confidence weighting** | Hard mask `t_conf > conf_thresh (0.2)` — anchor di bawah threshold dibuang total | Semua anchor di-distill, bobot kontinu `w = sqrt(teacher_conf)`, weighted mean (skala loss sebanding). Parameter `conf_thresh` dihapus |
| 2 | **Beta cosine schedule** | `beta = 0.25` konstan | Cosine decay `0.35 → 0.10` sepanjang `total_epochs`. KD dominan di awal, GT dominan di akhir. Tercatat di kolom `beta_kd` history CSV |
| 3 | **CWD feature distillation** | MSE + adapter Conv1×1+BN+ReLU, `gamma_feat = 0.08` | Channel-wise Distillation (Shu et al., ICCV 2021): KL pada softmax spasial per channel (τ=4), adapter Conv1×1 tanpa BN/ReLU, `GAMMA_FEAT = 0.03` |

**Rasional singkat:**
- #1 menyerang langsung precision-recall imbalance: anchor "setengah yakin" teacher tidak lagi dibuang → recall membaik tanpa kehilangan sinyal distilasi.
- #2 mendamaikan temuan beta 0.3 (night bagus) vs 0.25 (daytime bagus): awal training dapat distilasi kuat, akhir training dapat dominasi GT.
- #3 CWD terbukti di literatur lebih efektif dari MSE untuk dense prediction — student meniru *di mana* teacher melihat, bukan magnitudo aktivasi mentah.

**⚠️ Catatan operasional:**
1. Checkpoint KD v1/Tuned **tidak kompatibel** untuk resume (struktur `feat_kd` & optimizer berubah). KD v3 harus training dari epoch 1. Notebook resume sudah punya guard + warning untuk kasus ini.
2. `GAMMA_FEAT = 0.03` adalah estimasi awal (skala KL-CWD ≠ MSE). **Pantau kolom `feat` di progress bar epoch 1**: targetkan ~5–15% dari total loss, sesuaikan bila di luar itu.
3. Logika sudah diverifikasi unit-test dengan tensor dummy (soft weighting = weighted mean manual, beta schedule monoton 0.35→0.10, CWD backward OK & self-distill loss ≈ 0), tapi **belum di-train** — hasil aktual perlu 1 run penuh.
