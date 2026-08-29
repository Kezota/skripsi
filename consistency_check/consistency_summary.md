# Consistency Check — KD Student (3 independent runs, 9 classes, BDD100K)

n = 3 runs. Nilai = mean ± sd. Baseline & teacher masih 1 run (angka dari `final_evaluation_9class/`).

## 1. Overall

| Split | Metric | Mean ± sd | CV | Baseline | Δ | Δ/sd | % teacher |
|---|---|---|---|---|---|---|---|
| Day | mAP@0.5 | 0.5013 ± 0.0024 | 0.5% | 0.4893 | +0.0120 | 5.1σ | 89.6% |
| Day | mAP@0.5:0.95 | 0.2834 ± 0.0005 | 0.2% | 0.2771 | +0.0063 | 11.9σ | 86.9% |
| Day | F1 | 0.5399 ± 0.0023 | 0.4% | 0.5257 | +0.0142 | 6.1σ | 91.7% |
| Day | Precision | 0.6332 ± 0.0072 | 1.1% | 0.6344 | −0.0012 | −0.2σ | 94.7% |
| Day | Recall | 0.4747 ± 0.0057 | 1.2% | 0.4527 | +0.0220 | 3.9σ | 89.7% |
| Night | mAP@0.5 | 0.4780 ± 0.0038 | 0.8% | 0.4470 | +0.0310 | 8.1σ | 92.7% |
| Night | mAP@0.5:0.95 | 0.2586 ± 0.0037 | 1.4% | 0.2420 | +0.0166 | 4.4σ | 91.7% |
| Night | F1 | 0.5188 ± 0.0041 | 0.8% | 0.4838 | +0.0350 | 8.5σ | 94.6% |
| Night | Precision | 0.5966 ± 0.0128 | 2.1% | 0.5670 | +0.0296 | 2.3σ | 95.3% |
| Night | Recall | 0.4644 ± 0.0087 | 1.9% | 0.4281 | +0.0363 | 4.2σ | 94.5% |

## 2. Per run (mAP@0.5)

| Run | Day | Night | Gap |
|---|---|---|---|
| 1 | 0.5014 | 0.4740 | 0.0274 |
| 2 | 0.4989 | 0.4816 | 0.0173 |
| 3 | 0.5036 | 0.4783 | 0.0253 |
| **Mean** | **0.5013** | **0.4780** | **0.0233 ± 0.0053** |

## 3. Day→night gap antar model

| Model | Gap mAP@0.5 |
|---|---|
| Teacher YOLOv8l | 0.0441 |
| Student baseline | 0.0423 |
| KD tuned (1 run) | 0.0427 |
| KD v1 (1 run) | 0.0233 |
| **KD 3-run mean** | **0.0233** |

## 4. Per-class AP@0.5 (mean ± sd, 3 runs)

| Class | Day | sd | CV | Night | sd | CV |
|---|---|---|---|---|---|---|
| car | 0.7455 | 0.0019 | 0.3% | 0.7216 | 0.0008 | 0.1% |
| traffic sign | 0.5538 | 0.0047 | 0.8% | 0.6122 | 0.0014 | 0.2% |
| person | 0.5743 | 0.0034 | 0.6% | 0.4978 | 0.0060 | 1.2% |
| traffic light | 0.5200 | 0.0003 | 0.1% | 0.5285 | 0.0007 | 0.1% |
| truck | 0.5148 | 0.0024 | 0.5% | 0.4721 | 0.0098 | 2.1% |
| bus | 0.4810 | 0.0062 | 1.3% | 0.4137 | 0.0182 | 4.4% |
| rider | 0.3692 | 0.0163 | 4.4% | 0.4198 | 0.0063 | 1.5% |
| bike | 0.3492 | 0.0104 | 3.0% | 0.3561 | 0.0174 | 4.9% |
| motor | 0.4037 | 0.0106 | 2.6% | 0.2798 | 0.0131 | 4.7% |

## 5. Δ AP@0.5 vs baseline — 18/18 menang

| Class | Δ Day | Δ Night |
|---|---|---|
| bike | +0.0074 | +0.1082 |
| rider | +0.0193 | +0.1055 |
| motor | +0.0220 | +0.0169 |
| bus | +0.0194 | +0.0153 |
| truck | +0.0151 | +0.0029 |
| traffic sign | +0.0105 | +0.0069 |
| traffic light | +0.0071 | +0.0121 |
| person | +0.0046 | +0.0093 |
| car | +0.0024 | +0.0018 |

## 6. Catatan

- Semua CV overall < 2.2%; mAP@0.5 CV 0.5% (day) / 0.8% (night) → training stabil.
- Improvement 5–8× sd → di luar noise antar-run.
- sd hanya diukur untuk KD; baseline & teacher masih 1 run. Untuk klaim signifikansi penuh, baseline perlu 3 run juga.
- Gap day→night turun dari 0.0423 (baseline) ke 0.0233 (−45%), sementara teacher sendiri 0.0441.
