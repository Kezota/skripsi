# Revisi Naskah Lengkap: Judul, Abstract, Bab I sampai V, Diagram, Referensi

Teks berbahasa Inggris adalah teks naskah, siap tempel. Teks berbahasa Indonesia adalah catatan untuk kalian.

Gaya bahasa mengikuti draft *Skripsi_KWY.pdf*. Semua sebutan "v3", "KD v3", "Student KD v3" dihapus dan diganti **Student KD** atau **the distilled student**. KD tuned dan KD v1 tidak disebut sama sekali. Referensi [1] sampai [21] dipertahankan, ditambah **lima** referensi baru [22] sampai [26].

---

## 0. Yang harus dicek sebelum menempel teks

| # | Lokasi | Masalah | Tindakan |
|---|---|---|---|
| 1 | Ref [4] | Ditulis "in CVPR, 2024". Papernya preprint arXiv 2604.26857, April 2026 | Ganti entri, lihat Bagian 7 |
| 2 | Tabel V, baris No CWD | Naskah: 0,5010 siang, 0,4744 malam. Output notebook yang kalian kirim ke saya: 0,5003 siang, 0,4682 malam. **Beda dan mengubah kesimpulan** | Buka kembali file evaluasi ablation no-CWD, tentukan mana yang benar. Bab IV.C saya tulis dua versi |
| 3 | Bab IV.D | Test set 5-fold disebut berasal dari "~27,600 images from official validation and test directories", sedangkan Tabel I total 24.000 | Pastikan pool mana yang dipakai, lalu isi angka di kalimat yang saya tandai `[cek]` |
| 4 | Bab IV.F | Sitasi rusak `[?]` | Sudah saya ganti [4] di teks baru |
| 5 | Bab III.D lama | "10-class detection head" | Sudah diganti 9-class di teks baru |
| 6 | Tabel II lama | β = 0,3 konstan, γ = 0,1, patience 7 | Sudah diganti di teks baru |
| 7 | Bab III.E lama | Masih "MSE Loss" | Sudah diganti CWD di teks baru |
| 8 | Ref [16], [19], [21] | Venue dan nama penulis salah | Perbaikan ada di Bagian 7 |
| 9 | Fig 3 caption lama | Menyebut simbol (•) dan (♦) | Caption baru ada di Bab IV |

---

## 1. Judul

> **Narrowing the Day-to-Night Performance Gap of Lightweight YOLOv8 Detectors via Knowledge Distillation**

Alternatif lebih pendek: *Illumination-Robust Knowledge Distillation for Lightweight Driving Scene Object Detection*.

---

## 2. Abstract dan Index Terms

**Abstract.** Object detectors deployed on in-vehicle edge hardware must operate under both daylight and darkness with a single model, yet lightweight detectors degrade sharply at night. Prior work has demonstrated that knowledge distillation (KD) confers robustness to INT8 quantization on edge-deployable YOLOv8 students, but the effect of illumination change was left as future work. In this paper, we address that gap. A fine-tuned YOLOv8-L teacher (43.7M parameters) is distilled into a YOLOv8-S student (11.2M parameters) on a balanced daytime and nighttime subset of BDD100K, and every model is evaluated separately on each condition. The distillation framework integrates multi-level feature alignment with response distillation, and incorporates three mechanisms chosen against specific nighttime failure modes: soft confidence weighting, a cosine-decayed distillation weight, and channel-wise feature distillation. Averaged over three independent training runs, the distilled student improves nighttime mAP@0.5 from 0.4470 to 0.4780 (+0.0310, 8.1 standard deviations) while also improving daytime mAP@0.5 (+0.0120), and reduces the relative day-to-night drop from 8.65% to 4.65%, below the teacher's own 7.88%. Gains are observed across all nine classes in both conditions, with the largest improvements on vulnerable road users at night. A leave-one-out ablation shows the three mechanisms play distinct roles, and stratified five-fold cross-validation confirms the direction of the effect. No inference-time cost is added; the deployed model is an unmodified YOLOv8-S.

**Index Terms:** knowledge distillation, object detection, YOLOv8, low-light, day-to-night domain gap, edge deployment, BDD100K.

---

## 3. Bab I: Introduction

Object detection is one of the fundamental modules in computer vision for intelligent transport systems, enabling the real-time identification and localisation of obstacles and pedestrians in the vicinity of a vehicle. Although these perception systems demonstrate reliable performance under adequate lighting conditions, their reliability decreases significantly in low-light driving environments and at night [1], [2]. Insufficient lighting naturally results in dark and blurry visual images, thereby hindering the sensors' ability to capture image details clearly and causing a substantial drop in performance for standard detection models [1], [3]. Cui et al. [24] characterise the problem directly for BDD100K: detectors trained on well-lit daytime images suffer from poor performance on nighttime images owing to under- and over-exposure, motion blur, and headlamp glare. To ensure vehicle safety, such perception models must be able to operate effectively on in-vehicle edge devices, such as the NVIDIA Jetson platform, which have strict limitations in terms of memory and inference latency [4], [5]. Consequently, lightweight one-stage detectors, particularly the YOLO architecture, are more frequently chosen as they offer a good balance between speed and accuracy [5]–[7]. However, these compact models inherently have limited representational capacity, making them more susceptible to feature degradation and sensor noise commonly encountered in low-light driving conditions [1], [5].

In order to address the problem of poor performance in difficult lighting conditions, recent studies have considered Knowledge Distillation (KD) as an efficient method for model compression, without additional inference time cost. Several studies have developed unsupervised feature domain distillation for knowledge transfer from bright light conditions directly to the low-light domain and KD guided image enhancement neural networks [1], [3]. Some researchers have developed formulations where localization uncertainty is considered by distilling bounding box probabilities, and gradient based feature distillation to find the most informative knowledge transfer path [8], [9]. In a recent study on YOLOv8, Karjol and Hanna [4] demonstrated that distilling a YOLOv8-L teacher into a YOLOv8-S student on BDD100K confers robustness to INT8 quantization, with the distilled student losing only 5.6% mAP where the teacher loses 23%. Moreover, the combination of pruning and KD has proved useful in restoring the performance of very light YOLO models under low-light conditions [5], [10].

However, an important question remains open. The analysis in [4] treats daytime and nighttime images as a single pool, and its authors explicitly identify "stratified analysis by lighting conditions (day/night)" as future work. This matters for two reasons. First, an aggregate mAP can conceal a model that performs well by day and collapses at night; Liu et al. [26] show on BDD100K-C that high mAP does not guarantee robustness. Second, a single deployed model must serve both conditions, and training on one illumination condition alone is known to sacrifice the other [25]. Whether KD also transfers robustness to a change in illumination, on the same lightweight student and the same dataset as [4], has therefore not been examined.

In this paper, we address that question. We follow the YOLOv8-L to YOLOv8-S distillation pipeline of [4] on BDD100K, but train on a balanced daytime and nighttime subset and evaluate every model separately on each condition, reporting the relative day-to-night performance drop alongside absolute accuracy. The distillation framework combines multi-level feature alignment at the P3 to P5 neck layers with response distillation on classification logits and bounding-box distributions, and introduces three mechanism choices targeted at nighttime failure modes. The main contributions of this work are as follows:

- A day- and night-disaggregated evaluation of a distilled lightweight detector. Averaged over three independent training runs, the distilled YOLOv8-S improves nighttime mAP@0.5 by +0.0310 (8.1 standard deviations) without reducing daytime accuracy, and reduces the relative day-to-night drop from 8.65% to 4.65%.
- Evidence that the effect is illumination-specific: nighttime gains are nearly three times larger than daytime gains, and the distilled student degrades less under illumination change than its own teacher (7.88%).
- A component-level explanation through a leave-one-out ablation, showing that soft confidence weighting contributes most to nighttime accuracy, channel-wise distillation acts almost exclusively at night, and the cosine distillation schedule prevents the student from favouring daytime at the expense of robustness.
- Validation across protocols: the direction and magnitude of the effect hold under a fixed split with three repeated runs and under stratified five-fold cross-validation, with no inference-time cost added.

---

## 4. Bab II: Related Work

### A. Object Detection in Low-Light Driving Environments

Nighttime object detection is particularly difficult due to the lack of illumination, sensing noise, and motion blur phenomena [1], [2]. Due to technical challenges like camera gain and exposure problems, images usually end up being poorly lit and have backlighting problems, leading to detectors being unable to recognize important classes such as pedestrians [1]. For instance, classical detectors such as RetinaNet and YOLOv3 suffer from serious limitations in terms of correctly detecting road users, where the input images are under-illuminated [1].

In order to overcome these challenges, some methods take advantage of unsupervised domain adaptation (UDA) or image enhancement technologies, such as LIME and SID, to enhance the brightness of the input image prior to performing the detection [2], [3]. The NUDN algorithm and the LightImg data augmentation strategy proposed by Zhang and Lee are able to transform from nighttime features into daytime features, hence alleviating the problem of backlighting and poor lighting conditions [2]. RMD-Net proposed by Jaw et al., on the other hand, leverages GANs to perform feature transferring between high luminance domain and low luminance domain without introducing extra latency during testing [1]. More recently, the Debiased Teacher of Cui et al. [24] adapts a daytime-trained detector to unlabelled nighttime images through self-training, and establishes the practice of splitting BDD100K by its *day* and *night* labels as the standard day-to-night benchmark. However, enhancement methods often increase model complexity and computational cost, making them unsuitable for edge deployment [3], and domain adaptation methods address the setting in which nighttime labels are unavailable and report nighttime accuracy only. Neither reports how much a single deployed model loses when its operating condition changes from day to night.

### B. Knowledge Distillation for Object Detection

KD is a technique for model compression, where a small student network is trained to emulate a large teacher network in order to acquire "dark knowledge" [8], [11]. KD for object detection is broadly divided into feature imitation and logits mimicry methods [9], [11].

Feature imitation techniques ensure alignment between intermediate features through the use of spatial attention, focal distillation, or gradient guidance to enable the student to concentrate on learning foreground objects rather than background elements [8], [12], [13]. Yang et al. suggested FGD that segregates foreground and background information to assist students in concentrating on important pixels [12], while Guo et al. introduced Shared-KD to solve cross-layer feature discrepancies problems [14]. Most of these methods imitate activation maps with a mean-squared-error objective. Channel-wise distillation (CWD), proposed by Shu et al. [22], takes a different view: each channel of the feature map is normalised into a spatial probability map through a softmax, and the student is trained to minimise the KL divergence to the teacher's map. The student thereby learns where the teacher attends rather than the raw magnitude of its activations, which the authors show to be more effective for dense prediction tasks.

Logit mimicry has moved away from the usual method of imitating features. Localization Distillation (LD) was developed by Zheng et al., where it has been shown that distilling the distribution of bounding box probability can transfer information related to localization uncertainty, an aspect often ignored when using the feature imitation paradigm [9]. Furthermore, Decoupled Knowledge Distillation (DKD) was developed by Zhao et al., where knowledge distillation was divided into target and non-target classes for increased efficiency [11]. The CrossKD approach takes this idea further, where efficient learning occurs via forwarding student features directly to the detection head of the teacher to resolve optimization conflicts [15].

A further consideration is that not all anchors deserve equal distillation weight, and that the weight of distillation need not be constant over training. Adaptive Instance Distillation weights each instance according to the teacher's prediction rather than applying a fixed foreground mask [17], and Gradient-guided KD assigns weight according to the influence of each feature on the detection loss [8]. Annealing KD introduced a schedule under which the student is guided by soft targets early in training and is progressively handed to the hard-label objective [23]. The present framework draws on both of these ideas.

### C. Edge AI Deployment and Model Compression Strategies

In addition to several constraints imposed on AI implementation in in-vehicle settings, including memory capacity, latency, and model size, it should be emphasized that the application of INT8 quantization can contribute to the reduction of memory usage up to fourfold and provide efficient integer operations [4], [5]. Still, INT8 quantization can cause very substantial accuracy degradation in case of large models such as YOLOv8-L [4]. As shown by Karjol and Hanna, models that were trained through KD have lower chances of significant accuracy decline, hence better tolerance to INT8 quantization because of precision calibration transfer rather than recall capability [4].

In addition, pruning can decrease multiply-accumulate operations by approximately 40%, however, it often results in accuracy drop [5], [10]. According to the findings of Sun et al., KD plays an important role as a recovery tool that makes it possible to restore accuracy of the pruned lightweight model in low-light conditions while keeping very high efficiency (228 FPS achieved with the help of NVIDIA Jetson device) [5]. KD has likewise been applied to lightweight YOLOv8 for infrared detection on edge devices [21]. These works establish that distillation makes compact detectors more deployable; none of them separates the evaluation by illumination condition.

### D. BDD100K Dataset and Research Positioning

The BDD100K is a varied driver-oriented dataset with 100,000 video clips annotated with heterogeneous tasks such as object detection and lane segmentation [4], [10]. It is perfect for the evaluation of domain generalization and the daytime and nighttime performance degradation due to its large size and various lighting, weather, and time of day situations [1], [10], [24]. As opposed to other datasets that can include synthetic data and clear weather images, BDD100K provides a foundation for the evaluation of real-world low-visibility problems in the context of nighttime environments [4].

This study positions itself at the intersection of four requirements that no single prior work satisfies together: a lightweight edge-deployable detector, a natural day-to-night illumination shift on real driving images, distillation from a separate large teacher, and the relative performance drop as a reported metric. Karjol and Hanna [4] satisfy three of the four and name the fourth as future work; Cui et al. [24] address the illumination shift but on a two-stage detector without a compression objective and without reporting the drop; Cao et al. [21] distil a lightweight YOLOv8 but on infrared imagery. We focus on the question of what an integrated distillation framework with feature alignment at P3 to P5 layers and logit mimicking can achieve in minimizing the day-to-night performance degradation of a YOLOv8-S student.

---

## 5. Bab III: Methodology

### A. Dataset Acquisition and Partitioning

The data used in this work is from the BDD100K benchmark, which was specifically created for heterogeneous multitask learning in self-driving systems [1], [2]. BDD100K is ideal for testing the robustness of the algorithm in conditions of insufficient illumination since it includes real-world video sequences shot under different weather and lighting conditions throughout the day [3]. In order to examine the effect of insufficient illumination on perception performance, the data were first filtered to select only videos taken during the day, the sufficient illumination domain (P_S), and videos taken during the night, the insufficient illumination domain (P_L) [1], [3].

The class *train* was excluded from the study. It contributes only 25 training images and 3 test instances, an imbalance ratio of 791 against the class *car*, so that its average precision is statistically meaningless and distorts the class mean. Nine classes remain: bike, bus, car, motor, person, rider, traffic light, traffic sign, and truck. The entire pipeline was retrained with a nine-class head rather than filtered at evaluation time.

The data were separated into training, validation, and test sets that represent an even distribution across both P_S and P_L. As shown in Table I, 16,000 images, 8,000 daytime and 8,000 nighttime, were assigned to training, with the remaining split equally between validation and testing. Daytime and nighttime images were sampled in equal number at every split so that any performance difference between the two conditions can be attributed to illumination rather than to data quantity.

**TABLE I. BDD100K Dataset Partitioning** *(tidak berubah)*

| Split | Daytime Images (P_S) | Nighttime Images (P_L) | Total Images |
|---|---|---|---|
| Training | 8,000 | 8,000 | 16,000 |
| Validation | 2,000 | 2,000 | 4,000 |
| Testing | 2,000 | 2,000 | 4,000 |
| Total | 12,000 | 12,000 | 24,000 |

### B. Research Pipeline

The research pipeline is structured into three primary stages to facilitate effective knowledge transfer from a high-capacity architecture to a lightweight model. The initial stage involves establishing reference performance for the YOLOv8-L teacher and the YOLOv8-S student trained without teacher guidance, as illustrated in Fig. 1. The second stage distils the frozen teacher into a fresh YOLOv8-S student using the framework depicted in Fig. 2. The final stage evaluates every model separately on the daytime and nighttime test sets under the protocol of Section III-G, in order to quantify performance drops from day to night without the overhead cost of augmentation subnetworks.

**Fig. 1.** Workflow for data preparation and reference training. *(File: Fig1_pipeline.png)*

### C. Preprocessing and Specialized Augmentation

In order to boost the model's robustness to the technical difficulties associated with nighttime driving (noise, motion blur, etc.), a tailored augmentation technique is used [2], [8], [16]. Spatial transformations include letterbox resizing to 640 × 640, horizontal flipping, and affine transformation. Photometric augmentations include Color Jitter and Blur operations. Both augmentations simulate low-luminosity and blurry image regions. Ultimately, these transformations encourage the model to learn semantic features invariant to light changes [1].

### D. Training Configuration

This methodology uses the teacher and student model-based method of model compression [17], [18]. In our research, YOLOv8-L (43.7 million parameters) was chosen as the teacher owing to its superior representation power. YOLOv8-S (11.2 million parameters) acts as the student, a 3.9× parameter reduction. Both models are initialised from COCO-pretrained weights. The 80-class detection head is replaced by a 9-class head whose classification bias is initialised with a focal-loss prior, so that the initial objectness probability is low and early training is stable, after which all layers are fine-tuned. The optimization process involves the AdamW optimization technique with linear warm-up followed by cosine annealing, mixed precision, and gradient clipping. Early stopping monitors the mean of the daytime and nighttime validation loss. All the hyperparameters used in the process are listed in Table II.

**TABLE II. Hyperparameters and Training Configurations** *(ganti seluruh tabel lama)*

| Parameter | Value |
|---|---|
| Input resolution | 640 × 640 |
| Optimizer | AdamW, weight decay 0.05 |
| Initial learning rate | 1 × 10⁻⁴ |
| Learning rate schedule | Linear warm-up 3 epochs, then cosine annealing |
| Batch size | 16 |
| Teacher FT / Student Baseline / KD epochs | 30 / 50 / 50 |
| Early stopping | Patience 14 on mean validation loss |
| Task loss weight α | 0.7 |
| Response distillation weight β(t) | Cosine decay from 0.35 to 0.10 |
| Feature distillation weight γ | 0.03 |
| Classification temperature T | 4.0 |
| DFL temperature T_dfl | 2.0 |
| CWD temperature τ | 4.0 |
| Feature layers | 15, 18, 21 (neck P3, P4, P5) |
| Anchor weight | w = √(max teacher class confidence) |

### E. Integrated Knowledge Distillation Framework

The network applies a divide and conquer strategy for the transfer of semantic and localization information [17], [18], along the three paths shown in Fig. 2. All paths are active only during training; at inference the student runs unmodified.

**Feature Distillation:** Feature maps are extracted from the neck blocks at the levels of P3, P4, and P5 [18], [19], corresponding to layers 15, 18, and 21. Because the student has fewer channels than the teacher at each level, a 1 × 1 convolutional adapter without normalisation or activation projects the student features to the teacher's channel dimension. The projected student map and the teacher map are then compared with channel-wise distillation [22]: each channel is flattened over its spatial positions, passed through a softmax with temperature τ, and the KL divergence between the resulting student and teacher distributions is averaged over channels and levels. This choice is deliberate for the illumination setting. A change in lighting alters the magnitude of activations, so that a mean-squared-error objective would force the student to reproduce values that are themselves illumination-dependent, whereas the spatial softmax discards magnitude and transfers only the teacher's attention pattern [22].

**Response Distillation:** Kullback-Leibler (KL) Divergence is calculated between the classification logits and the DFL output values [18]. The classification term is softened by temperature T and scaled by T² [11]; the DFL term uses temperature T_dfl. DFL is used here because it offers improved localization in blurry conditions during night time compared to bounding box regression [20].

**Soft Confidence Weighting:** Rather than discarding anchors whose teacher confidence falls below a threshold, every anchor contributes to both response terms with a continuous weight w = √c_T, where c_T is the teacher's maximum class probability at that anchor, and the weighted sum is normalised by Σw [8], [17]. The square root raises the contribution of mid-confidence anchors. This matters at night, where the objects the framework aims to recover are precisely those on which the teacher is only moderately confident; a hard threshold would remove them from supervision.

**Cosine Distillation Schedule:** The response weight β is not constant. Following the annealing principle [23], it decays along a cosine curve from 0.35 at the first epoch to 0.10 at the last, so that the student is guided strongly by the teacher while its own representation is immature, and is increasingly governed by ground truth late in training. This also limits the influence of the teacher's own nighttime errors on the final student.

**Fig. 2.** Knowledge distillation framework integrating multi-level feature alignment and response mimicking. Shaded components denote the three mechanism choices of this work. *(File: Fig2_kd_framework.png)*

### F. Mathematical Formulation of the Objective Function

The overall training goal (L_total) for the distilled learner is an amalgamated value that consists of task supervision and teaching supervision:

$$L_{total} = \alpha\, L_{task} + \beta(t)\,\big(L_{cls} + L_{dfl}\big) + \gamma\, L_{CWD} \tag{1}$$

Where:

- L_task is the standard YOLOv8 loss derived from ground-truth labels.
- L_cls and L_dfl are the logit-based distillation losses on class scores and bounding-box distributions, weighted per anchor:

$$L_{cls} = \frac{T^2}{\sum_i w_i}\sum_i w_i\, \mathrm{KL}\!\left(\sigma\!\left(\tfrac{z^T_i}{T}\right)\,\Big\|\,\sigma\!\left(\tfrac{z^S_i}{T}\right)\right), \qquad w_i = \sqrt{\max_c \sigma(z^T_{i,c})} \tag{2}$$

$$L_{dfl} = \frac{T_{dfl}^2}{\sum_i w_i}\sum_i w_i \cdot \frac{1}{4}\sum_{k=1}^{4} \mathrm{KL}\!\left(\sigma\!\left(\tfrac{d^T_{i,k}}{T_{dfl}}\right)\,\Big\|\,\sigma\!\left(\tfrac{d^S_{i,k}}{T_{dfl}}\right)\right) \tag{3}$$

- L_CWD is the channel-wise feature distillation loss [22] over levels 𝓛 = {15, 18, 21}:

$$L_{CWD} = \frac{1}{|\mathcal{L}|}\sum_{l\in\mathcal{L}} \frac{\tau^2}{C_l}\sum_{c=1}^{C_l} \mathrm{KL}\!\left(\sigma\!\left(\tfrac{F^T_{l,c}}{\tau}\right)\,\Big\|\,\sigma\!\left(\tfrac{\phi_l(F^S_l)_c}{\tau}\right)\right) \tag{4}$$

- β(t) follows a cosine schedule over epochs t = 0, …, E − 1:

$$\beta(t) = \beta_{end} + \tfrac{1}{2}\,(\beta_{start} - \beta_{end})\left(1 + \cos\frac{\pi\, t}{E - 1}\right) \tag{5}$$

Here i indexes anchors, z are class logits, d_{i,k} are the DFL bin logits for side k of anchor i, F_l are feature maps flattened over spatial positions, φ_l is the 1 × 1 adapter, C_l the channel count at level l, and σ the softmax. The hyperparameters are α = 0.7, γ = 0.03, β_start = 0.35, and β_end = 0.10. The teacher is frozen throughout, and gradients flow to the student and the adapters only.

### G. Evaluation Metrics and Protocol

The performance of each model is measured based on well-known parameters such as Precision, Recall, mAP@0.5, and mAP@0.5:0.95. Separate evaluations are performed on the two test datasets consisting of day time and night time scenes to determine the decrease in performance between day and night datasets [21], [24]. The relative performance degradation across domains is formulated as:

$$\text{Relative Drop (\%)} = \frac{\mathrm{mAP}_{50,\text{Day}} - \mathrm{mAP}_{50,\text{Night}}}{\mathrm{mAP}_{50,\text{Day}}} \times 100 \tag{6}$$

Normalising by daytime performance allows models with different absolute accuracy, such as the teacher and the student, to be compared on the same footing. Four complementary protocols are used, as illustrated in Fig. 3:

1. **Fixed split with repeated training.** The distilled student is trained three times with identical hyperparameters under non-deterministic CUDA execution, and results are reported as mean ± standard deviation. Improvements are expressed in multiples of this standard deviation.
2. **Stratified five-fold cross-validation.** Folds are drawn with multilabel stratification over the *timeofday* attribute and per-class instance counts, so that every fold preserves the day-night balance and the class distribution.
3. **Leave-one-out ablation.** Each of the three mechanism choices in Section III-E is disabled in turn, with all other settings fixed, to measure its marginal contribution in each condition.
4. **Per-class analysis.** AP@0.5 is reported per class and per condition to identify where gains concentrate.

**Fig. 3.** Evaluation protocol. Every trained model is evaluated on the daytime and nighttime test sets separately, and the relative drop is computed from the pair. *(File: Fig3_evaluation_protocol.png)*

---

## 6. Bab IV: Experimental Results and Discussion

In this section, we evaluate the performance of Knowledge Distillation (KD) in mitigating the daytime-to-nighttime performance drop of lightweight object detectors. We compare the fine-tuned teacher model (YOLOv8-L, 43.7M parameters), the direct student baseline (YOLOv8-S, 11.2M parameters), and the distilled student (Student KD) across both fixed-split and five-fold cross-validation regimes on the BDD100K driving dataset.

### A. Primary Performance Comparison (Fixed-Split Analysis)

To isolate the impact of our integrated feature- and response-based knowledge distillation framework, all models were evaluated under identical input resolutions (640 × 640) on fixed daytime (P_S) and nighttime (P_L) test subsets (2,000 images per domain). Table III summarizes the primary detection metrics, performance gaps, and the relative drop defined in Eq. (6).

**TABLE III. Primary Detection Performance on BDD100K Day/Night Splits**

| Model | Params | mAP50 (Day) | mAP50 (Night) | mAP50-95 (Day) | mAP50-95 (Night) | Gap (Day − Night) | Rel. Drop (%) |
|---|---|---|---|---|---|---|---|
| Teacher YOLOv8-L FT | 43.7M | 0.5597 | 0.5156 | 0.3261 | 0.2819 | 0.0441 | 7.88% |
| Student Baseline (YOLOv8-S) | 11.2M | 0.4893 | 0.4470 | 0.2771 | 0.2420 | 0.0423 | 8.65% |
| Student KD (mean over 3 runs) | 11.2M | **0.5013** | **0.4780** | **0.2834** | **0.2586** | **0.0233** | **4.65%** |

As detailed in Table III, the Student Baseline suffers an 8.65% relative performance drop when transitioning from daytime (0.4893 mAP50) to nighttime (0.4470 mAP50), illustrating the vulnerability of lightweight architectures to low-luminance visual noise, reduced contrast, and feature degradation. In contrast, Student KD achieves significant accuracy gains over the baseline in both illumination domains: +0.0120 mAP50 in daytime and +0.0310 mAP50 in nighttime conditions.

Crucially, the absolute daytime-to-nighttime performance gap of Student KD (0.0233) is nearly half that of the Student Baseline (0.0423) and substantially lower than that of the Teacher model (0.0441). In relative terms, Student KD degrades by only 4.65% under nighttime conditions compared to 8.65% for the baseline and 7.88% for the teacher. It should be emphasized that, in relative terms, the teacher and the baseline are practically indistinguishable, so that model capacity alone does not determine illumination robustness; what determines it is the manner in which the student is trained. Despite offering a ∼3.9× parameter compression relative to the teacher, Student KD retains 89.6% of the teacher's daytime capacity and 92.7% of its nighttime performance, confirming that multi-level knowledge distillation effectively transfers domain-invariant representations that fortify compact student networks against illumination shifts.

**Fig. 4.** Comparison of mAP50 under daytime and nighttime conditions for the Teacher, the Student Baseline, and Student KD. Bars show mAP50 under each condition; the number above each pair is the absolute gap, with Student KD exhibiting the smallest gap (0.0233). *(ganti caption lama yang menyebut simbol)*

**Fig. 5.** Relative mAP50 performance drop from daytime to nighttime conditions across models. Student KD achieves a relative drop of 4.65% on fixed-split evaluation, nearly halving the 8.65% drop of the Student Baseline.

### B. Training Consistency and Statistical Significance

To verify that the performance gains of Student KD are statistically robust rather than artifacts of stochastic weight initialization or data augmentation noise, we conducted three independent training runs with identical hyperparameter configurations under unseeded CUDA nondeterminism. Table IV presents the run-to-run consistency results.

**TABLE IV. 3-Run Training Consistency of Student KD (Fixed-Split)**

| Run | mAP50 (Day) | mAP50 (Night) | mAP50-95 (Day) | mAP50-95 (Night) |
|---|---|---|---|---|
| Run 1 | 0.5014 | 0.4740 | 0.2836 | 0.2543 |
| Run 2 | 0.4989 | 0.4816 | 0.2838 | 0.2602 |
| Run 3 | 0.5036 | 0.4783 | 0.2828 | 0.2612 |
| **Mean ± SD** | **0.5013 ± 0.0024** | **0.4780 ± 0.0038** | **0.2834 ± 0.0005** | **0.2586 ± 0.0037** |

The empirical standard deviations across the three runs were exceptionally small: ±0.0024 for daytime mAP50 and ±0.0038 for nighttime mAP50. Comparing these variance margins against the performance gains over the Student Baseline (+0.0120 daytime and +0.0310 nighttime) reveals that the distillation gains correspond to 5.1 × SD in daytime and 8.1 × SD in nighttime environments. This confirms that the observed improvements and illumination resilience are statistically significant and attributable to the distillation mechanism. The relative drop is likewise consistent across runs (5.46%, 3.47%, and 5.02%), every value lying well below the 8.65% of the baseline.

**Fig. 6.** Results of three independent training runs for Student KD under fixed-split evaluation. The standard deviation across runs (±0.0024 daytime, ±0.0038 nighttime) is significantly smaller than the performance margin over the Student Baseline.

### C. Leave-One-Out Ablation Study

To evaluate the individual contributions of the three mechanism choices of Section III-E, namely soft confidence weighting, the cosine distillation schedule, and channel-wise distillation (CWD), we conducted a leave-one-out ablation study. Each variant disables one specific technique while keeping all other components active. Table V details the performance impact relative to the full Student KD.

> **Catatan penting:** baris *No CWD* di bawah memakai angka dari naskah kalian (0,5010 / 0,4744). Output notebook yang kalian kirim ke saya sebelumnya menunjukkan **0,5003 / 0,4682**. Kedua angka ini menghasilkan kesimpulan yang berbeda. Saya tulis kedua versi butir ketiga di bawah. Pilih satu setelah kalian cek file evaluasinya, lalu hapus yang lain.

**TABLE V. Leave-One-Out Ablation Analysis of Student KD Components**

| Ablated Variant | mAP50 (Day) | Δ vs. Student KD | mAP50 (Night) | Δ vs. Student KD | Impact |
|---|---|---|---|---|---|
| Full Student KD (Mean) | 0.5013 | — | 0.4780 | — | Reference |
| No Soft Weighting (→ Hard Mask) | 0.4973 | −0.0040 (−1.7 × SD) | 0.4628 | −0.0152 (−4.0 × SD) | Critical for Nighttime |
| No Cosine Schedule (→ Constant β) | 0.5107 | +0.0094 (+3.9 × SD) | 0.4716 | −0.0064 (−1.7 × SD) | Trade-off Balancing |
| No CWD (→ MSE Feature Loss) | 0.5010 `[cek]` | −0.0003 `[cek]` | 0.4744 `[cek]` | −0.0036 `[cek]` | `[cek]` |

The ablation findings yield the following architectural insights:

- **Soft Confidence Weighting is the primary nighttime contributor.** Disabling soft confidence weighting, replacing it with hard foreground masking, causes a severe degradation of −0.0152 mAP50 (−4.0 × SD) in nighttime conditions. Soft confidence weighting allows subtle, low-contrast dark-domain signals from the teacher to guide the student rather than abruptly filtering out low-confidence bounding regions.
- **The cosine schedule balances the domain trade-off.** Utilizing a constant distillation weight increases daytime performance (+0.0094) but degrades nighttime accuracy (−0.0064), and widens the relative drop from 4.65% to 7.66%, close to the baseline's 8.65%. The dynamic cosine schedule (β: 0.35 → 0.10) successfully regulates supervision intensity over epochs, establishing an optimal balance across lighting domains. This variant also illustrates why aggregate mAP alone is an inadequate selection criterion: it attains the highest daytime mAP50 of all configurations while being the least robust.
- **Channel-wise distillation effect, versi A (kalau angka naskah 0,5010 / 0,4744 yang benar):** Replacing CWD (τ = 4) with standard MSE feature loss yielded negligible variations (−0.0003 daytime, −0.0036 nighttime), both remaining within the noise band (≤ 0.9 × SD). Its contribution is therefore secondary to the two response-side mechanisms under the present configuration.
- **Channel-wise distillation effect, versi B (kalau angka notebook 0,5003 / 0,4682 yang benar):** Replacing CWD (τ = 4) with standard MSE feature loss leaves daytime accuracy practically unchanged (−0.0010, within noise) but reduces nighttime mAP50 by −0.0098 (−2.6 × SD), widening the relative drop to 6.42%. CWD is thus the most illumination-specific of the three components: it contributes little where activations are stable and substantially where lighting alters their magnitude, in agreement with the motivation of Section III-E.

If the three components addressed the same failure mode, removing any one of them would produce a similar signature. Instead, each removal yields a different one: soft weighting removal depresses nighttime most, the schedule removal raises daytime while widening the gap, and CWD removal `[cek: versi A atau B]`. This supports the claim that the components play distinct rather than redundant roles.

**Fig. 7.** Leave-one-out ablation study illustrating the impact of disabling each component on daytime and nighttime mAP50 relative to the full Student KD.

### D. Five-Fold Cross-Validation Analysis

To confirm generalization beyond a single fixed dataset split, Student KD was evaluated using a stratified five-fold cross-validation scheme. The test set for each fold was drawn from a pool of `[cek: 24,000 gambar Tabel I, atau 27,600 dari direktori validation dan test resmi; pastikan dan jelaskan]`, organized via stratified multi-label sampling over the *timeofday* attribute and per-class instance counts. Table VI presents the cross-validation metrics across all five folds.

**TABLE VI. Five-Fold Cross-Validation Results for Student KD**

| Fold | mAP50 (Day) | mAP50 (Night) | mAP50-95 (Day) | mAP50-95 (Night) | Rel. Drop (%) |
|---|---|---|---|---|---|
| Fold 0 | 0.5174 | 0.4814 | 0.2968 | 0.2612 | 6.96% |
| Fold 1 | 0.5249 | 0.4916 | 0.2968 | 0.2630 | 6.34% |
| Fold 2 | 0.5076 | 0.4800 | 0.2896 | 0.2644 | 5.44% |
| Fold 3 | 0.5153 | 0.4948 | 0.2927 | 0.2738 | 3.98% |
| Fold 4 | 0.5073 | 0.4958 | 0.2903 | 0.2648 | 2.27% |
| **Mean ± SD** | **0.5145 ± 0.0074** | **0.4887 ± 0.0075** | **0.2932 ± 0.0034** | **0.2654 ± 0.0049** | **5.00% ± 1.89%** |

The five-fold cross-validation demonstrates high consistency:

- **Daytime performance superior across all folds.** In 100% of folds, daytime mAP50 consistently outpaces nighttime mAP50, validating the directional stability of the illumination gap across dataset partitions.
- **Convergence of relative performance drop.** The mean relative drop across the five folds is 5.00% ± 1.89%, converging closely with the 4.65% drop observed in the three-run fixed-split evaluation. Both independent validation schemes confirm that Student KD achieves significantly lower relative degradation than the Student Baseline (8.65%).

**Fig. 8.** Five-fold cross-validation results for Student KD on unseen dataset splits. (Left) Daytime vs. nighttime mAP50 per fold. (Right) Relative performance drop per fold, remaining consistently below the Baseline drop of 8.65%.

### E. Per-Class Nighttime Performance Analysis (VRU Focus)

To assess the practical safety impact of knowledge distillation in nighttime driving scenarios, we examined category-specific Average Precision (AP50) across all 9 object classes under nighttime conditions. Table VII compares Student KD against the Student Baseline.

**TABLE VII. Nighttime AP50 Gain per Class (Student KD vs. Baseline)**

| Category | Object Class | Baseline | Student KD | Gain (Δ) | Safety Target |
|---|---|---|---|---|---|
| VRU | Bicycle (Bike) | 0.2479 | 0.3561 | +0.1082 | Vulnerable Road User |
| VRU | Rider | 0.3143 | 0.4198 | +0.1055 | Vulnerable Road User |
| VRU | Motorcycle (Motor) | 0.2629 | 0.2798 | +0.0169 | Vulnerable Road User |
| General | Bus | 0.3984 | 0.4137 | +0.0153 | Large Vehicle |
| Traffic | Traffic Light | 0.5164 | 0.5285 | +0.0121 | Infrastructure |
| VRU | Person (Pedestrian) | 0.4885 | 0.4978 | +0.0093 | Vulnerable Road User |
| Traffic | Traffic Sign | 0.6053 | 0.6122 | +0.0069 | Infrastructure |
| General | Truck | 0.4692 | 0.4721 | +0.0029 | Large Vehicle |
| General | Car | 0.7198 | 0.7216 | +0.0018 | Dominant Vehicle |

Student KD outperforms the baseline across all 9 categories under nighttime conditions, and likewise across all 9 categories under daytime conditions, so that no class regresses in either domain. Crucially, the most substantial performance jumps occur in Vulnerable Road User (VRU) classes: Bicycle (+0.1082 AP50) and Rider (+0.1055 AP50). In dark driving environments, small and non-rigid VRUs suffer heavily from low contrast and motion blur. Distillation from the high-capacity teacher transfers essential structural priors and confidence calibration that prevent compact student models from missing underrepresented road users at night.

**Fig. 9.** Per-class AP50 gain under nighttime conditions for Student KD over the Student Baseline. The largest improvements occur in VRU categories, notably bike (+0.1082) and rider (+0.1055).

### F. Discussion and Methodological Limitations

- **Precision calibration vs. raw capacity.** Our findings align with Karjol and Hanna [4], confirming that knowledge distillation primarily transfers confidence calibration and structural feature alignment rather than expanding the student's raw representational capacity. Where [4] observed this transfer under a numerical perturbation, INT8 quantization, the present work observes the same form of transfer under an input-domain perturbation, the change from daytime to nighttime illumination.
- **Evaluation metric divergence.** Raw mAP scores from the five-fold cross-validation (mean mAP50 = 0.5145 day / 0.4887 night) differ slightly from the fixed-split test set due to differing image distributions. Consequently, relative percentage drops (4.65% fixed-split vs. 5.00% k-fold) represent the valid cross-schema comparison metric.
- **Single-run baseline and ablation constraints.** While Student KD was validated over 3 fixed runs and 5 folds, the teacher, the baseline, and the ablation variants were evaluated on single runs. The standard deviations reported therefore characterise the distilled student only. Nevertheless, the nighttime gain (+0.0310) exceeds the multi-run variance by 8.1 × SD, confirming statistical validity. The ranking of ablation components by nighttime accuracy is robust; their ranking by relative drop, whose across-run standard deviation is 0.0053, remains indicative.
- **Scope of the illumination axis.** The present study isolates the day-to-night axis, which carries the highest concentration of safety risk and the cleanest label in BDD100K. Weather and dawn/dusk conditions are available in the same dataset and constitute a natural extension.

---

## 7. Bab V: Conclusion and Future Work

### A. Conclusion

This study evaluated the effectiveness of Knowledge Distillation (KD) in enhancing the low-light robustness of lightweight object detection models for autonomous driving applications. By distilling multi-level intermediate representations (at layers P3 to P5) and output response logits from a high-capacity YOLOv8-L teacher into a compact YOLOv8-S student on the BDD100K dataset, we achieved the following core findings:

- **Nighttime accuracy without daytime regression.** Student KD raised nighttime mAP50 from 0.4470 to 0.4780 (+0.0310, 8.1 × SD) while also raising daytime mAP50 (+0.0120), so that the improvement is not obtained by trading one condition for the other.
- **Illumination gap reduction.** Student KD reduced the daytime-to-nighttime performance drop to 4.65% on fixed-split evaluation and 5.00% on five-fold cross-validation, nearly halving the 8.65% degradation observed in the Student Baseline and falling below the teacher's own 7.88%. This addresses the stratified day/night analysis identified as future work in [4].
- **Component roles.** The leave-one-out ablation showed that the three mechanism choices play distinct roles: soft confidence weighting is the primary contributor to nighttime accuracy, the cosine distillation schedule prevents the student from favouring daytime at the expense of robustness, and channel-wise distillation `[cek: versi A atau B dari IV.C]`.
- **Model compression and efficiency.** Student KD retained 89.6% of daytime and 92.7% of nighttime teacher accuracy while achieving a ∼3.9× parameter compression (11.2M vs. 43.7M parameters), confirming suitability for real-time edge deployment without requiring computational preprocessing subnets.
- **Safety-critical VRU enhancement.** Distillation yielded nighttime accuracy gains across all 9 object categories, with the most dramatic improvements observed in Vulnerable Road User (VRU) classes, including Bicycle (+0.1082 AP50) and Rider (+0.1055 AP50).

### B. Future Work

Building upon these findings, future research directions include:

- **Edge hardware deployment and thermal profiling.** Deploying and benchmarking the distilled YOLOv8-S student on embedded automotive hardware (e.g., NVIDIA Jetson Orin) with TensorRT INT8 post-training quantization to measure real-world FPS, latency, power consumption, and thermal stability.
- **Quantization-aware distillation.** Integrating knowledge distillation directly into Quantization-Aware Training (QAT) to preserve low-light precision under low-precision integer arithmetic, combining the quantization robustness of [4] with the illumination robustness reported here.
- **Multi-run baseline and ablation.** Repeating the teacher, the baseline, and each ablation variant over three runs so that every comparison in this study carries a standard deviation on both sides.
- **Cross-dataset and adverse weather generalization.** Extending the validation framework to additional driving benchmarks (e.g., nuScenes, Waymo, KITTI) and evaluating performance under adverse weather conditions such as heavy rain, fog, and snow, which are available in BDD100K and require no retraining.

---

## 8. Referensi

### 8.1 Yang dipertahankan, dengan perbaikan

Semua referensi [1] sampai [21] tetap dipakai. Empat entri harus diperbaiki:

**[4]** A. Karjol and D. M. Hanna, "Edge AI for automotive vulnerable road user safety: Deployable detection via knowledge distillation," *arXiv preprint* arXiv:2604.26857, Apr. 2026.
*(Bukan CVPR 2024. Cek ulang statusnya menjelang finalisasi.)*

**[16]** J. Liang et al., "Exploring inconsistent knowledge distillation for object detection with data augmentation," in *Proc. ACM Int. Conf. Multimedia*, 2023.
*(Bukan CVPR 2024.)*

**[19]** L. Yao, F. Liu, C. Zhang, Z. Ou, and T. Wu, "Domain-invariant progressive knowledge distillation for UAV-based object detection," *IEEE Geosci. Remote Sens. Lett.*, vol. 21, 2024.
*(Bukan CVPR 2024.)*

**[21]** X. Cao, Y. Hu, and H. Zhang, "LKD-YOLOv8: A lightweight knowledge distillation-based method for infrared object detection," *Sensors*, vol. 25, no. 13, p. 4054, 2025.
*(Penulis pertama Xiancheng Cao.)*

### 8.2 Lima yang ditambahkan

**[22]** C. Shu, Y. Liu, J. Gao, Z. Yan, and C. Shen, "Channel-wise knowledge distillation for dense prediction," in *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, 2021, pp. 5311–5320.
*Dipakai di: II.B, III.E, III.F. Atribusi CWD, wajib.*

**[23]** A. Jafari, M. Rezagholizadeh, P. Sharma, and A. Ghodsi, "Annealing knowledge distillation," in *Proc. Conf. European Chapter Assoc. Computational Linguistics (EACL)*, 2021, pp. 2493–2504.
*Dipakai di: II.B, III.E. Landasan jadwal beta.*

**[24]** Y. Cui, L. Li, H. Yin, Y. Gao, Y. Sun, and C. Yan, "Debiased teacher for day-to-night domain adaptive object detection," in *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, 2025.
*Dipakai di: I, II.A, II.D, III.G. Pernyataan masalah day-to-night dan protokol pemisahan BDD100K. Cek tahun di halaman CVF.*

**[25]** E. Jones et al., "A study on data selection for object detection in various lighting conditions for autonomous vehicles," *J. Imaging*, vol. 10, no. 7, p. 153, 2024.
*Dipakai di: I. Bukti bahwa melatih satu kondisi saja mengorbankan kondisi lain.*

**[26]** J. Liu, Z. Wang, L. Ma, et al., "Benchmarking object detection robustness against real-world corruptions," *Int. J. Comput. Vis.*, 2024.
*Dipakai di: I. Bukti bahwa mAP tinggi tidak menjamin ketahanan.*

### 8.3 Yang sengaja tidak ditambahkan

Supaya kalian tahu apa yang dikorbankan karena batas lima:

- Soft weighting cukup disandarkan ke [17] AID dan [8] GKD yang sudah ada. Tidak perlu tambahan.
- Atribusi konsep KD dan temperature memakai [8], [11] yang sudah ada, bukan Hinton 2015.
- Dataset BDD100K masih disitasi lewat [4], [10] seperti draft. Kalau reviewer minta atribusi asli, tambahkan Yu et al., CVPR 2020.
- Urgensi regulasi FMVSS 127 dan statistik 76,3 persen tidak saya masukkan ke naskah karena butuh dua sitasi tambahan. Kalau mau dipakai di Bab I, tambahkan dua entri non-jurnal itu.
