# Real-Time Driver Distraction Detection: A 2D vs. 3D Vision Benchmark

**CS-412: Machine Learning Project**  
*Team Members:*
- **Doan Nhat Huy Pham** — 3D Spatio-Temporal Network (SlowFast Networks)
- **Ky An Dang** — 2D Spatial & Vision Transformer (ViT-DD, 2D CNN Baseline)

---

## 📌 Project Overview

Driver inattention (mobile phone usage, drowsiness, eating, reaching behind) is a leading cause of traffic accidents worldwide. Developing an intelligent, real-time in-cabin **Driver Monitoring System (DMS)** requires balancing three technical challenges:
1. **Spatial Complexity:** Distinguishing fine-grained hand-eye-object interactions (e.g., holding a phone vs. holding the steering wheel).
2. **Temporal Dynamics:** Disambiguating dynamic actions from static postures (e.g., a momentary glance at the rear-view mirror vs. prolonged distraction).
3. **Latency vs. Accuracy Trade-Off:** Achieving real-time inference (>30 FPS) within constrained in-cabin edge compute budgets.

This project delivers a head-to-head empirical benchmark between **Frame-Level 2D Spatial Architectures** and **3D Spatio-Temporal Video Architectures** on the standardized 10-class driver distraction taxonomy:
- **3D Spatio-Temporal:** **SlowFast Networks** (Feichtenhofer et al., ICCV 2019)
- **2D Spatial / Transformer:** **ViT-DD** (Ma & Wang, NeurIPS 2022 ML4AD Workshop) & 2D CNN baseline (Baheti et al., CVPRW 2018)

---

## 📂 Repository Structure

```
cs412-project/
├── data/
│   └── README.md                    # Dataset layout & download instructions (State Farm, AUCDD, Kinetics)
├── docs/
│   ├── notes/
│   │   ├── Personal Reading Report_ ViT-DD (Ky An Dang).pdf
│   │   └── SlowFast Reading Report.md.pdf
│   └── papers/                      # Reference research papers (SlowFast, ViT-DD, Baheti et al., AUCDD)
├── project_proposal/
│   └── Project Proposal - Doan Nhat Huy Pham, Ky An Dang (1).pdf
├── result/
│   ├── slowfast_summary.md          # Kinetics-400 dry-run benchmark report (SlowFast R50 8x8)
│   ├── slowfast_dryrun_preds.csv    # Per-video prediction logs and error breakdown
│   └── vit_dd_summary.md            # Dry-run benchmark report (ViT-DD 12-layer multi-task)
├── scripts/
│   └── run_vit_dd_dry_run.py        # ViT-DD profiling & synthetic benchmark script
├── src/
│   └── models/
│       ├── slowfast.py              # PySlowFast R50 wrapper adapted for 10-class DMS
│       └── vit_dd.py                # Dual-modality Multi-Task Vision Transformer (ViT-DD)
├── tests/
│   └── test_vit_dd.py               # Unit test suite for ViT-DD modules & loss functions
├── requirements.txt                 # Core environment dependencies
└── README.md
```

---

## 🔬 Architectures Under Benchmark

### 1. SlowFast Networks (3D Spatio-Temporal Baseline)
- **Concept:** Models visual inputs through two parallel pathways operating at different temporal rates:
  - **Slow Pathway:** Low frame rate (large temporal stride $\tau=16$), large channel capacity ($C$), captures rich spatial details (~80% FLOPs).
  - **Fast Pathway:** High temporal resolution ($\alpha=8\times$ frame refresh rate), lightweight channel capacity ($\beta=1/8$), captures rapid motion patterns (~20% FLOPs).
  - **Lateral Connections:** Unidirectional Fast-to-Slow feature fusion using Time-to-Channel (T2C) reshaping ($\alpha\beta=1$).
- **Adaptation:** Head projection modified from 400 classes (Kinetics) to **10 driver distraction classes** (C0–C9).

### 2. ViT-DD (Multi-Task Vision Transformer Baseline)
- **Concept:** Dual-modality tokenization with global Multi-Head Self-Attention (MHSA):
  - **Visual Tokenization:** Global cabin view ($224 \times 224 \rightarrow 196$ patches) + localized face crop ($32 \times 32 \rightarrow 4$ patches) = 200 visual tokens.
  - **Multi-Task Heads:** Prepends a Distraction Token (10 classes) and an Emotion Token (8 classes from AffectNet-7 teacher), yielding a 202-token sequence.
  - **Attention-Only Parameter Fine-Tuning:** Freezes ~67.2% of backbone parameters (MLP and projection layers), fine-tuning only MHSA blocks to prevent overfitting on driving datasets.

---

## 📊 Initial Dry-Run Profiling Summary

Both architectures were profiled on identical hardware (**NVIDIA GeForce RTX 5070 Ti Laptop GPU 12GB**, WSL2 Ubuntu, PyTorch 2.11):

| Metric | SlowFast R50 (8x8) | ViT-DD (Base-16) |
|---|---|---|
| **Input Modality** | 32-frame video clip ($[B, 3, 32, 256, 256]$) | Cabin image ($224^2$) + Face crop ($32^2$) |
| **Total Parameters** | 34.57 M | 86.41 M |
| **Trainable Parameters** | 34.57 M (or head-only) | 28.36 M (Attention-Only fine-tuning) |
| **GFLOPs** | 66.07 GFLOPs / view (~1,982 GFLOPs / 30-view video) | 36.24 GFLOPs / sample |
| **Inference Latency** | Median ~52 ms / clip (~0.418 s / iter) | Median ~7.90 ms / sample |
| **Throughput** | ~5.2 clips/s (CPU video decoding bottleneck) | ~116.8 samples/s |
| **Peak GPU VRAM** | 1.75 GB | 0.40 GB |
| **Target Evaluation** | Continuous temporal action transitions | Fine-grained spatial posture & facial emotion |

---

## 🗃️ Datasets

1. **State Farm Distracted Driver Detection** ([Kaggle](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data)):
   - 10 classes (`c0` Safe driving, `c1` Texting right, `c2` Talking phone right, `c3` Texting left, `c4` Talking phone left, `c5` Operating radio, `c6` Drinking, `c7` Reaching behind, `c8` Hair/makeup, `c9` Talking to passenger).
   - Structured by driver ID (`driver_imgs_list.csv`) for realistic split-by-driver evaluation.
2. **AUC Distracted Driver Dataset (AUCDD)** ([Abouelnaga et al., 2017](https://arxiv.org/abs/1706.09498)):
   - In-cabin videos and frame sequences recorded across diverse drivers, camera angles, and lighting conditions.
3. **Kinetics-400** ([CVDF Mirror](https://github.com/cvdfoundation/kinetics-dataset)):
   - Used for pre-trained weights and pipeline sanity verification.

*See [`data/README.md`](data/README.md) for data organization and download guidelines.*

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install required dependencies:
```bash
git clone https://github.com/Huypham310/cs412-project.git
cd cs412-project
pip install -r requirements.txt
```

> For PySlowFast, ensure `facebookresearch/SlowFast` is installed or set `export SLOWFAST_ROOT=/path/to/SlowFast`.

### 2. Verify SlowFast Model Builder
```bash
python src/models/slowfast.py --num-classes 10
```

### 3. Run ViT-DD Synthetic Profiling & Dry-Run
```bash
python scripts/run_vit_dd_dry_run.py --batch-size 8 --num-samples 50
```

### 4. Run Test Suite
```bash
pytest tests/ -v
```

---

## 📚 References

1. Feichtenhofer, C., Fan, H., Malik, J., & He, K. (2019). **SlowFast Networks for Video Recognition**. *ICCV 2019*. [arXiv:1812.03982](https://arxiv.org/abs/1812.03982).
2. Ma, Y., & Wang, Z. (2022). **ViT-DD: Multi-Task Vision Transformer for Semi-Supervised Driver Distraction Detection**. *NeurIPS 2022 ML4AD Workshop*. [arXiv:2209.09178](https://arxiv.org/abs/2209.09178).
3. Baheti, B., Gajre, S., & Talbar, S. (2018). **Detection of Distracted Driver Using Convolutional Neural Network**. *CVPR Workshops 2018*.
4. Abouelnaga, Y., Eraqi, H. M., & Moustafa, M. N. (2018). **Real-time Distracted Driver Posture Classification**. *CVPR Workshops 2018*. [arXiv:1706.09498](https://arxiv.org/abs/1706.09498).
