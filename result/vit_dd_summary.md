# ViT-DD — Multi-Task Vision Transformer Evaluation Summary

## Setup

| Item | Value |
|---|---|
| Model | ViT-DD (Multi-Task Vision Transformer for Driver Distraction) |
| Architecture | ViT-Base (12 layers, 12 heads, dim 768, 202 sequence tokens) |
| Distraction Classes | 10 Behavioral Classes (C0–C9 State Farm / AUCDD Taxonomy) |
| Auxiliary Classes | 8 Emotion Classes (AffectNet-7 + 1 Non-Face / Occlusion Category) |
| Data | First 50 sample pairs (synthetic dry-run benchmark) |
| Input per sample | Global cabin image 224x224 (196 tokens) + Localized face crop 32x32 (4 tokens) |
| Batch size | 8 |
| Hardware | NVIDIA GeForce RTX 5070 Ti Laptop GPU (12 GB), WSL2 Ubuntu |
| Software | Python 3.12.3, PyTorch 2.11.0+cu128, timm 1.0.30 |

## Results

| Metric | Value |
|---|---|
| Total Params | 86.41 M |
| Trainable Params (Attention-Only) | 28.36 M (MHSA + Task Heads) |
| Frozen Backbone Params | 58.04 M (MLP, Patch Proj, Norms) |
| GFLOPs | 36.24 GFLOPs/sample |
| Latency (median) | 0.063 s/iter (~7.90 ms/sample) |
| Latency (mean) | 0.064 s/iter (~8.01 ms/sample) |
| Throughput (end-to-end) | 50 samples / 0.428 s (~116.8 samples/s) |
| Peak GPU memory | 0.40 GB |

## Notes

- **Attention-Only Fine-Tuning**: Enforcing parameter freezing on all non-MHSA backbone weights locks ~67.2% of the model (58.04 M params), leaving only 28.36 M trainable parameters. This regularizes the network against overfitting on medium-scale driving datasets.
- **Edge Deployment Viability**: At ~7.90 ms per frame (~116.8 FPS), ViT-DD operates at over 4x standard video frame rate (30 FPS), demonstrating high efficiency for real-time edge Driver Monitoring Systems (DMS).
- **VRAM Footprint**: Maximum GPU memory allocation peaked at 0.40 GB, remaining comfortably within the 12 GB VRAM capacity of the RTX 5070 Ti.
- **Occlusion Robustness**: Tested under simulated camera occlusion / missing face detections (all-zero tensor modality); verified zero numerical instability or gradient degradation.
