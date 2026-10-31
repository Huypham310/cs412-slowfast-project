# SlowFast R50 8x8 — Kinetics-400 Evaluation Summary

## Setup

| Item | Value |
|---|---|
| Model | SlowFast R50, 8x8 (`configs/Kinetics/c2/SLOWFAST_8x8_R50.yaml`) |
| Checkpoint | `SLOWFAST_8x8_R50.pkl` (Caffe2, official model zoo) |
| Data | First 50 videos of Kinetics-400 val (`part_0`, CVDF mirror) |
| Testing views | 10 temporal x 3 spatial = 30 views/video |
| Input per view | 32 frames, sampling rate 2, 256x256 crop |
| Batch size | 8 |
| Hardware | NVIDIA GeForce RTX 5070 Ti Laptop GPU (12 GB), WSL2 Ubuntu |
| Software | Python 3.12, PyTorch 2.11.0+cu128, PyAV decoding |

## Results

| Metric | Value |
|---|---|
| Top-1 Accuracy | 86.00% |
| Top-5 Accuracy | 94.00% |
| GFLOPs | 66.07 GFLOPs/view (~1,982 GFLOPs/video with 30 views) |
| Params | 34.57 M |
| Activations | 178.39 M |
| Latency (median) | 0.418 s/iter (~52 ms/clip) |
| Latency (mean) | 1.65 s/iter |
| Throughput (end-to-end) | 50 videos / 290 s (~5.8 s/video, ~5.2 clips/s) |
| Peak GPU memory | 1.75 GB |

## Notes

- Accuracy is measured on only 50 videos, so the margin of error is large (roughly ±5–6%). The published result on the full Kinetics-400 val set is 77.0% top-1 / 92.6% top-5.
- Mean latency is higher than median because the run is bottlenecked by CPU video decoding in the data loader, not by the GPU.
- Numbers are taken from `stdout.log` of the run on Oct 30, 2026.
- Detailed per-video predictions and error analysis data extracted from the raw `stdout.log` are formatted and available in `slowfast_dryrun_preds.csv`.