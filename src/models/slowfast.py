"""
SlowFast R50 8x8 for Driver Distraction Detection, built on PySlowFast.

Uses the same code path as the Kinetics-400 evaluation in `result/`:
the local clone of facebookresearch/SlowFast, config
`configs/Kinetics/c2/SLOWFAST_8x8_R50.yaml` and the Caffe2 checkpoint
`SLOWFAST_8x8_R50.pkl`.

PySlowFast must be importable, either installed into the active environment
(`python setup.py build develop` inside the clone) or pointed to with the
`SLOWFAST_ROOT` environment variable.
"""

import argparse
import os
import sys
from pathlib import Path

import torch

try:
    import slowfast
except ImportError:
    if "SLOWFAST_ROOT" not in os.environ:
        raise ImportError(
            "PySlowFast not found. Install it from the SlowFast clone or set SLOWFAST_ROOT."
        )
    sys.path.insert(0, os.environ["SLOWFAST_ROOT"])
    import slowfast

from slowfast.config.defaults import get_cfg
from slowfast.datasets.utils import pack_pathway_output
from slowfast.models import build_model
from slowfast.utils.checkpoint import load_checkpoint

SLOWFAST_ROOT = Path(slowfast.__file__).resolve().parents[1]
DEFAULT_CFG = SLOWFAST_ROOT / "configs" / "Kinetics" / "c2" / "SLOWFAST_8x8_R50.yaml"
DEFAULT_CHECKPOINT = SLOWFAST_ROOT / "SLOWFAST_8x8_R50.pkl"


def load_cfg(cfg_file=DEFAULT_CFG, num_classes=10, num_gpus=None, opts=None):
    """
    Load the PySlowFast config and adapt it to driver distraction.
    - num_classes: 10 (corresponding to the 10 behavioral classes in State Farm/AUCDD datasets)
    - num_gpus: defaults to 1 if CUDA is available, else 0 (CPU)
    - opts: extra overrides in PySlowFast format, e.g. ["TEST.BATCH_SIZE", 8]
    """
    cfg = get_cfg()
    cfg.merge_from_file(str(cfg_file))
    if opts:
        cfg.merge_from_list(opts)
    cfg.MODEL.NUM_CLASSES = num_classes
    cfg.NUM_GPUS = (1 if torch.cuda.is_available() else 0) if num_gpus is None else num_gpus
    return cfg


def build_slowfast_dd(cfg, checkpoint=None, freeze_backbone=False):
    """
    Build SlowFast from `cfg` and optionally initialize it from a Kinetics checkpoint.

    The Kinetics head (2304 -> 400) does not match `cfg.MODEL.NUM_CLASSES`, so
    `load_checkpoint` skips it and `head.projection` stays randomly initialized
    for fine-tuning. Keeping the bare PySlowFast model (no wrapper) keeps
    state_dict keys compatible with `tools/train_net.py` and `tools/test_net.py`.
    """
    model = build_model(cfg)
    if checkpoint is not None:
        load_checkpoint(
            str(checkpoint),
            model,
            data_parallel=cfg.NUM_GPUS > 1,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
        )
    if freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith(("head.", "module.head."))
    return model


def pack_pathways(cfg, frames):
    """
    Split a clip into [slow_pathway, fast_pathway] inputs.
    - frames: C x T x H x W (single clip) or N x C x T x H x W (batch)
    """
    if frames.ndim == 4:
        return pack_pathway_output(cfg, frames)
    pathways = [pack_pathway_output(cfg, clip) for clip in frames]
    return [torch.stack(p) for p in zip(*pathways)]


def parse_args():
    parser = argparse.ArgumentParser(description="Dry-run SlowFast for driver distraction.")
    parser.add_argument("--cfg", default=DEFAULT_CFG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="PySlowFast config overrides")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_cfg(args.cfg, args.num_classes, 0 if args.cpu else None, args.opts)
    checkpoint = args.checkpoint if Path(args.checkpoint).exists() else None
    model = build_slowfast_dd(cfg, checkpoint).eval()

    size = cfg.DATA.TEST_CROP_SIZE
    clip = torch.rand(1, 3, cfg.DATA.NUM_FRAMES, size, size)
    inputs = pack_pathways(cfg, clip)
    if cfg.NUM_GPUS:
        inputs = [x.cuda(non_blocking=True) for x in inputs]
    with torch.no_grad():
        preds = model(inputs)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Checkpoint: {checkpoint or 'none (random init)'}")
    print(f"Input shapes: {[tuple(x.shape) for x in inputs]}")
    print(f"Output shape: {tuple(preds.shape)} | Params: {n_params:.2f} M")
