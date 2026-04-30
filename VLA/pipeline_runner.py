from __future__ import annotations

import os


def _configure_max_compute_threads() -> None:
    """Max out CPU parallelism for BLAS/OpenMP before NumPy/OpenCV import.

    Override with VLA_NUM_THREADS (integer) to cap threads on shared machines.
    """
    n = int(os.environ.get("VLA_NUM_THREADS", "0") or 0) or (os.cpu_count() or 1)
    n = max(1, n)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        if name not in os.environ:
            os.environ[name] = str(n)
    if "TOKENIZERS_PARALLELISM" not in os.environ:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"


_configure_max_compute_threads()

import argparse
from pathlib import Path

from pipeline.io_utils import ensure_output_dirs, load_config
from pipeline.orchestrator import HomeWorldModelPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Home World Model Builder from Video - Phase 1')
    parser.add_argument('--video', required=True, type=Path, help='Input RGB video path')
    parser.add_argument('--depth-video', type=Path, default=None, help='Optional depth video path')
    parser.add_argument('--config', required=True, type=Path, help='YAML config path')
    parser.add_argument('--output', required=True, type=Path, help='Output directory')
    return parser.parse_args()


def main() -> None:
    import cv2
    import torch

    n = int(os.environ.get("VLA_NUM_THREADS", "0") or 0) or (os.cpu_count() or 1)
    n = max(1, n)
    try:
        torch.set_num_threads(n)
    except RuntimeError:
        pass
    try:
        interop = max(1, min(8, max(1, n // 8)))
        torch.set_num_interop_threads(interop)
    except RuntimeError:
        pass
    try:
        cv2.setNumThreads(n)
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    args = parse_args()
    cfg = load_config(args.config)
    output_dirs = ensure_output_dirs(args.output)

    pipeline = HomeWorldModelPipeline(cfg=cfg, output_dirs=output_dirs)
    pipeline.initialize()
    summary = pipeline.run(video_path=args.video, depth_video_path=args.depth_video)

    print('Phase 1 pipeline complete.')
    print(summary)


if __name__ == '__main__':
    main()
