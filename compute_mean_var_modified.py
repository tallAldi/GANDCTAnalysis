"""Compute RAW DCT mean and variance from images under 'cropped' folders.

Usage:
  python compute_mean_var.py --data-root D:\dataset\test --out-dir output\mean_var\TEST
  # produces output\mean_var\TEST\mean.npy and var.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np

from src.dataset import image_paths
from src.image_np import dct2, load_image
from src.math import welford, welford_multidimensional


def _filter_cropped(root: Path, files: List[Path]) -> List[Path]:
    kept: List[Path] = []
    for p in files:
        rel = p.relative_to(root)
        if "cropped" in rel.parts:
            kept.append(p)
    return kept


def _stream_raw_dct(paths: List[Path], color: bool) -> Iterable[np.ndarray]:
    for p in paths:
        arr = load_image(str(p), grayscale=not color)
        arr = dct2(arr)  # RAW DCT coefficients (no log)
        yield arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute RAW DCT mean/var over 'cropped' images.")
    parser.add_argument("--data-root", required=True, type=str, help="Dataset root to scan (e.g., .../dataset/test).")
    parser.add_argument("--out-dir", required=True, type=str, help="Output directory to save mean.npy and var.npy.")
    parser.add_argument("--color", "-c", action="store_true", help="Use color images (per-channel stats).")
    parser.add_argument("--amount", "-n", type=int, default=0, help="Maximum number of images (0 = all).")
    args = parser.parse_args()

    root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_files = list(sorted(image_paths(root)))
    cropped_files = _filter_cropped(root, all_files)
    if args.amount > 0:
        cropped_files = cropped_files[: args.amount]

    if len(cropped_files) == 0:
        print(f"[warn] No 'cropped' images found under {root}")
        return

    stream = _stream_raw_dct(cropped_files, color=args.color)

    if args.color:
        mean, var = welford_multidimensional(stream)
    else:
        mean, var = welford(stream)

    np.save(out_dir / "mean.npy", mean)
    np.save(out_dir / "var.npy", var)
    print(f"[done] Saved mean/var to: {out_dir}")


if __name__ == "__main__":
    main()
