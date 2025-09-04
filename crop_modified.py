"""Crop images inside architecture subcategories and save to 'cropped' subfolders.

Expected layout (examples):
  <root>\test\1\GAN\Manual\*.png|jpg|jpeg      -> inputs (files directly in subcategory)
  <root>\test\1\GAN\Manual\resized\...         -> ignored
  <root>\test\1\GAN\Manual\cropped\...         -> outputs here

Notes:
  - Only subcategory folders are processed (…\<split>\1\<ARCH>\<SUBCATEGORY>\*).
  - Files in 'resized' or any nested subfolders are ignored.
  - Cropped outputs are saved as <subcategory>\cropped\<original_filename>.
  - Images already 128×128 are passed through unchanged; others are cropped using
    the original script's scheme.

Usage (Windows):
  python crop_subcategories.py --root "D:\dataset\test" --max-per-subcategory 0 --workers 8
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

EXTS = (".png", ".jpg", ".jpeg")


def _is_image_file(p: Path) -> bool:
    assert isinstance(p, Path)
    return p.is_file() and p.suffix.lower() in EXTS


def _enumerate_subcategory_dirs(split_root: Path) -> List[Path]:
    """
    Enumerate subcategory directories under BOTH classes 0 and 1:
      <split_root>/<class>/<ARCH>/<SUBCATEGORY>
    Skips folders named 'cropped' or 'resized'.
    """
    assert isinstance(split_root, Path)
    subs: List[Path] = []

    class_dirs = []
    for cls_name in ("0", "1"):
        candidate = split_root / cls_name
        if candidate.is_dir():
            class_dirs.append(candidate)
    assert isinstance(class_dirs, list)

    n_class = len(class_dirs)
    for i in range(n_class):
        class_dir = class_dirs[i]
        arch_dirs = [d for d in sorted(class_dir.iterdir()) if d.is_dir()]
        n_arch = len(arch_dirs)
        for j in range(n_arch):
            arch_dir = arch_dirs[j]
            sub_dirs = [d for d in sorted(arch_dir.iterdir()) if d.is_dir()]
            n_sub = len(sub_dirs)
            for k in range(n_sub):
                sub = sub_dirs[k]
                name = sub.name.lower()
                if name in ("cropped", "resized"):
                    continue
                subs.append(sub)
    # print all subdirs 
    for s in subs:
        print(f"[info] Found subcategory folder: {s}")
    return subs


def _gather_jobs_for_subcategory(sub_dir: Path, max_items: int) -> Tuple[List[Tuple[int, Path, Path]], Path]:
    """Collect (index, src_file, dst_dir) for files directly in sub_dir, excluding 'resized'."""
    assert isinstance(sub_dir, Path)
    assert isinstance(max_items, int) and max_items >= 0

    dst_dir = sub_dir / "cropped"
    files = [p for p in sorted(sub_dir.iterdir()) if _is_image_file(p)]
    # Exclude any files not at top-level (we are already non-recursive)
    # Exclude 'resized' by construction (we do not recurse into it)

    if max_items > 0 and len(files) > max_items:
        files = files[:max_items]

    jobs: List[Tuple[int, Path, Path]] = []
    n = len(files)
    for i in range(n):
        jobs.append((i, files[i], dst_dir))

    return jobs, dst_dir


def _crop_to_128(arr: np.ndarray) -> np.ndarray:
    """Apply original cropping scheme to produce 128×128."""
    assert isinstance(arr, np.ndarray)
    assert arr.ndim in (2, 3)

    h = int(arr.shape[0])
    w = int(arr.shape[1])
    if h == 128 and w == 128:
        return arr

    # Original heuristic from provided script
    x_upper = min(121 + 64, h)
    y_upper = min(89 + 64, w)
    x0 = max(0, x_upper - 128)
    y0 = max(0, y_upper - 128)
    x1 = min(h, x_upper)
    y1 = min(w, y_upper)

    cropped = np.copy(arr[x0:x1, y0:y1, ...])
    if cropped.shape[0] != 128 or cropped.shape[1] != 128:
        return np.empty((0, 0), dtype=np.uint8)  # sentinel for failure

    cropped = np.clip(cropped, 0, 255).astype(np.uint8)
    return cropped


def _process_one(task: Tuple[int, Path, Path]) -> int:
    """Worker: read, crop if needed, and save."""
    idx, src, dst_dir = task
    assert isinstance(idx, int) and idx >= 0
    assert isinstance(src, Path) and isinstance(dst_dir, Path)

    try:
        with Image.open(str(src)) as im:
            arr = np.asarray(im)
    except Exception:
        # Skip unreadable files
        return idx

    cropped = _crop_to_128(arr)
    if cropped.size == 0:
        # Failed to produce 128×128
        return idx

    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
        out_path = dst_dir / src.name
        Image.fromarray(cropped).save(str(out_path))
    except Exception:
        return idx

    return idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop images in architecture subcategories into 'cropped' folders.")
    parser.add_argument("--root", required=True, type=str, help="Root split directory to scan (e.g., ...\\dataset\\test).")
    parser.add_argument("--max-per-subcategory", type=int, default=0, help="Maximum files per subcategory (0 = all).")
    parser.add_argument("--workers", type=int, default=0, help="Number of parallel workers (0 = auto).")
    args = parser.parse_args()

    root = Path(args.root)
    assert root.exists() and root.is_dir()

    sub_dirs = _enumerate_subcategory_dirs(root)
    if len(sub_dirs) == 0:
        print("[info] No subcategory folders found. Nothing to do.")
        return

    all_tasks: List[Tuple[int, Path, Path]] = []
    # Build tasks with fixed bounds
    for sub in sub_dirs:
        jobs, dst = _gather_jobs_for_subcategory(sub, max_items=int(args.max_per_subcategory))
        if len(jobs) == 0:
            continue
        all_tasks.extend(jobs)
        # Ensure destination exists early to validate permissions
        dst.mkdir(parents=True, exist_ok=True)

    if len(all_tasks) == 0:
        print("[info] No image files found directly inside subcategory folders.")
        return

    # Execute
    if args.workers and args.workers > 0:
        max_workers = int(args.workers)
    else:
        max_workers = None  # let the executor choose

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for _ in pool.map(_process_one, all_tasks, chunksize=16):
            pass

    print("[done] Cropping completed.")


if __name__ == "__main__":
    main()
