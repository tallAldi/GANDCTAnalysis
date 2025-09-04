"""Prepare TEST images (evaluation) with per-architecture buckets.
Only images inside a 'cropped' folder are included.

Buckets:
  - 'real' for class '0'
  - '<ARCH>' for generated class (e.g., 'GAN', 'PixDiff', 'LatDiff', ...)
  - 'LatDiff_Commercial' for 'LatDiff/Commercial'
"""

from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import tensorflow as tf

from src.dataset import image_paths, serialize_data
from src.image_np import dct2, load_image, normalize, scale_image
from src.math import log_scale, welford

PairBucket = Tuple[str, int, str]


def _dct2_wrapper(image: np.ndarray, log: bool = False) -> np.ndarray:
    arr = np.asarray(image)
    out = dct2(arr)
    if log:
        out = log_scale(out)
    return out


def _determine_bucket(test_root: Path, path: Path) -> str:
    rel = path.relative_to(test_root)
    parts = rel.parts
    if len(parts) == 0:
        return "unknown"
    top = parts[0]
    if top == "0":
        return "real"
    if len(parts) >= 2:
        arch = parts[1]
        if arch == "LatDiff":
            if len(parts) >= 3 and parts[2] == "Commercial":
                return "LatDiff_Commercial"
            return "LatDiff"
        return arch
    return "generated"


def _is_in_cropped(test_root: Path, path: Path) -> bool:
    rel = path.relative_to(test_root)
    return "cropped" in rel.parts


def _collect_test_with_buckets_cropped(test_root: Path) -> List[PairBucket]:
    items: List[PairBucket] = []
    all_img_paths = list(sorted(image_paths(test_root)))
    total = len(all_img_paths)
    kept = 0
    for i in range(total):
        p = all_img_paths[i]
        if not _is_in_cropped(test_root, p):
            continue
        rel = p.relative_to(test_root)
        parts = rel.parts
        if len(parts) == 0:
            continue
        class_part = parts[0]
        label = int(class_part) if class_part.isdigit() else 0
        bucket = _determine_bucket(test_root, p)
        items.append((str(p), label, bucket))
        kept += 1
    if kept == 0:
        print(f"[warn] No images found under 'cropped' folders in: {test_root}")
    return items


def _compute_abs_scaler(
    items: Sequence[PairBucket],
    load_fn: Callable[..., np.ndarray],
    transform_fn: Callable[[np.ndarray], np.ndarray] | None
) -> Callable[[np.ndarray], np.ndarray]:
    if len(items) == 0:
        print("[warn] --abs requested but TEST set is empty; skipping abs scaling.")
        return lambda x: x
    first_path, _, _ = items[0]
    first = load_fn(first_path)
    if transform_fn is not None:
        first = transform_fn(first)
    current_max = np.abs(first).astype(np.float64)
    limit = len(items)
    for idx in range(1, limit):
        pth, _, _ = items[idx]
        arr = load_fn(pth)
        if transform_fn is not None:
            arr = transform_fn(arr)
        val = np.abs(arr).astype(np.float64)
        mask = current_max >= val
        current_max = mask * current_max + (~mask) * val

    def scale_by_absolute(img: np.ndarray) -> np.ndarray:
        denom = np.where(current_max == 0.0, 1.0, current_max)
        return img / denom

    return scale_by_absolute


def _compute_mean_std(
    items: Sequence[PairBucket],
    load_fn: Callable[..., np.ndarray],
    transform_fn: Callable[[np.ndarray], np.ndarray] | None,
    abs_fn: Callable[[np.ndarray], np.ndarray] | None
) -> Tuple[np.ndarray, np.ndarray]:
    if len(items) == 0:
        print("[warn] --normalize requested but TEST set is empty; skipping normalization.")
        return np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32)

    def _stream() -> Iterable[np.ndarray]:
        total = len(items)
        for i in range(total):
            pth, _, _ = items[i]
            arr = load_fn(pth)
            if transform_fn is not None:
                arr = transform_fn(arr)
            if abs_fn is not None:
                arr = abs_fn(arr)
            yield arr

    mean, var = welford(_stream())
    std = np.sqrt(var)
    return mean, std


def _encode_example(
    item: PairBucket,
    load_fn: Callable[..., np.ndarray],
    transform_fn: Callable[[np.ndarray], np.ndarray] | None,
    abs_fn: Callable[[np.ndarray], np.ndarray] | None,
    norm_fn: Callable[[np.ndarray], np.ndarray] | None
) -> Tuple[np.ndarray, int, str]:
    path, label, bucket = item
    img = load_fn(path)
    if transform_fn is not None:
        img = transform_fn(img)
    if abs_fn is not None:
        img = abs_fn(img)
    if norm_fn is not None:
        img = norm_fn(img)
    return img, int(label), bucket


def _make_output_stem(args: argparse.Namespace) -> str:
    base = args.output if args.output is not None else "dataset_test"
    parts: List[str] = [base]
    if args.color:
        parts.append("color")
    parts.append("raw" if args.raw else "dct")
    if (not args.raw) and args.log:
        parts.append("log_scaled")
    if args.abs:
        parts.append("abs_scaled")
    if args.normalize:
        parts.append("normalized")
    return "_".join(parts)


def _write_numpy_buckets(
    out_root: Path,
    items: Sequence[PairBucket],
    encode: Callable[[PairBucket], Tuple[np.ndarray, int, str]]
) -> None:
    buckets: Dict[str, List[int]] = {}
    for i in range(len(items)):
        _, _, b = items[i]
        if b not in buckets:
            buckets[b] = []
        buckets[b].append(i)

    summary: Dict[str, int] = {}
    for bucket, indices in sorted(buckets.items()):
        bucket_dir = out_root / bucket
        os.makedirs(bucket_dir, exist_ok=True)
        labels: List[int] = []
        total = len(indices)
        for j in range(total):
            idx = indices[j]
            arr, lbl, _ = encode(items[idx])
            print(f"\r[{bucket}] Converted {j + 1:06d}/{total:06d} images!", end="")
            with open(bucket_dir / f"{j:06}.npy", "wb+") as f:
                np.save(f, arr)
            labels.append(lbl)
        print("")
        with open(bucket_dir / "labels.npy", "wb+") as f:
            np.save(f, labels)
        summary[bucket] = total

    with open(out_root / "buckets_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _write_tfrecord_buckets(
    out_root: Path,
    items: Sequence[PairBucket],
    encode: Callable[[PairBucket], Tuple[np.ndarray, int, str]]
) -> None:
    buckets: Dict[str, List[int]] = {}
    for i in range(len(items)):
        _, _, b = items[i]
        if b not in buckets:
            buckets[b] = []
        buckets[b].append(i)

    summary: Dict[str, int] = {}
    for bucket, indices in sorted(buckets.items()):
        bucket_dir = out_root / f"{bucket}_tf"
        os.makedirs(bucket_dir, exist_ok=True)

        def _gen() -> Iterable[bytes]:
            total = len(indices)
            for j in range(total):
                idx = indices[j]
                arr, lbl, _ = encode(items[idx])
                print(f"\r[{bucket}] Converted {j + 1:06d}/{total:06d} images!", end="")
                yield serialize_data((arr, lbl))
            print("")

        dataset = tf.data.Dataset.from_generator(_gen, output_types=tf.string, output_shapes=())
        writer = tf.data.experimental.TFRecordWriter(str(bucket_dir / "data.tfrecords"))
        writer.write(dataset)
        summary[bucket] = len(indices)

    with open(out_root / "buckets_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare TEST images with per-architecture buckets using only 'cropped' subfolders."
    )
    parser.add_argument("--test-dir", required=True, type=str, help="Path to TEST root (e.g., .../dataset/test).")
    parser.add_argument("--output", required=False, type=str, help="Output name stem. Default: 'dataset_test'.")
    parser.add_argument("--raw", "-r", action="store_true", help="Save raw image data instead of DCT.")
    parser.add_argument("--log", "-l", action="store_true", help="Log scale (only for DCT).")
    parser.add_argument("--abs", "-a", action="store_true", help="Scale by global TEST-set max |value|.")
    parser.add_argument("--color", "-c", action="store_true", help="Load as color (default grayscale).")
    parser.add_argument("--normalize", "-n", action="store_true", help="Normalize using global TEST-set mean/std.")
    modes = parser.add_subparsers(dest="mode", help="Select output mode {normal|tfrecords}")
    _ = modes.add_parser("normal")
    _ = modes.add_parser("tfrecords")

    args = parser.parse_args()
    if args.mode not in {"normal", "tfrecords"}:
        print("[error] Please specify output mode: 'normal' or 'tfrecords'.")
        return

    outstem = _make_output_stem(args)
    out_root = Path(f"{outstem}_test")
    os.makedirs(out_root, exist_ok=True)

    load_fn = functools.partial(load_image, tf=(args.mode == "tfrecords"))
    if args.color:
        load_fn = functools.partial(load_fn, grayscale=False)

    transform_fn = None
    norm_fn = None
    abs_fn = None

    if args.raw:
        if args.normalize:
            norm_fn = scale_image
    else:
        transform_fn = _dct2_wrapper
        if args.log:
            transform_fn = functools.partial(_dct2_wrapper, log=True)

    test_root = Path(args.test_dir)
    items = _collect_test_with_buckets_cropped(test_root)

    if args.abs:
        abs_fn = _compute_abs_scaler(items, load_fn, transform_fn)
    if args.normalize and (not args.raw):
        mean, std = _compute_mean_std(items, load_fn, transform_fn, abs_fn)
        norm_fn = functools.partial(normalize, mean=mean, std=std)

    encode = functools.partial(
        _encode_example,
        load_fn=load_fn,
        transform_fn=transform_fn,
        abs_fn=abs_fn,
        norm_fn=norm_fn,
    )

    if args.mode == "normal":
        _write_numpy_buckets(out_root, items, encode)
        print(f"[done] Wrote NumPy buckets under: {out_root}")
    else:
        _write_tfrecord_buckets(out_root, items, encode)
        print(f"[done] Wrote TFRecord buckets under: {out_root}")


if __name__ == "__main__":
    main()
