# run_classifier_optimized.py
# Windows + CPU-only, Python 3.8 compatible.

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import Counter
from typing import Iterable, List, Optional, Sequence, Tuple

# ---- Enforce CPU-only BEFORE importing TensorFlow
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.dataset import image_paths
from src.image import dct2
from src.image_np import load_image


def _validate_cpu_visibility() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    assert isinstance(gpus, list)
    assert len(gpus) == 0, "GPU must be hidden; set CUDA_VISIBLE_DEVICES=-1"
    cpus = tf.config.list_physical_devices("CPU")
    assert isinstance(cpus, list)
    assert len(cpus) >= 1, "No CPU device visible"


class DCTLayer(tf.keras.layers.Layer):
    def __init__(self, mean: np.ndarray, var: np.ndarray) -> None:
        super().__init__()
        assert isinstance(mean, np.ndarray)
        assert isinstance(var, np.ndarray)
        self._mean_np = mean.astype(np.float32, copy=False)
        self._std_np = np.sqrt(var, dtype=np.float32)

    def build(self, input_shape: tf.TensorShape) -> None:
        assert isinstance(input_shape, (tuple, tf.TensorShape))
        assert len(input_shape) >= 2
        self.mean_w = self.add_weight(
            name="mean",
            shape=input_shape[1:],
            initializer=tf.keras.initializers.Constant(self._mean_np),
            trainable=False,
        )
        self.std_w = self.add_weight(
            name="std",
            shape=input_shape[1:],
            initializer=tf.keras.initializers.Constant(self._std_np),
            trainable=False,
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        assert isinstance(inputs, tf.Tensor)
        assert inputs.dtype in (tf.float32, tf.float16, tf.bfloat16)
        x = dct2(inputs)
        x = tf.abs(x)
        x = tf.math.log(x + tf.constant(1e-13, dtype=x.dtype))
        x = x - self.mean_w
        x = x / self.std_w
        return x


class PixelLayer(tf.keras.layers.Layer):
    def __init__(self) -> None:
        super().__init__()

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        assert isinstance(inputs, tf.Tensor)
        assert inputs.dtype in (tf.float32, tf.float16, tf.bfloat16)
        return (inputs / tf.constant(127.5, dtype=inputs.dtype)) - tf.constant(1.0, dtype=inputs.dtype)


def _wrap_pixel(model: tf.keras.Model) -> tf.keras.Sequential:
    assert isinstance(model, tf.keras.Model)
    seq = tf.keras.Sequential([PixelLayer(), model])
    assert isinstance(seq, tf.keras.Sequential)
    return seq


def _wrap_dct(model: tf.keras.Model, mean: np.ndarray, var: np.ndarray) -> tf.keras.Sequential:
    assert isinstance(model, tf.keras.Model)
    seq = tf.keras.Sequential([DCTLayer(mean, var), model])
    assert isinstance(seq, tf.keras.Sequential)
    return seq


def _gather_paths(directories: Sequence[str], per_dir_limit: Optional[int]) -> List[str]:
    assert isinstance(directories, (list, tuple))
    assert len(directories) >= 1
    collected: List[str] = []
    for d in directories:
        assert isinstance(d, str)
        assert os.path.isdir(d), f"Not a directory: {d}"
        paths = image_paths(d)
        if per_dir_limit is not None:
            assert per_dir_limit >= 0
            paths = paths[:per_dir_limit]
        collected.extend(paths)
    assert len(collected) >= 1, "No images found"
    assert all(isinstance(p, str) for p in collected)
    return collected


def _load_into_batch(paths: Sequence[str], batch_buffer: np.ndarray) -> Tuple[int, Tuple[int, int, int]]:
    assert isinstance(paths, (list, tuple))
    assert isinstance(batch_buffer, np.ndarray)
    assert batch_buffer.ndim == 4
    count = 0
    HWC: Optional[Tuple[int, int, int]] = None
    max_items = min(len(paths), batch_buffer.shape[0])
    assert max_items >= 0
    for i in range(max_items):
        img = load_image(paths[i])
        assert isinstance(img, np.ndarray)
        assert img.ndim in (2, 3)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        if HWC is None:
            HWC = (int(img.shape[0]), int(img.shape[1]), int(img.shape[2]))
            assert HWC[0] > 0 and HWC[1] > 0 and HWC[2] > 0
            assert (
                batch_buffer.shape[1] == HWC[0]
                and batch_buffer.shape[2] == HWC[1]
                and batch_buffer.shape[3] == HWC[2]
            ), "Batch buffer shape must match image shape"
        img_f32 = img.astype(np.float32, copy=False)
        batch_buffer[i, ...] = img_f32
        count += 1
    if HWC is None:
        HWC = (batch_buffer.shape[1], batch_buffer.shape[2], batch_buffer.shape[3])
    assert count >= 0
    assert isinstance(HWC, tuple)
    return count, HWC


def _predict_batch(model: tf.keras.Model, batch_buffer: np.ndarray, valid_count: int, tmp_preds: np.ndarray) -> int:
    assert isinstance(model, tf.keras.Model)
    assert isinstance(batch_buffer, np.ndarray)
    assert isinstance(tmp_preds, np.ndarray)
    assert valid_count >= 0 and valid_count <= batch_buffer.shape[0]
    batch_view = batch_buffer[:valid_count, ...]
    preds = model(batch_view, training=False)
    assert isinstance(preds, (tf.Tensor, np.ndarray))
    if isinstance(preds, tf.Tensor):
        preds = preds.numpy()
    assert preds.ndim >= 1
    if preds.shape[1] > 1:
        cls = np.argmax(preds, axis=1).astype(np.int64, copy=False)
    else:
        cls = np.rint(preds.reshape(-1)).astype(np.int64, copy=False)
    n_out = cls.shape[0]
    assert n_out == valid_count
    tmp_preds[:n_out] = cls.astype(np.int32, copy=False)
    assert tmp_preds.dtype == np.int32
    return n_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("MODEL", type=str, help="Path to a saved Keras model.")
    parser.add_argument("DATA", nargs="+", type=str, help="One or more directories to classify.")
    parser.add_argument("--size", "-s", type=int, default=None, help="Optional cap per directory.")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size (fixed upper bound).")
    parser.add_argument("--dct", "-d", type=str, default=None, help="Directory containing mean.npy / var.npy.")
    parser.add_argument("--csv_out", "-o", type=str, default=None, help="Output CSV path.")
    args = parser.parse_args()
    assert isinstance(args.MODEL, str)
    assert isinstance(args.DATA, list) and len(args.DATA) >= 1
    assert args.batch_size > 0
    if args.size is not None:
        assert args.size >= 0
    return args


def main() -> None:
    # CPU-only enforcement
    tf.config.set_visible_devices([], "GPU")
    _validate_cpu_visibility()

    args = parse_args()
    assert os.path.exists(args.MODEL), f"Model not found: {args.MODEL}"

    # Load model
    model = tf.keras.models.load_model(args.MODEL)
    assert isinstance(model, tf.keras.Model)
    if args.dct is not None:
        mean_p = os.path.join(args.dct, "mean.npy")
        var_p = os.path.join(args.dct, "var.npy")
        assert os.path.exists(mean_p), f"Missing mean.npy: {mean_p}"
        assert os.path.exists(var_p), f"Missing var.npy: {var_p}"
        mean = np.load(mean_p)
        var = np.load(var_p)
        model = _wrap_dct(model, mean, var)
    else:
        model = _wrap_pixel(model)
    assert isinstance(model, tf.keras.Model)

    # Gather paths
    paths = _gather_paths(args.DATA, args.size)
    total = len(paths)
    assert total >= 1

    # Inspect first image to fix H, W, C and pre-allocate reusable batch buffer
    first_img = load_image(paths[0])
    assert isinstance(first_img, np.ndarray)
    if first_img.ndim == 2:
        first_img = np.expand_dims(first_img, axis=-1)
    H, W, C = int(first_img.shape[0]), int(first_img.shape[1]), int(first_img.shape[2])
    assert H > 0 and W > 0 and C > 0

    batch_size = int(args.batch_size)
    batch_buffer = np.empty((batch_size, H, W, C), dtype=np.float32)
    tmp_preds = np.empty((batch_size,), dtype=np.int32)
    assert batch_buffer.shape[0] == batch_size
    assert tmp_preds.shape[0] == batch_size

    # CSV target
    csv_out = args.csv_out
    if csv_out is None:
        csv_out = os.path.join(os.path.abspath(args.DATA[0]), "predictions.csv")
    out_dir = os.path.dirname(csv_out) or "."
    assert os.path.isdir(out_dir), f"Output directory missing: {out_dir}"

    # Fixed number of steps (statically bounded loop)
    steps = (total + batch_size - 1) // batch_size
    assert steps >= 1

    class_counts: Counter[int] = Counter()
    with open(csv_out, mode="w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["path", "prediction"])
        for s in tqdm(range(steps)):
            start = s * batch_size
            end = min(start + batch_size, total)
            cur_paths = paths[start:end]
            assert len(cur_paths) >= 1
            # Load images into the reusable buffer (no per-batch reallocation)
            valid_count, shape_tuple = _load_into_batch(cur_paths, batch_buffer)
            assert valid_count == len(cur_paths)
            assert shape_tuple == (H, W, C)
            # Predict
            n_out = _predict_batch(model, batch_buffer, valid_count, tmp_preds)
            assert n_out == valid_count
            # Write and update counts
            for i in range(n_out):
                pred_i = int(tmp_preds[i])
                writer.writerow([cur_paths[i], pred_i])
                class_counts[pred_i] += 1

    # Report
    for c, amount in sorted(class_counts.items(), key=lambda x: x[0]):
        frac = amount / float(total)
        print(f"{frac:3.2%} of the images are from class {c} ({amount})")
    print(f"Saved predictions to: {csv_out}")


if __name__ == "__main__":
    # Single, explicit entry point
    main()
