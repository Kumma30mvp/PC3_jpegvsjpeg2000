"""Image quality and compression metrics."""

from __future__ import annotations

import math
import os
from typing import Union

import numpy as np
from skimage.metrics import structural_similarity as _sk_ssim

PathLike = Union[str, os.PathLike]


def mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    if original.shape != reconstructed.shape:
        raise ValueError(
            f"shape mismatch in mse: {original.shape} vs {reconstructed.shape}"
        )
    a = original.astype(np.float64)
    b = reconstructed.astype(np.float64)
    return float(np.mean((a - b) ** 2))


def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    err = mse(original, reconstructed)
    if err == 0.0:
        return math.inf
    return 10.0 * math.log10((255.0 ** 2) / err)


def ssim_rgb(original: np.ndarray, reconstructed: np.ndarray) -> float:
    if original.shape != reconstructed.shape:
        raise ValueError(
            f"shape mismatch in ssim_rgb: {original.shape} vs {reconstructed.shape}"
        )
    return float(
        _sk_ssim(original, reconstructed, data_range=255, channel_axis=-1)
    )


def file_size_bytes(path: PathLike) -> int:
    return int(os.path.getsize(path))


def compression_ratio_file(original_path: PathLike, compressed_path: PathLike) -> float:
    o = file_size_bytes(original_path)
    c = file_size_bytes(compressed_path)
    if c <= 0:
        raise ValueError(f"compressed file {compressed_path} has zero size")
    return o / c


def compression_ratio_raw(image_array: np.ndarray, compressed_path: PathLike) -> float:
    if image_array.ndim != 3:
        raise ValueError("compression_ratio_raw expects an HxWxC array")
    h, w, c = image_array.shape
    raw = h * w * c
    cs = file_size_bytes(compressed_path)
    if cs <= 0:
        raise ValueError(f"compressed file {compressed_path} has zero size")
    return raw / cs
