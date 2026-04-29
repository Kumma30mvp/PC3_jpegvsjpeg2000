"""Educational DCT module that illustrates the JPEG pipeline on the Y channel.

This module is for visual explanation only — production metrics in the CSVs come
from the real Pillow-JPEG codec, never from these functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .io_utils import load_rgb_image

# Standard JPEG luminance quantization matrix (ISO/IEC 10918-1 Annex K).
STD_LUMA_Q = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float64,
)


def rgb_to_ycbcr(image: np.ndarray) -> np.ndarray:
    """ITU-R BT.601 RGB → YCbCr conversion. Input/output in [0, 255] float64."""
    img = image.astype(np.float64)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return np.stack([y, cb, cr], axis=-1)


def ycbcr_to_rgb(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float64)
    y, cb, cr = img[..., 0], img[..., 1], img[..., 2]
    r = y + 1.402 * (cr - 128.0)
    g = y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0)
    b = y + 1.772 * (cb - 128.0)
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0.0, 255.0)


def split_into_blocks(channel: np.ndarray, block_size: int = 8) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad ``channel`` to a multiple of ``block_size`` and split into blocks.

    Returns the array of shape ``(nby, nbx, block_size, block_size)`` and the
    original shape so :func:`merge_blocks` can crop back.
    """
    h, w = channel.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    padded = np.pad(channel, ((0, pad_h), (0, pad_w)), mode="edge")
    nby = padded.shape[0] // block_size
    nbx = padded.shape[1] // block_size
    blocks = padded.reshape(nby, block_size, nbx, block_size).transpose(0, 2, 1, 3)
    return blocks, (h, w)


def merge_blocks(blocks: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
    """Inverse of :func:`split_into_blocks`."""
    nby, nbx, bh, bw = blocks.shape
    merged = blocks.transpose(0, 2, 1, 3).reshape(nby * bh, nbx * bw)
    h, w = original_shape
    return merged[:h, :w]


def block_dct_2d(block: np.ndarray) -> np.ndarray:
    return cv2.dct(block.astype(np.float32))


def block_idct_2d(coeffs: np.ndarray) -> np.ndarray:
    return cv2.idct(coeffs.astype(np.float32))


def quality_to_scale(quality: int) -> float:
    """Standard libjpeg quality → quantization-scale formula."""
    q = max(1, min(100, int(quality)))
    if q < 50:
        return 5000.0 / q
    return 200.0 - 2.0 * q


def scaled_quantization_matrix(quality: int) -> np.ndarray:
    scale = quality_to_scale(quality)
    qm = np.floor((STD_LUMA_Q * scale + 50.0) / 100.0)
    qm[qm < 1.0] = 1.0
    return qm


def quantize_block(dct_block: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
    return np.round(dct_block / q_matrix)


def dequantize_block(q_block: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
    return q_block * q_matrix


def reconstruct_channel_with_dct_quantization(
    channel: np.ndarray, quality: int
) -> np.ndarray:
    """Run DCT → quantize → dequantize → IDCT on every 8×8 block of ``channel``."""
    qm = scaled_quantization_matrix(quality)
    blocks, original_shape = split_into_blocks(channel.astype(np.float32) - 128.0)
    nby, nbx, bh, bw = blocks.shape
    out = np.empty_like(blocks)
    for by in range(nby):
        for bx in range(nbx):
            d = block_dct_2d(blocks[by, bx])
            q = quantize_block(d, qm)
            dq = dequantize_block(q, qm)
            out[by, bx] = block_idct_2d(dq)
    merged = merge_blocks(out, original_shape) + 128.0
    return np.clip(merged, 0.0, 255.0).astype(np.uint8)


def create_dct_analysis_figure(
    image_path: str | Path,
    output_dir: str | Path,
    quality: int = 30,
) -> Path:
    """Save a 2×3 figure that walks through the JPEG DCT pipeline on one block."""
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb = load_rgb_image(image_path)
    ycbcr = rgb_to_ycbcr(rgb)
    y = ycbcr[..., 0]

    h, w = y.shape
    by = (h // 2) - ((h // 2) % 8)
    bx = (w // 2) - ((w // 2) % 8)
    block = y[by : by + 8, bx : bx + 8].astype(np.float32) - 128.0
    qm = scaled_quantization_matrix(quality)
    dct = block_dct_2d(block)
    q = quantize_block(dct, qm)
    dq = dequantize_block(q, qm)
    recon_block = block_idct_2d(dq) + 128.0

    full_recon = reconstruct_channel_with_dct_quantization(y.astype(np.uint8), quality)
    error = np.abs(y.astype(np.float64) - full_recon.astype(np.float64))

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(
        f"Educational DCT pipeline (Y channel, quality={quality})\n{image_path.name}"
    )

    axes[0, 0].imshow(y, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Y channel (luma)")
    axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("y")

    axes[0, 1].imshow(block + 128.0, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title(f"Selected 8x8 block @ ({by},{bx})")
    axes[0, 1].set_xlabel("col"); axes[0, 1].set_ylabel("row")

    im2 = axes[0, 2].imshow(np.log1p(np.abs(dct)), cmap="viridis")
    axes[0, 2].set_title("DCT coefficients (log|.|)")
    axes[0, 2].set_xlabel("u"); axes[0, 2].set_ylabel("v")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    axes[1, 0].imshow(q, cmap="coolwarm")
    axes[1, 0].set_title("Quantized DCT coeffs")
    axes[1, 0].set_xlabel("u"); axes[1, 0].set_ylabel("v")
    for (i, j), v in np.ndenumerate(q):
        axes[1, 0].text(j, i, f"{int(v)}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 1 else "black")

    axes[1, 1].imshow(recon_block, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title("Reconstructed block (IDCT)")
    axes[1, 1].set_xlabel("col"); axes[1, 1].set_ylabel("row")

    im4 = axes[1, 2].imshow(error, cmap="hot")
    axes[1, 2].set_title("|Y - reconstructed Y| (full image)")
    axes[1, 2].set_xlabel("x"); axes[1, 2].set_ylabel("y")
    fig.colorbar(im4, ax=axes[1, 2], fraction=0.046)

    fig.tight_layout()
    out_path = output_dir / f"{image_path.stem}_dct_analysis.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
