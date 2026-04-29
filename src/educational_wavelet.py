"""Educational wavelet module that illustrates the JPEG2000 sub-band idea.

Like ``educational_dct``, this module is for visual explanation only — production
metrics in the CSVs come from the real JPEG2000 codec, never from these functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pywt

from .io_utils import load_rgb_image


def wavelet_decompose_channel(
    channel: np.ndarray, wavelet: str = "haar", level: int = 2
) -> list:
    return pywt.wavedec2(channel.astype(np.float64), wavelet=wavelet, level=level)


def wavelet_reconstruct_channel(coeffs: list, wavelet: str = "haar") -> np.ndarray:
    return pywt.waverec2(coeffs, wavelet=wavelet)


def quantize_wavelet_coeffs(coeffs: list, step: float) -> list:
    """Uniform mid-tread quantization on the LL band and every detail tuple."""
    step = float(step)
    if step <= 0:
        return coeffs
    out = [np.round(coeffs[0] / step) * step]
    for detail in coeffs[1:]:
        out.append(tuple(np.round(d / step) * step for d in detail))
    return out


def reconstruct_with_wavelet_quantization(
    image: np.ndarray, wavelet: str = "haar", level: int = 2, step: float = 10.0
) -> np.ndarray:
    if image.ndim == 2:
        coeffs = wavelet_decompose_channel(image, wavelet, level)
        rq = quantize_wavelet_coeffs(coeffs, step)
        recon = wavelet_reconstruct_channel(rq, wavelet)
        return np.clip(recon, 0, 255).astype(np.uint8)

    out_channels = []
    for c in range(image.shape[2]):
        coeffs = wavelet_decompose_channel(image[..., c], wavelet, level)
        rq = quantize_wavelet_coeffs(coeffs, step)
        recon = wavelet_reconstruct_channel(rq, wavelet)
        out_channels.append(np.clip(recon, 0, 255))
    out = np.stack(out_channels, axis=-1)
    return out.astype(np.uint8)[: image.shape[0], : image.shape[1]]


def _normalise_for_display(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float64)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def create_wavelet_analysis_figure(
    image_path: str | Path,
    output_dir: str | Path,
    wavelet: str = "haar",
    level: int = 2,
    step: float = 10.0,
) -> Path:
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb = load_rgb_image(image_path)
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float64)

    coeffs = wavelet_decompose_channel(gray, wavelet=wavelet, level=level)
    ll = coeffs[0]
    lh, hl, hh = coeffs[-1]  # finest-scale detail bands

    rq = quantize_wavelet_coeffs(coeffs, step=step)
    recon = wavelet_reconstruct_channel(rq, wavelet=wavelet)
    recon = recon[: gray.shape[0], : gray.shape[1]]
    error = np.abs(gray - recon)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        f"Educational wavelet decomposition ({wavelet}, level={level}, step={step})"
        f"\n{image_path.name}"
    )

    axes[0, 0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Original (grayscale)")
    axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("y")

    axes[0, 1].imshow(_normalise_for_display(ll), cmap="gray")
    axes[0, 1].set_title("LL approximation")

    axes[0, 2].imshow(_normalise_for_display(lh), cmap="gray")
    axes[0, 2].set_title("LH (horizontal detail)")

    axes[0, 3].imshow(_normalise_for_display(hl), cmap="gray")
    axes[0, 3].set_title("HL (vertical detail)")

    axes[1, 0].imshow(_normalise_for_display(hh), cmap="gray")
    axes[1, 0].set_title("HH (diagonal detail)")

    axes[1, 1].imshow(np.clip(recon, 0, 255), cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title(f"Reconstruction (step={step})")

    im_err = axes[1, 2].imshow(error, cmap="hot")
    axes[1, 2].set_title("|Original - reconstructed|")
    fig.colorbar(im_err, ax=axes[1, 2], fraction=0.046)

    axes[1, 3].axis("off")
    axes[1, 3].text(
        0.0,
        0.5,
        f"Wavelet: {wavelet}\nLevels: {level}\nQuant step: {step}\n"
        "Note: educational only;\nthe real JPEG2000 codec\nhandles encoding.",
        fontsize=11,
        va="center",
    )

    fig.tight_layout()
    out_path = output_dir / f"{image_path.stem}_wavelet_analysis.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
