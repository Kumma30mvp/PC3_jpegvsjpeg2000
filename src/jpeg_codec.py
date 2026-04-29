"""Real JPEG (DCT) codec using Pillow."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

from . import config
from .io_utils import save_png

JPEG_BACKEND_NAME = "Pillow-JPEG"


def check_jpeg_backend() -> str:
    """Round-trip a small RGB image through Pillow's JPEG encoder."""
    test = (np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8))
    from io import BytesIO
    buf = BytesIO()
    Image.fromarray(test, "RGB").save(buf, format="JPEG", quality=80, optimize=True)
    buf.seek(0)
    decoded = np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)
    if decoded.shape != test.shape or decoded.dtype != np.uint8:
        raise RuntimeError("Pillow JPEG round-trip failed")
    return JPEG_BACKEND_NAME


def encode_decode_jpeg(
    image_uint8: np.ndarray,
    quality: int,
    image_stem: str,
    level_name: str,
) -> Dict:
    """Encode ``image_uint8`` as JPEG at ``quality``, decode it, and save outputs.

    Returns a dict with paths, timings, the reconstructed array, and the backend name.
    """
    if image_uint8.dtype != np.uint8 or image_uint8.ndim != 3 or image_uint8.shape[2] != 3:
        raise ValueError("encode_decode_jpeg expects an HxWx3 uint8 RGB array")

    compressed_path = (config.COMPRESSED_DIR / "jpeg" / f"{image_stem}__{level_name}.jpg").resolve()
    reconstructed_path = (
        config.RECONSTRUCTED_DIR / "jpeg" / f"{image_stem}__{level_name}.png"
    ).resolve()
    compressed_path.parent.mkdir(parents=True, exist_ok=True)
    reconstructed_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.fromarray(image_uint8, mode="RGB")

    t0 = time.perf_counter()
    img.save(str(compressed_path), format="JPEG", quality=int(quality), optimize=True)
    encoding_time_ms = (time.perf_counter() - t0) * 1000.0

    if not compressed_path.exists() or compressed_path.stat().st_size <= 0:
        raise RuntimeError(f"JPEG encoding produced no output at {compressed_path}")

    t0 = time.perf_counter()
    with Image.open(str(compressed_path)) as decoded:
        recon = np.asarray(decoded.convert("RGB"), dtype=np.uint8)
    decoding_time_ms = (time.perf_counter() - t0) * 1000.0

    if recon.shape != image_uint8.shape:
        raise RuntimeError(
            f"JPEG decoded shape {recon.shape} != original {image_uint8.shape}"
        )

    save_png(recon, reconstructed_path)

    return {
        "compressed_path": str(compressed_path),
        "reconstructed_path": str(reconstructed_path),
        "encoding_time_ms": float(encoding_time_ms),
        "decoding_time_ms": float(decoding_time_ms),
        "backend": JPEG_BACKEND_NAME,
        "reconstructed_array": recon,
    }
