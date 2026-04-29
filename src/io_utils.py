"""Dataset discovery and image loading utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from .config import ALLOWED_EXTENSIONS, IGNORED_EXTENSIONS


def find_images(input_dir: str | os.PathLike, recursive: bool = True) -> List[Path]:
    """Return a sorted list of image paths under ``input_dir``.

    Only files whose suffix is in :data:`ALLOWED_EXTENSIONS` are returned.
    Files whose suffix is in :data:`IGNORED_EXTENSIONS` (e.g. ``archive.zip``)
    are explicitly skipped.
    """
    root = Path(input_dir)
    if not root.exists():
        return []

    results: List[Path] = []
    if recursive:
        walker = (Path(dp) / fn for dp, _dn, fns in os.walk(root) for fn in fns)
    else:
        walker = (p for p in root.iterdir() if p.is_file())

    for p in walker:
        suffix = p.suffix.lower()
        if suffix in IGNORED_EXTENSIONS:
            continue
        if suffix in ALLOWED_EXTENSIONS:
            results.append(p)

    return sorted(results)


def validate_dataset(input_dir: str | os.PathLike) -> List[Path]:
    """Verify the dataset folder, print a status line, and return image paths."""
    root = Path(input_dir)
    if not root.exists():
        print(f"Dataset folder does not exist: {root}")
        return []

    images = find_images(root, recursive=True)
    n = len(images)
    if n == 0:
        print(
            "No images were found in data/kodak/raw/. "
            "Place kodim01.png to kodim24.png in this folder."
        )
        return []

    print(f"Found {n} images in {root}.")
    if n == 24:
        print("Kodak dataset correctly detected: 24 images found.")
    return images


def load_rgb_image(path: str | os.PathLike) -> np.ndarray:
    """Load an image as an RGB ``uint8`` array, no resize, no histogram changes."""
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
        raise ValueError(f"Loaded image at {path} is not RGB uint8 (got {arr.shape}, {arr.dtype})")
    return arr


def save_png(arr: np.ndarray, path: str | os.PathLike) -> None:
    """Write an RGB uint8 array to disk as a lossless PNG."""
    if arr.dtype != np.uint8:
        raise ValueError("save_png expects uint8 input")
    Image.fromarray(arr, mode="RGB").save(str(path), format="PNG")
