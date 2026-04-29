"""Image-complexity scores used to pick representative images and crops.

Note: HOG and LBP are used here only to *select* visually interesting regions.
They are never used as a compression method.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .io_utils import find_images, load_rgb_image


def gradient_energy(image: np.ndarray) -> float:
    """Mean Sobel-magnitude over the grayscale image."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    gray = gray.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return float(mag.mean())


def texture_lbp_score(image: np.ndarray) -> float:
    """Simple 8-neighbor LBP, returning the entropy of the normalized histogram."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    gray = gray.astype(np.int16)
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0

    center = gray[1:-1, 1:-1]
    neighbours = [
        gray[0:-2, 0:-2],  # top-left      bit 0
        gray[0:-2, 1:-1],  # top           bit 1
        gray[0:-2, 2:  ],  # top-right     bit 2
        gray[1:-1, 2:  ],  # right         bit 3
        gray[2:  , 2:  ],  # bottom-right  bit 4
        gray[2:  , 1:-1],  # bottom        bit 5
        gray[2:  , 0:-2],  # bottom-left   bit 6
        gray[1:-1, 0:-2],  # left          bit 7
    ]
    code = np.zeros_like(center, dtype=np.uint8)
    for i, n in enumerate(neighbours):
        code |= ((n >= center).astype(np.uint8) << i)

    hist, _ = np.histogram(code.ravel(), bins=256, range=(0, 256))
    p = hist.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    nz = p > 0
    return float(-np.sum(p[nz] * np.log2(p[nz])))


def color_diversity_score(image: np.ndarray) -> float:
    """Mean of per-channel standard deviation."""
    if image.ndim != 3:
        return 0.0
    return float(np.mean([image[..., c].astype(np.float64).std() for c in range(image.shape[2])]))


def select_representative_images(input_dir: str | Path) -> Dict[str, Path]:
    """Pick four images that exhibit different visual properties."""
    paths = find_images(input_dir, recursive=True)
    if not paths:
        return {}

    grad: List[Tuple[Path, float]] = []
    tex: List[Tuple[Path, float]] = []
    col: List[Tuple[Path, float]] = []
    for p in paths:
        img = load_rgb_image(p)
        grad.append((p, gradient_energy(img)))
        tex.append((p, texture_lbp_score(img)))
        col.append((p, color_diversity_score(img)))

    grad_sorted = sorted(grad, key=lambda x: x[1])
    tex_sorted = sorted(tex, key=lambda x: x[1])
    col_sorted = sorted(col, key=lambda x: x[1])

    return {
        "highest_gradient": grad_sorted[-1][0],
        "highest_texture": tex_sorted[-1][0],
        "lowest_gradient": grad_sorted[0][0],
        "highest_color_diversity": col_sorted[-1][0],
    }


def find_interesting_crops(image: np.ndarray, size: int = 160) -> Dict[str, Tuple[int, int, int, int]]:
    """Slide a non-overlapping ``size × size`` window over ``image`` and pick three crops.

    Returns a dict of crop boxes (``y0, x0, y1, x1``) keyed by:
      - ``"edge_rich"``      → highest gradient_energy
      - ``"texture_rich"``   → highest texture_lbp_score
      - ``"smooth"``         → lowest gradient_energy
    """
    h, w = image.shape[:2]
    size = min(size, h, w)
    if size < 32:
        size = min(h, w)

    scores: List[Tuple[Tuple[int, int, int, int], float, float]] = []
    step = max(size // 2, 32)
    for y in range(0, h - size + 1, step):
        for x in range(0, w - size + 1, step):
            patch = image[y : y + size, x : x + size]
            scores.append(((y, x, y + size, x + size),
                           gradient_energy(patch),
                           texture_lbp_score(patch)))

    if not scores:
        # fall back to a single full-image crop
        return {"edge_rich": (0, 0, h, w), "texture_rich": (0, 0, h, w), "smooth": (0, 0, h, w)}

    edge_box = max(scores, key=lambda s: s[1])[0]
    texture_box = max(scores, key=lambda s: s[2])[0]
    smooth_box = min(scores, key=lambda s: s[1])[0]
    return {
        "edge_rich": edge_box,
        "texture_rich": texture_box,
        "smooth": smooth_box,
    }
