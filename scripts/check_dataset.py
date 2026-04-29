"""Standalone dataset sanity-check script.

Run after installing requirements::

    python scripts/check_dataset.py --input_dir data/kodak/raw

Uses Pillow to read image dimensions, so dependencies must be installed first.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

from PIL import Image

ALLOWED_EXTENSIONS = {".png", ".tif", ".tiff", ".bmp", ".ppm"}
IGNORED_EXTENSIONS = {".zip", ".rar", ".7z", ".txt", ".csv"}


def _find_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    out: List[Path] = []
    for dp, _dn, fns in os.walk(root):
        for fn in fns:
            p = Path(dp) / fn
            suf = p.suffix.lower()
            if suf in IGNORED_EXTENSIONS:
                continue
            if suf in ALLOWED_EXTENSIONS:
                out.append(p)
    return sorted(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the Kodak dataset folder.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/kodak/raw",
        help="Folder to scan. Default: data/kodak/raw",
    )
    args = parser.parse_args()

    root = Path(args.input_dir)
    print(f"Scanning: {root.resolve()}")
    if not root.exists():
        print(f"ERROR: folder does not exist: {root}")
        return 1

    images = _find_images(root)
    n = len(images)
    print(f"Number of images found: {n}")

    if n == 0:
        print(
            "No images were found in data/kodak/raw/. "
            "Place kodim01.png to kodim24.png in this folder."
        )
        return 2

    print(f"\n{'name':<24} {'WxH':>15} {'size (bytes)':>15}")
    print("-" * 60)
    for p in images:
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception as e:  # noqa: BLE001
            w, h = -1, -1
            print(f"  WARNING: could not read dimensions of {p.name}: {e}")
        size = p.stat().st_size
        print(f"{p.name:<24} {f'{w}x{h}':>15} {size:>15}")

    if n == 24:
        print("\nKodak dataset correctly detected: 24 images found.")
    else:
        print(f"\nFound {n} images (expected 24 for the full Kodak dataset).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
