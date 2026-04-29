"""Project paths, compression levels, and folder bootstrapping."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "kodak" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
COMPRESSED_DIR = OUTPUT_DIR / "compressed"
RECONSTRUCTED_DIR = OUTPUT_DIR / "reconstructed"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"

ALLOWED_EXTENSIONS = {".png", ".tif", ".tiff", ".bmp", ".ppm"}
IGNORED_EXTENSIONS = {".zip", ".rar", ".7z", ".txt", ".csv"}

JPEG_LEVELS = {
    "low_compression_q90": 90,
    "medium_compression_q60": 60,
    "high_compression_q30": 30,
}

JPEG2000_LEVELS = {
    "low_compression_cr5": 5,
    "medium_compression_cr10": 10,
    "high_compression_cr20": 20,
}

ALL_SUBDIRS = [
    COMPRESSED_DIR / "jpeg",
    COMPRESSED_DIR / "jpeg2000",
    RECONSTRUCTED_DIR / "jpeg",
    RECONSTRUCTED_DIR / "jpeg2000",
    METRICS_DIR,
    FIGURES_DIR / "comparisons",
    FIGURES_DIR / "error_maps",
    FIGURES_DIR / "crops",
    FIGURES_DIR / "plots",
    FIGURES_DIR / "dct_analysis",
    FIGURES_DIR / "wavelet_analysis",
]


def ensure_output_dirs() -> None:
    """Create every folder under outputs/ if it does not already exist."""
    for d in ALL_SUBDIRS:
        d.mkdir(parents=True, exist_ok=True)
