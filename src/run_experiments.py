"""End-to-end pipeline: encode/decode → metrics CSVs → figures → metadata JSON.

CLI entry point. Run with::

    python -m src.run_experiments --help
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config
from .complexity_analysis import find_interesting_crops, select_representative_images
from .educational_dct import create_dct_analysis_figure
from .educational_wavelet import create_wavelet_analysis_figure
from .io_utils import find_images, load_rgb_image, validate_dataset
from .jpeg2000_codec import check_jpeg2000_backend, encode_decode_jpeg2000
from .jpeg_codec import JPEG_BACKEND_NAME, check_jpeg_backend, encode_decode_jpeg
from .metrics import (
    compression_ratio_file,
    compression_ratio_raw,
    file_size_bytes,
    mse,
    psnr,
    ssim_rgb,
)
from . import visualization


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_experiments",
        description="Compare JPEG (DCT) and JPEG2000 (wavelet) on the Kodak dataset.",
    )
    p.add_argument(
        "--input_dir",
        type=str,
        default=str(config.RAW_DATA_DIR),
        help="Folder containing the input images. Default: data/kodak/raw",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=str(config.OUTPUT_DIR),
        help="Folder where outputs/ will be written. Default: outputs",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        choices=["jpeg", "jpeg2000"],
        default=["jpeg", "jpeg2000"],
        help="Which compression methods to run. Default: jpeg jpeg2000",
    )
    p.add_argument(
        "--make_figures",
        action="store_true",
        help="Generate plots, error maps, and crop comparisons.",
    )
    p.add_argument(
        "--make_algorithm_analysis",
        action="store_true",
        help="Generate the educational DCT and wavelet figures.",
    )
    p.add_argument(
        "--only_check_backends",
        action="store_true",
        help="Only verify codec backends; do not require the dataset folder.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N images (after sorting). Useful for smoke tests.",
    )
    return p


# ---------------------------------------------------------------------------
# Single-image processing
# ---------------------------------------------------------------------------


def _process_jpeg(
    image: np.ndarray,
    image_path: Path,
    rows: List[Dict],
    recon_paths: Dict[Tuple[str, str], Path],
) -> None:
    h, w, c = image.shape
    raw_size = h * w * c
    orig_size = file_size_bytes(image_path)

    for level_name, quality in config.JPEG_LEVELS.items():
        result = encode_decode_jpeg(image, quality, image_path.stem, level_name)
        recon = result["reconstructed_array"]

        cr_file = compression_ratio_file(image_path, result["compressed_path"])
        cr_raw = compression_ratio_raw(image, result["compressed_path"])
        m = mse(image, recon)
        p = psnr(image, recon)
        s = ssim_rgb(image, recon)
        if math.isnan(p) or math.isnan(s):
            raise RuntimeError(f"NaN metric for {image_path.name} jpeg/{level_name}")

        rows.append({
            "image_name": image_path.name,
            "method": "jpeg",
            "level_name": level_name,
            "level_value": quality,
            "codec_backend": result["backend"],
            "width": w,
            "height": h,
            "channels": c,
            "original_file_size_bytes": orig_size,
            "raw_size_bytes": raw_size,
            "compressed_file_size_bytes": file_size_bytes(result["compressed_path"]),
            "compression_ratio_file": cr_file,
            "compression_ratio_raw": cr_raw,
            "mse": m,
            "psnr": p,
            "ssim": s,
            "encoding_time_ms": result["encoding_time_ms"],
            "decoding_time_ms": result["decoding_time_ms"],
            "compressed_path": result["compressed_path"],
            "reconstructed_path": result["reconstructed_path"],
        })
        recon_paths[("jpeg", level_name)] = Path(result["reconstructed_path"])


def _process_jpeg2000(
    image: np.ndarray,
    image_path: Path,
    rows: List[Dict],
    recon_paths: Dict[Tuple[str, str], Path],
) -> None:
    h, w, c = image.shape
    raw_size = h * w * c
    orig_size = file_size_bytes(image_path)

    for level_name, cr in config.JPEG2000_LEVELS.items():
        result = encode_decode_jpeg2000(image, cr, image_path.stem, level_name)
        recon = result["reconstructed_array"]

        cr_file = compression_ratio_file(image_path, result["compressed_path"])
        cr_raw = compression_ratio_raw(image, result["compressed_path"])
        m = mse(image, recon)
        p = psnr(image, recon)
        s = ssim_rgb(image, recon)
        if math.isnan(p) or math.isnan(s):
            raise RuntimeError(f"NaN metric for {image_path.name} jpeg2000/{level_name}")

        rows.append({
            "image_name": image_path.name,
            "method": "jpeg2000",
            "level_name": level_name,
            "level_value": cr,
            "codec_backend": result["backend"],
            "width": w,
            "height": h,
            "channels": c,
            "original_file_size_bytes": orig_size,
            "raw_size_bytes": raw_size,
            "compressed_file_size_bytes": file_size_bytes(result["compressed_path"]),
            "compression_ratio_file": cr_file,
            "compression_ratio_raw": cr_raw,
            "mse": m,
            "psnr": p,
            "ssim": s,
            "encoding_time_ms": result["encoding_time_ms"],
            "decoding_time_ms": result["decoding_time_ms"],
            "compressed_path": result["compressed_path"],
            "reconstructed_path": result["reconstructed_path"],
        })
        recon_paths[("jpeg2000", level_name)] = Path(result["reconstructed_path"])


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _build_summary(per_image_df: pd.DataFrame) -> pd.DataFrame:
    grouped = per_image_df.groupby(["method", "level_name"], as_index=False)
    summary = grouped.agg(
        number_of_images=("image_name", "count"),
        mean_compressed_size_bytes=("compressed_file_size_bytes", "mean"),
        std_compressed_size_bytes=("compressed_file_size_bytes", "std"),
        mean_compression_ratio_file=("compression_ratio_file", "mean"),
        std_compression_ratio_file=("compression_ratio_file", "std"),
        mean_compression_ratio_raw=("compression_ratio_raw", "mean"),
        std_compression_ratio_raw=("compression_ratio_raw", "std"),
        mean_mse=("mse", "mean"),
        std_mse=("mse", "std"),
        mean_psnr=("psnr", "mean"),
        std_psnr=("psnr", "std"),
        mean_ssim=("ssim", "mean"),
        std_ssim=("ssim", "std"),
        mean_encoding_time_ms=("encoding_time_ms", "mean"),
        mean_decoding_time_ms=("decoding_time_ms", "mean"),
    )
    return summary


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _generate_aggregate_figures(per_image_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    plots_dir = config.FIGURES_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    visualization.plot_mean_psnr(summary_df, plots_dir / "mean_psnr_by_method_level.png")
    visualization.plot_mean_ssim(summary_df, plots_dir / "mean_ssim_by_method_level.png")
    visualization.plot_mean_compression_ratio(
        summary_df, plots_dir / "mean_cr_by_method_level.png"
    )
    visualization.plot_cr_vs_psnr(per_image_df, plots_dir / "cr_vs_psnr.png")
    visualization.plot_cr_vs_ssim(per_image_df, plots_dir / "cr_vs_ssim.png")


def _generate_visual_figures(
    representative: Dict[str, Path],
    per_image_recons: Dict[str, Dict[Tuple[str, str], Path]],
) -> None:
    comparisons_dir = config.FIGURES_DIR / "comparisons"
    error_dir = config.FIGURES_DIR / "error_maps"
    crops_dir = config.FIGURES_DIR / "crops"
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    for label, image_path in representative.items():
        recons = per_image_recons.get(image_path.name)
        if not recons:
            continue
        visualization.make_visual_comparison(
            image_path,
            recons,
            comparisons_dir / f"{image_path.stem}__{label}.png",
        )

        original = load_rgb_image(image_path)
        for (method, level_name), recon_path in recons.items():
            recon = load_rgb_image(recon_path)
            visualization.make_error_map(
                original,
                recon,
                error_dir / f"{image_path.stem}__{method}__{level_name}.png",
                title=f"{image_path.name} — {method} {level_name}",
            )

        crop_boxes = find_interesting_crops(original, size=160)
        for crop_name, box in crop_boxes.items():
            visualization.make_crop_comparison(
                image_path,
                recons,
                box,
                crop_name,
                crops_dir / f"{image_path.stem}__{label}__{crop_name}.png",
            )


def _generate_algorithm_analysis(representative: Dict[str, Path]) -> None:
    dct_dir = config.FIGURES_DIR / "dct_analysis"
    wav_dir = config.FIGURES_DIR / "wavelet_analysis"
    dct_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    chosen: List[Path] = []
    seen = set()
    for key in ("highest_gradient", "highest_texture"):
        p = representative.get(key)
        if p is not None and p not in seen:
            chosen.append(p)
            seen.add(p)
    if not chosen and representative:
        chosen = [next(iter(representative.values()))]

    for p in chosen:
        create_dct_analysis_figure(p, dct_dir, quality=30)
        create_wavelet_analysis_figure(p, wav_dir, wavelet="haar", level=2, step=10.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    config.OUTPUT_DIR = Path(args.output_dir)
    # Re-derive subpaths from the (possibly overridden) output dir.
    config.COMPRESSED_DIR = config.OUTPUT_DIR / "compressed"
    config.RECONSTRUCTED_DIR = config.OUTPUT_DIR / "reconstructed"
    config.METRICS_DIR = config.OUTPUT_DIR / "metrics"
    config.FIGURES_DIR = config.OUTPUT_DIR / "figures"
    config.ALL_SUBDIRS = [
        config.COMPRESSED_DIR / "jpeg",
        config.COMPRESSED_DIR / "jpeg2000",
        config.RECONSTRUCTED_DIR / "jpeg",
        config.RECONSTRUCTED_DIR / "jpeg2000",
        config.METRICS_DIR,
        config.FIGURES_DIR / "comparisons",
        config.FIGURES_DIR / "error_maps",
        config.FIGURES_DIR / "crops",
        config.FIGURES_DIR / "plots",
        config.FIGURES_DIR / "dct_analysis",
        config.FIGURES_DIR / "wavelet_analysis",
    ]
    config.ensure_output_dirs()

    if args.only_check_backends:
        # Backend checks must NOT require the dataset.
        print("Checking codec backends...")
        try:
            jpeg_name = check_jpeg_backend()
            print(f"  JPEG backend:     OK  ({jpeg_name})")
        except Exception as e:  # noqa: BLE001
            print(f"  JPEG backend:     FAIL  ({type(e).__name__}: {e})")
            return 1
        try:
            j2k_name = check_jpeg2000_backend()
            print(f"  JPEG2000 backend: OK  ({j2k_name})")
        except RuntimeError as e:
            print(f"  JPEG2000 backend: FAIL  ({e})")
            return 2
        return 0

    images = validate_dataset(args.input_dir)
    if not images:
        print("No images to process. Exiting.")
        return 3

    if args.limit is not None and args.limit > 0:
        images = images[: args.limit]
        print(f"--limit {args.limit} active: processing first {len(images)} images.")

    jpeg_backend_name = check_jpeg_backend()
    jpeg2000_backend_name: Optional[str] = None
    if "jpeg2000" in args.methods:
        jpeg2000_backend_name = check_jpeg2000_backend()
        print(f"Active JPEG2000 backend: {jpeg2000_backend_name}")

    rows: List[Dict] = []
    per_image_recons: Dict[str, Dict[Tuple[str, str], Path]] = {}

    for image_path in tqdm(images, desc="Encoding"):
        image = load_rgb_image(image_path)
        recon_paths: Dict[Tuple[str, str], Path] = {}

        if "jpeg" in args.methods:
            _process_jpeg(image, image_path, rows, recon_paths)
        if "jpeg2000" in args.methods:
            _process_jpeg2000(image, image_path, rows, recon_paths)

        per_image_recons[image_path.name] = recon_paths

    if not rows:
        print("No metrics rows produced. Exiting.")
        return 4

    per_image_df = pd.DataFrame(rows)

    expected_rows = len(images) * len(args.methods) * 3
    if per_image_df.shape[0] != expected_rows:
        print(
            f"WARNING: per_image_results.csv has {per_image_df.shape[0]} rows, "
            f"expected {expected_rows} (images × methods × levels)."
        )

    summary_df = _build_summary(per_image_df)
    if summary_df.empty:
        raise RuntimeError("summary_by_method_level.csv would be empty; aborting.")

    per_image_csv = config.METRICS_DIR / "per_image_results.csv"
    summary_csv = config.METRICS_DIR / "summary_by_method_level.csv"
    per_image_df.to_csv(per_image_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    print(f"Wrote {per_image_csv}  ({per_image_df.shape[0]} rows)")
    print(f"Wrote {summary_csv}  ({summary_df.shape[0]} rows)")

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "input_dir": str(Path(args.input_dir).resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
        "methods": list(args.methods),
        "limit": args.limit,
        "number_of_images_processed": len(images),
        "expected_csv_rows": expected_rows,
        "actual_csv_rows": int(per_image_df.shape[0]),
        "jpeg_backend": jpeg_backend_name,
        "jpeg2000_backend": jpeg2000_backend_name,
        "jpeg_levels": config.JPEG_LEVELS,
        "jpeg2000_levels": config.JPEG2000_LEVELS,
    }
    metadata_path = config.METRICS_DIR / "experiment_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {metadata_path}")

    if args.make_figures or args.make_algorithm_analysis:
        representative = select_representative_images(args.input_dir)
        # Filter to images we actually processed (limit-aware).
        processed_names = {p.name for p in images}
        representative = {k: v for k, v in representative.items() if v.name in processed_names}

        if args.make_figures:
            _generate_aggregate_figures(per_image_df, summary_df)
            _generate_visual_figures(representative, per_image_recons)
            print(f"Figures written under {config.FIGURES_DIR}")

        if args.make_algorithm_analysis:
            _generate_algorithm_analysis(representative)
            print(f"Algorithm analysis figures under {config.FIGURES_DIR}")

    print("\n=== Final summary (mean per method × level) ===")
    cols = [
        "method", "level_name", "number_of_images",
        "mean_compression_ratio_file", "mean_psnr", "mean_ssim",
        "mean_encoding_time_ms", "mean_decoding_time_ms",
    ]
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print(summary_df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
