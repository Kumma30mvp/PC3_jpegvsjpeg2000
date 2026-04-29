"""Plots, error maps, and crop figures for the JPEG vs JPEG2000 comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURES_DIR, JPEG_LEVELS, JPEG2000_LEVELS
from .io_utils import load_rgb_image


# ---------------------------------------------------------------------------
# Aggregate plots driven by the summary CSV
# ---------------------------------------------------------------------------


_LEVEL_ORDER = list(JPEG_LEVELS.keys()) + list(JPEG2000_LEVELS.keys())


def _bar_plot_metric(summary_df: pd.DataFrame, metric_col: str, title: str, ylabel: str, output_path: Path) -> None:
    methods = ["jpeg", "jpeg2000"]
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.35
    x_positions = np.arange(3)

    for i, method in enumerate(methods):
        sub = summary_df[summary_df["method"] == method]
        if sub.empty:
            continue
        levels = list(JPEG_LEVELS.keys()) if method == "jpeg" else list(JPEG2000_LEVELS.keys())
        values = []
        for lv in levels:
            row = sub[sub["level_name"] == lv]
            values.append(float(row[metric_col].iloc[0]) if not row.empty else np.nan)
        offset = (i - 0.5) * width
        ax.bar(x_positions + offset, values, width=width, label=method.upper())
        for xi, v in zip(x_positions + offset, values):
            if np.isfinite(v):
                ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(["low", "medium", "high"])
    ax.set_xlabel("Compression level (low → high)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_mean_psnr(summary_df: pd.DataFrame, output_path: Path) -> None:
    _bar_plot_metric(
        summary_df,
        "mean_psnr",
        "Mean PSNR by method and compression level (higher is better)",
        "PSNR (dB)",
        output_path,
    )


def plot_mean_ssim(summary_df: pd.DataFrame, output_path: Path) -> None:
    _bar_plot_metric(
        summary_df,
        "mean_ssim",
        "Mean SSIM by method and compression level (higher is better)",
        "SSIM",
        output_path,
    )


def plot_mean_compression_ratio(summary_df: pd.DataFrame, output_path: Path) -> None:
    _bar_plot_metric(
        summary_df,
        "mean_compression_ratio_file",
        "Mean Compression Ratio by method and compression level (file-based)",
        "Compression Ratio (original / compressed)",
        output_path,
    )


def _scatter_quality_vs_cr(per_image_df: pd.DataFrame, quality_metric: str, ylabel: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"jpeg": "tab:blue", "jpeg2000": "tab:orange"}
    for method, sub in per_image_df.groupby("method"):
        ax.scatter(
            sub["compression_ratio_file"],
            sub[quality_metric],
            alpha=0.6,
            c=colors.get(method, "gray"),
            label=method.upper(),
        )
    ax.set_xlabel("Compression Ratio (file-based)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs Compression Ratio (per image)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_cr_vs_psnr(per_image_df: pd.DataFrame, output_path: Path) -> None:
    df = per_image_df.copy()
    df = df[np.isfinite(df["psnr"])]
    _scatter_quality_vs_cr(df, "psnr", "PSNR (dB)", output_path)


def plot_cr_vs_ssim(per_image_df: pd.DataFrame, output_path: Path) -> None:
    _scatter_quality_vs_cr(per_image_df, "ssim", "SSIM", output_path)


# ---------------------------------------------------------------------------
# Per-image visual figures
# ---------------------------------------------------------------------------


def make_visual_comparison(
    image_path: Path,
    recon_paths: Dict[Tuple[str, str], Path],
    output_path: Path,
) -> None:
    """Build a 1×7 grid: original + 3 JPEG + 3 JPEG2000 reconstructions."""
    original = load_rgb_image(image_path)
    fig, axes = plt.subplots(1, 7, figsize=(22, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    jpeg_levels = list(JPEG_LEVELS.keys())
    j2k_levels = list(JPEG2000_LEVELS.keys())

    for i, lv in enumerate(jpeg_levels):
        ax = axes[1 + i]
        p = recon_paths.get(("jpeg", lv))
        if p is not None and Path(p).exists():
            ax.imshow(load_rgb_image(p))
        ax.set_title(f"JPEG\n{lv}")
        ax.axis("off")

    for i, lv in enumerate(j2k_levels):
        ax = axes[4 + i]
        p = recon_paths.get(("jpeg2000", lv))
        if p is not None and Path(p).exists():
            ax.imshow(load_rgb_image(p))
        ax.set_title(f"JPEG2000\n{lv}")
        ax.axis("off")

    fig.suptitle(f"Visual comparison — {image_path.name}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)


def make_error_map(
    original: np.ndarray,
    reconstructed: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    if original.shape != reconstructed.shape:
        raise ValueError("error map needs matching shapes")
    err = np.abs(original.astype(np.int16) - reconstructed.astype(np.int16))
    err_visual = np.clip(err * 4, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(reconstructed); axes[1].set_title("Reconstructed"); axes[1].axis("off")
    axes[2].imshow(err_visual); axes[2].set_title("|err| × 4 (clipped)"); axes[2].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)


def make_crop_comparison(
    image_path: Path,
    recon_paths: Dict[Tuple[str, str], Path],
    crop_box: Tuple[int, int, int, int],
    crop_name: str,
    output_path: Path,
) -> None:
    y0, x0, y1, x1 = crop_box
    original = load_rgb_image(image_path)[y0:y1, x0:x1]
    fig, axes = plt.subplots(1, 7, figsize=(22, 4))
    axes[0].imshow(original); axes[0].set_title("Original"); axes[0].axis("off")

    jpeg_levels = list(JPEG_LEVELS.keys())
    j2k_levels = list(JPEG2000_LEVELS.keys())

    for i, lv in enumerate(jpeg_levels):
        ax = axes[1 + i]
        p = recon_paths.get(("jpeg", lv))
        if p is not None and Path(p).exists():
            ax.imshow(load_rgb_image(p)[y0:y1, x0:x1])
        ax.set_title(f"JPEG\n{lv}")
        ax.axis("off")

    for i, lv in enumerate(j2k_levels):
        ax = axes[4 + i]
        p = recon_paths.get(("jpeg2000", lv))
        if p is not None and Path(p).exists():
            ax.imshow(load_rgb_image(p)[y0:y1, x0:x1])
        ax.set_title(f"JPEG2000\n{lv}")
        ax.axis("off")

    fig.suptitle(f"Crop: {crop_name} ({y0}:{y1}, {x0}:{x1}) — {image_path.name}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)
