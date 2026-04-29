"""Re-render summary figures from the existing metrics CSVs.

This script does not re-run the JPEG/JPEG2000 experiment. It reads:

    outputs/metrics/per_image_results.csv
    outputs/metrics/summary_by_method_level.csv

and writes the following PNG files into ``outputs/figures/plots/``:

    summary_metrics_dashboard.png
    compression_ratio_barplot.png
    psnr_barplot.png
    ssim_barplot.png
    cr_vs_psnr_scatter.png
    cr_vs_ssim_scatter.png
    summary_table.png

Usage::

    python scripts/generate_summary_report.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
DEFAULT_PLOTS_DIR = PROJECT_ROOT / "outputs" / "figures" / "plots"

LEVELS = {
    "jpeg": ["low_compression_q90", "medium_compression_q60", "high_compression_q30"],
    "jpeg2000": ["low_compression_cr5", "medium_compression_cr10", "high_compression_cr20"],
}
METHOD_COLORS = {"jpeg": "tab:blue", "jpeg2000": "tab:orange"}
METHOD_LABELS = {"jpeg": "JPEG", "jpeg2000": "JPEG2000"}
LEVEL_TICK_LABELS = ["low", "medium", "high"]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _grouped_bars(
    summary_df: pd.DataFrame,
    metric_col: str,
    ax: plt.Axes,
    *,
    title: str,
    ylabel: str,
    annotate_fmt: str = "{:.2f}",
) -> None:
    width = 0.38
    x_positions = np.arange(3)
    methods = ["jpeg", "jpeg2000"]

    for i, method in enumerate(methods):
        sub = summary_df[summary_df["method"] == method]
        values = []
        for lv in LEVELS[method]:
            row = sub[sub["level_name"] == lv]
            values.append(float(row[metric_col].iloc[0]) if not row.empty else np.nan)
        offset = (i - 0.5) * width
        bars = ax.bar(
            x_positions + offset,
            values,
            width=width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.6,
        )
        for rect, v in zip(bars, values):
            if np.isfinite(v):
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    rect.get_height(),
                    annotate_fmt.format(v),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(LEVEL_TICK_LABELS)
    ax.set_xlabel("Compression level (low → high)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)


def _scatter_cr_vs(
    per_image_df: pd.DataFrame,
    quality_metric: str,
    ax: plt.Axes,
    *,
    title: str,
    ylabel: str,
) -> None:
    df = per_image_df.copy()
    df = df[np.isfinite(df[quality_metric])]
    for method, sub in df.groupby("method"):
        ax.scatter(
            sub["compression_ratio_file"],
            sub[quality_metric],
            alpha=0.65,
            s=32,
            c=METHOD_COLORS.get(method, "gray"),
            label=METHOD_LABELS.get(method, method),
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xlabel("Compression Ratio (file size based)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)


def make_compression_ratio_barplot(summary_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    _grouped_bars(
        summary_df,
        "mean_compression_ratio_file",
        ax,
        title="Mean Compression Ratio by method and level (file-size based)",
        ylabel="Compression Ratio (original / compressed)",
        annotate_fmt="{:.2f}×",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def make_psnr_barplot(summary_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    _grouped_bars(
        summary_df,
        "mean_psnr",
        ax,
        title="Mean PSNR by method and level (higher is better)",
        ylabel="PSNR (dB)",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def make_ssim_barplot(summary_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    _grouped_bars(
        summary_df,
        "mean_ssim",
        ax,
        title="Mean SSIM by method and level (higher is better)",
        ylabel="SSIM (0…1)",
        annotate_fmt="{:.4f}",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def make_cr_vs_psnr_scatter(per_image_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6))
    _scatter_cr_vs(
        per_image_df,
        "psnr",
        ax,
        title="Compression Ratio vs PSNR (per image)",
        ylabel="PSNR (dB)",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def make_cr_vs_ssim_scatter(per_image_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6))
    _scatter_cr_vs(
        per_image_df,
        "ssim",
        ax,
        title="Compression Ratio vs SSIM (per image)",
        ylabel="SSIM (0…1)",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def make_dashboard(per_image_df: pd.DataFrame, summary_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "JPEG vs JPEG2000 — Summary Dashboard (Kodak dataset)",
        fontsize=16,
        fontweight="bold",
    )

    _grouped_bars(
        summary_df, "mean_compression_ratio_file", axes[0, 0],
        title="Mean Compression Ratio",
        ylabel="CR (×)", annotate_fmt="{:.2f}×",
    )
    _grouped_bars(
        summary_df, "mean_psnr", axes[0, 1],
        title="Mean PSNR (dB)",
        ylabel="PSNR (dB)",
    )
    _grouped_bars(
        summary_df, "mean_ssim", axes[0, 2],
        title="Mean SSIM",
        ylabel="SSIM",
        annotate_fmt="{:.4f}",
    )
    _scatter_cr_vs(
        per_image_df, "psnr", axes[1, 0],
        title="CR vs PSNR (per image)",
        ylabel="PSNR (dB)",
    )
    _scatter_cr_vs(
        per_image_df, "ssim", axes[1, 1],
        title="CR vs SSIM (per image)",
        ylabel="SSIM",
    )

    # Bottom-right cell: encoding/decoding mean times.
    ax = axes[1, 2]
    width = 0.38
    x_positions = np.arange(3)
    methods = ["jpeg", "jpeg2000"]
    for i, method in enumerate(methods):
        sub = summary_df[summary_df["method"] == method]
        enc = []
        dec = []
        for lv in LEVELS[method]:
            row = sub[sub["level_name"] == lv]
            if row.empty:
                enc.append(np.nan); dec.append(np.nan)
            else:
                enc.append(float(row["mean_encoding_time_ms"].iloc[0]))
                dec.append(float(row["mean_decoding_time_ms"].iloc[0]))
        offset = (i - 0.5) * width
        ax.bar(
            x_positions + offset, enc, width=width,
            label=f"{METHOD_LABELS[method]} encode",
            color=METHOD_COLORS[method], edgecolor="black", linewidth=0.6,
        )
        ax.bar(
            x_positions + offset, dec, width=width, bottom=enc,
            label=f"{METHOD_LABELS[method]} decode",
            color=METHOD_COLORS[method], alpha=0.45, edgecolor="black", linewidth=0.6,
            hatch="//",
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(LEVEL_TICK_LABELS)
    ax.set_xlabel("Compression level")
    ax.set_ylabel("Mean time per image (ms)")
    ax.set_title("Encode + decode time")
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def make_summary_table_png(summary_df: pd.DataFrame, out_path: Path) -> None:
    cols = [
        "method",
        "level_name",
        "number_of_images",
        "mean_compression_ratio_file",
        "mean_psnr",
        "mean_ssim",
        "mean_encoding_time_ms",
        "mean_decoding_time_ms",
    ]
    df = summary_df[cols].copy()
    method_order = {"jpeg": 0, "jpeg2000": 1}
    level_order = {
        "low_compression_q90": 0, "medium_compression_q60": 1, "high_compression_q30": 2,
        "low_compression_cr5": 0, "medium_compression_cr10": 1, "high_compression_cr20": 2,
    }
    df = df.assign(
        _m=df["method"].map(method_order),
        _l=df["level_name"].map(level_order),
    ).sort_values(["_m", "_l"]).drop(columns=["_m", "_l"]).reset_index(drop=True)

    pretty_headers = [
        "method", "level", "n_images",
        "mean CR (×)", "mean PSNR (dB)", "mean SSIM",
        "mean encode (ms)", "mean decode (ms)",
    ]
    cell_text = []
    for _, row in df.iterrows():
        cell_text.append([
            str(row["method"]),
            str(row["level_name"]),
            f"{int(row['number_of_images'])}",
            f"{row['mean_compression_ratio_file']:.2f}",
            f"{row['mean_psnr']:.2f}",
            f"{row['mean_ssim']:.4f}",
            f"{row['mean_encoding_time_ms']:.2f}",
            f"{row['mean_decoding_time_ms']:.2f}",
        ])

    fig, ax = plt.subplots(figsize=(13, 0.55 * (len(cell_text) + 2) + 1.5))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=pretty_headers,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # Header styling.
    n_cols = len(pretty_headers)
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#264653")
        cell.set_text_props(color="white", weight="bold")
    # Method coloring on the data rows.
    for i, row in enumerate(cell_text, start=1):
        bg = "#e8f1fb" if row[0] == "jpeg" else "#fdecd2"
        for j in range(n_cols):
            table[i, j].set_facecolor(bg)

    ax.set_title(
        "Summary by method and compression level (from summary_by_method_level.csv)",
        pad=14, fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--metrics_dir", type=str, default=str(DEFAULT_METRICS_DIR),
        help="Folder containing the CSVs. Default: outputs/metrics",
    )
    parser.add_argument(
        "--plots_dir", type=str, default=str(DEFAULT_PLOTS_DIR),
        help="Folder where PNGs will be written. Default: outputs/figures/plots",
    )
    args = parser.parse_args(argv)

    metrics_dir = Path(args.metrics_dir)
    plots_dir = _ensure_dir(Path(args.plots_dir))

    per_image_csv = metrics_dir / "per_image_results.csv"
    summary_csv = metrics_dir / "summary_by_method_level.csv"

    if not per_image_csv.exists():
        print(f"ERROR: missing {per_image_csv}", file=sys.stderr)
        return 1
    if not summary_csv.exists():
        print(f"ERROR: missing {summary_csv}", file=sys.stderr)
        return 2

    per_image_df = pd.read_csv(per_image_csv)
    summary_df = pd.read_csv(summary_csv)
    if per_image_df.empty:
        print(f"ERROR: {per_image_csv} is empty", file=sys.stderr)
        return 3
    if summary_df.empty:
        print(f"ERROR: {summary_csv} is empty", file=sys.stderr)
        return 4

    out_paths = {
        "compression_ratio_barplot.png":
            (make_compression_ratio_barplot, (summary_df,)),
        "psnr_barplot.png":
            (make_psnr_barplot, (summary_df,)),
        "ssim_barplot.png":
            (make_ssim_barplot, (summary_df,)),
        "cr_vs_psnr_scatter.png":
            (make_cr_vs_psnr_scatter, (per_image_df,)),
        "cr_vs_ssim_scatter.png":
            (make_cr_vs_ssim_scatter, (per_image_df,)),
        "summary_metrics_dashboard.png":
            (make_dashboard, (per_image_df, summary_df)),
        "summary_table.png":
            (make_summary_table_png, (summary_df,)),
    }

    for filename, (fn, args_tuple) in out_paths.items():
        out_path = plots_dir / filename
        fn(*args_tuple, out_path)
        print(f"  wrote {out_path}")

    print(f"\nDone. {len(out_paths)} figures written under {plots_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
