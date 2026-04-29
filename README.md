# JPEG vs JPEG2000 Compression Analysis on Kodak Dataset

A reproducible Python project that compares **JPEG** (DCT-based) and **JPEG2000**
(wavelet-based) compression on the *Kodak Lossless True Color Image Dataset*
(24 PNGs, 24-bit RGB). For every image and every compression level, the
pipeline encodes, decodes, measures the resulting file size, and computes
fidelity metrics (PSNR, SSIM) — then renders comparative plots, error maps,
and artifact crops.

The project is designed to run from VS Code on Windows and to be cloned and
reproduced by classmates.

---

## Academic context

Image compression reduces the storage footprint of an image, ideally without
visibly altering its content. Two foundational lossy methods are studied here:

- **JPEG** — block-based **Discrete Cosine Transform (DCT)**, scalar
  quantization on each 8×8 block, entropy coding. Standardized in 1992.
- **JPEG2000** — image-wide **discrete wavelet transform (DWT)**, embedded
  bit-plane coding (EBCOT). Standardized in 2000. Designed to address JPEG's
  blocking artifacts and to support progressive decoding.

The goal of this project is to evaluate, on the same images:

1. **Compression efficiency** — how small are the files?
2. **Image quality** — how much information is lost?
3. **Visual artifacts** — what kind of degradation does each method introduce?

---

## Key concepts

| Term | Short definition |
|---|---|
| **Image compression** | Encoding an image with fewer bytes than its raw pixel storage. |
| **Lossy compression** | The decoded image is *similar* but not bit-identical to the original. Some information is permanently discarded in exchange for a smaller file. |
| **JPEG** | Lossy codec based on 8×8 block DCT, quantization, and entropy coding. |
| **DCT** | Discrete Cosine Transform; expresses an 8×8 block as a sum of cosine basis functions. Most natural-image energy concentrates in low frequencies. |
| **JPEG2000** | Lossy/lossless codec based on a multi-resolution wavelet transform. Replaces DCT blocking with smoother, frequency-localized degradation. |
| **Wavelet transform** | A multi-scale decomposition into approximation (LL) and detail (LH/HL/HH) sub-bands. JPEG2000 uses the irreversible 9/7 wavelet for lossy coding. |
| **Compression Ratio (CR)** | `original_file_size_bytes / compressed_file_size_bytes`. Higher = stronger compression. |
| **MSE** | Mean squared error between original and reconstructed pixels. Lower = closer. |
| **PSNR** | `10·log10(255² / MSE)`. Pixel-level fidelity in dB. Higher is better. |
| **SSIM** | Structural Similarity Index (0…1). Models luminance/contrast/structure perception. Higher is better. |
| **Error map** | `\|original − reconstructed\|`, often amplified, to visualize where the codec lost information. |
| **Blocking artifacts** | Visible 8×8 grid pattern produced by JPEG at high compression. |
| **Blur** | Loss of high-frequency detail; common to both codecs at very high compression. |
| **Ringing** | Oscillations near sharp edges, especially in wavelet-coded images. |
| **Loss of fine texture** | Smoothing of small repetitive patterns (fabric, foliage, skin pores). |

---

## Dataset

Place the 24 Kodak images in:

```
data/kodak/raw/kodim01.png
data/kodak/raw/kodim02.png
...
data/kodak/raw/kodim24.png
```

**The dataset is *not* committed to GitHub** (see `.gitignore`). Each user
should download it locally.

- Kaggle source:
  <https://www.kaggle.com/datasets/sherylmehta/kodak-dataset/data>

If a Kaggle archive (`archive.zip`) is left in the project root, the pipeline
ignores it.

---

## Folder structure

```
pc3/
├── data/kodak/raw/                  # 24 PNGs (not committed)
├── src/                             # Python source modules
├── scripts/                         # standalone CLI scripts
└── outputs/                         # all generated artefacts
    ├── compressed/                  # real .jpg and .jp2 files (not committed)
    │   ├── jpeg/
    │   └── jpeg2000/
    ├── reconstructed/               # decoded reconstructions, saved as PNG (not committed)
    │   ├── jpeg/
    │   └── jpeg2000/
    ├── metrics/                     # CSVs + JSON metadata (committed)
    └── figures/                     # all figures (committed)
        ├── plots/                   # bar charts, scatter plots, dashboard
        ├── comparisons/             # original vs all 6 reconstructions, side by side
        ├── error_maps/              # |original − reconstructed| ×4
        ├── crops/                   # 160×160 patches (edge / texture / smooth)
        ├── dct_analysis/            # educational JPEG/DCT block walkthrough
        └── wavelet_analysis/        # educational wavelet sub-band decomposition
```

---

## Source code

| File | Purpose |
|---|---|
| `src/config.py` | Single source of truth for project paths, compression levels (`q90/q60/q30` for JPEG; `cr5/cr10/cr20` for JPEG2000), allowed/ignored extensions, and `ensure_output_dirs()`. |
| `src/io_utils.py` | `find_images()`, `validate_dataset()`, `load_rgb_image()` — strict RGB uint8 loading, no resizing or histogram changes. |
| `src/metrics.py` | MSE, PSNR, SSIM (`scikit-image`, `data_range=255, channel_axis=-1`), file-size based CR, raw-size based CR. |
| `src/jpeg_codec.py` | Real JPEG codec via Pillow. Saves `.jpg`, decodes back to RGB, saves the reconstruction as a lossless PNG so it is never recompressed. |
| `src/jpeg2000_codec.py` | Real JPEG2000 codec with multiple backends (glymur → imagecodecs → Pillow JPEG2000 → OpenCV). Strictly validates the output: file exists, size > 0, JP2/J2K magic bytes are present, decoded shape matches the input. |
| `src/educational_dct.py` | Illustrative DCT pipeline (RGB↔YCbCr, 8×8 split, `cv2.dct`, standard JPEG luma quantization matrix). Used only for the analysis figure, **not** for measured metrics. |
| `src/educational_wavelet.py` | Illustrative `pywt` 2-level wavelet decomposition + uniform quantization. Used only for the analysis figure. |
| `src/complexity_analysis.py` | Sobel gradient energy, 8-neighbour LBP entropy, color-diversity score; selects representative images and 160×160 crops (edge-rich / texture-rich / smooth). |
| `src/visualization.py` | All matplotlib plots: bar charts of mean CR/PSNR/SSIM, scatter plots, side-by-side comparisons, error maps, crop comparisons. No seaborn. |
| `src/run_experiments.py` | argparse CLI orchestrator. Validates dataset, checks codec backends, runs the full encode/decode loop, writes `per_image_results.csv`, `summary_by_method_level.csv`, and `experiment_metadata.json`. |
| `scripts/check_dataset.py` | Standalone dataset sanity check: prints file count, names, dimensions, sizes, and confirms the 24 Kodak images. |
| `scripts/generate_summary_report.py` | Re-renders summary plots (dashboard, bar charts, scatter plots, summary table image) from the existing CSVs without re-running the experiment. |

---

## Methodology

For each image and each (method, level) pair:

1. **Load** the original PNG as an RGB uint8 array.
2. **Encode** with JPEG at three quality levels (`q=90, 60, 30`).
3. **Encode** with JPEG2000 at three target compression levels (`cr=5, 10, 20`).
4. **Decode** every compressed file back to RGB uint8.
5. **Save** the reconstruction as a lossless PNG (so the recorded array equals
   what would render on screen).
6. **Measure**:
   - on-disk compressed file size,
   - file-based and raw-based compression ratios,
   - MSE, PSNR, SSIM (RGB),
   - encoding and decoding time in milliseconds.
7. **Aggregate** into `per_image_results.csv` (one row per
   image × method × level) and `summary_by_method_level.csv` (one row per
   method × level with mean and std).
8. **Render** plots, error maps, and artifact crops for representative images.

### Important note about JPEG2000 levels

The labels `cr5`, `cr10`, and `cr20` are **target** compression configurations.
Different JPEG2000 backends accept different parameters:

- `glymur` accepts `cratios=[cr]` directly, so the target is honoured.
- `imagecodecs` accepts a quality `level` parameter; the project maps `cr` to
  a quality level (`cr5 → 90`, `cr10 → 60`, `cr20 → 30`) and prints a warning
  that the actual ratio is whatever the encoder produces.
- `Pillow` and `OpenCV` use their own knobs.

**The official Compression Ratio used in the report is always
`original_file_size_bytes / compressed_file_size_bytes`** computed from the
real `.jp2` file on disk, not from the encoder argument. The active backend
is recorded in the `codec_backend` CSV column and in
`outputs/metrics/experiment_metadata.json`.

---

## Installation (Windows + VS Code)

```bat
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `pillow`, `opencv-python`, `scikit-image`,
`matplotlib`, `tqdm`, `imagecodecs`, `glymur`, `PyWavelets`.

---

## Running the experiments

### 1. Sanity check the dataset

```bat
python scripts/check_dataset.py --input_dir data/kodak/raw
```

Expected end of output:

```
Kodak dataset correctly detected: 24 images found.
```

### 2. Verify codec backends without needing the dataset

```bat
python -m src.run_experiments --only_check_backends
```

### 3. Smoke test on 3 images

```bat
python -m src.run_experiments --input_dir data/kodak/raw --methods jpeg jpeg2000 --make_figures --make_algorithm_analysis --limit 3
```

### 4. Full run (24 images, all 6 method × level combinations)

```bat
python -m src.run_experiments --input_dir data/kodak/raw --methods jpeg jpeg2000 --make_figures --make_algorithm_analysis
```

### 5. Re-render summary figures from the CSVs (no re-encoding)

```bat
python scripts/generate_summary_report.py
```

This script reads
`outputs/metrics/per_image_results.csv` and
`outputs/metrics/summary_by_method_level.csv` and writes:

- `outputs/figures/plots/summary_metrics_dashboard.png`
- `outputs/figures/plots/compression_ratio_barplot.png`
- `outputs/figures/plots/psnr_barplot.png`
- `outputs/figures/plots/ssim_barplot.png`
- `outputs/figures/plots/cr_vs_psnr_scatter.png`
- `outputs/figures/plots/cr_vs_ssim_scatter.png`
- `outputs/figures/plots/summary_table.png`

---

## Results

The numbers below come directly from `outputs/metrics/summary_by_method_level.csv`
on the committed run. Active codec backends were `Pillow-JPEG` and
`imagecodecs-JPEG2000` (Python 3.14.4, Windows 11).

| method   | level_name              | number_of_images | mean_compression_ratio_file | mean_psnr (dB) | mean_ssim |
|----------|-------------------------|------------------|-----------------------------|----------------|-----------|
| jpeg     | low_compression_q90     | 24               | 5.81                        | 38.03          | 0.9634    |
| jpeg     | medium_compression_q60  | 24               | 13.67                       | 32.91          | 0.9117    |
| jpeg     | high_compression_q30    | 24               | 22.70                       | 30.49          | 0.8651    |
| jpeg2000 | low_compression_cr5     | 24               | 2.23                        | 51.05          | 0.9965    |
| jpeg2000 | medium_compression_cr10 | 24               | 2.36                        | 50.60          | 0.9962    |
| jpeg2000 | high_compression_cr20   | 24               | 59.65                       | 29.84          | 0.8206    |

> The `imagecodecs` JPEG2000 backend uses a quality knob rather than an
> explicit compression-ratio target, which is why `cr5` and `cr10` produce
> very similar measured ratios (~2.2–2.4×) on this dataset. With
> `glymur`, the measured ratios would track the targets more closely. The
> values above are exactly what the real codec produced — no values are
> hand-tuned.

### How to interpret

- **Higher Compression Ratio** → stronger compression (smaller file).
- **Higher PSNR** → less pixel-level error.
- **Higher SSIM** → better structural similarity to the original (perception
  of edges, contrast, texture).
- **JPEG** at high compression typically shows **blocking artifacts** along
  the 8×8 grid and a loss of color fidelity.
- **JPEG2000** typically degrades **more smoothly** than JPEG; instead of
  blocking it shows blur, ringing near edges, and a loss of fine texture.
  At extreme compression (e.g. `cr20`) it can fall behind JPEG-q30 on PSNR
  but usually preserves edge structure better.

These are *expected trends*, not guarantees — the actual ordering depends on
the active backend, the parameter mapping, and the measured file sizes.

---

## Visual results

Generated figures (committed under `outputs/figures/`):

- `outputs/figures/plots/` — bar charts and scatter plots summarising the run,
  plus the summary dashboard and table image produced by
  `scripts/generate_summary_report.py`.
- `outputs/figures/comparisons/` — original alongside the six reconstructions,
  per representative image.
- `outputs/figures/error_maps/` — amplified absolute-error visualizations.
- `outputs/figures/crops/` — 160×160 patches (edge-rich, texture-rich, smooth)
  for visual inspection of artifacts.
- `outputs/figures/dct_analysis/` — educational walkthrough of the JPEG DCT
  pipeline on a single 8×8 luma block.
- `outputs/figures/wavelet_analysis/` — educational sub-band decomposition
  used by JPEG2000.

---

## Reproducibility

- **Expected number of rows in `per_image_results.csv`**:
  `24 images × 2 methods × 3 levels = 144`. The committed run has 144 rows,
  reported in `experiment_metadata.json`.
- All metrics come from real `.jpg` and `.jp2` files on disk — no values are
  invented.
- The JPEG2000 codec performs strict validation (file existence, size > 0,
  JP2 magic bytes, shape match). A non-JP2 file silently saved with a `.jp2`
  extension would be rejected.
- The active backends, image count, level definitions, Python version, and
  timestamp are written to `outputs/metrics/experiment_metadata.json`.

---

## GitHub note

The repository is structured so that the heavy or environment-specific
artefacts are **not** committed:

- `data/kodak/raw/` — dataset images (each user downloads them).
- `archive.zip` — Kaggle dataset archive.
- `.venv/`, `__pycache__/`, `*.pyc` — local environment state.
- `outputs/compressed/` — generated `.jpg` / `.jp2` files (regenerated by the
  pipeline).
- `outputs/reconstructed/` — generated PNG reconstructions (regenerated).

But the *summary* of the run is committed:

- `outputs/metrics/*.csv`, `outputs/metrics/*.json` — all numerical results.
- `outputs/figures/**/*.png` — every figure.

---

## Publishing to GitHub

After cloning or completing a fresh run, the typical first publication
sequence is:

```bat
git init
git status
git add README.md requirements.txt .gitignore src scripts outputs/metrics outputs/figures
git commit -m "feat: add JPEG vs JPEG2000 compression analysis"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repository-name>.git
git push -u origin main
```

Replace `<your-username>` and `<your-repository-name>` with your actual GitHub
account and repository name before running the `git remote add` line.

---

## Troubleshooting JPEG2000

If `python -m src.run_experiments --only_check_backends` reports
`No functional JPEG2000 backend was found`:

1. `pip install --upgrade imagecodecs glymur Pillow`.
2. `glymur` requires the OpenJPEG shared library. On Windows the wheel usually
   bundles it; otherwise install OpenJPEG and ensure it is on `PATH`.
3. As a last resort, run with `--methods jpeg` to get the JPEG-only results.
