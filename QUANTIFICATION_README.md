# Signal Spread and Decay Quantification

## Overview

The `plot_distance_heatmap.py` module now includes Gaussian-based quantification for analyzing photoconversion data that spreads spatially over time and decays.

## Key Features

### 1. **Random Sampling for Unbiased Measurements**
- Uses random sampling (default: 20 pixels per distance bin) to avoid bias from:
  - Pixel count variations between bins
  - Bright debris or outliers
  - Geometric effects at different distances
- Fixed random seed (default: 42) ensures reproducibility

### 2. **Gaussian Fitting for Peak Tracking**
- Fits Gaussian distribution to intensity profile at each distance
- Tracks three key lines over time:
  - **Peak trajectory**: Center of Gaussian (where signal is strongest)
  - **FWHM upper bound**: Peak + half of FWHM
  - **FWHM lower bound**: Peak - half of FWHM
- FWHM (Full Width at Half Maximum) quantifies signal spread

### 3. **No T0 Normalization in Quantification**
- Quantification uses absolute intensities (no T0 baseline subtraction)
- Makes sense for photoconversion: you're tracking appearance and spread, not relative changes

## Usage

### Basic Quantification
```bash
python standard_code/python/plots/plot_distance_heatmap.py \
  --input-search-pattern "./images/*.tif" \
  --distance-search-pattern "./distances/*_geodesic.tif" \
  --quantify \
  --output-folder ./quantification
```

### Custom Parameters
```bash
python standard_code/python/plots/plot_distance_heatmap.py \
  --input-search-pattern "./images/*.tif" \
  --distance-search-pattern "./distances/*_geodesic.tif" \
  --quantify \
  --output-folder ./quantification \
  --max-measure-pixels 30 \
  --smooth-sigma 2.0 \
  --remove-first-n-bins 10
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--quantify` | False | Enable Gaussian quantification analysis |
| `--max-measure-pixels` | 20 | Number of pixels to randomly sample per distance bin |
| `--smooth-sigma` | 1.5 | Gaussian smoothing sigma for distance dimension only (each timepoint processed independently) |
| `--remove-first-n-bins` | 5 | Remove N closest distance bins (often noisy) |
| `--channel` | 0 | Which channel to analyze |
| `--force-show` | False | Display plots even when saving |

## Outputs

### 1. Quantification Plot (`*_quantification.png`)
A comprehensive figure with 4 subplots:
- **Top**: Smoothed heatmap with overlaid peak trajectory and FWHM bounds
- **Middle**: Peak position and FWHM bounds over time
- **Bottom Left**: FWHM width over time (signal spread)
- **Bottom Right**: Total intensity and Gaussian fit quality (R²)

### 2. Metrics TSV (`*_quantification_metrics.tsv`)
Tab-separated file with columns:
- `Timepoint`: Frame number
- `Peak_Distance`: Gaussian peak position
- `FWHM_Lower`: Lower FWHM bound
- `FWHM_Upper`: Upper FWHM bound
- `FWHM_Width`: FWHM spread
- `Gaussian_Amplitude`: Fitted amplitude
- `Gaussian_Sigma`: Fitted sigma
- `Fit_R_Squared`: Quality of Gaussian fit (0-1)
- `Total_Intensity`: Sum of all sampled intensities

### 3. Raw Heatmap (`*_distance_time_heatmap.png`)
Saved (but not displayed) when `--output-folder` is specified

## Interpretation

### Peak Trajectory
- Shows where signal is strongest at each timepoint
- For photoconversion: tracks the "front" of spreading signal

### FWHM Bounds
- Upper and lower bounds show the extent of signal spread
- Wider separation = more diffuse signal
- Narrower = more concentrated signal

### Fit Quality (R²)
- Values close to 1.0 indicate good Gaussian fits
- Lower values suggest:
  - Complex/multimodal distributions
  - Very weak signal
  - Non-Gaussian spreading behavior

## Example Interpretation

```
T=0:  Peak=10, FWHM=5  → Signal centered at distance 10, narrow spread
T=5:  Peak=15, FWHM=8  → Signal moved to distance 15, wider spread
T=10: Peak=20, FWHM=12 → Continued spreading and diffusion
```

## Technical Notes

### Why Random Sampling?
- Avoids bias from variable pixel counts per distance bin
- More robust than using all pixels (sum) or top-N brightest pixels
- Avoids bias from bright debris or outliers
- Fixed seed ensures reproducibility across runs

### Why Distance-Only Smoothing?
- Smoothing is applied only in the distance dimension, not time
- Each timepoint is processed independently
- Prevents T0 from influencing T1, T1 from influencing T2, etc.
- Preserves temporal dynamics while reducing noise in spatial measurements

### Why FWHM?
- Standard measure of peak width
- More robust than using percentiles
- Directly related to Gaussian sigma: FWHM ≈ 2.355 × σ

### When Gaussian Fitting Fails
- Returns NaN for that timepoint
- Common reasons: very weak signal, flat distribution, multiple peaks
- Check fit quality (R²) to identify problematic timepoints

## Troubleshooting

### Poor Fit Quality
- Try adjusting `--smooth-sigma` (increase for noisy data)
- Check if signal is actually Gaussian-distributed
- Ensure sufficient pixels per bin (check logs)

### Unexpected Peak Positions
- Verify distance matrix is correct
- Check if `--remove-first-n-bins` is appropriate
- Inspect raw heatmap for quality

### Inconsistent Results
- Random seed is fixed (42) by default
- If changing parameters, results should be reproducible
- Different `--max-measure-pixels` values will give different absolute intensities but similar relative patterns

## Citation

```
BIPHUB Pipeline Manager
Bioimage Informatics Hub, University of Oslo
Author: Øyvind Fiksdahl Østerås
License: MIT
```
