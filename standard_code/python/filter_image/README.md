# Image Filtering Modules

This folder contains individual filter scripts for various image filtering operations.

## Structure

Each filter is implemented as a standalone Python script that can be run independently or integrated into pipelines via the pipeline designer GUI.

## Available Filters

### Smoothing Filters

- **`gaussian_filter.py`** - Gaussian blur filter for general smoothing
  - Parameter: `sigma` (standard deviation)
  
- **`median_filter.py`** - Median filter for noise reduction (preserves edges)
  - Parameter: `size` (kernel size)
  
- **`mean_filter.py`** - Mean (uniform) filter for basic smoothing
  - Parameter: `size` (kernel size)
  
- **`bilateral_filter.py`** - Edge-preserving bilateral filter
  - Parameters: `sigma-spatial`, `sigma-intensity`
  - Requires: OpenCV (opencv-python)

### Edge Detection Filters

- **`sobel_filter.py`** - Sobel edge detection
  - Parameter: `axis` (-1 for magnitude, 0/1 for specific axis)
  
- **`laplacian_filter.py`** - Laplacian edge detection
  - No parameters

### Enhancement Filters

- **`unsharp_filter.py`** - Unsharp mask for image sharpening
  - Parameters: `sigma` (blur amount), `amount` (sharpening strength)
  
- **`tophat_filter.py`** - Morphological top-hat filter
  - Parameters: `size` (structuring element size), `mode` (white/black)

## Usage

All filters follow the same interface pattern:

```bash
python <filter_name>.py --input-pattern "path/to/images/*.tif" --output-folder "path/to/output" [OPTIONS]
```

### Common Arguments

- `--input-pattern` (required): Glob pattern for input images
- `--output-folder` (required): Output directory for filtered images

### Filter-Specific Arguments

Each filter has its own parameters (see individual scripts for details).

## Environment

All filters use the `filter_image` environment which includes:
- Python 3.11
- NumPy, SciPy
- OpenCV (for bilateral filter)
- bioio and related plugins for image I/O

## Output

Filtered images are saved with a suffix indicating the filter type:
- `_gaussian`, `_median`, `_mean`, `_bilateral`
- `_sobel`, `_laplacian`
- `_unsharp`, `_tophat_white`, `_tophat_black`

## Integration

These filters are integrated into the pipeline designer GUI under the category:
**Image Processing - Filtering**

Each filter appears as a separate node type with appropriate parameter controls.
