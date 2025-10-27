# Filter Image Migration Guide

## Overview

The `filter_image.py` script has been refactored into individual filter scripts for better modularity and maintainability.

## Changes

### Before (Monolithic)
- Single script: `standard_code/python/filter_image.py`
- Single CLI definition: `filter_image.json`
- All filters accessed via `--filter-type` parameter

### After (Modular)
- Individual scripts in `standard_code/python/filter_image/` folder:
  - `gaussian_filter.py`
  - `median_filter.py`
  - `mean_filter.py`
  - `bilateral_filter.py`
  - `sobel_filter.py`
  - `laplacian_filter.py`
  - `unsharp_filter.py`
  - `tophat_filter.py`

- Individual CLI definitions in `cli_definitions/` folder:
  - `gaussian_filter.json`
  - `median_filter.json`
  - `mean_filter.json`
  - `bilateral_filter.json`
  - `sobel_filter.json`
  - `laplacian_filter.json`
  - `unsharp_filter.json`
  - `tophat_filter.json`

## Pipeline Designer UI

### Old Behavior
- Single node type: "Filter Image"
- Filter type selected via dropdown parameter
- All filter parameters visible (even irrelevant ones)

### New Behavior
- Eight separate node types under "Image Processing - Filtering" category:
  - "Gaussian Filter"
  - "Median Filter"
  - "Mean Filter"
  - "Bilateral Filter"
  - "Sobel Edge Detection"
  - "Laplacian Edge Detection"
  - "Unsharp Mask"
  - "Top-Hat Filter"

- Each node shows only relevant parameters for that specific filter
- Cleaner, more intuitive interface

## Migration Instructions

### For Pipeline Designers GUI Users
1. Rebuild the application using `build.ps1`
2. The new filter nodes will appear in the "Image Processing - Filtering" category
3. The old "Filter Image" node is still available for backward compatibility

### For Command Line Users
The old script still works:
```bash
python filter_image.py --input-pattern "*.tif" --output-folder "./output" --filter-type gaussian --sigma 2.0
```

New modular scripts provide cleaner interfaces:
```bash
python filter_image/gaussian_filter.py --input-pattern "*.tif" --output-folder "./output" --sigma 2.0
```

### For Pipeline YAML Configurations
Old pipelines using `filter_image.py` will continue to work. New pipelines should use individual filter scripts.

## Benefits of New Structure

1. **Better Organization**: Each filter is self-contained
2. **Clearer Parameters**: Only relevant parameters per filter
3. **Easier Testing**: Can test individual filters independently
4. **Better Documentation**: Each script has focused documentation
5. **GUI Clarity**: Separate nodes reduce parameter confusion
6. **Maintainability**: Easier to update individual filters

## Backward Compatibility

The original `filter_image.py` script is kept for backward compatibility and will continue to work with existing pipelines.

## Future Plans

The monolithic `filter_image.py` may be deprecated in a future release once all pipelines have been migrated to use the individual filter scripts.
