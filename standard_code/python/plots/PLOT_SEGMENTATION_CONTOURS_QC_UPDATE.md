# plot_segmentation_contours.py - QC Filtering Update

## Overview

The `plot_segmentation_contours.py` module has been updated to integrate with the BIPHUB QC system, automatically filtering out images that failed QC before creating overlay plots.

## Key Changes

### 1. **QC Integration Functions** ⭐ NEW

Added three new helper functions:

```python
def get_qc_file_path(image_path: str) -> str:
    """Get the QC YAML file path for an image."""

def load_qc_status(image_path: str, qc_key: str) -> Optional[str]:
    """Load QC status for a specific QC key from the QC file."""

def filter_by_qc_status(grouped_files: Dict[str, Dict[str, str]], qc_key: str) -> Dict[str, Dict[str, str]]:
    """Filter grouped files to exclude those that failed QC."""
```

### 2. **Input-Driven File Matching** ⭐ NEW

Changed from folder-based search to pattern-based matching:

**OLD:**
```bash
python plot_segmentation_contours.py -i ./quantification_output -o overlay.png
```

**NEW:**
```bash
python plot_segmentation_contours.py \
  --input-search-pattern "./input/*.tif" \
  --mask-search-pattern "./output/*_mask.tif" \
  --qc-key nuc_segmentation \
  -o overlay.png
```

### 3. **Automatic QC Filtering**

- Uses `rp.get_grouped_files_to_process` to match input files to masks
- Reads QC files for each input image
- Excludes images with `status: failed` for the specified QC key
- Logs filtering results (passed, unreviewed, failed counts)

### 4. **Updated CLI Arguments**

**Required Arguments:**
- `--input-search-pattern`: Glob pattern for input images
- `--mask-search-pattern`: Glob pattern for mask images  
- `--qc-key`: QC key to use for filtering (e.g., "nuc_segmentation", "track_completeness")

**New Optional Arguments:**
- `--search-subfolders`: Enable recursive search
- `-o`, `--output`: Output path for plot

**Removed Arguments:**
- `-i`, `--input-folder`: Replaced by search patterns
- `--pattern`: Replaced by mask-search-pattern

## Usage Examples

### Basic Usage with QC Filtering

```bash
python plot_segmentation_contours.py \
  --input-search-pattern "./input/*.tif" \
  --mask-search-pattern "./output/*_mask.tif" \
  --qc-key nuc_segmentation \
  -o overlay.png
```

### With Track Completeness QC

```bash
python plot_segmentation_contours.py \
  --input-search-pattern "./input/*.tif" \
  --mask-search-pattern "./tracked_masks/*_tracked_mask.tif" \
  --qc-key track_completeness \
  -o tracked_overlay.png
```

### Customize Appearance

```bash
python plot_segmentation_contours.py \
  --input-search-pattern "./input/*.tif" \
  --mask-search-pattern "./masks/*_mask.tif" \
  --qc-key nuc_segmentation \
  -o overlay.png \
  --linewidth 3.0 \
  --alpha 0.9 \
  --colormap tab20
```

### Recursive Search

```bash
python plot_segmentation_contours.py \
  --input-search-pattern "./experiments/**/*.tif" \
  --mask-search-pattern "./output/**/*_mask.tif" \
  --qc-key nuc_segmentation \
  --search-subfolders \
  -o overlay.png
```

### Mode Selection

```bash
# Process all files (default)
python plot_segmentation_contours.py \
  --input-search-pattern "./input/*.tif" \
  --mask-search-pattern "./masks/*_mask.tif" \
  --qc-key nuc_segmentation \
  --mode all \
  -o overlay.png

# Process 1 sample per group
python plot_segmentation_contours.py \
  --input-search-pattern "./input/*.tif" \
  --mask-search-pattern "./masks/*_mask.tif" \
  --qc-key nuc_segmentation \
  --mode group1 \
  -o overlay.png
```

## QC Filtering Logic

1. **Load Files**: Use `get_grouped_files_to_process` to match inputs to masks
2. **Check QC Status**: For each input file, load `<filename>_QC.yaml`
3. **Filter**: 
   - **Include**: Files with `status: passed` or no QC file (unreviewed)
   - **Exclude**: Files with `status: failed`
4. **Log Results**: Report counts of passed, unreviewed, and failed files
5. **Create Plot**: Generate overlay with only QC-passed files

## Example Console Output

```
================================================================================
Segmentation Contour Overlay Plotter with QC Filtering
================================================================================
Searching for files with patterns:
  Input: ./input/*.tif
  Mask:  ./output/*_mask.tif
  QC Key: nuc_segmentation
Found 50 input files
  45 with masks, 5 without masks

QC Filtering Results (key: nuc_segmentation):
  Passed: 38
  Unreviewed: 2
  Failed (excluded): 5
  Total included: 40/45

After QC filtering: 40 files
Found 40 mask files to process
Found 4 experimental groups
  Group 'WT': selected 1/12 samples
  Group '3SA': selected 1/10 samples
  Group 'L58R': selected 1/9 samples
  Group 'LKO8': selected 1/9 samples
Selected 4 mask files for visualization (mode: group1)

Creating average mask overlay plot from 4 mask files
Processing: SP20250625_PC_R3_WT011_mask.tif
  Group: WT, Shape: (50, 100), Pixels: 4523
Processing: SP20250625_PC_R3_3SA_001_mask.tif
  Group: 3SA, Shape: (50, 100), Pixels: 3891
...

Saved contour overlay plot to: overlay.png
Done!
```

## Integration with Pipeline YAML

```yaml
pipeline_name: "Create Segmentation Overlay Plot"
steps:
  - module: "plots.plot_segmentation_contours"
    parameters:
      input_search_pattern: "./input/*.tif"
      mask_search_pattern: "./quantification_output/*_mask.tif"
      qc_key: "nuc_segmentation"
      output: "./plots/segmentation_overlay.png"
      mode: "group1"
      linewidth: 2.5
      alpha: 0.7
      colormap: "tab10"
      search_subfolders: false
```

## Benefits

1. **Automatic QC Integration**: No manual file exclusion needed
2. **Consistent with Other Tools**: Uses same QC system as `qc_mask.py` and `qc_tracking.py`
3. **Complete Visibility**: Logs show exactly which files are excluded and why
4. **Reproducible**: QC decisions documented in YAML files
5. **Flexible**: Works with any QC key (segmentation, tracking, etc.)

## Notes

- Only images with `status: failed` are excluded
- Images without QC files are included (treated as unreviewed)
- Images with `status: passed` are included
- QC files must be in same directory as input files with suffix `_QC.yaml`
- The mask files are what get plotted, but QC status is checked on the input files

## Migration from Old Version

**Old Command:**
```bash
python plot_segmentation_contours.py -i ./quantification_output -o overlay.png
```

**New Command:**
```bash
python plot_segmentation_contours.py \
  --input-search-pattern "./input/*.tif" \
  --mask-search-pattern "./quantification_output/*_mask.tif" \
  --qc-key nuc_segmentation \
  -o overlay.png
```

## Author

BIPHUB - Bioimage Informatics Hub, University of Oslo  
License: MIT
