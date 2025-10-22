# QC Tracking - Track Completeness Validation

## Overview

The `qc_tracking.py` module automatically detects incomplete tracks in tracked mask files and integrates with the BIPHUB QC system. It creates heatmaps with visual indicators (red text) for files that fail track completeness validation.

**Major Update**: Now uses `rp.get_grouped_files_to_process` to ensure **ALL input images** are shown in the heatmap, even if masks are missing or tracking failed. This provides complete visibility into which files have issues.

## Key Features

### 1. **Input-Driven Analysis** ⭐ NEW
   - Shows ALL input images in the heatmap
   - Matches input images to corresponding mask files using wildcard patterns
   - Marks missing masks as FAILED with reason "Mask file not found"
   - Ensures no files are silently skipped

### 2. **Comprehensive Track Validation**
   - Analyzes each tracked mask file to ensure all objects appear in every timepoint
   - **NEW**: Detects empty masks (zero objects in all timepoints) and marks as FAILED
   - Detects missing timepoints for any tracked object
   - Creates detailed failure reports for incomplete tracks

### 3. **QC File Integration**
   - Creates/updates `<filename>_QC.yaml` files next to mask files
   - Preserves existing QC data for other keys
   - Supports custom QC keys (e.g., `track_completeness`, `tracking_validation`)

### 4. **Visual QC Status**
   - **Black text**: Tracks passed validation
   - **Red bold text**: Tracks failed validation
   - Summary statistics in plot title
   - Detailed failure report printed to console

### 5. **Smart QC Workflow**
   - Only validates files with missing QC or `passed` status
   - Respects previously failed QC (doesn't re-validate)
   - Updates QC files only when status changes

## Usage

### Basic Usage (Required Parameters)

```bash
python qc_tracking.py \
  --input-search-pattern "./input/*.tif" \
  --mask-search-pattern "./masks/*_tracked.tif" \
  --qc-key track_completeness
```

### Save Output to File

```bash
python qc_tracking.py \
  --input-search-pattern "./input/*.tif" \
  --mask-search-pattern "./masks/*_tracked.tif" \
  --qc-key track_completeness \
  --output track_qc.png
```

### Custom Figure Size

```bash
python qc_tracking.py \
  --input-search-pattern "./input/*.tif" \
  --mask-search-pattern "./masks/*_tracked.tif" \
  --qc-key track_completeness \
  --width 20 \
  --height 15
```

### Recursive Search

```bash
python qc_tracking.py \
  --input-search-pattern "./input/**/*.tif" \
  --mask-search-pattern "./masks/**/*_tracked.tif" \
  --qc-key track_completeness \
  --search-subfolders
```

### Different QC Key

```bash
python qc_tracking.py \
  --input-search-pattern "./input/*.tif" \
  --mask-search-pattern "./masks/*_tracked.tif" \
  --qc-key tracking_validation
```

### File Selection Modes

```bash
# Process all files (default)
python qc_tracking.py \
  --input-search-pattern "./input/*.tif" \
  --mask-search-pattern "./masks/*_tracked.tif" \
  --qc-key track_completeness \
  --mode all

# Process first 10 files
python qc_tracking.py \
  --input-search-pattern "./input/*.tif" \
  --mask-search-pattern "./masks/*_tracked.tif" \
  --qc-key track_completeness \
  --mode first10

# Process 5 random samples
python qc_tracking.py \
  --input-search-pattern "./input/*.tif" \
  --mask-search-pattern "./masks/*_tracked.tif" \
  --qc-key track_completeness \
  --mode random5

# Process 2 samples per experimental group
python qc_tracking.py \
  --input-search-pattern "./input/*.tif" \
  --mask-search-pattern "./masks/*_tracked.tif" \
  --qc-key track_completeness \
  --mode group2
```

## How File Matching Works

The script uses `rp.get_grouped_files_to_process` to match files by their basename (the part matching the `*` wildcard):

**Example:**
```
Input Pattern:  "./input/*.tif"
Mask Pattern:   "./masks/*_tracked.tif"

Files found:
- input/experiment_A_001.tif
- input/experiment_A_002.tif
- input/experiment_A_003.tif
- masks/experiment_A_001_tracked.tif
- masks/experiment_A_003_tracked.tif

Basename extraction:
- experiment_A_001.tif → basename: "experiment_A_001"
- experiment_A_002.tif → basename: "experiment_A_002"
- experiment_A_003.tif → basename: "experiment_A_003"

Matching:
- experiment_A_001: ✓ Has both input and mask → Validate tracks
- experiment_A_002: ✗ Missing mask → Mark as FAILED ("Mask file not found")
- experiment_A_003: ✓ Has both input and mask → Validate tracks

Note: If a mask exists without a corresponding input, it won't be shown in the heatmap.
```

## QC File Format

The script creates/updates YAML files with the following structure:

```yaml
qc:
  track_completeness:
    status: failed
    comment: "No objects detected in any timepoint (empty mask across all 50 timepoints)"
    mask_search_pattern: "./masks/*_tracked.tif"
  
  # Other QC keys are preserved
  nuc_segmentation:
    status: passed
    comment: "User defined that segmentation is correct"
```

## Track Validation Logic

1. **Group Files**: Use `rp.get_grouped_files_to_process` to match input files to masks

2. **For Each Input File**:
   
   a. **Check for Mask File**:
      - If no mask file matches → Mark as FAILED ("Mask file not found")
   
   b. **Load QC File**: Check if `<mask_filename>_QC.yaml` exists
      - If status is `failed` → Skip validation (already marked)
   
   c. **Validate Tracks** (for files with masks and not already failed):
      - Load mask file
      - Extract all unique object labels across all timepoints
      - **Check for empty mask**: If zero objects → Mark as FAILED
      - For each label, verify it appears in every timepoint
      - Record missing timepoints for incomplete tracks
   
   d. **Update QC Status**:
      - **PASSED**: All objects present in all timepoints
      - **FAILED**: One of:
        - Mask file not found
        - No objects in any timepoint (empty mask)
        - One or more objects missing from timepoints
        - Error loading/processing file
   
   e. **Save QC File**: Create or update the QC YAML file with results

## Example Output

### Console Output

```
Searching for files with patterns:
  Input: ./input/*.tif
  Mask:  ./masks/*_tracked.tif

Found 15 input files
  11 with masks, 4 without masks

Processing 15 input files...
Checking track completeness and QC status (key: track_completeness)...

experiment_A_001: FAILED - Mask file not found
experiment_A_002: PASSED - All 23 tracks complete across 50 timepoints
experiment_A_003: FAILED - No objects detected in any timepoint (empty mask across all 50 timepoints)
experiment_A_004: FAILED - Incomplete tracks detected: Object 5 missing at T=[10, 11, 12]
experiment_A_005: FAILED - Mask file not found
experiment_B_001: Already marked as FAILED in QC
experiment_B_002: PASSED - All 18 tracks complete across 50 timepoints
...

Created data matrix: 15 images × 50 timepoints
QC Results: 9 PASSED, 6 FAILED

============================================================
FAILED TRACKS SUMMARY:
============================================================
❌ experiment_A_001
   Reason: Mask file not found
❌ experiment_A_003
   Reason: No objects detected in any timepoint (empty mask across all 50 timepoints)
❌ experiment_A_004
   Reason: Incomplete tracks detected: Object 5 missing at T=[10, 11, 12]
❌ experiment_A_005
   Reason: Mask file not found
❌ experiment_B_003
   Reason: Incomplete tracks detected: Object 12 missing at T=[45]; Object 15 missing at T=[47, 48]
❌ experiment_C_001
   Reason: Previously failed
============================================================
```

### Visual Output

The heatmap shows:
- **Heatmap colors**: Based on max object size (pixels)
- **Cell annotations**: Object count per timepoint
- **Y-axis labels**: 
  - **Black**: Passed QC
  - **Red bold**: Failed QC
- **Title**: Includes pass/fail counts

## Integration with Pipeline

To integrate into a YAML pipeline configuration:

```yaml
pipeline_name: "Track Validation QC"
steps:
  - module: "plots.qc_tracking"
    parameters:
      input_search_pattern: "./input/*.tif"
      mask_search_pattern: "./output/tracked_masks/*_tracked.tif"
      qc_key: "track_completeness"
      output: "./qc_reports/track_completeness_heatmap.png"
      mode: "all"
      search_subfolders: false
```

## Common Failure Reasons

| Failure Reason | Explanation | How to Fix |
|---------------|-------------|------------|
| **Mask file not found** | No mask file matches the input file basename | Check tracking pipeline, verify file naming |
| **No objects detected in any timepoint** | The mask exists but contains only zeros (no segmented objects) | Check segmentation pipeline, verify input quality |
| **Incomplete tracks detected: Object X missing at T=[...]** | Object appears in some timepoints but not others | Check tracking algorithm, may need gap filling |
| **Error processing file** | Exception occurred during file loading or analysis | Check file integrity, verify file format |

## API Reference

### Main Functions

#### `check_track_completeness(mask_path, qc_key, mask_search_pattern, basename)`
Validates track completeness for a single mask file.

**Parameters**:
- `mask_path`: Path to mask file (None if missing)
- `qc_key`: QC key for storing results
- `mask_search_pattern`: Search pattern for documentation
- `basename`: Base filename for identification

**Returns**: `(filename, is_complete, failure_reason)`

#### `plot_mask_count_heatmap(grouped_files, qc_key, mask_search_pattern, output_path, figsize)`
Creates QC heatmap with track validation.

**Parameters**:
- `grouped_files`: Dict mapping basename to {'input': path, 'mask': path}
- `qc_key`: QC key for storing results
- `mask_search_pattern`: Search pattern for documentation
- `output_path`: Optional save path (None for interactive)
- `figsize`: Optional tuple (width, height) in inches

### Helper Functions

#### `get_qc_file_path(image_path)`
Returns the path to the QC YAML file for a given mask file.

#### `load_existing_qc(image_path, qc_key)`
Loads existing QC data for the specified key.

#### `save_qc_result(image_path, qc_key, status, comment, mask_search_pattern)`
Saves/updates QC results while preserving other QC keys.

#### `count_masks_in_image(mask_path, basename)`
Counts masks and max object sizes per timepoint.

**Parameters**:
- `mask_path`: Path to mask file (None if missing)
- `basename`: Base filename for identification

**Returns**: `(filename, list of counts, list of max sizes)`

## Notes

- All input files are shown, even without masks (complete visibility)
- Track validation assumes indexed masks where each object has a unique label
- Missing timepoints are detected when a label present in some timepoints is absent in others
- Empty masks (no objects) are now detected and marked as FAILED
- The validation logic handles 5D TCZYX images (considers all Z slices)
- QC files are saved in the same directory as the mask files
- Previously failed QC is respected unless manually changed to `passed`

## Troubleshooting

**Q: Why are some of my input files showing as FAILED with "Mask file not found"?**
A: This means the tracking pipeline didn't produce a mask for these files. Check your tracking pipeline logs.

**Q: Why are some files showing zero objects in the heatmap?**
A: This indicates either empty masks (tracking failed) or the tracking algorithm didn't detect any objects. Check marked as FAILED.

**Q: How do I re-run QC after fixing tracking issues?**
A: Either delete the `_QC.yaml` files or manually change the status from `failed` to `passed` in the YAML file.

## Author

BIPHUB - Bioimage Informatics Hub, University of Oslo  
License: MIT
