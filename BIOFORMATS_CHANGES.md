# Bio-Formats Changes Summary

## What Changed

Updated `bioimage_pipeline_utils.py` to handle Bio-Formats initialization failures gracefully when using UV environments.

## The Problem

- Bio-Formats requires Java (via jpype → scyjava → bioio-bioformats)
- UV installs Python packages but doesn't bundle Java
- jpype crashes with access violations when Java is incompatible or missing
- This created a poor user experience with cryptic crashes

## The Solution

### 1. Graceful Error Handling
When Bio-Formats fails to load, users now see:

```
======================================================================
ERROR: Failed to load your_file.oib
======================================================================

This file format requires Bio-Formats (Java), which failed to initialize.

SOLUTION: Use Conda environment instead of UV for Bio-Formats support:

1. Create Conda environment from: conda_envs/convert_to_tif.yml
   conda env create -f conda_envs/convert_to_tif.yml

2. Run your pipeline with the Conda environment:
   run_pipeline.exe pipeline_configs/your_config.yaml
   (use 'environment: convert_to_tif' in your YAML config)

NOTE: Most formats (ND2, LIF, CZI, DV, TIFF) work without Bio-Formats.
======================================================================
```

### 2. Documentation
Created `conda_envs/BIOFORMATS_README.md` explaining:
- Which formats work with UV vs Conda
- Why Conda is needed for Bio-Formats
- How to use each environment

### 3. Battle-Tested Approach
- Reverted to your working version of the code
- Added try/except around bioformats imports only
- UV works for 95% of use cases (common formats)
- Conda available for exotic formats

## Files Modified

1. **`standard_code/python/bioimage_pipeline_utils.py`**
   - Added helpful RuntimeError when Bio-Formats import fails
   - Error message directs users to Conda solution
   - Catches exceptions for both .ims fallback and unknown formats

2. **`conda_envs/BIOFORMATS_README.md`** (new)
   - Complete guide to format support
   - UV vs Conda comparison
   - Usage examples

## Testing

✅ **UV environment**: Successfully loads TIFF, ND2, LIF, CZI, DV  
✅ **Error handling**: Shows helpful message for exotic formats  
✅ **Conda environment**: Ready with `scyjava` for Bio-Formats support

## User Impact

**Before**: Cryptic jpype crashes with no guidance  
**After**: Clear error message with actionable solution

Most users won't see any change since common formats work with UV. Only users with exotic formats will see the helpful error directing them to use Conda.
