# Drift Correction - Quick Start Guide

## ✅ What's Implemented

**Minimal translation-only drift correction** using established libraries:

1. **`drift_correction.py`** - Entry point (230 lines)
   - Simple CLI for single or multiple files
   - Routes to phase_cross_correlation or stackreg
   - Applies shifts to all channels

2. **`phase_cross_correlation.py`** - scikit-image & cuCIM wrapper (220 lines)
   - CPU: scikit-image
   - GPU: cuCIM (automatic fallback to CPU)
   - Subpixel accuracy

3. **`pystackreg.py`** - ImageJ TurboReg wrapper (180 lines)
   - TRANSLATION mode only
   - Per-slice registration for 3D data

**Total**: ~600 lines wrapping thousands of lines from battle-tested libraries

## 🚀 Quick Usage

### Command Line

```bash
# Process multiple files with glob pattern
python drift_correction.py --input-search-pattern "data/*.tif" --output-folder output/ --method phase_cross_correlation --reference first

# With specific reference channel (e.g., channel 1 for multi-channel data)
python drift_correction.py --input-search-pattern "data/*.tif" --output-folder output/ --method phase_cross_correlation --reference first --reference-channel 1

# With GPU acceleration
python drift_correction.py --input-search-pattern "data/*.tif" --output-folder output/ --method phase_cross_correlation --reference previous --gpu

# StackReg method
python drift_correction.py --input-search-pattern "data/*.tif" --output-folder output/ --method stackreg_translation --reference first
```

### Python API

```python
import bioimage_pipeline_utils as rp
from standard_code.python.drift_correction import drift_correct

# Load TCZYX image
img = rp.load_tczyx_image("timelapse.tif")

# Simple drift correction
corrected = drift_correct(img, method="phase_cross_correlation", reference="first")

# GPU-accelerated
corrected = drift_correct(img, method="phase_cross_correlation", reference="first", use_gpu=True)

# Sequential registration
corrected = drift_correct(img, method="phase_cross_correlation", reference="previous")

# Save result
rp.save_tczyx_image(corrected, "corrected.tif")
```

### Pipeline YAML

```yaml
run:
- name: Drift correction
  environment: uv:drift_correct
  commands:
  - python: '%REPO%/standard_code/python/drift_correction.py'
  - --input-search-pattern: '%TEST_FOLDER%/data/*.tif'  # Input pattern
  - --output-folder: '%TEST_FOLDER%/output'              # Output folder
  - --method: phase_cross_correlation
  - --reference: previous
  - --reference-channel: 0  # Channel to use for registration (0-based)
  - --upsample-factor: 10
  # - --gpu  # Uncomment for GPU acceleration
```

Run with:
```bash
.\run_pipeline.exe drift_correct.yaml
```

## 📋 Parameters

### Entry Point (drift_correction.py)

**Allowed methods**:
- `phase_cross_correlation` (default, recommended)
- `stackreg_translation`

**Allowed references**:
- `first` - Register all frames to T=0 (most common)
- `previous` - Register each frame to previous (sequential)

**Other options**:
- `--reference-channel` - Channel index to use for registration (0-based, default: 0)
- `--gpu` - Use GPU acceleration (requires cuCIM)
- `--upsample-factor` - Subpixel precision (10 = 0.1 pixel, 100 = 0.01 pixel)

### Advanced API (Direct Algorithm Access)

For power users who need advanced features like mean/median references:

```python
from standard_code.python.drift_correction.phase_cross_correlation import phase_cross_correlation_cpu

stack = img.get_image_data("TZYX", C=0)

# Register to mean projection (robust for noisy data)
shifts, corrected = phase_cross_correlation_cpu(stack, reference="mean", upsample_factor=20)

# Register to median projection (robust to outliers)
shifts, corrected = phase_cross_correlation_cpu(stack, reference="median")

# Register to specific timepoint
shifts, corrected = phase_cross_correlation_cpu(stack, reference=5)
```

## 🔍 What Got Deleted

All the old `_*.py` files should be deleted as they were non-working implementations:

```bash
# DELETE these files from drift_correction/ folder:
_demons.py
_drift_correction_Legacy.py
_enhanced_optical_flow.py
_feature_based.py
_feature_matching.py
_fourier_shift.py
_gradient_descent.py
_mock_correction.py
_mutual_information.py
_normalized_cross_correlation.py
_optical_flow.py
_phase_cross_correlation.py  # Old version
_sum_squared_differences.py
_template_matching.py
_variational.py
```

Keep only:
- `__init__.py`
- `drift_correct_utils.py`
- `synthetic_data_generators.py`
- `phase_cross_correlation.py` (NEW)
- `pystackreg.py` (NEW)
- `drift_correction_implementation_plan.md`

## ⚠️ Translation Only

**This module ONLY performs XYZ translation (rigid shifts).**

No rotation, scaling, affine, or non-linear transformations are permitted.

This is mathematically guaranteed by:
- Phase cross-correlation uses Fourier phase shift theorem (pure translation)
- StackReg TRANSLATION mode is explicitly enforced
- All other StackReg modes (RIGID_BODY, AFFINE, etc.) are disabled

## 📦 Dependencies

**Required** (in drift_correct.yml):
- numpy
- scipy
- scikit-image

**Optional** (for additional features):
- pystackreg (for StackReg algorithm)
- cupy + cucim (for GPU acceleration)

All dependencies are already in the `drift_correct` conda environment!

## 🧪 Testing

```python
# Test with synthetic data
from standard_code.python.drift_correction.synthetic_data_generators import generate_drifting_stack

# Generate test data with known drift
stack = generate_drifting_stack(T=50, Z=10, Y=256, X=256, drift_per_frame=2.5)

# Apply correction
from standard_code.python.drift_correction.phase_cross_correlation import phase_cross_correlation_cpu
shifts, corrected = phase_cross_correlation_cpu(stack, reference="first", upsample_factor=10)

# Verify shifts match expected drift
print(f"Expected drift per frame: 2.5 pixels")
print(f"Detected mean drift: {np.mean(np.abs(shifts)):.2f} pixels")
```

## 📚 Documentation

See `drift_correction_implementation_plan.md` for complete implementation details.

## 🎯 Summary

✅ **Minimal code** (~600 lines)  
✅ **Maximum reliability** (uses scikit-image, cuCIM, pystackreg)  
✅ **Translation only** (mathematically guaranteed)  
✅ **GPU support** (automatic fallback)  
✅ **Works with TCZYX** (direct input/output)  
✅ **CLI + Python API** (flexible usage)  
✅ **Updated YAML** (ready to run)

**Ready to use! Just run:**
```bash
.\run_pipeline.exe drift_correct.yaml
```
