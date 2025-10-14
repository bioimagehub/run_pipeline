# Drift Correction Implementation Plan

## Core Philosophy

**Use battle-tested, widely-adopted algorithms instead of reimplementing from scratch.**

The drift correction module leverages established libraries:
- **scikit-image** (`phase_cross_correlation`) - CPU, reliable, well-tested, **TRANSLATION ONLY**
- **cuCIM** (`phase_cross_correlation`) - GPU-accelerated version of scikit-image, **TRANSLATION ONLY**
- **pystackreg** - Python wrapper for ImageJ's TurboReg algorithm, **TRANSLATION MODE ONLY**

## ⚠️ CRITICAL CONSTRAINT: TRANSLATION ONLY

**This module ONLY performs XYZ translation (rigid shifts).** 

**NO rotation, scaling, affine, or non-linear transformations are permitted.**

This ensures:
- Preservation of original image geometry
- No interpolation artifacts beyond basic shifting
- Predictable, reversible transformations
- Biologically meaningful drift correction (sample movement only)

## Architecture

### Entry Point: `drift_correction.py`

**Location**: `standard_code/python/drift_correction.py`

**Input**: TCZYX image object (from `bioimage_pipeline_utils.load_tczyx_image()`)

**Output**: Drift-corrected TCZYX image object with **translation-only** shifts applied

**Allowed Reference Modes (Unified API)**:
- ✅ `"first"` - Register all timepoints to T=0 (supported by both algorithms)
- ✅ `"previous"` - Register each frame to previous frame (supported by both algorithms)

**Responsibilities**:
1. Accept TCZYX image as input
2. Route to appropriate **translation-only** algorithm based on user parameters
3. Handle iteration over T, C dimensions
4. Return corrected TCZYX image object
5. **Restrict reference options to only "first" and "previous"** for API simplicity

### Algorithm Implementations

#### 1. `phase_cross_correlation.py`

**Location**: `standard_code/python/drift_correction/phase_cross_correlation.py`

**Purpose**: Fast, subpixel-accurate **TRANSLATION-ONLY** drift correction

**Transformation Type**: Pure translation (X, Y, Z shifts only)

**Native Reference Support**:
- ✅ `"first"` - Register all frames to T=0 (direct implementation)
- ✅ `"previous"` - Register each frame to previous frame (loop implementation)
- ✅ `"mean"` - Register to mean/median projection (advanced option)
- ✅ `"median"` - Register to median projection (robust to outliers)
- ✅ Custom integer - Register all frames to specific timepoint T=N

**Functions**:

```python
def phase_cross_correlation_cpu(
    zyx_stack: np.ndarray,  # Shape: (T, Z, Y, X) or (T, Y, X)
    reference: Union[Literal["first", "previous", "mean", "median"], int] = "first",
    upsample_factor: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """CPU-based phase cross-correlation using scikit-image."""

def phase_cross_correlation_gpu(
    zyx_stack: np.ndarray,
    reference: Union[Literal["first", "previous", "mean", "median"], int] = "first",
    upsample_factor: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """GPU-accelerated phase cross-correlation using cuCIM."""
```

**Why Phase Cross-Correlation?**
- Mathematically pure translation detection via Fourier phase shift theorem
- No possibility of detecting rotation or non-linear deformations
- Subpixel accuracy without interpolation artifacts
- Fast and robust for most drift correction scenarios
- Flexible reference strategies including mean/median for noisy data

**Z-dimension Handling**:
- If Z=1: Treat as 2D registration (YX only, returns shifts shape (T, 2))
- If Z>1: Full 3D registration (ZYX shifts, returns shifts shape (T, 3))

#### 2. `pystackreg.py`

**Location**: `standard_code/python/drift_correction/pystackreg.py`

**Purpose**: ImageJ TurboReg integration for **TRANSLATION MODE ONLY**

**Transformation Type**: TRANSLATION ONLY (StackReg.TRANSLATION mode)

**Native Reference Support**:
- ✅ `"first"` - Register all frames to T=0 (direct: `reference='first'`)
- ✅ `"previous"` - Register each frame to previous frame (direct: `reference='previous'`)
- ✅ `"mean"` - Register to mean projection (direct: `reference='mean'`)

**⚠️ CRITICAL**: Only the `StackReg.TRANSLATION` mode is exposed. All other modes (RIGID_BODY, SCALED_ROTATION, AFFINE, BILINEAR) are **explicitly disabled** to prevent non-translation transforms.

**Functions**:

```python
def stackreg_register(
    zyx_stack: np.ndarray,  # Shape: (T, Z, Y, X) or (T, Y, X)
    reference: Literal["first", "previous", "mean"] = "first"
) -> Tuple[np.ndarray, np.ndarray]:
    """StackReg TRANSLATION-ONLY registration using pystackreg."""
```

**Why StackReg Translation Mode?**
- Proven algorithm from ImageJ/Fiji community
- Deterministic and reproducible
- Good for noisy images where phase correlation may struggle
- Native support for "first", "previous", and "mean" references
- Strictly enforces translation-only constraint when using TRANSLATION mode

**Z-dimension Handling**:
- StackReg is inherently 2D
- For Z>1: Apply translation registration independently to each Z-slice
- Returns per-slice XY shifts (shape: T, Z, 2)

## Reference Strategy Comparison

### Entry Point (`drift_correction.py`)

**Allowed**: `"first"`, `"previous"` only

**Rationale**: These are the only two strategies supported by BOTH algorithms, ensuring a consistent, predictable API regardless of which algorithm is chosen.

### Algorithm-Specific (`phase_cross_correlation.py`)

**Allowed**: `"first"`, `"previous"`, `"mean"`, `"median"`, or integer

**Use Cases**:
- `"mean"`: Good for noisy data, creates stable reference from all frames
- `"median"`: Robust to outlier frames (e.g., debris, sudden movements)
- Integer (e.g., `reference=5`): Register to specific stable timepoint

### Algorithm-Specific (`pystackreg.py`)

**Allowed**: `"first"`, `"previous"`, `"mean"`

**Use Cases**:
- `"mean"`: StackReg computes this internally for stability
- Native support via pystackreg API

## API Design Decision

```python
# drift_correction.py - Simple, predictable API
drift_correct(img, reference="first")   # ✅ Supported
drift_correct(img, reference="previous")  # ✅ Supported
drift_correct(img, reference="mean")     # ❌ Not supported (use algorithm directly)

# phase_cross_correlation.py - Advanced users
phase_cross_correlation_cpu(stack, reference="mean")    # ✅ Supported
phase_cross_correlation_cpu(stack, reference="median")  # ✅ Supported
phase_cross_correlation_cpu(stack, reference=5)         # ✅ Supported

# pystackreg.py - Advanced users
stackreg_register(stack, reference="mean")  # ✅ Supported
stackreg_register(stack, reference="median")  # ❌ Not supported (use phase_cross_correlation)
```

**Philosophy**: The entry point API is **simple and consistent**. Power users who need advanced reference strategies can call algorithm-specific functions directly.

## Implementation Status

### ✅ Completed (Phase 1)

1. ✅ Created `drift_correction.py` entry point with `reference="first"` and `reference="previous"` only
2. ✅ Implemented `phase_cross_correlation.py` with CPU and GPU versions
   - Supports all native references: "first", "previous", "mean", "median", int
   - ~200 lines (mostly docstrings and logging)
3. ✅ Implemented `pystackreg.py` for TRANSLATION mode ONLY
   - Supports native references: "first", "previous", "mean"
   - ~180 lines (mostly docstrings and logging)
4. ✅ Handles both Z=1 and Z>1 cases transparently
5. ✅ Entry point applies shifts to all channels

### 🔄 TODO (Phase 2)

6. ⏸️ Add quality metrics (translation magnitude, registration confidence)
7. ⏸️ Add visualization tools (drift plots showing XYZ shifts over time)
8. ⏸️ Add chunked processing for large datasets
9. ⏸️ Comprehensive testing with synthetic data (known translation patterns)

### 🔄 TODO (Phase 3)

10. ⏸️ Add Dask integration for out-of-core processing
11. ⏸️ Add multi-GPU support for cuCIM
12. ⏸️ Optimize memory usage for large TCZYX stacks

## Code Statistics

**Total lines written**: ~600 lines across 3 files

**Lines per file**:
- `drift_correction.py`: ~230 lines (entry point with CLI)
- `phase_cross_correlation.py`: ~220 lines (CPU + GPU implementations)
- `pystackreg.py`: ~180 lines (StackReg wrapper)

**Code vs. Library**:
- Our code: ~600 lines (mostly API, logging, error handling)
- External libraries doing the work: scikit-image, cuCIM, pystackreg (thousands of lines, battle-tested)

**Ratio**: We wrote ~0.1% of the actual implementation, leveraging 99.9% from established libraries.

## Usage Examples

### Entry Point (Simple API)

```python
import bioimage_pipeline_utils as rp
from standard_code.python.drift_correction import drift_correct

# Load TCZYX image
img = rp.load_tczyx_image("timelapse.tif")

# Simple translation-only drift correction (register to first frame)
corrected = drift_correct(img, method="phase_cross_correlation", reference="first")

# Sequential registration (register to previous frame)
corrected = drift_correct(img, method="phase_cross_correlation", reference="previous")

# GPU-accelerated
corrected = drift_correct(img, method="phase_cross_correlation", reference="first", use_gpu=True)

# StackReg translation mode
corrected = drift_correct(img, method="stackreg_translation", reference="first")

# Register using specific channel, apply to all
corrected = drift_correct(
    img, 
    method="phase_cross_correlation",
    reference="first",
    channels_to_register=[1]
)

# Save result
rp.save_tczyx_image(corrected, "timelapse_corrected.tif")
```

### Command Line

```bash
python drift_correction.py input.tif output.tif --method phase_cross_correlation --reference first
python drift_correction.py input.tif output.tif --method stackreg_translation --reference previous --gpu
```

### Advanced API (Direct Algorithm Access)

```python
from standard_code.python.drift_correction.phase_cross_correlation import phase_cross_correlation_cpu
from standard_code.python.drift_correction.pystackreg import stackreg_register

# Phase cross-correlation with mean reference (not available in entry point)
stack = img.get_image_data("TZYX", C=0)  # Get first channel as TZYX
shifts, corrected = phase_cross_correlation_cpu(stack, reference="mean", upsample_factor=20)

# Phase cross-correlation with median reference (robust to outliers)
shifts, corrected = phase_cross_correlation_cpu(stack, reference="median")

# Register to specific timepoint
shifts, corrected = phase_cross_correlation_cpu(stack, reference=5)  # Register to T=5

# StackReg with mean reference
shifts, corrected = stackreg_register(stack, reference="mean")
```

## Dependencies

**Required**:
- `numpy`
- `scipy` (for `scipy.ndimage.shift` - applies pure translation)
- `scikit-image` (for phase_cross_correlation - translation only)

**Optional**:
- `pystackreg` (for StackReg TRANSLATION mode only)
- `cucim` (for GPU acceleration - translation only)
- `cupy` (for GPU arrays)

## Migration from Existing Code

All `_*.py` files in the drift_correction folder should be **deleted** as they are non-working implementations:

**Files to DELETE**:
- `_demons.py`
- `_drift_correction_Legacy.py`
- `_enhanced_optical_flow.py`
- `_feature_based.py`
- `_feature_matching.py`
- `_fourier_shift.py`
- `_gradient_descent.py`
- `_mock_correction.py`
- `_mutual_information.py`
- `_normalized_cross_correlation.py`
- `_optical_flow.py`
- `_phase_cross_correlation.py` (replaced by `phase_cross_correlation.py`)
- `_sum_squared_differences.py`
- `_template_matching.py`
- `_variational.py`

**Files to KEEP**:
- `__init__.py`
- `drift_correct_utils.py` (utility functions)
- `synthetic_data_generators.py` (for testing)
- `phase_cross_correlation.py` (NEW - our minimal implementation)
- `pystackreg.py` (NEW - our minimal implementation)

## Testing Strategy

All tests must verify **translation-only** constraint:

1. **Synthetic data with known translations**: Use `synthetic_data_generators.py`
2. **Reference strategy validation**: Test "first" and "previous" modes
3. **Algorithm consistency**: Verify similar results across algorithms
4. **2D vs 3D**: Test both Z=1 and Z>1 cases
5. **GPU vs CPU**: Ensure GPU and CPU versions give identical translation values
6. **Edge cases**: T=1 (should return original), Z=1 (2D mode)

## Error Handling

Comprehensive error messages guide users:

```python
# Invalid reference
ValueError: Invalid reference 'mean'. Entry point only supports 'first' or 'previous'. 
For advanced reference strategies ('mean', 'median', specific timepoint), 
use algorithm-specific functions in drift_correction/ folder

# GPU not available
WARNING: GPU (cuCIM/CuPy) not available: No module named 'cucim'
WARNING: Falling back to CPU implementation

# StackReg not installed
ImportError: pystackreg is not installed. Install with: pip install pystackreg
```

## Logging

All implementations use Python logging:

```python
INFO: Applying translation-only drift correction using phase_cross_correlation
INFO: Image shape: T=100, C=2, Z=5, Y=512, X=512
INFO: Reference strategy: first
INFO: Using channel 0 for registration
INFO: Phase cross-correlation (CPU) - 3D registration
INFO: Computed 100 shifts - Mean: [0.12 -0.05 0.31], Max: 2.45 px
INFO: Applying shifts to all channels...
INFO: Drift correction complete
```

## Notes

- **Translation only**: Primary constraint - no exceptions
- **Minimal code**: ~600 lines wrapping established libraries
- **Simple entry point**: Only "first" and "previous" for consistency
- **Advanced options**: Direct algorithm access for power users
- **Subpixel accuracy**: upsample_factor parameter for precision
- **Fail gracefully**: Automatic CPU fallback if GPU unavailable
- **Type safety**: Comprehensive type hints throughout
- **Biological relevance**: Translation-only matches physical sample drift

---

**Author**: AI Agent for BIPHUB Pipeline  
**Date**: 2025-10-14  
**Status**: Phase 1 Complete - Minimal Translation-Only Implementation
