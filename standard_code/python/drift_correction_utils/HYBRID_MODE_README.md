# Hybrid Mode Drift Correction - Implementation Details

## Overview

The hybrid mode in `phase_cross_correlation_v2.py` implements a **two-pass drift correction strategy** that combines the smoothness of frame-to-frame tracking with linear drift correction to prevent cumulative error accumulation.

## Problem Statement

### "Previous" Mode Issues
- ✅ Smooth local corrections (good for small frame-to-frame motion)
- ❌ Accumulates drift over time (small errors compound)
- Result: Systematic drift in one direction over long time series

### "First" Mode Issues
- ✅ No drift accumulation (always relative to frame 0)
- ❌ Fails when displacement becomes large (>25% of image)
- ❌ Phase cross-correlation produces spurious results for large shifts
- Result: Works for first few frames, then catastrophic jumps

## Solution: Two-Pass Hybrid Approach

### Pass 1: Frame-to-Frame Registration
```
Apply 'previous' mode:
- Register each frame to its previous frame
- Accumulate shifts: shift[t] = shift[t-1] + delta[t]
- Result: Smoothly corrected stack with potential residual drift
```

### Pass 2: Linear Drift Correction
```
1. Measure residual drift at intervals (default: every 10th frame):
   - Frame 0 vs Frame 10
   - Frame 0 vs Frame 20
   - Frame 0 vs Frame 30
   - ...
   
2. Linear interpolation of drift correction:
   - Frame 0: 0 pixels (by definition)
   - Frame 10: measured_drift pixels
   - Frame 5: (measured_drift / 2) pixels (interpolated)
   
3. Apply proportional correction to all frames
```

## Mathematical Details

Given:
- `shifts_pass1[t]` = accumulated shifts from frame-to-frame tracking
- `residual[k]` = measured drift at anchor frames k = [0, 10, 20, 30, ...]

For any frame `t`, the final shift is:
```
drift_correction[t] = linear_interpolation(t, anchor_frames, residuals)
final_shift[t] = shifts_pass1[t] + drift_correction[t]
```

Example with 30 frames, interval=10:
```
Anchor frames: [0, 10, 20, 30]
Residual drift: [0, -10, -20, -28] pixels in Y

Interpolated corrections:
Frame 0:  0 px
Frame 1:  -1 px
Frame 2:  -2 px
...
Frame 10: -10 px
Frame 15: -15 px
Frame 20: -20 px
...
Frame 30: -28 px
```

## Parameters

### `hybrid_reanchor_interval` (default: 10)
Controls how often residual drift is measured.

- **Smaller values (5)**: 
  - More frequent corrections
  - Better for severe, non-linear drift
  - Slightly more computation
  
- **Larger values (20)**:
  - Fewer corrections
  - Assumes more linear drift
  - Faster computation

## Usage

```python
import bioimage_pipeline_utils as rp
from drift_correction_utils.phase_cross_correlation_v2 import register_image_xy

# Load image
img = rp.load_tczyx_image("timelapse.tif")

# Apply hybrid correction
registered, shifts = register_image_xy(
    img, 
    reference='hybrid',
    channel=0,
    hybrid_reanchor_interval=10
)

# Save result
registered.save("corrected.tif")
```

## Comparison with Other Modes

| Mode | Smoothness | Drift Prevention | Large Displacements | Speed |
|------|-----------|------------------|---------------------|-------|
| **first** | ❌ Can jump | ✅ No drift | ❌ Fails > 25% | Fast |
| **previous** | ✅ Very smooth | ❌ Accumulates drift | ✅ Works well | Fast |
| **hybrid** | ✅ Smooth | ✅ Corrects drift | ✅ Works well | Slightly slower |

## Performance

### CPU Version
- Pass 1: ~0.1s per frame (phase cross-correlation)
- Pass 2: ~0.1s per anchor frame + interpolation
- Total: ~10-20% slower than pure 'previous' mode

### GPU Version
- Pass 1: ~0.01s per frame (CuPy-accelerated)
- Pass 2: All operations on GPU, minimal overhead
- Total: ~5-10% slower than pure 'previous' mode

## Output Files from Test

When running `test_code()`:
```
1_Meng_PCC_v2_PREVIOUS_corrected.tif  - Baseline (frame-to-frame only)
1_Meng_PCC_v2_HYBRID_corrected.tif    - Two-pass corrected

1_Meng_PCC_v2_PREVIOUS_shifts.npy     - Shift vectors from Pass 1
1_Meng_PCC_v2_HYBRID_shifts.npy       - Combined shift vectors
```

Compare these in ImageJ/Fiji to verify drift correction!

## Troubleshooting

### If hybrid mode shows artifacts:
1. Check `hybrid_reanchor_interval` - try smaller values
2. Verify input image quality (signal-to-noise ratio)
3. Try `crop_fraction=0.8` to ignore noisy edges

### If residual drift persists:
1. Reduce `hybrid_reanchor_interval` to 5 or even 3
2. Check that anchor frames show good correlation to frame 0
3. Consider if drift is non-linear (exponential, sinusoidal, etc.)

## Future Enhancements

Possible improvements:
- Non-linear drift models (polynomial, spline fitting)
- Adaptive interval selection based on correlation quality
- Weighted interpolation based on confidence scores
- Multi-resolution pyramid for very large displacements

---

**Implementation by**: Øyvind Fiksdahl Østerås for BIPHUB  
**Date**: October 2025  
**License**: MIT
