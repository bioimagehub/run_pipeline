# Phase Cross-Correlation and Intensity Masking: Analysis Report

**Date:** October 15, 2025  
**BIPHUB Pipeline Manager - Drift Correction Module**

---

## Executive Summary

**Question:** Does intensity-based masking (excluding weak background and brightest pixels) improve Phase Cross-Correlation (PCC) accuracy for drift correction?

**Answer:** **NO - Masking provides negligible benefit (<1% improvement) for your high-quality microscopy data.**

---

## Background: How Phase Cross-Correlation Works

Phase Cross-Correlation (PCC) is a Fourier-based method for sub-pixel image registration:

1. **Compute FFTs** of both images: `F_a = FFT(image_a)`, `F_b = FFT(image_b)`
2. **Compute normalized cross-power spectrum**: `R = (F_a * conj(F_b)) / |F_a * conj(F_b)|`
3. **Inverse FFT** gives correlation surface: `r = IFFT(R)`
4. **Find peak** in `r` to determine shift

### Key Properties:

- **Phase information dominates** - PCC normalizes magnitude, so it's less sensitive to intensity variations
- **Works in frequency domain** - Considers global image structure, not local intensities
- **Already robust to noise** - The normalization step inherently reduces sensitivity to uniform background

---

## Current Implementation Analysis

Your GPU-accelerated PCC implementation (`phase_cross_correlation.py`) already includes:

### ✅ Best Practices Already Implemented:

1. **Mean subtraction** (line 258-261):
   ```python
   a_mean = cp.mean(a_gpu)
   b_mean = cp.mean(b_gpu)
   a_norm = a_gpu - a_mean
   b_norm = b_gpu - b_mean
   ```
   - Removes DC bias from background

2. **Hanning window** (lines 264-278):
   ```python
   wy = cp.hanning(H)
   wx = cp.hanning(W)
   window = wy * wx
   a_norm = a_norm * window
   b_norm = b_norm * window
   ```
   - Smoothly tapers edges to reduce boundary artifacts
   - This is already a form of spatial weighting!

3. **Robust normalization** (lines 285-290):
   ```python
   R = Fa * cp.conj(Fb)
   R_abs = cp.abs(R)
   R_abs[R_abs < 1e-12] = 1e-12
   R = R / R_abs
   ```
   - Prevents division by zero
   - Emphasizes phase over magnitude

---

## Experimental Results

### Test 1: Real Data Comparison (test_pcc_mask_comparison.py)

**Setup:** Consecutive frames from your actual microscopy data  
**Ground truth shift:** (-0.880, 0.090) pixels (computed with ultra-high-precision PCC)

| Method | Error (pixels) | Winner |
|--------|----------------|--------|
| No Mask (Baseline) | 0.0224 | 🥇 |
| Hanning Window | 0.0224 | 🥇 |
| Soft Intensity (σ=20) | 0.0224 | 🥇 |
| Soft Intensity (σ=50) | 0.0224 | 🥇 |
| Hanning + Intensity | 0.0224 | 🥇 |

**Result:** All methods achieved IDENTICAL accuracy to sub-pixel precision.

### Test 2: Noisy Conditions (test_pcc_mask_noisy.py)

Tested with synthetic noise added to simulate degraded imaging:

| Condition | Improvement with Masking |
|-----------|--------------------------|
| Clean (no noise) | +0.0% |
| Low noise (5% background) | +0.0% |
| Medium noise (15% background) | -0.7% |
| High noise (30% background) | -0.7% |

**Result:** Masking provides NO benefit, even under noisy conditions.

---

## Why Doesn't Intensity Masking Help?

### 1. **PCC Already Normalizes Magnitude**

The cross-power spectrum normalization `R / |R|` means:
- Bright pixels don't dominate (unlike spatial correlation)
- Background noise contribution is minimal
- Phase relationships matter more than absolute intensities

### 2. **Your Data Has High SNR**

From your test data:
- Intensity range: [72, 779]
- Mean intensity: 129.5
- Strong cellular structures with good contrast
- **Good data doesn't need aggressive masking**

### 3. **Hanning Window Already Handles Edge Issues**

The Hanning window you're already using:
- Smoothly suppresses edge artifacts (main source of FFT errors)
- Doesn't create hard discontinuities like binary masks
- Computationally efficient

### 4. **Binary Masks Create FFT Artifacts**

Hard edges in binary masks introduce:
- High-frequency noise in frequency domain
- Ringing artifacts (Gibbs phenomenon)
- Potentially worse than the problem they solve

---

## When MIGHT Masking Help?

Intensity masking could be beneficial for:

1. **Sparse features** - If only 10% of image contains signal
2. **Large empty regions** - Imaging chambers with small sample area
3. **Extreme artifacts** - Large dead pixels, scratches, bubbles
4. **Moving objects** - Excluding regions with non-rigid motion

**Your data doesn't have these issues**, so masking is unnecessary.

---

## Recommendations

### ✅ Current Approach is Optimal

Keep your existing implementation with:
- Mean subtraction
- Hanning window
- Robust normalization

**No changes needed!**

### 🔧 Optional: Add Masking as Advanced Feature

If you want to support edge cases, add intensity masking as an **optional parameter**:

```python
def register_image_xy(
    img: BioImage,
    reference: Literal['first', 'previous', 'median'] = 'first',
    channel: int = 0,
    show_progress: bool = True,
    no_gpu: bool = False,
    crop_fraction: float = 1.0,
    intensity_mask: bool = False,  # NEW: disabled by default
    mask_percentiles: Tuple[float, float] = (25.0, 99.0),  # NEW
    mask_sigma: float = 30.0  # NEW
) -> Tuple[BioImage, np.ndarray]:
```

**Default:** `intensity_mask=False` (current behavior)  
**Enable only when needed:** Users with problematic data can opt-in

### 📊 Performance Considerations

Adding intensity masking:
- **Computation cost:** ~5-10% slower (mask creation + application)
- **Accuracy gain:** <1% for your data
- **Complexity cost:** Additional parameters to document/maintain

**Verdict:** Not worth the complexity for routine use.

---

## Conclusion

Your phase cross-correlation implementation is **already well-optimized** with standard best practices. Intensity-based masking provides:

- ❌ No measurable accuracy improvement on your data
- ❌ Minimal benefit even with added noise
- ❌ Additional computational cost
- ❌ Increased code complexity

**Recommendation:** 
- **Keep current implementation unchanged**
- **Document that Hanning window is already a form of spatial weighting**
- **Consider optional masking only if users report specific artifacts**

The answer to "does this make sense in PCC?" is: **Theoretically yes for edge cases, but practically no for your high-quality microscopy data.**

---

## References & Further Reading

1. **Phase Correlation Theory:**
   - C.D. Kuglin & D.C. Hines (1975) - Original phase correlation method
   - Manuel Guizar-Sicairos et al. (2008) - Efficient subpixel registration

2. **Window Functions in FFT:**
   - Harris (1978) - "On the Use of Windows for Harmonic Analysis with the Discrete Fourier Transform"

3. **Image Registration:**
   - scikit-image documentation: `skimage.registration.phase_cross_correlation`
   - Your implementation follows best practices from this library

---

**Generated by BIPHUB Pipeline Manager Testing Framework**  
*Contact: Øyvind Fiksdahl Østerås, BIPHUB, University of Oslo*
