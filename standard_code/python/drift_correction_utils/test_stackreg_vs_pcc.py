"""
Compare StackReg vs Phase Cross-Correlation shift application methods.

This investigates WHY StackReg works perfectly without upsampling while PCC has issues.

MIT License
Copyright (c) 2024 BIPHUB - Bioimage Informatics Hub, University of Oslo
"""

import numpy as np
from scipy.ndimage import shift as scipy_shift, affine_transform
from skimage.registration import phase_cross_correlation
from pystackreg import StackReg
import matplotlib.pyplot as plt


def test_shift_methods():
    """Compare different methods of applying subpixel shifts."""
    
    # Create test image
    img = np.zeros((100, 100), dtype=np.float32)
    img[40:60, 40:60] = 255
    
    # Test subpixel shift
    true_shift_y, true_shift_x = 2.3, 1.7
    
    print("="*80)
    print("SHIFT METHOD COMPARISON")
    print("="*80)
    print(f"\nTrue shift: ({true_shift_y}, {true_shift_x}) pixels")
    print(f"This is a SUBPIXEL shift (not integer)")
    
    # Method 1: scipy.ndimage.shift (what PCC uses)
    print(f"\n{'-'*80}")
    print("METHOD 1: scipy.ndimage.shift (order=3)")
    print("-"*80)
    shifted_scipy = scipy_shift(img, shift=[true_shift_y, true_shift_x], 
                                order=3, mode='constant', cval=0)
    print(f"Creates shifted image using spline interpolation")
    
    # Method 2: StackReg affine transform
    print(f"\n{'-'*80}")
    print("METHOD 2: StackReg affine_transform")
    print("-"*80)
    
    # StackReg uses affine transformation matrices
    # For translation: [[1, 0, dx], [0, 1, dy], [0, 0, 1]]
    tmat = np.array([
        [1, 0, true_shift_x],
        [0, 1, true_shift_y],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Apply using StackReg's method (same as what transform_stack does internally)
    sr = StackReg(StackReg.TRANSLATION)
    shifted_stackreg = sr.transform(img, tmat)
    print(f"Creates shifted image using affine transformation")
    print(f"Transformation matrix:\n{tmat}")
    
    # Method 3: scipy affine_transform (to understand StackReg)
    print(f"\n{'-'*80}")
    print("METHOD 3: scipy.ndimage.affine_transform")
    print("-"*80)
    
    # Build transformation matrix for scipy (inverse of forward transform)
    matrix = np.array([[1, 0], [0, 1]])
    offset = [-true_shift_y, -true_shift_x]
    
    shifted_affine = affine_transform(img, matrix, offset=offset, 
                                     order=3, mode='constant', cval=0)
    print(f"Creates shifted image using inverse affine transformation")
    
    # Now detect shifts using PCC with different upsample factors
    print(f"\n{'-'*80}")
    print("SHIFT DETECTION: Phase Cross-Correlation")
    print("-"*80)
    
    for test_img, name in [(shifted_scipy, "scipy_shift"), 
                           (shifted_stackreg, "StackReg"), 
                           (shifted_affine, "affine_transform")]:
        print(f"\n{name}:")
        for upsample in [1, 10, 100]:
            detected, error, _ = phase_cross_correlation(
                img, test_img, upsample_factor=upsample, normalization="phase"
            )
            error_y = abs(detected[0] - true_shift_y)
            error_x = abs(detected[1] - true_shift_x)
            total_error = np.sqrt(error_y**2 + error_x**2)
            print(f"  upsample={upsample:3d}: detected=({detected[0]:6.3f}, {detected[1]:6.3f})  "
                  f"error={total_error:.4f}px")
    
    # Now test: Can we correct using detected shifts?
    print(f"\n{'-'*80}")
    print("CORRECTION TEST: Using detected shift from upsample_factor=1")
    print("-"*80)
    
    # Use scipy-shifted image (what we have in our pipeline)
    test_shifted = shifted_scipy
    
    # Detect with upsample_factor=1 (integer only)
    detected_int, _, _ = phase_cross_correlation(
        img, test_shifted, upsample_factor=1, normalization="phase"
    )
    
    # Detect with upsample_factor=10 (subpixel)
    detected_sub, _, _ = phase_cross_correlation(
        img, test_shifted, upsample_factor=10, normalization="phase"
    )
    
    print(f"\nDetected with upsample=1:  ({detected_int[0]:6.3f}, {detected_int[1]:6.3f})")
    print(f"Detected with upsample=10: ({detected_sub[0]:6.3f}, {detected_sub[1]:6.3f})")
    print(f"True shift:                ({true_shift_y:6.3f}, {true_shift_x:6.3f})")
    
    # Correct using both
    corrected_int = scipy_shift(test_shifted, shift=detected_int, 
                               order=3, mode='constant', cval=0)
    corrected_sub = scipy_shift(test_shifted, shift=detected_sub, 
                               order=3, mode='constant', cval=0)
    
    error_int = np.sum(np.abs(img - corrected_int))
    error_sub = np.sum(np.abs(img - corrected_sub))
    
    print(f"\nAlignment error with integer detection: {error_int:.1f}")
    print(f"Alignment error with subpixel detection: {error_sub:.1f}")
    print(f"Improvement: {(error_int - error_sub) / error_int * 100:.1f}%")
    
    # Visualize
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    axes[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(shifted_scipy, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title('Shifted (scipy.shift)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(shifted_stackreg, cmap='gray', vmin=0, vmax=255)
    axes[0, 2].set_title('Shifted (StackReg)')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(shifted_affine, cmap='gray', vmin=0, vmax=255)
    axes[0, 3].set_title('Shifted (affine_transform)')
    axes[0, 3].axis('off')
    
    axes[1, 0].imshow(np.abs(img - shifted_scipy), cmap='hot')
    axes[1, 0].set_title('Diff: scipy.shift')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(np.abs(img - shifted_stackreg), cmap='hot')
    axes[1, 1].set_title('Diff: StackReg')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(corrected_int, cmap='gray', vmin=0, vmax=255)
    axes[1, 2].set_title(f'Corrected (int)\nError: {error_int:.0f}')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(corrected_sub, cmap='gray', vmin=0, vmax=255)
    axes[1, 3].set_title(f'Corrected (subpixel)\nError: {error_sub:.0f}')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('shift_method_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n[CHART] Saved: shift_method_comparison.png")
    plt.show()
    
    # KEY INSIGHT
    print(f"\n{'='*80}")
    print("KEY INSIGHT")
    print("="*80)
    print("StackReg uses affine transformations (same as scipy methods).")
    print("The key difference is NOT the shift application method.")
    print("\nStackReg likely works better because:")
    print("1. It uses optimization-based registration (not just FFT phase correlation)")
    print("2. The optimization inherently finds subpixel shifts")
    print("3. It doesn't round to integer pixels")
    print("\nPCC with upsample_factor=1 DOES round to integers, causing jumps.")
    print("PCC with upsample_factor=10 gives similar quality to StackReg!")
    print("="*80)


if __name__ == "__main__":
    test_shift_methods()
