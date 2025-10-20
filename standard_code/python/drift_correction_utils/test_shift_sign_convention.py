"""
Test to verify shift sign conventions and identify the source of drift correction errors.

This script tests:
1. How np.roll applies shifts
2. How scipy.ndimage.shift applies shifts
3. How phase_cross_correlation reports shifts
4. Whether the sign convention is correct in our drift correction

MIT License
Copyright (c) 2024 BIPHUB - Bioimage Informatics Hub, University of Oslo
"""

import numpy as np
from scipy.ndimage import shift as scipy_shift
from skimage.registration import phase_cross_correlation
import matplotlib.pyplot as plt


def test_shift_conventions():
    """Test and visualize the sign conventions of different shift operations."""
    
    # Create simple test image with a bright square offset from center
    img = np.zeros((100, 100), dtype=np.float32)
    img[30:40, 30:40] = 255  # Square in upper-left quadrant
    
    print("="*70)
    print("SIGN CONVENTION TEST")
    print("="*70)
    print("\nOriginal square position: rows 30-40, cols 30-40")
    
    # Test 1: np.roll with positive shift
    print("\n" + "-"*70)
    print("TEST 1: np.roll(img, shift=10, axis=0)")
    print("-"*70)
    rolled = np.roll(img, shift=10, axis=0)
    # Find where the square moved
    rows, cols = np.where(rolled > 0)
    print(f"Square moved to: rows {rows.min()}-{rows.max()}, cols {cols.min()}-{cols.max()}")
    print(f"-> Positive shift in np.roll moves content DOWN (increases row index)")
    
    # Test 2: scipy.ndimage.shift with positive shift
    print("\n" + "-"*70)
    print("TEST 2: scipy_shift(img, shift=[10, 0])")
    print("-"*70)
    shifted = scipy_shift(img, shift=[10, 0], order=1, mode='constant', cval=0)
    rows, cols = np.where(shifted > 0)
    print(f"Square moved to: rows {rows.min()}-{rows.max()}, cols {cols.min()}-{cols.max()}")
    print(f"-> Positive shift in scipy_shift moves content DOWN (increases row index)")
    
    # Test 3: What does phase_cross_correlation report?
    print("\n" + "-"*70)
    print("TEST 3: phase_cross_correlation between original and np.roll(+10)")
    print("-"*70)
    
    # Create reference and shifted image
    reference = img.copy()
    shifted_img = np.roll(img, shift=10, axis=0)  # Shift down by 10 pixels
    
    # What shift does PCC detect?
    detected_shift, error, phasediff = phase_cross_correlation(
        reference, shifted_img, upsample_factor=1, normalization="phase"
    )
    
    print(f"Image was shifted DOWN by 10 pixels (using np.roll)")
    print(f"PCC detected shift: [{detected_shift[0]:.1f}, {detected_shift[1]:.1f}]")
    
    if detected_shift[0] > 0:
        print(f"-> PCC reports POSITIVE shift when content moves DOWN")
        print(f"-> This means: to UNDO the shift, we need to shift by NEGATIVE of this value")
    else:
        print(f"-> PCC reports NEGATIVE shift when content moves DOWN")
        print(f"-> This means: to UNDO the shift, we can use the detected shift directly")
    
    # Test 4: Apply correction using detected shift
    print("\n" + "-"*70)
    print("TEST 4: Correcting the shifted image")
    print("-"*70)
    
    # Try correcting with the detected shift directly
    corrected_direct = scipy_shift(shifted_img, shift=detected_shift, order=1, mode='constant', cval=0)
    
    # Try correcting with negated shift
    corrected_negated = scipy_shift(shifted_img, shift=-detected_shift, order=1, mode='constant', cval=0)
    
    # Compute alignment errors
    error_direct = np.sum(np.abs(reference - corrected_direct))
    error_negated = np.sum(np.abs(reference - corrected_negated))
    
    print(f"Error using detected shift directly: {error_direct:.1f}")
    print(f"Error using NEGATED shift: {error_negated:.1f}")
    
    if error_direct < error_negated:
        print(f"[OK] Correction works with detected shift DIRECTLY")
        print(f"-> No sign flip needed")
        sign_flip_needed = False
    else:
        print(f"[WARNING] Correction works with NEGATED shift")
        print(f"-> Sign flip IS needed!")
        sign_flip_needed = True
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(reference, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Original (Reference)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(shifted_img, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title(f'Shifted (np.roll +10)\nPCC detected: [{detected_shift[0]:.1f}, {detected_shift[1]:.1f}]')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(np.abs(reference - shifted_img), cmap='hot')
    axes[0, 2].set_title('Difference\n(uncorrected)')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(corrected_direct, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title(f'Corrected (direct shift)\nError: {error_direct:.0f}')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(corrected_negated, cmap='gray', vmin=0, vmax=255)
    axes[1, 1].set_title(f'Corrected (negated shift)\nError: {error_negated:.0f}')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(np.abs(reference - (corrected_direct if not sign_flip_needed else corrected_negated)), 
                      cmap='hot')
    axes[1, 2].set_title('Difference\n(best correction)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('shift_convention_test.png', dpi=150, bbox_inches='tight')
    print(f"\n[CHART] Visualization saved to: shift_convention_test.png")
    plt.show()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    if sign_flip_needed:
        print("[ERROR] PROBLEM FOUND: Current implementation needs sign correction!")
        print("   phase_cross_correlation returns shift to ALIGN images")
        print("   But we're applying it directly, which moves in WRONG direction")
        print("\n[FIX] Negate the shift before applying with scipy_shift")
        print("   Change: scipy_shift(img, shift=[shifts[t, 0], shifts[t, 1]], ...)")
        print("   To:     scipy_shift(img, shift=[-shifts[t, 0], -shifts[t, 1]], ...)")
    else:
        print("[OK] Sign convention is CORRECT in current implementation")
    print("="*70)
    
    return sign_flip_needed


if __name__ == "__main__":
    sign_flip_needed = test_shift_conventions()
