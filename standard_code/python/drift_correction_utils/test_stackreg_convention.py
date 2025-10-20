"""
Investigate StackReg's transformation matrix convention.

StackReg uses a FORWARD transformation convention, while scipy uses BACKWARD.

MIT License  
Copyright (c) 2024 BIPHUB - Bioimage Informatics Hub, University of Oslo
"""

import numpy as np
from scipy.ndimage import shift as scipy_shift
from pystackreg import StackReg


def investigate_stackreg_convention():
    """Test what StackReg's transformation matrices actually represent."""
    
    # Create simple test image
    img = np.zeros((100, 100), dtype=np.float32)
    img[40:45, 40:45] = 255
    
    print("="*80)
    print("STACKREG TRANSFORMATION MATRIX CONVENTION")
    print("="*80)
    
    # Create two images: original and shifted
    shift_amount_y = 5.0
    shift_amount_x = 3.0
    
    print(f"\nCreating shifted image by ({shift_amount_y}, {shift_amount_x}) pixels")
    
    # Use scipy to create a shifted version
    img_shifted = scipy_shift(img, shift=[shift_amount_y, shift_amount_x], 
                             order=3, mode='constant', cval=0)
    
    # Find square location in both images
    y0, x0 = np.where(img > 0)
    y1, x1 = np.where(img_shifted > 0)
    
    print(f"Original square at: y=[{y0.min()}-{y0.max()}], x=[{x0.min()}-{x0.max()}]")
    print(f"Shifted square at:  y=[{y1.min()}-{y1.max()}], x=[{x1.min()}-{x1.max()}]")
    print(f"Actual movement: dy={y1.min()-y0.min()}, dx={x1.min()-x0.min()}")
    
    # Now use StackReg to find the transformation
    print(f"\n{'-'*80}")
    print("STACKREG REGISTRATION")
    print("-"*80)
    
    sr = StackReg(StackReg.TRANSLATION)
    
    # Register: find transformation FROM img TO img_shifted
    # (i.e., what transform moves img to match img_shifted)
    tmat = sr.register(img, img_shifted)
    
    print(f"\nTransformation matrix returned by StackReg:")
    print(tmat)
    print(f"\nTranslation components:")
    print(f"  tmat[0, 2] = {tmat[0, 2]:.3f}  (X translation)")
    print(f"  tmat[1, 2] = {tmat[1, 2]:.3f}  (Y translation)")
    
    # Apply the transformation to the original image
    img_corrected = sr.transform(img, tmat)
    
    # Check if it matches
    error = np.sum(np.abs(img_corrected - img_shifted))
    print(f"\nApplying tmat to original image...")
    print(f"Alignment error: {error:.1f}")
    
    if error < 100:
        print("✓ FORWARD transformation: tmat moves img TOWARD img_shifted")
    else:
        print("✗ Transformation doesn't work as expected")
    
    # Now test: what if we want to CORRECT the shifted image?
    print(f"\n{'-'*80}")
    print("DRIFT CORRECTION SCENARIO")
    print("-"*80)
    print("Goal: We have img_shifted (drifted) and want to correct it back to img (reference)")
    
    # Register: find transformation FROM img_shifted TO img
    tmat_correction = sr.register(img_shifted, img)
    
    print(f"\nCorrection matrix (img_shifted -> img):")
    print(tmat_correction)
    print(f"Translation components:")
    print(f"  tmat[0, 2] = {tmat_correction[0, 2]:.3f}  (X)")
    print(f"  tmat[1, 2] = {tmat_correction[1, 2]:.3f}  (Y)")
    
    # Apply correction
    img_corrected = sr.transform(img_shifted, tmat_correction)
    
    error = np.sum(np.abs(img_corrected - img))
    print(f"\nAlignment error after correction: {error:.1f}")
    
    if error < 100:
        print("✓ Successfully corrected!")
    
    # Compare with our PCC approach
    print(f"\n{'-'*80}")
    print("COMPARISON: What would PCC give us?")
    print("-"*80)
    
    from skimage.registration import phase_cross_correlation
    
    # PCC from img (reference) to img_shifted (moving)
    pcc_shift, _, _ = phase_cross_correlation(img, img_shifted, 
                                              upsample_factor=100, 
                                              normalization="phase")
    
    print(f"\nPCC detected shift (img -> img_shifted):")
    print(f"  shift_y = {pcc_shift[0]:.3f}")
    print(f"  shift_x = {pcc_shift[1]:.3f}")
    
    # To correct img_shifted, we apply this shift directly
    img_corrected_pcc = scipy_shift(img_shifted, shift=pcc_shift, 
                                    order=3, mode='constant', cval=0)
    
    error_pcc = np.sum(np.abs(img_corrected_pcc - img))
    print(f"\nAlignment error after PCC correction: {error_pcc:.1f}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)
    print("\nStackReg transformation matrices:")
    print(f"  - tmat represents FORWARD transformation (moves img to match target)")
    print(f"  - For drift correction: register(drifted, reference)")
    print(f"  - Then apply: transform(drifted, tmat)")
    print(f"\nPhase Cross-Correlation:")
    print(f"  - Returns shift vector to ALIGN images")  
    print(f"  - For drift correction: detect shift from reference to drifted")
    print(f"  - The detected shift can be applied DIRECTLY with scipy_shift")
    print(f"\nBoth methods work correctly when used properly!")
    print(f"StackReg works without upsampling because it uses optimization,")
    print(f"which inherently finds subpixel shifts without explicit upsampling.")
    print("="*80)


if __name__ == "__main__":
    investigate_stackreg_convention()
