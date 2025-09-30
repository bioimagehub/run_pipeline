"""
Debug script with verbose output to trace the sign issue
"""
import numpy as np
import sys
import os

sys.path.append(r'e:\Oyvind\OF_git\run_pipeline\standard_code\python')
sys.path.append(r'e:\Oyvind\OF_git\run_pipeline\standard_code\python\drift_correction')
import bioimage_pipeline_utils as rp
import apply_shifts

# Import the internal function for testing
from phase_cross_correlation import _phase_cross_correlation_3d_cupy

def debug_phase_correlation():
    """Test phase cross-correlation internal function directly"""
    
    print("=== DEBUGGING PHASE CORRELATION INTERNALS ===")
    
    # Create a simple test case
    Z, Y, X = 1, 50, 50
    
    # Reference image: bright square at (20, 20)
    reference = np.zeros((Z, Y, X), dtype=np.float32)
    reference[0, 20:25, 20:25] = 1000.0
    
    # Test image: same square shifted by (+3, +5) pixels
    test_image = np.zeros((Z, Y, X), dtype=np.float32)
    test_image[0, 23:28, 25:30] = 1000.0  # Shifted by (dy=+3, dx=+5)
    
    print("Reference square: [20:25, 20:25]")  
    print("Test square:      [23:28, 25:30]")  # Shift: dy=+3, dx=+5
    print("Expected detected shift (how much image moved): [0, +3, +5]")
    print("Expected correction shift (for apply_shifts): [0, -3, -5]")
    
    # Call the internal function directly
    result = _phase_cross_correlation_3d_cupy(
        reference, test_image, 
        upsample_factor=1, 
        use_triangle_threshold=False
    )
    
    detected_raw = result[0]  # Raw shift detected by phase correlation
    print(f"\nPhase correlation raw result: {detected_raw}")
    print(f"This represents: how much the test image moved relative to reference")
    
    # Apply the sign convention from phase_cross_correlation function
    correction_shift = -detected_raw
    print(f"Correction shift (negated): {correction_shift}")
    print(f"This should be applied to test image to align it with reference")
    
    # Test if the correction works
    print(f"\nTesting correction...")
    
    # Apply the correction shift to the test image
    test_corrected = apply_shifts.apply_shift(
        test_image[0], 
        correction_shift[1:],  # Only Y,X for 2D image (skip Z)
        mode='constant', 
        order=1
    )
    
    # Compare corrected image with reference
    diff = np.abs(test_corrected - reference[0])
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"After correction:")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    
    # Find bright square positions
    ref_pos = np.unravel_index(np.argmax(reference[0]), reference[0].shape)
    test_pos = np.unravel_index(np.argmax(test_image[0]), test_image[0].shape)  
    corrected_pos = np.unravel_index(np.argmax(test_corrected), test_corrected.shape)
    
    print(f"\nSquare positions:")
    print(f"  Reference: {ref_pos}")
    print(f"  Original test: {test_pos}")
    print(f"  Corrected test: {corrected_pos}")
    
    if ref_pos == corrected_pos:
        print("+ SUCCESS: Correction worked perfectly!")
    else:
        print("! FAILURE: Correction did not align the squares")
        
        # Let's try the opposite sign
        print("\nTrying opposite sign...")
        wrong_correction = detected_raw  # Don't negate
        test_wrong_corrected = apply_shifts.apply_shift(
            test_image[0], 
            wrong_correction[1:], 
            mode='constant', 
            order=1
        )
        wrong_pos = np.unravel_index(np.argmax(test_wrong_corrected), test_wrong_corrected.shape)
        print(f"  With opposite sign: {wrong_pos}")

if __name__ == "__main__":
    debug_phase_correlation()