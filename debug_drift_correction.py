"""
Debug script to test drift correction step by step
"""
import numpy as np
import sys
import os

sys.path.append(r'e:\Oyvind\OF_git\run_pipeline\standard_code\python')
sys.path.append(r'e:\Oyvind\OF_git\run_pipeline\standard_code\python\drift_correction')
import bioimage_pipeline_utils as rp
import apply_shifts
import phase_cross_correlation as pcc

def debug_drift_correction():
    """Test drift correction with a simple known case"""
    
    # Create a simple test image with a clear feature
    print("=== DEBUGGING DRIFT CORRECTION ===")
    
    # Create test data: a bright square in a dark background
    Z, Y, X = 1, 100, 100
    T, C = 3, 1
    
    test_stack = np.zeros((T, C, Z, Y, X), dtype=np.float32)
    
    # Put a bright 10x10 square at position (40, 40) in first frame
    test_stack[0, 0, 0, 40:50, 40:50] = 1000.0
    
    # Apply known shifts to create drifted frames
    known_shifts = np.array([
        [0.0, 0.0, 0.0],  # T=0: no shift
        [0.0, 5.0, 3.0],  # T=1: shift by +5 in Y, +3 in X  
        [0.0, -4.0, 7.0]  # T=2: shift by -4 in Y, +7 in X
    ])
    
    print("Creating test data with known shifts:")
    for t in range(T):
        dz, dy, dx = known_shifts[t]
        print(f"  T={t}: Applying shift (dz={dz}, dy={dy}, dx={dx})")
        
        # Apply shifts manually using numpy roll to simulate drift
        if t == 0:
            # First frame is reference - no shift
            continue
        else:
            # Create drifted frame by shifting the reference
            shifted_frame = test_stack[0, 0, 0].copy()  # Start with reference frame
            
            # Apply shifts (use integer shifts for simplicity)
            if int(dy) != 0:
                shifted_frame = np.roll(shifted_frame, int(dy), axis=0)  # Y direction
            if int(dx) != 0:
                shifted_frame = np.roll(shifted_frame, int(dx), axis=1)  # X direction
                
            test_stack[t, 0, 0] = shifted_frame
    
    print("\nTest stack created. Now detecting shifts...")
    
    # Detect shifts using phase cross-correlation
    detected_shifts = pcc.phase_cross_correlation(
        test_stack, 
        reference_frame='first', 
        channel=0, 
        upsample_factor=1,  # Use integer accuracy for this test
        use_triangle_threshold=False  # Disable thresholding for simple test case
    )
    
    print("\nShift detection results:")
    for t in range(T):
        print(f"  T={t}: Known {known_shifts[t]} -> Detected {detected_shifts[t]}")
    
    # The detected shifts should be the NEGATIVE of the applied shifts
    # because phase correlation detects "how much the image moved"
    # but we need "how much to move it back" for correction
    expected_corrections = -known_shifts  # Should be negative of applied shifts
    
    print("\nShift validation:")
    for t in range(T):
        error = np.abs(detected_shifts[t] - expected_corrections[t])
        print(f"  T={t}: Expected correction {expected_corrections[t]} -> Detected {detected_shifts[t]}")
        print(f"        Error: {error} (magnitude: {np.linalg.norm(error):.3f})")
    
    # Apply the detected shifts to correct the drift
    print("\nApplying detected shifts for correction...")
    corrected_stack = apply_shifts.apply_shifts_to_tczyx_stack(
        test_stack, detected_shifts, mode='constant', order=1
    )
    
    # Check if correction worked by comparing all frames to first frame
    print("\nValidating drift correction:")
    reference_frame = corrected_stack[0, 0, 0]
    
    for t in range(1, T):
        corrected_frame = corrected_stack[t, 0, 0]
        frame_diff = np.abs(corrected_frame - reference_frame)
        max_diff = np.max(frame_diff)
        mean_diff = np.mean(frame_diff)
        
        print(f"  T={t} vs T=0: Max difference = {max_diff:.6f}, Mean difference = {mean_diff:.6f}")
        
        # Find the position of the bright square in corrected frame
        max_pos = np.unravel_index(np.argmax(corrected_frame), corrected_frame.shape)
        ref_max_pos = np.unravel_index(np.argmax(reference_frame), reference_frame.shape)
        
        print(f"  T={t}: Bright square at {max_pos}, reference at {ref_max_pos}")
        
        if max_diff < 1e-6:
            print(f"  + T={t}: EXCELLENT correction (near-perfect alignment)")
        elif max_diff < 0.1:
            print(f"  + T={t}: GOOD correction")
        else:
            print(f"  ! T={t}: POOR correction (drift not corrected)")
    
    # Save test data for visual inspection
    test_output_path = r"E:\Oyvind\BIP-hub-test-data\drift\debug_test_original.tif"
    corrected_output_path = r"E:\Oyvind\BIP-hub-test-data\drift\debug_test_corrected.tif"
    
    rp.save_tczyx_image(test_stack, test_output_path)
    rp.save_tczyx_image(corrected_stack, corrected_output_path)
    
    print(f"\nTest data saved:")
    print(f"  Original (drifted): {test_output_path}")
    print(f"  Corrected: {corrected_output_path}")
    print(f"Open these in ImageJ and play as movie to visually verify correction")
    
    return detected_shifts, corrected_stack

if __name__ == "__main__":
    debug_drift_correction()