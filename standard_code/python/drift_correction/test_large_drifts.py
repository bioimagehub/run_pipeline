"""
Create synthetic test data with large known drifts to verify the algorithm works correctly.
"""
import sys
sys.path.append(r'e:\Oyvind\OF_git\run_pipeline\standard_code\python')
import bioimage_pipeline_utils as rp
import numpy as np
from phase_cross_correlation import phase_cross_correlation
import apply_shifts

def create_large_drift_test():
    """Create synthetic data with large known shifts and test our algorithm."""
    
    print("=== CREATING SYNTHETIC DATA WITH LARGE DRIFTS ===")
    
    # Load a real frame to use as base
    input_file = r"E:\Oyvind\BIP-hub-test-data\drift\input\live_cells\1_Meng.nd2"
    img = rp.load_tczyx_image(input_file)
    base_frame = img.data[0, 0, 0].astype(np.float32)  # Use first frame
    
    H, W = base_frame.shape
    print(f"Base frame size: {H}x{W}")
    
    # Create synthetic drift pattern: large random shifts
    T = 10
    large_shifts = np.array([
        [0, 0],      # Reference frame
        [25, -15],   # Large shifts in pixels  
        [-30, 20],
        [15, 35], 
        [-45, -10],
        [20, -25],
        [-15, 40],
        [35, 15],
        [-25, -30],
        [40, 25]
    ], dtype=np.float32)
    
    print("Applied large shifts (dy, dx):")
    for t, shift in enumerate(large_shifts):
        print(f"  T={t}: {shift}")
    
    # Create synthetic stack with applied shifts
    synthetic_stack = np.zeros((T, 1, 1, H, W), dtype=np.float32)
    
    for t in range(T):
        dy, dx = large_shifts[t]
        # Apply shifts using roll (simple integer shifts for ground truth)
        shifted_frame = np.roll(np.roll(base_frame, int(dy), axis=0), int(dx), axis=1)
        synthetic_stack[t, 0, 0] = shifted_frame
        
    print(f"\nCreated synthetic stack with shape: {synthetic_stack.shape}")
    
    # Test our phase correlation algorithm
    print("\n=== TESTING PHASE CORRELATION ON LARGE DRIFT DATA ===")
    
    detected_shifts = phase_cross_correlation(
        synthetic_stack, 
        reference_frame='first', 
        channel=0, 
        upsample_factor=100
    )
    
    print("\nDetected vs Applied shifts:")
    print("T   Applied(dy,dx)   Detected(dz,dy,dx)    Error(dy,dx)")
    print("-" * 65)
    
    errors = []
    for t in range(T):
        applied_dy, applied_dx = large_shifts[t]
        detected_dz, detected_dy, detected_dx = detected_shifts[t]
        
        error_dy = abs(detected_dy - applied_dy)  
        error_dx = abs(detected_dx - applied_dx)
        errors.append([error_dy, error_dx])
        
        print(f"{t:2d}  ({applied_dy:6.1f},{applied_dx:6.1f})   ({detected_dz:6.2f},{detected_dy:6.2f},{detected_dx:6.2f})   ({error_dy:5.2f},{error_dx:5.2f})")
    
    # Analyze accuracy
    errors = np.array(errors)
    mean_error = np.mean(errors, axis=0)
    max_error = np.max(errors, axis=0)
    
    print(f"\nAccuracy Analysis:")
    print(f"  Mean error (dy,dx): ({mean_error[0]:.2f}, {mean_error[1]:.2f}) pixels")
    print(f"  Max error (dy,dx):  ({max_error[0]:.2f}, {max_error[1]:.2f}) pixels")
    
    # Test correction
    print("\n=== TESTING SHIFT CORRECTION ===")
    corrected_stack = apply_shifts.apply_shifts_to_tczyx_stack(
        synthetic_stack, detected_shifts, mode='constant'
    )
    
    # Measure correction quality 
    reference_repeated = np.tile(synthetic_stack[0:1], (T, 1, 1, 1, 1))
    mse = np.mean((corrected_stack - reference_repeated)**2)
    print(f"Correction MSE: {mse:.6f}")
    
    if mse < 1.0:
        print("✓ Excellent correction quality")
    elif mse < 10.0:
        print("✓ Good correction quality")
    else:
        print("⚠ Poor correction quality")
    
    # Determine if our algorithm works with large shifts
    if np.all(mean_error < 1.0):
        print(f"\n✅ ALGORITHM WORKS CORRECTLY FOR LARGE SHIFTS")
        print(f"   Mean detection error < 1 pixel for {np.max(np.abs(large_shifts)):.0f} pixel drifts")
        print(f"   ➡️  THE ORIGINAL DATA LIKELY HAS ONLY SUB-PIXEL DRIFTS")
    else:
        print(f"\n❌ ALGORITHM HAS ISSUES WITH LARGE SHIFTS") 
        print(f"   Mean error {np.max(mean_error):.2f} pixels - needs debugging")
        
    return detected_shifts, large_shifts, errors

if __name__ == "__main__":
    create_large_drift_test()