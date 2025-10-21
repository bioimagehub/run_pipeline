"""
Test drift correction on real ND2 file
"""
import sys
sys.path.insert(0, r'e:\Oyvind\OF_git\run_pipeline\standard_code\python')

from drift_correction_utils import phase_cross_correlation_v2 as pcc
import logging

# Set up logging to see progress
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Input file
input_path = r"Z:\Schink\Meng\driftcor\input\1.nd2"
output_path = r"Z:\Schink\Meng\driftcor\input\1_corrected.tif"

print("=" * 60)
print("Testing drift correction on real ND2 file")
print("=" * 60)
print(f"Input: {input_path}")
print(f"Output: {output_path}")
print()

try:
    # Run drift correction with default parameters
    print("Starting drift correction...")
    print("Parameters:")
    print("  - Channel: 0")
    print("  - Correct only XY: True")
    print("  - Multi-time-scale: False")
    print("  - Subpixel: True")
    print("  - Upsample factor: 10")
    print("  - Max shifts: [20, 20, 5]")
    print("  - GPU acceleration:", pcc.GPU_AVAILABLE)
    print()
    
    img_registered, shifts = pcc.run(
        path=input_path,
        output_path=output_path,
        channel=0,
        correct_only_xy=True,
        multi_time_scale=False,
        subpixel=True,
        upsample_factor=10,
        max_shifts=[20, 20, 5],
        use_gpu=True
    )
    
    print()
    print("=" * 60)
    print("SUCCESS! Drift correction completed")
    print("=" * 60)
    print(f"Registered image shape: {img_registered.data.shape}")
    print(f"Number of timepoints: {img_registered.data.shape[0]}")
    print(f"Number of channels: {img_registered.data.shape[1]}")
    print(f"Z-slices: {img_registered.data.shape[2]}")
    print(f"Dimensions: {img_registered.data.shape[3]}x{img_registered.data.shape[4]}")
    print()
    print("Detected shifts (X, Y, Z) for each frame:")
    for i, shift in enumerate(shifts):
        print(f"  Frame {i}: [{shift[0]:.2f}, {shift[1]:.2f}, {shift[2]:.2f}]")
    print()
    print(f"Output saved to: {output_path}")
    
except FileNotFoundError as e:
    print(f"ERROR: File not found - {e}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
