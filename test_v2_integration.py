"""
Test phase_cross_correlation_v2 integration with drift_correction.py
"""
import sys
sys.path.insert(0, r'e:\Oyvind\OF_git\run_pipeline\standard_code\python')

from drift_correction_utils.phase_cross_correlation_v2 import register_image_xy
import bioimage_pipeline_utils as rp
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Load test file
input_path = r"Z:\Schink\Meng\driftcor\input\1.nd2"

print("=" * 60)
print("Testing phase_cross_correlation_v2.register_image_xy()")
print("=" * 60)

try:
    print(f"Loading image: {input_path}")
    img = rp.load_tczyx_image(input_path)
    print(f"Image shape: {img.data.shape}")
    print()
    
    print("Testing with 'first' reference...")
    registered_img, tmats = register_image_xy(
        img,
        channel=0,
        show_progress=True,
        no_gpu=False,
        reference='first',
        crop_fraction=1.0
    )
    
    print()
    print("=" * 60)
    print("SUCCESS!")
    print("=" * 60)
    print(f"Registered image shape: {registered_img.data.shape}")
    print(f"Transformation matrices shape: {tmats.shape}")
    print(f"First 5 shifts:")
    for i in range(min(5, len(tmats))):
        print(f"  Frame {i}: [{tmats[i][0]:.2f}, {tmats[i][1]:.2f}, {tmats[i][2]:.2f}]")
    
    # Test saving
    output_path = r"Z:\Schink\Meng\driftcor\input\1_v2_test.tif"
    print(f"\nSaving to: {output_path}")
    registered_img.save(output_path)
    print("Saved successfully!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
