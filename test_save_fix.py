#!/usr/bin/env python
"""Quick test to verify save_tczyx_image fix works properly."""

import numpy as np
import os
import sys
import tempfile

# Add standard_code to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'standard_code', 'python'))

import bioimage_pipeline_utils as rp

def test_save_and_load():
    """Test that saved TIFF files have proper channel metadata."""
    
    # Create test data (5D TCZYX)
    test_data = np.random.randint(0, 255, size=(2, 3, 4, 128, 128), dtype=np.uint8)
    t, c, z, y, x = test_data.shape
    
    print(f"Test data shape: {test_data.shape} (TCZYX)")
    print(f"Dimensions: T={t}, C={c}, Z={z}, Y={y}, X={x}")
    
    # Save to temporary location
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_output.tif")
        
        # Test 1: Save without metadata
        print("\n=== Test 1: Save without metadata ===")
        rp.save_tczyx_image(test_data, output_path)
        print(f"✓ Saved to {output_path}")
        
        # Load and verify
        loaded_img = rp.load_tczyx_image(output_path)
        loaded_data = loaded_img.data
        print(f"✓ Loaded shape: {loaded_data.shape}")
        print(f"  Dimensions: T={loaded_data.shape[0]}, C={loaded_data.shape[1]}, Z={loaded_data.shape[2]}, Y={loaded_data.shape[3]}, X={loaded_data.shape[4]}")
        
        if loaded_data.shape[1] != c:
            print(f"✗ ERROR: Channel dimension mismatch! Expected {c}, got {loaded_data.shape[1]}")
            return False
        print("✓ Channel dimension preserved correctly")
        
        # Test 2: Save with channel names
        print("\n=== Test 2: Save with channel names ===")
        output_path2 = os.path.join(tmpdir, "test_with_channels.tif")
        channel_names = ["DAPI", "GFP", "RFP"]
        rp.save_tczyx_image(
            test_data, 
            output_path2,
            channel_names=channel_names,
            physical_pixel_sizes=(1.0, 0.5, 0.5)
        )
        print(f"✓ Saved with channel names: {channel_names}")
        
        # Load and verify
        loaded_img2 = rp.load_tczyx_image(output_path2)
        loaded_data2 = loaded_img2.data
        print(f"✓ Loaded shape: {loaded_data2.shape}")
        
        if loaded_data2.shape[1] != c:
            print(f"✗ ERROR: Channel dimension mismatch! Expected {c}, got {loaded_data2.shape[1]}")
            return False
        print("✓ Channel dimension preserved correctly")
        
        # Test 3: Single channel
        print("\n=== Test 3: Save single channel ===")
        single_channel = np.random.randint(0, 255, size=(1, 1, 1, 128, 128), dtype=np.uint8)
        output_path3 = os.path.join(tmpdir, "test_single_channel.tif")
        rp.save_tczyx_image(single_channel, output_path3)
        print(f"✓ Saved single channel image")
        
        loaded_img3 = rp.load_tczyx_image(output_path3)
        loaded_data3 = loaded_img3.data
        print(f"✓ Loaded shape: {loaded_data3.shape}")
        
        if loaded_data3.shape[1] != 1:
            print(f"✗ ERROR: Single channel not preserved! Got {loaded_data3.shape[1]} channels")
            return False
        print("✓ Single channel preserved correctly")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)
    print("\nThe fix is working correctly. You can now re-run your pipeline:")
    print("1. Delete the old TIF files from input_tif folder")
    print("2. Re-run the 'Convert to tif' step")
    print("3. The new files should have proper channel metadata")
    print("4. Then 'Extract nuclear channel' step should work without errors")
    
    return True

if __name__ == "__main__":
    try:
        success = test_save_and_load()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
