#!/usr/bin/env python3
"""
Quick validation script to check the pipeline drift correction results.
"""

import numpy as np
import yaml
from pathlib import Path
import bioimage_pipeline_utils as rp

def validate_drift_correction():
    """Validate that the pipeline drift correction worked correctly."""
    
    # Define paths
    input_dir = Path("E:/Oyvind/BIP-hub-test-data/drift/synthetic/input")
    output_dir = Path("E:/Oyvind/BIP-hub-test-data/drift/synthetic/corrected")
    
    # Test files to validate
    test_cases = [
        ("squares_2d.tif", "squares_2d_corrected.tif"),
        ("squares_3d.tif", "squares_3d_corrected.tif"),
        ("cells_2d.tif", "cells_2d_corrected.tif"),
        ("cells_3d.tif", "cells_3d_corrected.tif")
    ]
    
    print("🔍 Validating Pipeline Drift Correction Results\n")
    
    for input_name, output_name in test_cases:
        input_path = input_dir / input_name
        output_path = output_dir / output_name
        metadata_path = output_dir / f"{output_name.replace('.tif', '_drift_metadata.yaml')}"
        
        if not input_path.exists() or not output_path.exists():
            print(f"❌ {input_name}: Files missing")
            continue
            
        # Load images
        try:
            input_img = rp.load_tczyx_image(str(input_path))
            output_img = rp.load_tczyx_image(str(output_path))
            
            print(f"📁 {input_name}:")
            print(f"   Input shape:  {input_img.shape}")
            print(f"   Output shape: {output_img.shape}")
            
            # Load metadata if available
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = yaml.safe_load(f)
                
                print(f"   Algorithm: {metadata.get('algorithm', 'Unknown')}")
                print(f"   Processing time: {metadata.get('processing_time_seconds', 'N/A'):.2f}s")
                
                if 'drift_stats' in metadata:
                    stats = metadata['drift_stats']
                    print(f"   Max drift: {stats.get('max_drift_magnitude', 'N/A'):.2f} pixels")
                    print(f"   Mean drift: {stats.get('mean_drift_magnitude', 'N/A'):.2f} pixels")
            
            # Calculate basic quality metrics
            input_data = input_img.get_image_data("CZYX", T=0)
            output_data = output_img.get_image_data("CZYX", T=0)
            
            # Check if we have time dimension for drift analysis
            if input_img.shape[0] > 1:  # Multiple time points
                # Compare first and last frames alignment
                first_frame = input_img.get_image_data("CZYX", T=0)
                last_frame_input = input_img.get_image_data("CZYX", T=-1)
                last_frame_output = output_img.get_image_data("CZYX", T=-1)
                
                # MSE between first and last frame
                mse_before = np.mean((first_frame - last_frame_input) ** 2)
                mse_after = np.mean((first_frame - last_frame_output) ** 2)
                
                improvement = (mse_before - mse_after) / mse_before * 100 if mse_before > 0 else 0
                
                print(f"   MSE before correction: {mse_before:.3f}")
                print(f"   MSE after correction:  {mse_after:.3f}")
                print(f"   Improvement: {improvement:.1f}%")
                
                if improvement > 50:
                    print(f"   ✅ Excellent correction")
                elif improvement > 10:
                    print(f"   ✅ Good correction")
                elif improvement > 0:
                    print(f"   ⚠️  Modest correction")
                else:
                    print(f"   ❌ No improvement or worsening")
            
            print()
            
        except Exception as e:
            print(f"❌ {input_name}: Error loading - {e}")
            print()

if __name__ == "__main__":
    validate_drift_correction()