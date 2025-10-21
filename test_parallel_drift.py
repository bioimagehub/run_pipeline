"""
Test parallel drift correction with progress bars per file.

This test creates synthetic drift data and processes multiple files in parallel
to verify the progress_manager integration works correctly.

MIT License
Copyright (c) 2024 BIPHUB - Bioimage Informatics Hub, University of Oslo
"""
import sys
sys.path.insert(0, r'e:\Oyvind\OF_git\run_pipeline\standard_code\python')

import os
import numpy as np
import logging
from pathlib import Path
import bioimage_pipeline_utils as rp

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_synthetic_drifting_image(output_path: str, n_timepoints: int = 5, size: tuple = (64, 64)):
    """
    Create a synthetic image with simulated drift for testing.
    
    Args:
        output_path: Where to save the test image
        n_timepoints: Number of timepoints
        size: Image dimensions (Y, X)
    """
    from scipy.ndimage import shift as scipy_shift
    
    # Create a pattern with features
    base = np.zeros(size, dtype=np.float32)
    base[size[0]//4:3*size[0]//4, size[1]//4:3*size[1]//4] = 1.0
    base[size[0]//3:2*size[0]//3, size[1]//3:2*size[1]//3] = 0.5
    
    # Create drifting sequence
    data = []
    for t in range(n_timepoints):
        # Add random drift
        drift_y = t * 2.5  # 2.5 pixels per frame
        drift_x = t * 1.5
        shifted = scipy_shift(base, [drift_y, drift_x], order=1, mode='constant')
        data.append(shifted)
    
    # Stack as TCZYX
    data = np.array(data)[:, np.newaxis, np.newaxis, :, :]  # Add C and Z dims
    
    # Save using tifffile
    import tifffile
    tifffile.imwrite(output_path, data, imagej=True, metadata={'axes': 'TCZYX'})
    print(f"Created synthetic image: {output_path}")


def main():
    """Test parallel drift correction with multiple files."""
    
    # Setup test directories
    test_dir = Path("test_data/parallel_drift_test")
    input_dir = test_dir / "input"
    output_dir = test_dir / "output"
    
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Testing Parallel Drift Correction with Progress Manager")
    print("=" * 60)
    
    # Create test files
    n_files = 3
    test_files = []
    
    print(f"\nCreating {n_files} synthetic test images...")
    for i in range(n_files):
        file_path = input_dir / f"test_drift_{i+1}.tif"
        # Vary timepoints for better progress bar testing
        n_timepoints = 5 + i * 2
        create_synthetic_drifting_image(str(file_path), n_timepoints=n_timepoints)
        test_files.append(str(file_path))
    
    print(f"\nTest files created in: {input_dir}")
    
    # Test 1: Sequential processing
    print("\n" + "=" * 60)
    print("TEST 1: Sequential Processing (--no-parallel)")
    print("=" * 60)
    
    from drift_correction import process_files
    import argparse
    
    args_seq = argparse.Namespace(
        input_search_pattern=str(input_dir / "*.tif"),
        output_folder=str(output_dir / "sequential"),
        output_suffix="_corrected",
        method="phase_cross_correlation",
        reference_channel=0,
        reference="first",
        no_save_tmats=False,
        no_gpu=False,
        no_parallel=True,  # Sequential
        crop_fraction=1.0,
        upsample_factor=10,
        max_shift=50.0
    )
    
    process_files(args_seq)
    
    # Test 2: Parallel processing
    print("\n" + "=" * 60)
    print("TEST 2: Parallel Processing (default)")
    print("=" * 60)
    print("You should see one progress bar per file updating independently!")
    print()
    
    args_par = argparse.Namespace(
        input_search_pattern=str(input_dir / "*.tif"),
        output_folder=str(output_dir / "parallel"),
        output_suffix="_corrected",
        method="phase_cross_correlation",
        reference_channel=0,
        reference="first",
        no_save_tmats=False,
        no_gpu=False,
        no_parallel=False,  # Parallel!
        crop_fraction=1.0,
        upsample_factor=10,
        max_shift=50.0
    )
    
    process_files(args_par)
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
    print(f"Sequential output: {output_dir / 'sequential'}")
    print(f"Parallel output: {output_dir / 'parallel'}")
    print("\nVerify that:")
    print("  1. Both methods produced identical results")
    print("  2. Parallel processing showed multiple progress bars")
    print("  3. Each file had its own independent progress bar")
    

if __name__ == "__main__":
    main()
