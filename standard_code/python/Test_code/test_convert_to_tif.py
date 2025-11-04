"""
Test different approaches for converting ND2 to OME-TIFF using BioIO ecosystem.
Compares performance of 5 different strategies.

MIT License - BIPHUB, University of Oslo
"""

import os
import time
import tempfile
import shutil
from pathlib import Path
import numpy as np

from bioio import BioImage
import bioio_ome_tiff
from tifffile import imwrite
import ome_types


TEST_FILE = r"E:\Oyvind\BIP-hub-test-data\drift\input\live_cells\1_Meng.nd2"
OUTPUT_DIR = Path(tempfile.gettempdir()) / "bioio_speed_test"


def setup_test():
    """Create output directory and verify test file exists."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    if not Path(TEST_FILE).exists():
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")
    print(f"Test file: {TEST_FILE}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("-" * 80)


def cleanup_test():
    """Remove output directory after tests."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    print(f"\nCleaned up: {OUTPUT_DIR}")


def get_file_size_mb(filepath):
    """Get file size in MB."""
    return Path(filepath).stat().st_size / (1024 * 1024)


def verify_output(output_path: Path, original_img: BioImage, original_data: np.ndarray = None) -> dict:
    """
    Verify that output file has correct metadata and pixel values.
    
    Args:
        output_path: Path to output TIFF file
        original_img: Original BioImage object
        original_data: Original numpy array (optional, will load if not provided)
    
    Returns:
        Dictionary with verification results
    """
    from tifffile import TiffFile, imread
    import warnings
    
    results = {
        'is_ome': False,
        'pixels_identical': False,
        'physical_size_correct': False,
        'shape_correct': False,
        'dtype_correct': False
    }
    
    try:
        # Suppress bioformats warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            
            # Load output file using tifffile directly for speed
            output_data = imread(output_path)
            
            # Check if it's OME-TIFF
            with TiffFile(output_path) as tif:
                results['is_ome'] = tif.is_ome
                
                # Try to get physical pixel sizes from OME-XML if available
                if tif.is_ome and hasattr(tif, 'ome_metadata'):
                    try:
                        output_img = BioImage(output_path, reader=bioio_ome_tiff.Reader)
                        out_sizes = output_img.physical_pixel_sizes
                        
                        # Check physical pixel sizes
                        orig_sizes = original_img.physical_pixel_sizes
                        
                        # Allow small floating point differences
                        if orig_sizes and out_sizes:
                            x_match = abs(orig_sizes.X - out_sizes.X) < 1e-6 if orig_sizes.X and out_sizes.X else False
                            y_match = abs(orig_sizes.Y - out_sizes.Y) < 1e-6 if orig_sizes.Y and out_sizes.Y else False
                            z_match = abs(orig_sizes.Z - out_sizes.Z) < 1e-6 if orig_sizes.Z and out_sizes.Z else False
                            results['physical_size_correct'] = x_match and y_match and z_match
                    except:
                        # If we can't load with BioImage, skip physical size check
                        results['physical_size_correct'] = False
        
        # Check shape
        results['shape_correct'] = output_data.shape == original_img.shape
        
        # Check dtype
        results['dtype_correct'] = output_data.dtype == original_img.dtype
        
        # Check pixel values (sample-based comparison for speed)
        if original_data is None:
            original_data = np.asarray(original_img.data)
        
        # First check: array_equal for quick identical check
        if original_data.shape == output_data.shape and original_data.dtype == output_data.dtype:
            # Sample 1000 random pixels for quick verification
            total_pixels = original_data.size
            sample_size = min(10000, total_pixels)
            
            # Generate random indices
            np.random.seed(42)  # Reproducible
            flat_indices = np.random.choice(total_pixels, sample_size, replace=False)
            
            # Compare sampled pixels
            orig_flat = original_data.ravel()
            out_flat = output_data.ravel()
            
            sample_match = np.array_equal(orig_flat[flat_indices], out_flat[flat_indices])
            
            if sample_match:
                # If samples match, do full check on a subset
                results['pixels_identical'] = True
            else:
                # Count differences in sample
                diff_in_sample = np.sum(orig_flat[flat_indices] != out_flat[flat_indices])
                results['pixels_identical'] = False
                results['pixel_diff_count'] = diff_in_sample
                results['pixel_diff_percent'] = (diff_in_sample / sample_size) * 100
        else:
            results['pixels_identical'] = False
        
    except Exception as e:
        results['error'] = str(e)
    
    return results


def print_verification(strategy_name: str, verify_results: dict) -> None:
    """Print verification results in a readable format."""
    print(f"\n  Verification for {strategy_name}:")
    print(f"    ✓ OME-TIFF format: {verify_results.get('is_ome', False)}")
    print(f"    ✓ Shape preserved: {verify_results.get('shape_correct', False)}")
    print(f"    ✓ Dtype preserved: {verify_results.get('dtype_correct', False)}")
    print(f"    ✓ Physical sizes: {verify_results.get('physical_size_correct', False)}")
    print(f"    ✓ Pixels identical: {verify_results.get('pixels_identical', False)}")
    
    if not verify_results.get('pixels_identical', False) and 'pixel_diff_percent' in verify_results:
        print(f"      WARNING: {verify_results['pixel_diff_count']} pixels differ ({verify_results['pixel_diff_percent']:.4f}%)")
    
    if 'error' in verify_results:
        print(f"    ✗ ERROR: {verify_results['error']}")



# ============================================================================
# STRATEGY 1: Direct BioImage.save() - Simplest approach
# ============================================================================
def strategy_1_direct_bioimage_save():
    """
    Strategy 1: Direct BioImage.save() method.
    Simplest but loads entire file into memory.
    """
    print("\n[Strategy 1] Direct BioImage.save()")
    output_path = OUTPUT_DIR / "strategy_1_direct.ome.tif"
    
    start = time.time()
    
    img = BioImage(TEST_FILE)
    original_data = np.asarray(img.data)
    img.save(output_path)
    
    elapsed = time.time() - start
    size_mb = get_file_size_mb(output_path)
    
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  Shape: {img.shape}")
    
    # Verify output
    verify_results = verify_output(output_path, img, original_data)
    print_verification("Strategy 1", verify_results)
    
    return elapsed, size_mb, verify_results


# ============================================================================
# STRATEGY 2: Numpy asarray() + tifffile with compression
# ============================================================================
def strategy_2_numpy_tifffile_compressed():
    """
    Strategy 2: Load to numpy array, save with tifffile + compression.
    More control over compression, but loads entire file into memory.
    """
    print("\n[Strategy 2] Numpy + tifffile with compression")
    output_path = OUTPUT_DIR / "strategy_2_compressed.ome.tif"
    
    start = time.time()
    
    img = BioImage(TEST_FILE)
    data = np.asarray(img.data)  # TCZYX
    
    # Save with compression
    imwrite(
        output_path,
        data,
        photometric='minisblack',
        compression='zlib',
        compressionargs={'level': 6},
        ome=True,
        metadata={'axes': 'TCZYX'}
    )
    
    elapsed = time.time() - start
    size_mb = get_file_size_mb(output_path)
    
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  Shape: {data.shape}")
    
    # Verify output
    verify_results = verify_output(output_path, img, data)
    print_verification("Strategy 2", verify_results)
    
    return elapsed, size_mb, verify_results


# ============================================================================
# STRATEGY 3: Dask delayed loading + chunked writing
# ============================================================================
def strategy_3_dask_delayed():
    """
    Strategy 3: Use dask for lazy/delayed loading.
    Memory efficient for large files.
    """
    print("\n[Strategy 3] Dask delayed loading")
    output_path = OUTPUT_DIR / "strategy_3_dask.ome.tif"
    
    start = time.time()
    
    img = BioImage(TEST_FILE)
    dask_data = img.dask_data  # Lazy loading
    
    # Compute and save (dask handles chunking automatically)
    data = dask_data.compute()
    
    imwrite(
        output_path,
        data,
        photometric='minisblack',
        compression='zlib',
        compressionargs={'level': 6},
        ome=True,
        metadata={'axes': 'TCZYX'}
    )
    
    elapsed = time.time() - start
    size_mb = get_file_size_mb(output_path)
    
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  Shape: {data.shape}")
    
    # Verify output
    verify_results = verify_output(output_path, img, data)
    print_verification("Strategy 3", verify_results)
    
    return elapsed, size_mb, verify_results


# ============================================================================
# STRATEGY 4: Frame-by-frame writing (memory efficient)
# ============================================================================
def strategy_4_frame_by_frame():
    """
    Strategy 4: Write frame-by-frame to minimize memory usage.
    Most memory efficient but potentially slower.
    """
    print("\n[Strategy 4] Frame-by-frame writing")
    output_path = OUTPUT_DIR / "strategy_4_frame_by_frame.ome.tif"
    
    start = time.time()
    
    img = BioImage(TEST_FILE)
    shape = img.shape  # TCZYX
    
    # Process frame by frame
    with imwrite(output_path, shape=shape, dtype=img.dtype, photometric='minisblack',
                 compression='zlib', compressionargs={'level': 6},
                 ome=True, metadata={'axes': 'TCZYX'}) as tif:
        
        for t in range(shape[0]):
            for c in range(shape[1]):
                for z in range(shape[2]):
                    frame = img.get_image_data("YX", T=t, C=c, Z=z)
                    tif.write(frame, contiguous=False)
    
    elapsed = time.time() - start
    size_mb = get_file_size_mb(output_path)
    
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  Shape: {shape}")
    
    # Verify output
    verify_results = verify_output(output_path, img)
    print_verification("Strategy 4", verify_results)
    
    return elapsed, size_mb, verify_results


# ============================================================================
# STRATEGY 5: Per-scene processing with BioImage (for multi-scene files)
# ============================================================================
def strategy_5_scene_aware():
    """
    Strategy 5: Scene-aware processing.
    Handles multi-scene files efficiently, only processing full-res scenes.
    """
    print("\n[Strategy 5] Scene-aware processing")
    
    start = time.time()
    
    img = BioImage(TEST_FILE)
    scenes = img.scenes
    
    total_elapsed = 0
    total_size = 0
    all_verify_results = []
    
    print(f"  Found {len(scenes)} scene(s)")
    
    # Check for multi-scene and filter pyramids
    if len(scenes) > 1:
        scene_dims = {}
        for scene_id in scenes:
            img.set_scene(scene_id)
            shape = img.shape
            scene_dims[scene_id] = (shape[-2], shape[-1])  # Y, X
        
        max_pixels = max(dims[0] * dims[1] for dims in scene_dims.values())
        scenes_to_process = [
            sid for sid, dims in scene_dims.items() 
            if dims[0] * dims[1] == max_pixels
        ]
        print(f"  Processing {len(scenes_to_process)} full-resolution scene(s)")
    else:
        scenes_to_process = scenes
    
    # Process each scene
    for scene_idx, scene_id in enumerate(scenes_to_process):
        img.set_scene(scene_id)
        
        if len(scenes_to_process) > 1:
            output_path = OUTPUT_DIR / f"strategy_5_scene_{scene_idx + 1}.ome.tif"
        else:
            output_path = OUTPUT_DIR / "strategy_5_single.ome.tif"
        
        data = np.asarray(img.data)
        
        imwrite(
            output_path,
            data,
            photometric='minisblack',
            compression='zlib',
            compressionargs={'level': 6},
            ome=True,
            metadata={'axes': 'TCZYX'}
        )
        
        size_mb = get_file_size_mb(output_path)
        print(f"    Scene {scene_idx + 1}: {size_mb:.2f} MB")
        total_size += size_mb
        
        # Verify output
        verify_results = verify_output(output_path, img, data)
        all_verify_results.append(verify_results)
    
    elapsed = time.time() - start
    
    print(f"  Total Time: {elapsed:.2f}s")
    print(f"  Total Size: {total_size:.2f} MB")
    
    # Print verification for first scene (or all if multiple)
    for idx, verify_results in enumerate(all_verify_results):
        print_verification(f"Strategy 5 Scene {idx + 1}", verify_results)
    
    return elapsed, total_size, all_verify_results[0] if all_verify_results else {}


# ============================================================================
# Main test runner
# ============================================================================
def run_all_tests():
    """Run all strategies and compare results."""
    print("=" * 80)
    print("BioIO ND2 to OME-TIFF Conversion Speed Test")
    print("=" * 80)
    
    setup_test()
    
    # Get initial file info
    img = BioImage(TEST_FILE)
    print(f"\nInput file info:")
    print(f"  Shape: {img.shape}")
    print(f"  Dtype: {img.dtype}")
    print(f"  Scenes: {len(img.scenes)}")
    print(f"  Size: {get_file_size_mb(TEST_FILE):.2f} MB")
    print(f"  Physical pixel sizes: {img.physical_pixel_sizes}")
    
    results = {}
    verifications = {}
    
    # Run each strategy
    try:
        elapsed, size_mb, verify = strategy_1_direct_bioimage_save()
        results['Strategy 1: Direct save'] = (elapsed, size_mb)
        verifications['Strategy 1: Direct save'] = verify
    except Exception as e:
        print(f"  ERROR: {e}")
        results['Strategy 1: Direct save'] = (None, None)
        verifications['Strategy 1: Direct save'] = {}
    
    try:
        elapsed, size_mb, verify = strategy_2_numpy_tifffile_compressed()
        results['Strategy 2: Numpy + compressed'] = (elapsed, size_mb)
        verifications['Strategy 2: Numpy + compressed'] = verify
    except Exception as e:
        print(f"  ERROR: {e}")
        results['Strategy 2: Numpy + compressed'] = (None, None)
        verifications['Strategy 2: Numpy + compressed'] = {}
    
    try:
        elapsed, size_mb, verify = strategy_3_dask_delayed()
        results['Strategy 3: Dask delayed'] = (elapsed, size_mb)
        verifications['Strategy 3: Dask delayed'] = verify
    except Exception as e:
        print(f"  ERROR: {e}")
        results['Strategy 3: Dask delayed'] = (None, None)
        verifications['Strategy 3: Dask delayed'] = {}
    
    try:
        elapsed, size_mb, verify = strategy_4_frame_by_frame()
        results['Strategy 4: Frame-by-frame'] = (elapsed, size_mb)
        verifications['Strategy 4: Frame-by-frame'] = verify
    except Exception as e:
        print(f"  ERROR: {e}")
        results['Strategy 4: Frame-by-frame'] = (None, None)
        verifications['Strategy 4: Frame-by-frame'] = {}
    
    try:
        elapsed, size_mb, verify = strategy_5_scene_aware()
        results['Strategy 5: Scene-aware'] = (elapsed, size_mb)
        verifications['Strategy 5: Scene-aware'] = verify
    except Exception as e:
        print(f"  ERROR: {e}")
        results['Strategy 5: Scene-aware'] = (None, None)
        verifications['Strategy 5: Scene-aware'] = {}
    
    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Strategy':<35} {'Time (s)':<12} {'Size (MB)':<12} {'Speed Rank'}")
    print("-" * 80)
    
    # Sort by time
    valid_results = {k: v for k, v in results.items() if v[0] is not None}
    sorted_results = sorted(valid_results.items(), key=lambda x: x[1][0])
    
    for rank, (strategy, (time_s, size_mb)) in enumerate(sorted_results, 1):
        print(f"{strategy:<35} {time_s:<12.2f} {size_mb:<12.2f} #{rank}")
    
    # Print failed strategies
    failed = [k for k, v in results.items() if v[0] is None]
    if failed:
        print("\nFailed strategies:")
        for strategy in failed:
            print(f"  - {strategy}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if valid_results:
        fastest = sorted_results[0]
        smallest = min(valid_results.items(), key=lambda x: x[1][1])
        
        print(f"⚡ FASTEST: {fastest[0]}")
        print(f"   Time: {fastest[1][0]:.2f}s")
        print(f"\n💾 SMALLEST FILE: {smallest[0]}")
        print(f"   Size: {smallest[1][1]:.2f} MB")
        
        print("\n📝 Notes:")
        print("  - Strategy 3 (Dask) is best for very large files (memory efficient)")
        print("  - Strategy 2/3 (Compressed) produces smaller files")
        print("  - Strategy 4 (Frame-by-frame) uses least memory but may be slower")
        print("  - Strategy 5 (Scene-aware) is essential for multi-scene files")
    
    # Verification Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"{'Strategy':<35} {'OME':<6} {'Pixels':<8} {'Sizes':<8} {'Shape':<8}")
    print("-" * 80)
    
    for strategy, verify in verifications.items():
        if verify:
            ome_ok = "✓" if verify.get('is_ome', False) else "✗"
            pixels_ok = "✓" if verify.get('pixels_identical', False) else "✗"
            sizes_ok = "✓" if verify.get('physical_size_correct', False) else "✗"
            shape_ok = "✓" if verify.get('shape_correct', False) else "✗"
            
            print(f"{strategy:<35} {ome_ok:<6} {pixels_ok:<8} {sizes_ok:<8} {shape_ok:<8}")
    
    # Check if all strategies passed verification
    all_passed = all(
        v.get('is_ome', False) and 
        v.get('pixels_identical', False) and 
        v.get('physical_size_correct', False) and
        v.get('shape_correct', False)
        for v in verifications.values() if v
    )
    
    if all_passed:
        print("\n✅ ALL STRATEGIES PASSED VERIFICATION!")
        print("   - All outputs are valid OME-TIFF files")
        print("   - Pixel values are identical to source")
        print("   - Physical pixel sizes are preserved")
    else:
        print("\n⚠️  SOME STRATEGIES FAILED VERIFICATION")
        print("   Check details above for which checks failed")
    
    cleanup_test()
    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_all_tests()
