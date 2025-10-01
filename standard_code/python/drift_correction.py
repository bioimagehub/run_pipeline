import argparse
import os
import sys
import time
import logging
from typing import Optional, Tuple, Any, Dict, List, Union, Literal
from pathlib import Path

import numpy as np
import dask.array as da
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# Universal import fix - add parent directories to sys.path
from pathlib import Path

# Add standard_code directory to path  
current_dir = Path(__file__).parent
standard_code_dir = current_dir
project_root = standard_code_dir.parent
for path in [str(project_root), str(standard_code_dir)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Local helper used throughout the repo
import bioimage_pipeline_utils as rp
from drift_correction.drift_correct_utils import apply_shifts_to_tczyx_stack
from drift_correction.phase_cross_correlation import phase_cross_correlation

# Additional algorithm imports
#TODO check that all imports are functions that accept 2D or 3D numpy arrays and return shifts
# from drift_correction.optical_flow import optical_flow
# from drift_correction.mutual_information import mutual_information
# from drift_correction.normalized_cross_correlation import normalized_cross_correlation
# from drift_correction.sum_squared_differences import sum_squared_differences
# from drift_correction.enhanced_optical_flow import enhanced_optical_flow
# from drift_correction.template_matching import template_matching
# from drift_correction.gradient_descent import gradient_descent
# from drift_correction.fourier_shift import fourier_shift
# from drift_correction.feature_based import feature_based
# from drift_correction.demons import demons
# from drift_correction.variational import variational
from drift_correction.mock_correction import mock_drift_correction


# Set up logging before imports
logger = logging.getLogger(__name__)

# Drift correction algorithm registry
ALGORITHMS = {
    'phase_correlation': {
        '2d': phase_cross_correlation, # resolves both 2d and 3d
        '3d': phase_cross_correlation, # resolves both 2d and 3d
        'description': 'GPU-accelerated phase cross-correlation (CuPy required)'
    },
    # 'optical_flow': {
    #     '2d': optical_flow,
    #     '3d': None,  # Not implemented for 3D
    #     'description': 'OpenCV optical flow-based drift correction with synthetic data optimization'
    # },
    # 'mutual_information': {
    #     '2d': mutual_information,
    #     '3d': None,
    #     'description': 'Mutual information-based drift correction with multi-modal optimization'
    # },
    # 'normalized_cross_correlation': {
    #     '2d': normalized_cross_correlation,
    #     '3d': None,
    #     'description': 'Normalized cross-correlation with template matching and multi-scale analysis'
    # },
    # 'sum_squared_differences': {
    #     '2d': sum_squared_differences,
    #     '3d': None,
    #     'description': 'Sum of squared differences with hierarchical search and multiple variants'
    # },
    # 'enhanced_optical_flow': {
    #     '2d': enhanced_optical_flow,
    #     '3d': None,
    #     'description': 'Enhanced optical flow methods including TV-L1 and Farneback with robust preprocessing'
    # },
    # 'template_matching': {
    #     '2d': template_matching,
    #     '3d': None,
    #     'description': 'Template matching with multiple OpenCV methods and multi-scale approaches'
    # },
    # 'gradient_descent': {
    #     '2d': gradient_descent,
    #     '3d': None,
    #     'description': 'Gradient descent optimization for drift correction with various cost functions'
    # },
    # 'fourier_shift': {
    #     '2d': fourier_shift,
    #     '3d': None,
    #     'description': 'Fourier domain shift estimation with GPU acceleration and windowing'
    # },
    # 'feature_based': {
    #     '2d': feature_based,
    #     '3d': None,
    #     'description': 'Feature-based matching using SIFT, ORB, AKAZE and other feature detectors'
    # },
    # 'demons': {
    #     '2d': demons,
    #     '3d': None,
    #     'description': 'Demons algorithm for non-rigid registration with translation extraction'
    # },
    # 'variational': {
    #     '2d': variational,
    #     '3d': None,
    #     'description': 'Variational optical flow methods including Horn-Schunck and TV-L1 with regularization'
    # },
    'mock': {
        '2d': mock_drift_correction,
        '3d': mock_drift_correction,
        'description': 'Mock drift correction that returns the input image unchanged'
    }

}

def process_file(
    image_path: str,
    output_path: Optional[str] = None,
    reference_channel: int = 0,
    reference_frame: str = 'first',
    algorithm: str = 'phase_correlation',
    upsample_factor: int = 1,
    apply_correction: bool = True,
    shift_mode: str = 'constant',
    gaussian_sigma: float = 0.0,
    use_gpu: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply drift correction to a TCZYX image stack.
    
    Args:
        image_path: path to image file (will be loaded internally)
        reference_channel: Channel index to use for drift detection (0-based)
        reference_frame: Frame to use as reference ('first', 'previous', 'mean', 'mean10')
        algorithm: Drift correction algorithm ('phase_correlation')
        upsample_factor: Subpixel precision factor for phase correlation
        apply_correction: Whether to apply shifts to all channels (if False, only compute shifts)
        shift_mode: How to handle out-of-bounds pixels when applying shifts (default: 'constant')
        gaussian_sigma: Gaussian smoothing for preprocessing before drift detection (use -1 to disable, default: -1)
        no_gpu: Do not use GPU acceleration (default: False)
    
    Returns:
        Tuple of (corrected_image, shifts_array, metadata_dict)
        - corrected_image: Drift-corrected image stack (TCZYX)
        - shifts_array: Detected shifts for each timepoint
        - metadata_dict: Processing metadata and statistics
    """
    
    # Validate inputs
    if reference_frame not in ['first', 'previous', 'mean', 'mean10']:
        raise ValueError(f"Reference frame '{reference_frame}' not recognized. Choose from: 'first', 'previous', 'mean', 'mean10'.")
    if algorithm not in ALGORITHMS:
        available = list(ALGORITHMS.keys())
        raise ValueError(f"Algorithm '{algorithm}' not available. Choose from: {available}")

    # Load image
    img = rp.load_tczyx_image(image_path)
    T, C, Z, Y, X = img.shape

    # validate image
    if reference_channel >= C or reference_channel < 0:
        raise ValueError(f"Reference channel {reference_channel} out of range [0, {C-1}]")

    logger.info(f"Starting drift correction: T={T}, C={C}, Z={Z}, Y={Y}, X={X}")
    logger.info(f"Using algorithm: {algorithm}, reference channel: {reference_channel}, reference frame: {reference_frame}")

    # Select algorithm functions
    algo_info = ALGORITHMS[algorithm]
    
    # Resolve 2D vs 3D mode and algorithm choice
     
    is_3d = Z > 1
    if is_3d and algo_info['3d'] is None:
        logger.info(f"Algorithm '{algorithm}' does not support 3D images, falling back to 2D mode (looping over Z slice)")
    # Select function key based on 2D/3D mode
    func_key = '3d' if is_3d and algo_info['3d'] else '2d' #used later in loop to choose z loop or not
    drift_func = algo_info[func_key]

    # Preprocess image data once on the entire dataset
    # Extract single channel data in TCZYX format - work with numpy arrays directly
    img_data = img.data  # Full 5D TCZYX numpy array
    img_single_channel = img_data[:, reference_channel, :, :, :]  # Extract single channel: TZYX

    if gaussian_sigma >= 0:
        logger.info(f"Applying Gaussian smoothing with sigma={gaussian_sigma} for preprocessing")
        
        # Apply Gaussian filtering to the numpy array
        img_single_channel = gaussian_filter(
            img_single_channel,
            sigma=(0, 0, gaussian_sigma, gaussian_sigma),  # 4D sigma for TZYX (T=0, Z=0, Y=sigma, X=sigma)
            mode='nearest'
        )

    # Extract reference image (single channel, single timepoint)
    # convert to 3D array (Z, Y, X)
    reference_timepoint = None  # For methods with fixed reference
    if reference_frame == 'first':
        reference_3dstack = img_single_channel[0]  # Use first timepoint as reference: ZYX
        reference_timepoint = 0
    elif reference_frame == 'last':
        reference_3dstack = img_single_channel[T-1]  # Use last timepoint as reference: ZYX
        reference_timepoint = T - 1
    elif reference_frame == 'mean':
        reference_3dstack = np.mean(img_single_channel, axis=0, keepdims=False).astype(img.dtype)
    elif reference_frame == 'mean10':
        n = min(10, img.shape[0]) # reduce n if less than 10 frames
        reference_3dstack = np.mean(img_single_channel[:n], axis=0, keepdims=False).astype(img.dtype) # mean of first 10 or n frames
    elif reference_frame == 'previous':
        # Special case: use previous frame as reference (iterative processing)
        # We'll handle this in the loop below
        reference_3dstack = None  # Will be set iteratively
    else:
        raise ValueError(f"Invalid reference_frame: {reference_frame}")


    # Compute shifts for each timepoint
    errors = []
    shifts = np.zeros((T, 3), dtype=np.float32)  # Store shifts as (dz, dy, dx) for each timepoint

    logger.info(f"Computing shifts using {func_key.upper()} {algorithm}...")
    
    for t in tqdm(range(T), desc="Computing drift"):
        if t == reference_timepoint:
            # Reference frame has zero shift
            shifts[t] = [0, 0, 0] 
            continue
        if reference_3dstack is None: # only for 'previous' case
            if t == 0:
                # shifts can be set to zero for first frame
                shifts[t] = [0, 0, 0] 
                continue
            else:
                reference_3dstack = img_single_channel[t-1]  # Use previous timepoint as reference: ZYX

        current_image = img_single_channel[t]  # ZYX

        # Compute shift using selected algorithm
        # For 2D algorithms on 3D data, we should loop over Z slices

        if func_key == '2d': # 2d image or 3d image with 2d algorithm
            if Z == 1:
                # Single Z slice - phase correlation returns [dy, dx] for 2D case
                shift, error, snr = drift_func(reference_3dstack[0], current_image[0], upsample_factor=upsample_factor)
                if len(shift) == 2:  # 2D shift [dy, dx]
                    shifts[t] = [0, shift[0], shift[1]]  # Convert to [dz, dy, dx]
                else:  # 3D shift already
                    shifts[t] = shift
                errors.append(error)
            else:
                # Multiple Z slices - average shifts across slices or use central slice
                # For now, use central slice to avoid complexity
                z_center = Z // 2
                shift, error, snr = drift_func(reference_3dstack[z_center], current_image[z_center], upsample_factor=upsample_factor)
                if len(shift) == 2:  # 2D shift [dy, dx]
                    shifts[t] = [0, shift[0], shift[1]]  # Convert to [dz, dy, dx]
                else:  # 3D shift already
                    shifts[t] = shift
                errors.append(error)

        elif func_key == '3d': # 3d image with 3d algorithm
            shift, error, snr = drift_func(reference_3dstack, current_image, upsample_factor=upsample_factor)
            shifts[t] = shift
            errors.append(error)
        else:
            raise ValueError(f"Invalid function key: {func_key}")
    # Apply shifts if requested
    
    corrected_image = None
    
    if apply_correction:
        # Convert the original image data to dask array for correction
        corrected_image = da.from_array(img.data)

        logger.info("Applying drift correction to all channels...")
        # use shifts and apply_shifts_to_tczyx_stack function
        corrected_image = apply_shifts_to_tczyx_stack(
            corrected_image,
            shifts,
            mode=shift_mode,
            order=3)

    # Compile metadata
    # TODO: Ensure that this is still valid after update
    metadata = {
        'algorithm': algorithm,
        'algorithm_description': algo_info['description'],
        'reference_channel': reference_channel,
        'reference_timepoint': reference_timepoint,
        'upsample_factor': upsample_factor,
        'gaussian_sigma': gaussian_sigma,
        'use_gpu': use_gpu,
        'image_dimensions': {'T': T, 'C': C, 'Z': Z, 'Y': Y, 'X': X},
        'is_3d': is_3d,
        'shifts_applied': apply_correction,
        'shift_mode': shift_mode if apply_correction else None,
        'mean_error': float(np.mean(errors)) if errors else 0.0,
        'max_error': float(np.max(errors)) if errors else 0.0,
        'max_shift': float(np.max(np.abs(shifts))) if len(shifts) > 0 else 0.0,
        'processing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    logger.info(f"Drift correction completed. Mean error: {metadata['mean_error']:.4f}, Max shift: {metadata['max_shift']:.2f} pixels")
    
    if corrected_image: # 
        # Save corrected image if requested
        if output_path:
            pps = img.physical_pixel_sizes if img.physical_pixel_sizes is not None else (None, None, None)

            rp.save_tczyx_image(corrected_image, output_path, physical_pixel_sizes=pps)
            logger.info(f"Saved corrected image to: {output_path}")
        else:
            logger.warning("Output path not provided, corrected image not saved.")

    return shifts, metadata


def process_folder(
    input_files: List[str],
    output_dir: str,
    base_folder: str,
    collapse_delimiter: str = "__",
    output_extension: str = "",
    n_jobs: int = -1,
    **kwargs
) -> None:
    """
    Process multiple files in parallel.
    
    Args:
        input_files: List of input file paths
        output_dir: Output directory
        base_folder: Base folder for relative path calculation
        collapse_delimiter: Delimiter for collapsing nested folder names
        output_extension: Extension to add to output filenames
        n_jobs: Number of parallel jobs (-1 for all cores)
        **kwargs: Additional arguments passed to process_single_file
    """
    def process_file_wrapper(input_path: str) -> None:
        # Generate output path
        collapsed = rp.collapse_filename(input_path, base_folder, collapse_delimiter)
        output_filename = os.path.splitext(collapsed)[0] + output_extension + ".tif"
        output_path = os.path.join(output_dir, output_filename)
        
        # Process the file
        return process_single_file(input_path, output_path, **kwargs)
    
    # Process files in parallel
    if n_jobs == 1:
        # Sequential processing
        for input_path in tqdm(input_files, desc="Processing files"):
            process_file_wrapper(input_path)
    else:
        # Parallel processing
        logger.info(f"Processing {len(input_files)} files with {n_jobs} parallel jobs")
        Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(process_file_wrapper)(input_path) for input_path in input_files
        )


def try_all_algorithms(test_image_path: str = "E:/Oyvind/BIP-hub-test-data/drift/input/live_cells/1_Meng_with_known_drift.tif", 
                      ground_truth_json: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Test all available drift correction algorithms on a test image and return performance scores.
    
    This function systematically tests every algorithm in the ALGORITHMS registry on the specified
    test image (default: known drift test case) and compares results against ground truth if available.
    
    Args:
        test_image_path: Path to test image file (default uses known drift test case)
        ground_truth_json: Optional path to ground truth shifts JSON file (auto-detected if None)
    
    Returns:
        Dictionary containing results for each algorithm:
        {
            'algorithm_name': {
                'success': bool,
                'score': float,  # drift_correction_score result
                'execution_time': float,
                'detected_shifts': np.ndarray,
                'error': str  # if failed
            }
        }
    """
    from drift_correction.drift_correct_utils import drift_correction_score
    import json
    
    # Load test image
    try:
        img = rp.load_tczyx_image(test_image_path)
        video = img.data  # 5D TCZYX array
        logger.info(f"Loaded test image: {video.shape} (TCZYX)")
    except Exception as e:
        logger.error(f"Failed to load test image {test_image_path}: {e}")
        return {}
    
    # Try to load ground truth shifts
    ground_truth = None
    if ground_truth_json is None:
        # Auto-detect JSON file with same name as image
        json_path = str(Path(test_image_path).with_suffix('.json'))
        if Path(json_path).exists():
            ground_truth_json = json_path
    
    if ground_truth_json and Path(ground_truth_json).exists():
        try:
            with open(ground_truth_json, 'r') as f:
                gt_data = json.load(f)
            if 'shifts' in gt_data:
                ground_truth = np.array(gt_data['shifts'])
                logger.info(f"Loaded ground truth shifts: {ground_truth.shape}")
        except Exception as e:
            logger.warning(f"Failed to load ground truth from {ground_truth_json}: {e}")
    
    results = {}
    
    # Test each algorithm
    for algo_name, algo_info in ALGORITHMS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing algorithm: {algo_name}")
        logger.info(f"Description: {algo_info['description']}")
        logger.info(f"{'='*60}")
        
        result = {
            'success': False,
            'score': None,
            'execution_time': None,
            'detected_shifts': None,
            'error': None
        }
        
        try:
            # Get 2D algorithm function
            algo_func = algo_info.get('2d')
            if algo_func is None:
                result['error'] = "2D implementation not available"
                results[algo_name] = result
                continue
            
            # Prepare video data based on algorithm requirements
            if algo_name in ['phase_correlation', 'optical_flow']:
                # These algorithms expect full 5D TCZYX arrays
                test_video = video  # Full 5D array
                logger.info(f"Testing on full 5D video shape: {test_video.shape}")
            else:
                # Other algorithms expect 3D arrays (T, Y, X)
                test_video = video[0, 0]  # Shape: (T, Z, Y, X) -> (T, Y, X) for 2D
                if test_video.ndim == 4:  # Handle 3D case
                    test_video = test_video[:, 0]  # Take first Z slice
                logger.info(f"Testing on 3D video shape: {test_video.shape}")
            
            # Time the execution
            start_time = time.time()
            
            # Call the algorithm with appropriate parameters
            if algo_name == 'phase_correlation':
                # Phase correlation expects 5D array and specific parameters
                detected_shifts = algo_func(test_video, upsample_factor=1)[1]  # Returns (corrected, shifts, metadata)
            elif algo_name == 'optical_flow':
                # Optical flow expects 5D array but different calling pattern
                detected_shifts = algo_func(
                    test_video,
                    reference_frame='first',
                    channel=0,
                    max_shift_per_frame=25.0
                )
            else:
                # Most other algorithms follow this pattern with 3D arrays
                detected_shifts = algo_func(
                    test_video,
                    reference_frame='first',
                    max_shift_per_frame=25.0
                )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Validate output
            if detected_shifts is None or len(detected_shifts) == 0:
                result['error'] = "Algorithm returned no shifts"
                results[algo_name] = result
                continue
            
            # Calculate score if ground truth is available
            score = None
            if ground_truth is not None:
                try:
                    # Use the image path for drift_correction_score (it may load the file internally)
                    score = drift_correction_score(detected_shifts, test_image_path, ground_truth)
                    logger.info(f"Score vs ground truth: {score:.4f}")
                except Exception as e:
                    logger.warning(f"Failed to calculate score: {e}")
                    score = None
            
            # Success!
            result['success'] = True
            result['score'] = score
            result['execution_time'] = execution_time
            result['detected_shifts'] = detected_shifts
            
            logger.info(f"✓ SUCCESS - Time: {execution_time:.2f}s, Score: {score}")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"✗ FAILED: {e}")
        
        results[algo_name] = result
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY OF ALL ALGORITHMS")
    logger.info(f"{'='*60}")
    
    successful = [(name, res) for name, res in results.items() if res['success']]
    failed = [(name, res) for name, res in results.items() if not res['success']]
    
    logger.info(f"Successful algorithms: {len(successful)}/{len(results)}")
    
    if ground_truth is not None and successful:
        # Rank by score (lower is better for drift_correction_score)
        ranked = sorted(successful, key=lambda x: x[1]['score'] if x[1]['score'] is not None else float('inf'))
        logger.info("\nRanking by performance (lower score = better):")
        for i, (name, res) in enumerate(ranked, 1):
            score_str = f"{res['score']:.4f}" if res['score'] is not None else "N/A"
            logger.info(f"{i:2d}. {name:25s} - Score: {score_str}, Time: {res['execution_time']:.2f}s")
    
    if failed:
        logger.info(f"\nFailed algorithms ({len(failed)}):")
        for name, res in failed:
            logger.info(f"   ✗ {name}: {res['error']}")
    
    return results

def test_drift_correction() -> None:
    """
    Test function to verify drift correction implementation.
    """
   
    print("Testing drift correction implementation...")

    # Test 1: Synthetic test data
    from drift_correction.synthetic_data_generators import create_simple_squares  

    drifted_video, ground_truth, applied_shifts = create_simple_squares()

    tmpdir = os.path.join(os.path.dirname(__file__), "temp_test_data")
    os.makedirs(tmpdir, exist_ok=True)
    test1_image_path = os.path.join(tmpdir, "synthetic_test_drift_correction.tif")

    # Save the drifted video as a test image
    rp.save_tczyx_image(drifted_video, test1_image_path)

    shifts, metadata = process_file(test1_image_path,     
                                           output_path=None,  # No need to save output
                                           reference_channel=0,
                                           reference_frame='first',
                                           algorithm='phase_correlation',
                                           upsample_factor=1,
                                           apply_correction=False,  # Only compute shifts
                                           gaussian_sigma=-1.0)  # No smoothing

    # Validate detected shifts against ground truth
    # Convert applied shifts to expected correction shifts format [dz, dy, dx]
    # Correction shifts are the negative of applied shifts
    gt_correction_3d = np.array([[0, -dy, -dx] for dx, dy in applied_shifts])
    
    if np.allclose(shifts, gt_correction_3d, atol=0.5):  # Allow 0.5 pixel tolerance
        print("✓ DRIFT CORRECTION TEST PASSED: Detected shifts match expected correction shifts!")
        print(f"Perfect accuracy achieved with phase correlation algorithm")
    else:
        print("✗ DRIFT CORRECTION TEST FAILED: Shifts do not match expected values.")
        mae = np.mean(np.abs(shifts - gt_correction_3d))
        print(f"Mean absolute error: {mae:.3f} pixels")
        print(f"Detected: {shifts.tolist()}")
        print(f"Expected: {gt_correction_3d.tolist()}")
    
    
    # Test 2: make a np.roll test from first frame of known image
    
    from drift_correction.synthetic_data_generators import create_drift_image_from_template
    import json
    template_path ="E:/Oyvind/BIP-hub-test-data/drift/input/live_cells/1_Meng_with_known_drift.tif"
    test2_image_path = os.path.join(tmpdir, "rolled_test_stack.tif")
    test2_image_path_shifts = os.path.join(tmpdir, "rolled_test_stack_known_shifts.json")

    
    


    # Create a rolled image stack for testing
    _ = create_drift_image_from_template(input_file=template_path, 
                                                    output_file=test2_image_path)
    applied_shifts = json.load(open(test2_image_path_shifts, 'r'))['generated_shifts']
    # remove z shifts from dz dy dx to just dx dy
    applied_shifts = [(dx, dy) for dz, dy, dx in applied_shifts]


    print(f"Applied shifts for rolled test stack: {applied_shifts}")
    shifts, metadata = process_file(test2_image_path,     
                                       output_path=None,  # No need to save output
                                       reference_channel=0,
                                           reference_frame='first',
                                           algorithm='phase_correlation',
                                           upsample_factor=1,
                                           apply_correction=False,  # Only compute shifts
                                           gaussian_sigma=-1.0)  # No smoothing

    
    
    # Validate detected shifts against ground truth
    # Convert applied shifts to expected correction shifts format [dz, dy, dx]
    # Correction shifts are the negative of applied shifts
    gt_correction_3d = np.array([[0, -dy, -dx] for dx, dy in applied_shifts])
    
    if np.allclose(shifts, gt_correction_3d, atol=0.5):  # Allow 0.5 pixel tolerance
        print("✓ DRIFT CORRECTION TEST PASSED: Detected shifts match expected correction shifts!")
        print(f"Perfect accuracy achieved with phase correlation algorithm")
    else:
        print("✗ DRIFT CORRECTION TEST FAILED: Shifts do not match expected values.")
        mae = np.mean(np.abs(shifts - gt_correction_3d))
        print(f"Mean absolute error: {mae:.3f} pixels")
        print(f"Detected: {shifts.tolist()}")
        print(f"Expected: {gt_correction_3d.tolist()}")
    

    # Delete temporary test data
    try:
        os.remove(test1_image_path)
        os.remove(test2_image_path)
        os.remove(test2_image_path_shifts)

        os.rmdir(tmpdir)
    except Exception as e:
        print(f"Warning: Failed to clean up temporary files: {e}")  

    


def main() -> None:
    """
    Main entry point for drift correction pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Apply drift correction to image time series using various algorithms."
    )
    
    # Input/output arguments
    parser.add_argument("--input-search-pattern", type=str,help="Input file pattern or folder path")
    parser.add_argument("--output-folder", type=str,help="Output folder (default: input_folder + '_drift_corrected')")
    parser.add_argument("--search-subfolders", action="store_true",help="Search subfolders recursively")
    parser.add_argument("--collapse-delimiter", type=str, default="__",help="Delimiter for collapsing nested folder names")
    parser.add_argument("--output-file-name-extension", type=str, default="_drift_corrected",help="Extension to add to output filenames")
    
    # general drift correction parameters
    parser.add_argument("--algorithm", type=str, default="phase_correlation", choices=list(ALGORITHMS.keys()),
                       help="Drift correction algorithm to use")

    parser.add_argument("--reference-channel", type=int, default=0,help="Channel index to use for drift detection (0-based)")
    parser.add_argument("--reference-timepoint", type=int, default=0,help="Timepoint to use as reference (0-based)")

    parser.add_argument("--upsample-factor", type=int, default=1,
                       help="Subpixel precision factor (1=pixel, higher=subpixel)")
    parser.add_argument("--no-apply-correction", action="store_true",
                       help="Only compute shifts, don't apply correction")
    parser.add_argument("--shift-mode", type=str, default="constant",
                       choices=["constant", "nearest", "reflect", "wrap"],
                       help="How to handle out-of-bounds pixels when applying shifts")
    parser.add_argument("--shift-cval", type=float, default=0.0,
                       help="Constant value for out-of-bounds pixels (when shift-mode=constant)")
    parser.add_argument("--gaussian-sigma", type=float, default=1.0,
                       help="Gaussian smoothing sigma for preprocessing (use -1 to disable, default: 1.0)")
    parser.add_argument("--no-gpu", action="store_true",
                       help="Disable GPU acceleration for shift application")
    
    # Output options
    parser.add_argument("--no-save-shifts", action="store_true",
                       help="Don't save shift arrays as .npy files")
    parser.add_argument("--no-save-metadata", action="store_true",
                       help="Don't save processing metadata as .yaml files")
    
    # Processing options
    parser.add_argument("--no-parallel", action="store_true",
                       help="Disable parallel processing")
    parser.add_argument("--n-jobs", type=int, default=-1,
                       help="Number of parallel jobs (-1 for all cores)")
    
    # Utility options
    parser.add_argument("--list-algorithms", action="store_true",
                       help="List available drift correction algorithms and exit")
    parser.add_argument("--test-all", action="store_true",
                       help="Test all algorithms on known drift test data and exit")
    parser.add_argument("--test-image", type=str,
                       help="Custom test image path for --test-all (default: known drift test case)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print planned actions but don't process files")
    parser.add_argument("--version", action="store_true",
                       help="Print version and exit")
    
    args = parser.parse_args()
    
    # Handle utility options
    if args.list_algorithms:
        print("Available drift correction algorithms:")
        for name, info in ALGORITHMS.items():
            dims = []
            if '2d' in info: dims.append('2D')
            if '3d' in info: dims.append('3D')
            print(f"  {name}: {info['description']} (supports: {', '.join(dims)})")
        return
    
    if args.version:
        version_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "VERSION")
        try:
            with open(version_file, "r") as f:
                version = f.read().strip()
        except:
            version = "unknown"
        print(f"drift_correction.py version: {version}")
        return
    
    if args.test_all:
        # Setup logging for test output
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        test_image_path = args.test_image if args.test_image else "E:/Oyvind/BIP-hub-test-data/drift/input/live_cells/1_Meng_with_known_drift.tif"
        
        logger.info("Testing all drift correction algorithms...")
        logger.info(f"Test image: {test_image_path}")
        
        results = try_all_algorithms(test_image_path)
        
        # Print final summary
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST RESULTS")
        print("="*80)
        
        successful = {name: res for name, res in results.items() if res['success']}
        failed = {name: res for name, res in results.items() if not res['success']}
        
        print(f"Total algorithms tested: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            # Rank by score if available
            ranked = sorted(successful.items(), 
                          key=lambda x: x[1]['score'] if x[1]['score'] is not None else float('inf'))
            
            print("\nBest performing algorithms (by score):")
            for i, (name, res) in enumerate(ranked[:5], 1):  # Top 5
                score_str = f"{res['score']:.4f}" if res['score'] is not None else "N/A"
                print(f"  {i}. {name:30s} Score: {score_str:>8s} Time: {res['execution_time']:.2f}s")
        
        return
    
    if args.list_algorithms:
        print("Available drift correction algorithms:")
        for name, info in ALGORITHMS.items():
            dims = []
            if '2d' in info: dims.append('2D')
            if '3d' in info: dims.append('3D')
            print(f"  {name}: {info['description']} (supports: {', '.join(dims)})")
        return
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Find input files
    is_glob = any(ch in args.input_search_pattern for ch in "*?[")
    if is_glob:
        input_files = rp.get_files_to_process2(
            args.input_search_pattern, 
            args.search_subfolders
        )
        base_folder = os.path.dirname(args.input_search_pattern) or "."
    else:
        # Check if it's a single file or a directory
        if os.path.isfile(args.input_search_pattern):
            # Single file
            input_files = [args.input_search_pattern]
            base_folder = os.path.dirname(args.input_search_pattern) or "."
        else:
            # Directory - search within it
            pattern = os.path.join(args.input_search_pattern, "**", "*") if args.search_subfolders else os.path.join(args.input_search_pattern, "*")
            input_files = rp.get_files_to_process2(pattern, args.search_subfolders)
            base_folder = args.input_search_pattern

    
    # Determine output directory
    if args.output_folder:
        output_dir = args.output_folder
    else:
        # For glob patterns, use a sensible default
        if is_glob:
            if base_folder == ".":
                output_dir = "drift_corrected"
            else:
                parent_dir = os.path.dirname(base_folder) or "."
                output_dir = os.path.join(parent_dir, "drift_corrected")
        else:
            output_dir = base_folder + "_drift_corrected"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Dry run
    if args.dry_run:
        print(f"[DRY RUN] Would process {len(input_files)} files")
        print(f"[DRY RUN] Output directory: {output_dir}")
        print(f"[DRY RUN] Algorithm: {args.algorithm}")
        for input_path in input_files:
            collapsed = rp.collapse_filename(input_path, base_folder, args.collapse_delimiter)
            output_filename = os.path.splitext(collapsed)[0] + args.output_file_name_extension + ".tif"
            output_path = os.path.join(output_dir, output_filename)
            print(f"[DRY RUN] {input_path} -> {output_path}")
        return
    
    # Process files
    n_jobs = 1 if args.no_parallel else args.n_jobs
    
    process_files_parallel(
        input_files=input_files,
        output_dir=output_dir,
        base_folder=base_folder,
        collapse_delimiter=args.collapse_delimiter,
        output_extension=args.output_file_name_extension,
        n_jobs=n_jobs,
        reference_channel=args.reference_channel,
        reference_timepoint=args.reference_timepoint,
        algorithm=args.algorithm,
        upsample_factor=args.upsample_factor,
        apply_correction=not args.no_apply_correction,
        shift_mode=args.shift_mode,
        gaussian_sigma=args.gaussian_sigma,
        use_gpu=not args.no_gpu,
        save_shifts=not args.no_save_shifts,
        save_metadata=not args.no_save_metadata
    )
    
    logger.info(f"Drift correction pipeline completed. Processed {len(input_files)} files.")

if __name__ == "__main__":
    main()  # Run as script with argument parsing

