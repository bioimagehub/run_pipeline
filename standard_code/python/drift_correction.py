import argparse
import os
import sys
import time
import logging
from typing import Optional, Tuple, Any, Dict, List, Union, Literal, Callable
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
from progress_manager import process_folder_unified

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
from drift_correction._mock_correction import mock_drift_correction


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
    upsample_factor: int = 5,
    shift_mode: str = 'constant',
    gaussian_sigma: float = -1.0,
    use_gpu: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply drift correction to a TCZYX image stack.
    
    Args:
        image_path: path to image file (will be loaded internally)
        output_path: path to save corrected image (if None, only shifts are computed and returned, correction is not applied)
        reference_channel: Channel index to use for drift detection (0-based)
        reference_frame: Frame to use as reference ('first', 'previous', 'mean', 'mean10')
        algorithm: Drift correction algorithm ('phase_correlation')
        upsample_factor: Subpixel precision factor for phase correlation
        shift_mode: How to handle out-of-bounds pixels when applying shifts (default: 'constant')
        gaussian_sigma: Gaussian smoothing for preprocessing before drift detection (use -1 to disable, default: -1)
        use_gpu: Use GPU acceleration (default: True)
    
    Returns:
        Tuple of (shifts_array, metadata_dict)
        - shifts_array: Detected drift for each timepoint [dz, dy, dx] or [dy, dx]
                       For 'previous' mode: cumulative shifts relative to frame 0
                       For other modes: shifts relative to the reference frame
                       Positive values indicate movement in positive axis direction
                       Note: The NEGATIVE of these shifts is applied for correction
        - metadata_dict: Processing metadata and statistics
    """
    
    # Hardcoded internal parameter (not exposed to users)
    return_raw_shifts = False  # Always return cumulative shifts for 'previous' mode
    
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
    
    # Debug: Track shift statistics
    debug_shifts = []
    large_shift_frames = []
    
    # STEP 1: Detect frame-to-frame shifts (t vs t-1 for 'previous' mode, or t vs reference for others)
    for t in tqdm(range(T), desc="Computing drift"):
        if t == reference_timepoint:
            # Reference frame has zero shift
            shifts[t] = [0, 0, 0] 
            debug_shifts.append((t, "reference", [0, 0, 0]))
            continue
        
        # For 'previous' mode, update reference to previous frame each iteration
        if reference_frame == 'previous':
            if t == 0:
                # shifts can be set to zero for first frame
                shifts[t] = [0, 0, 0] 
                debug_shifts.append((t, "first", [0, 0, 0]))
                continue
            # Update reference to previous frame for this iteration
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
            
        # Debug: Log shift details for problematic frames
        shift_magnitude = np.linalg.norm(shifts[t])
        debug_shifts.append((t, f"error={error:.4f}", shifts[t].tolist()))
        if shift_magnitude > 2.0:  # Large shift
            large_shift_frames.append((t, shifts[t].tolist(), error))
    
    # STEP 2: Convert frame-to-frame shifts to cumulative shifts relative to frame 0 (for 'previous' mode)
    if reference_frame == 'previous':
        logger.info("Converting frame-to-frame shifts to cumulative shifts relative to frame 0...")
        
        # At this point, shifts[t] contains CORRECTION shifts from t to t-1
        # Save these raw frame-to-frame shifts before cumulative conversion
        frame_to_frame_corrections = shifts.copy()  # Save raw shifts
        
        # Convert to cumulative shifts
        for t in range(1, T):
            # Convert to detected drift (negate), accumulate, convert back to correction
            detected_drift_this_frame = -frame_to_frame_corrections[t]
            # Use previous cumulative from shifts[t-1] which we're building up
            previous_total_detected_drift = -shifts[t-1]
            total_detected_drift = previous_total_detected_drift + detected_drift_this_frame
            shifts[t] = -total_detected_drift
            
            # Debug: Log the transformation for first few frames
            if t <= 5:
                logger.info(f"Frame {t} cumulative conversion:")
                logger.info(f"  Frame-to-frame correction (t→t-1): {frame_to_frame_corrections[t]}")
                logger.info(f"  Previous cumulative correction (t-1→0): {shifts[t-1]}")
                logger.info(f"  New cumulative correction (t→0): {shifts[t]}")
        
        shifts_to_return = shifts
    else:
        shifts_to_return = shifts
    
    # Debug: Print statistics
    logger.info(f"Shift detection completed. Large shifts (>2px): {len(large_shift_frames)}")
    if large_shift_frames:
        logger.info("Frames with large shifts:")
        for frame, shift, error in large_shift_frames[:5]:  # Show first 5
            logger.info(f"  Frame {frame}: shift={shift}, error={error:.4f}")
    
    # Debug: Print first 10 shifts for inspection
    logger.info("First 10 detected shifts:")
    for t, note, shift in debug_shifts[:10]:
        logger.info(f"  Frame {t:3d} ({note:12s}): {shift}")
    
    # Apply shifts and save if output_path is provided
    corrected_image = None
    apply_correction = output_path is not None
    
    if apply_correction:
        # Convert the original image data to dask array for correction
        corrected_image = da.from_array(img.data)

        logger.info("Applying drift correction to all channels...")
        logger.info(f"Shifts summary - Shape: {shifts.shape}, Max: {np.max(np.abs(shifts)):.3f}, Mean: {np.mean(np.abs(shifts)):.3f}")
        logger.info(f"Non-zero shifts: {np.count_nonzero(shifts)} / {shifts.size} elements")
        
        # Debug: Check if image actually changes
        original_mean = np.mean(img.data.astype(np.float64))
        logger.info(f"Original image mean: {original_mean:.6f}")
        
        # use shifts and apply_shifts_to_tczyx_stack function
        corrected_image = apply_shifts_to_tczyx_stack(
            corrected_image,
            shifts,
            mode=shift_mode,
            order=3)
            
        # Debug: Check if corrected image is different
        if hasattr(corrected_image, 'compute'):
            corrected_mean = np.mean(corrected_image.compute().astype(np.float64))
        else:
            corrected_mean = np.mean(corrected_image.astype(np.float64))
        logger.info(f"Corrected image mean: {corrected_mean:.6f}")
        logger.info(f"Image changed: {abs(corrected_mean - original_mean) > 1e-10}")

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
        'max_shift': float(np.max(np.abs(shifts_to_return))) if len(shifts_to_return) > 0 else 0.0,
        'processing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    logger.info(f"Drift correction completed. Mean error: {metadata['mean_error']:.4f}, Max shift: {metadata['max_shift']:.2f} pixels")
    
    if corrected_image is not None and output_path is not None:
        # Save corrected image
        pps = img.physical_pixel_sizes if img.physical_pixel_sizes is not None else (None, None, None)
        rp.save_tczyx_image(corrected_image, output_path, physical_pixel_sizes=pps)
        logger.info(f"Saved corrected image to: {output_path}")

    # Return the appropriate shifts (raw or cumulative depending on return_raw_shifts parameter)
    return shifts_to_return, metadata


def estimate_timepoints_drift_correction(input_path: str) -> int:
    """
    Estimate number of timepoints in an image for progress tracking.
    
    Args:
        input_path: Path to the image file
        
    Returns:
        Number of timepoints (T dimension)
    """
    try:
        img = rp.load_tczyx_image(input_path)
        return img.shape[0]  # T is first dimension in TCZYX
    except Exception as e:
        logger.warning(f"Failed to estimate timepoints for {input_path}: {e}")
        return 1  # Fallback


def process_file_with_progress(
    input_path: str,
    output_path: str,
    progress_callback: Optional[Callable[[int], None]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Wrapper around process_file that supports progress callbacks.
    
    This adapter allows the existing process_file function to work with
    the unified progress tracking system without modifying its signature.
    
    Args:
        input_path: Path to input image
        output_path: Path to output image
        progress_callback: Function to call when timepoints are processed
        **kwargs: Additional arguments passed to process_file
        
    Returns:
        Dict with 'success' boolean and optional error information
    """
    try:
        # The current process_file doesn't support granular progress updates
        # during processing, so we'll just call it and report completion
        # Future enhancement: modify process_file internals to call progress_callback
        # after each timepoint is processed
        
        shifts, metadata = process_file(
            image_path=input_path,
            output_path=output_path,
            **kwargs
        )
        
        # Report full completion if callback provided
        if progress_callback:
            # Estimate based on metadata if available
            T = metadata.get('image_dimensions', {}).get('T', 1)
            progress_callback(T)
        
        return {
            'success': True,
            'shifts': shifts,
            'metadata': metadata
        }
        
    except Exception as e:
        logger.error(f"Error processing {input_path}: {e}")
        return {
            'success': False,
            'error': str(e)
        }


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
    Process multiple files in parallel using the unified progress tracking system.
    
    Args:
        input_files: List of input file paths
        output_dir: Output directory
        base_folder: Base folder for relative path calculation
        collapse_delimiter: Delimiter for collapsing nested folder names
        output_extension: Extension to add to output filenames (before .tif extension)
        n_jobs: Number of parallel jobs (-1 for all cores)
        **kwargs: Additional arguments passed to process_file
    """
    # Determine parallel settings
    parallel = n_jobs != 1
    n_workers = None if n_jobs == -1 else max(1, n_jobs)
    
    # Custom output path function that replaces extension with .tif instead of appending
    def create_drift_output_path(input_path: str, base_folder: str, output_folder: str, collapse_delimiter: str) -> str:
        """Create output path with .tif extension, replacing original extension."""
        collapsed = rp.collapse_filename(input_path, base_folder, collapse_delimiter)
        base_name = os.path.splitext(collapsed)[0]  # Remove original extension
        output_filename = base_name + output_extension + ".tif"  # Add extension + .tif
        return os.path.join(output_folder, output_filename)
    
    # Use unified processing with progress tracking
    results = process_folder_unified(
        input_files=input_files,
        output_folder=output_dir,
        base_folder=base_folder,
        file_processor=process_file_with_progress,
        collapse_delimiter=collapse_delimiter,
        output_extension=output_extension,  # Pass as-is, custom function handles .tif
        create_output_path_func=create_drift_output_path,  # Use custom path function
        parallel=parallel,
        n_jobs=n_workers,
        use_processes=False,  # Use threads for I/O-bound operations
        estimate_timepoints_func=estimate_timepoints_drift_correction,
        **kwargs
    )
    
    # Log any failures
    failures = [r for r in results if not r.get('success', True)]
    if failures:
        logger.warning(f"{len(failures)} files failed to process:")
        for result in failures:
            logger.warning(f"  - {result['input_path']}: {result.get('error', 'Unknown error')}")


def test_1_synthetic() -> bool:
    """
    Test 1: Synthetic simple squares test
    Tests drift correction on clean synthetic data with known shifts.
    
    Returns:
        True if test passed, False otherwise
    """

    # File paths

    test_folder = "E:/Oyvind/BIP-hub-test-data/drift/input/test_1"
    test1_image_path = os.path.join(test_folder, "test1_synthetic_squares.tif")
    test1_corrected_path = os.path.join(test_folder, "test1_synthetic_squares_corrected.tif")

    print("\n" + "="*80)
    print("TEST 1: SYNTHETIC SIMPLE SQUARES")
    print("="*80)

    from drift_correction.synthetic_data_generators import create_simple_squares  

    # Create synthetic drifted video with known shifts
    drifted_video, ground_truth, applied_shifts = create_simple_squares()
    
    print(f"Created synthetic video: shape={drifted_video.shape}")
    print(f"Applied shifts (dy, dx format): {applied_shifts}")


    # Save the drifted video as a test image
    rp.save_tczyx_image(drifted_video, test1_image_path)
    print(f"Saved synthetic test image to: {test1_image_path}")

    # Run drift correction
    shifts, metadata = process_file(
        test1_image_path,     
        output_path=test1_corrected_path,
        reference_channel=0,
        reference_frame='first',
        algorithm='phase_correlation',
        upsample_factor=1,
        gaussian_sigma=-1.0)

    corrected_img = rp.load_tczyx_image(test1_corrected_path)

    identical = np.allclose(corrected_img.data, ground_truth, atol=1)

    if identical:
        print("✓ TEST 1 PASSED: Corrected image matches ground truth!")
    else:
        print("✗ TEST 1 FAILED: Corrected image does not match ground truth.")


    
    return identical
    
def test_2_template_based() -> bool:
    """
    Test 2: Template-based synthetic drift
    Uses a real image and applies known shifts to create synthetic drift.
    
    Returns:
        True if test passed, False otherwise
    """
    print("\n" + "="*80)
    print("TEST 2: TEMPLATE-BASED SYNTHETIC DRIFT")
    print("="*80)
    
    # File paths
    template_path = "E:/Oyvind/BIP-hub-test-data/drift/input/live_cells/1_Meng_timecrop.tif"


    test_folder = "E:/Oyvind/BIP-hub-test-data/drift/input/test_2"

    test2_image_path = os.path.join(test_folder, os.path.basename(template_path).replace(".tif", "_template_rolled.tif"))
    test2_ground_truth_path = os.path.join(test_folder, os.path.basename(template_path).replace(".tif", "_template_rolled_ground_truth.tif"))
    
    
    test2_corrected_path = os.path.join(test_folder, os.path.basename(template_path).replace(".tif", "_template_rolled_corrected.tif"))
    test2_shifts_path = os.path.join(test_folder, os.path.basename(template_path).replace(".tif", "_template_rolled_known_shifts.json"))

    from drift_correction.synthetic_data_generators import create_drift_image_from_template
    from drift_correction.drift_correct_utils import drift_correction_score
    import json
    
    # Create synthetic drift from template
    if os.path.exists(test2_image_path) and os.path.exists(test2_ground_truth_path):
        print(f"Using existing synthetic test image: {test2_image_path}")
        img_data = rp.load_tczyx_image(test2_image_path).data
        ground_truth_img_data = rp.load_tczyx_image(test2_ground_truth_path).data
        applied_shifts = json.load(open(test2_shifts_path))  # Not loading shifts from file for now
    else:
        print(f"Creating synthetic drift from template: {template_path}")
        img_data, ground_truth_img_data, applied_shifts = create_drift_image_from_template(
            input_file=template_path, 
            output_file=test2_image_path,
                T=10,
        max_shift=15.0
    )
    

    print(f"Using image with {img_data.shape[0]} timepoints with max shift ~15 pixels")
    
    # Find shifts
    shifts, metadata = process_file(
        test2_image_path,     
        output_path=test2_corrected_path,
        reference_channel=0,
        reference_frame='previous', 
        algorithm='phase_correlation',
        upsample_factor=5,  # Use subpixel for better accuracy
        gaussian_sigma=0)  # Smooth real data a bit

    # Load corrected image
    corrected_img = rp.load_tczyx_image(test2_corrected_path)

    drift_correction_score_before = drift_correction_score(image_path=test2_image_path,)
    drift_correction_score_after = drift_correction_score(image_path=test2_corrected_path)  
    print(f"Drift correction score before: {drift_correction_score_before:.4f}")
    print(f"Drift correction score after: {drift_correction_score_after:.4f}")

    improved = drift_correction_score_after > drift_correction_score_before

    return improved

def test_3_real_data() -> bool:
    """
    Test 3: Real data drift correction
    Tests on actual microscopy data without ground truth.
    Evaluates based on drift score improvement.

    Returns:
        True if test passed, False otherwise
    """
    test_folder = "E:/Oyvind/BIP-hub-test-data/drift/input/test_3"

    test_image_path = os.path.join(test_folder, "1_Meng_timecrop.tif")
    output_path = os.path.splitext(test_image_path)[0] + "_corrected.tif"


    print("\n" + "="*80)
    print("TEST 3: REAL DATA DRIFT CORRECTION")
    print("="*80)
    
    from drift_correction.drift_correct_utils import drift_correction_score

    output_path = os.path.splitext(test_image_path)[0] + "_corrected.tif"

    
    # Get score before correction
    print(f"Input file: {test_image_path}")
    score_before = drift_correction_score(image_path=test_image_path)
    print(f"Drift score before correction: {score_before:.4f}")
    
    # Run drift correction
    shifts, metadata = process_file(
        test_image_path,
        output_path=output_path,  # Don't save yet - we want raw shifts first
        reference_channel=0,
        reference_frame='previous',
        algorithm='phase_correlation',
        upsample_factor=5,
        gaussian_sigma=0
        )
            
    # Get score after correction
    score_after = drift_correction_score(image_path=output_path, reference="previous")
    improvement = score_after - score_before

    # Success if improvement is positive
    if improvement > 0:
        print(f"\n✓ TEST 3 PASSED: Drift score improved by {improvement:.4f}")
        return True
    else:
        print(f"\n✗ TEST 3 WARNING: Small or negative improvement ({improvement:+.4f})")
        print(f"\n💡 TIP: Try running with 'first' reference mode instead:")
        print(f"  The frame-to-frame shifts show wild oscillations (max {np.max(np.abs(shifts)):.1f} px)")
        print(f"  This suggests 'previous' mode may be accumulating errors")
        return False


def run_all_tests():
    """
    Run all drift correction tests and report results.
    """
    print("\n" + "="*80)
    print("DRIFT CORRECTION TEST SUITE")
    print("="*80)
    print("\nRunning comprehensive drift correction tests...")
    
    results = {}
    
    # Test 1: Synthetic simple squares
    try:
        print("\n" + "-"*80)
        result1 = test_1_synthetic()
        results['Test 1: Synthetic Squares'] = {'passed': result1, 'error': None}
    except Exception as e:
        print(f"\n✗ TEST 1 ERROR: {e}")
        results['Test 1: Synthetic Squares'] = {'passed': False, 'error': str(e)}
    
    # Test 2: Template-based synthetic drift
    try:
        print("\n" + "-"*80)
        result2 = test_2_template_based()
        results['Test 2: Template-Based'] = {'passed': result2, 'error': None}
    except FileNotFoundError as e:
        print(f"\n⚠ TEST 2 SKIPPED: Template file not found - {e}")
        results['Test 2: Template-Based'] = {'passed': None, 'error': 'File not found'}
    except Exception as e:
        print(f"\n✗ TEST 2 ERROR: {e}")
        results['Test 2: Template-Based'] = {'passed': False, 'error': str(e)}
    
    # Test 3: Real data
    try:
        print("\n" + "-"*80)
        result3 = test_3_real_data()
        results['Test 3: Real Data'] = {'passed': result3, 'error': None}
    except FileNotFoundError as e:
        print(f"\n⚠ TEST 3 SKIPPED: Data file not found - {e}")
        results['Test 3: Real Data'] = {'passed': None, 'error': 'File not found'}
    except Exception as e:
        print(f"\n✗ TEST 3 ERROR: {e}")
        results['Test 3: Real Data'] = {'passed': False, 'error': str(e)}
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, result in results.items():
        if result['passed'] is None:
            status = "⚠ SKIPPED"
        elif result['passed']:
            status = "✓ PASSED"
        else:
            status = "✗ FAILED"
        
        error_msg = f" ({result['error']})" if result['error'] else ""
        print(f"{status:12s} {test_name}{error_msg}")
    
    # Overall result
    passed_count = sum(1 for r in results.values() if r['passed'] is True)
    failed_count = sum(1 for r in results.values() if r['passed'] is False)
    skipped_count = sum(1 for r in results.values() if r['passed'] is None)
    
    print(f"\nResults: {passed_count} passed, {failed_count} failed, {skipped_count} skipped")
    
    if failed_count == 0 and passed_count > 0:
        print("\n🎉 ALL TESTS PASSED!")
    elif failed_count > 0:
        print(f"\n⚠ {failed_count} test(s) failed - review output above")
    
    return failed_count == 0


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
    parser.add_argument("--reference-frame", type=str, default="first", 
                       choices=["first", "previous", "mean", "mean10"],
                       help="Frame to use as reference ('first', 'previous', 'mean', 'mean10')")

    parser.add_argument("--upsample-factor", type=int, default=5,
                       help="Subpixel precision factor (1=pixel, 5=recommended subpixel, higher=more precise)")
    parser.add_argument("--shift-mode", type=str, default="constant",
                       choices=["constant", "nearest", "reflect", "wrap"],
                       help="How to handle out-of-bounds pixels when applying shifts")
    parser.add_argument("--shift-cval", type=float, default=0.0,
                       help="Constant value for out-of-bounds pixels (when shift-mode=constant)")
    parser.add_argument("--gaussian-sigma", type=float, default=-1.0,
                       help="Gaussian smoothing sigma for preprocessing (use -1 to disable, default: -1.0=disabled)")
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
        # Run the comprehensive test suite instead
        print("Running comprehensive drift correction test suite...")
        run_all_tests()
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
    
    process_folder(
        input_files=input_files,
        output_dir=output_dir,
        base_folder=base_folder,
        collapse_delimiter=args.collapse_delimiter,
        output_extension=args.output_file_name_extension,
        n_jobs=n_jobs,
        reference_channel=args.reference_channel,
        reference_frame=args.reference_frame,
        algorithm=args.algorithm,
        upsample_factor=args.upsample_factor,
        shift_mode=args.shift_mode,
        gaussian_sigma=args.gaussian_sigma,
        use_gpu=not args.no_gpu
    )
    
    logger.info(f"Drift correction pipeline completed. Processed {len(input_files)} files.")

if __name__ == "__main__":
    # Run test suite when executed directly (without arguments)
    import sys
    if len(sys.argv) == 1:
        # No arguments provided - run tests
        run_all_tests()
    else:
        # Arguments provided - run main CLI
        main()

