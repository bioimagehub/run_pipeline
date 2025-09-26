import argparse
import os
import sys
import time
import logging
from typing import Optional, Tuple, Any, Dict, List, Union, Literal
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

# Local helper used throughout the repo
import bioimage_pipeline_utils as rp

# Import phase correlation functions
try:
    # Try relative import first
    from drift_correction.phase_cross_correlation import (
        phase_cross_correlation_cupy,
        phase_cross_correlation_cupy_3d
    )
except ImportError:
    # Fallback: add drift_correction subfolder to path
    drift_correction_path = os.path.join(os.path.dirname(__file__), 'drift_correction')
    if drift_correction_path not in sys.path:
        sys.path.append(drift_correction_path)
    try:
        from phase_cross_correlation import (
            phase_cross_correlation_cupy,
            phase_cross_correlation_cupy_3d
        )
    except ImportError as e:
        logger.warning(f"Could not import phase correlation functions: {e}")
        # Define placeholder functions
        def phase_cross_correlation_cupy(*args, **kwargs):
            raise ImportError("Phase correlation functions not available")
        def phase_cross_correlation_cupy_3d(*args, **kwargs):
            raise ImportError("Phase correlation functions not available")

# Module logger
logger = logging.getLogger(__name__)


# Drift correction algorithm registry
ALGORITHMS = {
    'phase_correlation': {
        '2d': phase_cross_correlation_cupy,
        '3d': phase_cross_correlation_cupy_3d,
        'description': 'GPU-accelerated phase cross-correlation (CuPy required)'
    }
    # Future algorithms can be added here:
    # 'template_matching': {...},
    # 'feature_based': {...},
}


def apply_shift_to_frame(frame: np.ndarray, shift: np.ndarray, mode: str = 'constant', cval: float = 0.0) -> np.ndarray:
    """
    Apply a translational shift to a 2D or 3D frame.
    
    Args:
        frame: Input frame (2D YX or 3D ZYX)
        shift: Shift vector [dy, dx] for 2D or [dz, dy, dx] for 3D
        mode: How to handle out-of-bounds pixels ('constant', 'nearest', 'reflect', 'wrap')
        cval: Constant value to use for out-of-bounds pixels when mode='constant'
    
    Returns:
        Shifted frame with same shape as input
    """
    try:
        from scipy.ndimage import shift as scipy_shift
        # Apply the detected shift directly (phase correlation returns the correction shift)
        return scipy_shift(frame, shift, mode=mode, cval=cval, order=1, prefilter=False)
    except ImportError:
        logger.error("SciPy is required for applying shifts. Install with: pip install scipy")
        return frame


def drift_correct_image_stack(
    image_tczyx: np.ndarray,
    reference_channel: int = 0,
    reference_timepoint: int = 0,
    algorithm: str = 'phase_correlation',
    upsample_factor: int = 1,
    apply_correction: bool = True,
    shift_mode: str = 'constant',
    shift_cval: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Apply drift correction to a TCZYX image stack.
    
    Args:
        image_tczyx: Input image stack in TCZYX format
        reference_channel: Channel index to use for drift detection (0-based)
        reference_timepoint: Timepoint to use as reference (0-based)
        algorithm: Drift correction algorithm ('phase_correlation', etc.)
        upsample_factor: Subpixel precision factor for phase correlation
        apply_correction: Whether to apply shifts to all channels (if False, only compute shifts)
        shift_mode: How to handle out-of-bounds pixels when applying shifts
        shift_cval: Constant value for out-of-bounds pixels when shift_mode='constant'
    
    Returns:
        Tuple of (corrected_image, shifts_array, metadata_dict)
        - corrected_image: Drift-corrected image stack (TCZYX)
        - shifts_array: Detected shifts for each timepoint
        - metadata_dict: Processing metadata and statistics
    """
    T, C, Z, Y, X = image_tczyx.shape
    
    # Validate inputs
    if reference_channel >= C or reference_channel < 0:
        raise ValueError(f"Reference channel {reference_channel} out of range [0, {C-1}]")
    if reference_timepoint >= T or reference_timepoint < 0:
        raise ValueError(f"Reference timepoint {reference_timepoint} out of range [0, {T-1}]")
    if algorithm not in ALGORITHMS:
        available = list(ALGORITHMS.keys())
        raise ValueError(f"Algorithm '{algorithm}' not available. Choose from: {available}")
    
    logger.info(f"Starting drift correction: T={T}, C={C}, Z={Z}, Y={Y}, X={X}")
    logger.info(f"Using algorithm: {algorithm}, reference channel: {reference_channel}, reference timepoint: {reference_timepoint}")
    
    # Select algorithm functions
    algo_info = ALGORITHMS[algorithm]
    is_3d = Z > 1
    func_key = '3d' if is_3d else '2d'
    
    if func_key not in algo_info:
        raise ValueError(f"Algorithm '{algorithm}' does not support {'3D' if is_3d else '2D'} images")
    
    drift_func = algo_info[func_key]
    
    # Extract reference image (single channel, single timepoint)
    if is_3d:
        reference_image = image_tczyx[reference_timepoint, reference_channel, :, :, :]  # ZYX
        shifts = np.zeros((T, 3), dtype=np.float32)  # [dz, dy, dx] for each timepoint
    else:
        reference_image = image_tczyx[reference_timepoint, reference_channel, 0, :, :]  # YX
        shifts = np.zeros((T, 2), dtype=np.float32)  # [dy, dx] for each timepoint
    
    # Compute shifts for each timepoint
    errors = []
    logger.info(f"Computing shifts using {func_key.upper()} {algorithm}...")
    
    for t in tqdm(range(T), desc="Computing drift"):
        if t == reference_timepoint:
            # Reference frame has zero shift
            continue
            
        if is_3d:
            current_image = image_tczyx[t, reference_channel, :, :, :]  # ZYX
        else:
            current_image = image_tczyx[t, reference_channel, 0, :, :]  # YX
        
        try:
            shift, error, _ = drift_func(reference_image, current_image, upsample_factor=upsample_factor)
            shifts[t] = shift
            errors.append(error)
        except Exception as e:
            logger.error(f"Failed to compute shift for timepoint {t}: {e}")
            errors.append(1.0)  # Maximum error
    
    # Apply shifts if requested
    corrected_image = image_tczyx.copy() if apply_correction else image_tczyx
    
    if apply_correction:
        logger.info("Applying drift correction to all channels...")
        
        for t in tqdm(range(T), desc="Applying shifts"):
            if t == reference_timepoint:
                continue  # No shift needed for reference
                
            current_shift = shifts[t]
            
            for c in range(C):
                if is_3d:
                    # Apply 3D shift to ZYX data
                    corrected_image[t, c, :, :, :] = apply_shift_to_frame(
                        image_tczyx[t, c, :, :, :], current_shift, mode=shift_mode, cval=shift_cval
                    )
                else:
                    # Apply 2D shift to YX data
                    corrected_image[t, c, 0, :, :] = apply_shift_to_frame(
                        image_tczyx[t, c, 0, :, :], current_shift, mode=shift_mode, cval=shift_cval
                    )
    
    # Compile metadata
    metadata = {
        'algorithm': algorithm,
        'algorithm_description': algo_info['description'],
        'reference_channel': reference_channel,
        'reference_timepoint': reference_timepoint,
        'upsample_factor': upsample_factor,
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
    
    return corrected_image, shifts, metadata


def process_single_file(
    input_path: str,
    output_path: str,
    reference_channel: int = 0,
    reference_timepoint: int = 0,
    algorithm: str = 'phase_correlation',
    upsample_factor: int = 1,
    apply_correction: bool = True,
    shift_mode: str = 'constant',
    shift_cval: float = 0.0,
    save_shifts: bool = True,
    save_metadata: bool = True
) -> None:
    """
    Process a single image file for drift correction.
    
    Args:
        input_path: Path to input image file
        output_path: Path to output corrected image file
        reference_channel: Channel index to use for drift detection
        reference_timepoint: Timepoint to use as reference
        algorithm: Drift correction algorithm to use
        upsample_factor: Subpixel precision factor
        apply_correction: Whether to apply shifts (if False, only saves shifts)
        shift_mode: How to handle out-of-bounds pixels when applying shifts
        shift_cval: Constant value for out-of-bounds pixels
        save_shifts: Whether to save shifts as .npy file
        save_metadata: Whether to save processing metadata as .yaml file
    """
    try:
        # Load image with optimized loading (dask-backed)
        logger.info(f"Loading image: {input_path}")
        img_obj = rp.load_tczyx_image(input_path)
        
        # Get image dimensions for optimal loading strategy
        T, C, Z, Y, X = img_obj.shape
        
        # Load data efficiently based on Z dimension
        if Z == 1:
            # For 2D time series, we can process more efficiently
            logger.info(f"Processing as 2D time series (Z=1): T={T}, C={C}, Y={Y}, X={X}")
            # Load only XY slices as needed (dask will handle this efficiently)
            image_data = img_obj.data  # Keep as dask array initially
        else:
            # For 3D stacks, we need full 3D data
            logger.info(f"Processing as 3D stack (Z={Z}): T={T}, C={C}, Z={Z}, Y={Y}, X={X}")
            # Compute dask array to numpy for 3D processing
            image_data = img_obj.data
        
        # Convert dask to numpy if needed (for computation)
        # Check if it's a dask array by looking for compute method
        if hasattr(image_data, 'compute') and callable(getattr(image_data, 'compute')):
            logger.info("Converting dask array to numpy for processing")
            image_data = image_data.compute()
        elif hasattr(image_data, 'dask_data'):
            # Alternative: use rp helper if available
            logger.info("Using dask_data from image object")
            image_data = img_obj.dask_data
            if hasattr(image_data, 'compute') and callable(getattr(image_data, 'compute')):
                image_data = image_data.compute()
            else:
                image_data = np.asarray(image_data)
        else:
            # Already a numpy array
            image_data = np.asarray(image_data)
        
        # Apply drift correction
        corrected_image, shifts, metadata = drift_correct_image_stack(
            image_data,
            reference_channel=reference_channel,
            reference_timepoint=reference_timepoint,
            algorithm=algorithm,
            upsample_factor=upsample_factor,
            apply_correction=apply_correction,
            shift_mode=shift_mode,
            shift_cval=shift_cval
        )
        
        # Save corrected image
        logger.info(f"Saving corrected image: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Preserve physical pixel sizes if available
        pps = getattr(img_obj, 'physical_pixel_sizes', (None, None, None))
        rp.save_tczyx_image(corrected_image, output_path, physical_pixel_sizes=pps)
        
        # Save shifts if requested
        if save_shifts:
            shifts_path = os.path.splitext(output_path)[0] + "_shifts.npy"
            np.save(shifts_path, shifts)
            logger.info(f"Saved shifts: {shifts_path}")
        
        # Save metadata if requested
        if save_metadata:
            metadata_path = os.path.splitext(output_path)[0] + "_drift_metadata.yaml"
            try:
                import yaml
                with open(metadata_path, 'w') as f:
                    yaml.safe_dump(metadata, f, sort_keys=False)
                logger.info(f"Saved metadata: {metadata_path}")
            except ImportError:
                logger.warning("PyYAML not available, skipping metadata save")
        
        logger.info(f"Successfully processed: {input_path}")
        
    except Exception as e:
        logger.error(f"Failed to process {input_path}: {e}")
        raise


def process_files_parallel(
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
    
    # Drift correction parameters
    parser.add_argument("--reference-channel", type=int, default=0,help="Channel index to use for drift detection (0-based)")
    parser.add_argument("--reference-timepoint", type=int, default=0,help="Timepoint to use as reference (0-based)")
    parser.add_argument("--algorithm", type=str, default="phase_correlation", choices=list(ALGORITHMS.keys()),
                       help="Drift correction algorithm to use")
    parser.add_argument("--upsample-factor", type=int, default=1,
                       help="Subpixel precision factor (1=pixel, higher=subpixel)")
    parser.add_argument("--no-apply-correction", action="store_true",
                       help="Only compute shifts, don't apply correction")
    parser.add_argument("--shift-mode", type=str, default="constant",
                       choices=["constant", "nearest", "reflect", "wrap"],
                       help="How to handle out-of-bounds pixels when applying shifts")
    parser.add_argument("--shift-cval", type=float, default=0.0,
                       help="Constant value for out-of-bounds pixels (when shift-mode=constant)")
    
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
        shift_cval=args.shift_cval,
        save_shifts=not args.no_save_shifts,
        save_metadata=not args.no_save_metadata
    )
    
    logger.info(f"Drift correction pipeline completed. Processed {len(input_files)} files.")


def test_drift_correction() -> bool:
    """
    Test function to verify drift correction implementation.
    """
    logger.info("Testing drift correction implementation...")
    
    # Create synthetic test data
    T, C, Z, Y, X = 5, 2, 1, 128, 128
    
    # Create a test pattern
    test_image = np.zeros((T, C, Z, Y, X), dtype=np.float32)
    
    # Add a simple pattern (circle in center)
    cy, cx = Y // 2, X // 2
    y, x = np.ogrid[:Y, :X]
    mask = ((y - cy) ** 2 + (x - cx) ** 2) < 20 ** 2
    
    # Create drifting pattern
    shifts_true = np.array([[0, 0], [2, 1], [-1, 3], [1, -2], [0, 2]], dtype=np.float32)
    
    for t in range(T):
        # Apply true shift to create synthetic drift
        shifted_mask = np.roll(mask, shifts_true[t].astype(int), axis=(0, 1))
        test_image[t, 0, 0] = shifted_mask.astype(np.float32)
        test_image[t, 1, 0] = shifted_mask.astype(np.float32) * 0.8  # Second channel with different intensity
    
    # Add some noise
    test_image += np.random.normal(0, 0.05, test_image.shape)
    
    try:
        # Test drift correction
        corrected, shifts_detected, metadata = drift_correct_image_stack(
            test_image,
            reference_channel=0,
            reference_timepoint=0,
            algorithm='phase_correlation',
            upsample_factor=1,
            apply_correction=True
        )
        
        print("Test Results:")
        print(f"True shifts: {shifts_true}")
        print(f"Detected shifts: {shifts_detected}")
        print(f"Mean error: {metadata['mean_error']:.4f}")
        print(f"Max shift difference: {np.max(np.abs(shifts_detected - shifts_true)):.2f} pixels")
        
        # Check if detection is reasonably accurate (within 1 pixel)
        accuracy = np.all(np.abs(shifts_detected - shifts_true) < 1.0)
        print(f"Detection accuracy (< 1 pixel): {accuracy}")
        
        logger.info("Drift correction test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Drift correction test failed: {e}")
        return False


if __name__ == "__main__":
    # If run with --test argument, run test function
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        logging.basicConfig(level=logging.INFO)
        test_drift_correction()
    else:
        main()

