"""
Drift Correction Entry Point - Translation-Only Registration

⚠️ TRANSLATION ONLY: This module only corrects for XYZ shifts.
No rotation, scaling, or non-linear transformations are applied.

MIT License
Copyright (c) 2024 BIPHUB - Bioimage Informatics Hub, University of Oslo
"""

import logging
from typing import Literal, Optional, List, Union, Dict, Any, Callable
import numpy as np

from bioio import BioImage
import bioimage_pipeline_utils as rp
from progress_manager import process_folder_unified

logger = logging.getLogger(__name__)


def process_single_file_drift_correction(
    input_path: str,
    output_path: str,
    *,
    progress_callback: Optional[Callable[[int], None]] = None,
    method: str = "phase_cross_correlation",
    reference_channel: int = 0,
    reference: Literal["first", "previous", "median"] = "first",
    no_gpu: bool = False,
    crop_fraction: float = 1.0,
    upsample_factor: int = 10,
    max_shift: float = 50.0,
    no_save_tmats: bool = False
) -> Dict[str, Any]:
    """
    Process a single file for drift correction.
    
    This function matches the FileProcessor protocol for use with progress_manager.
    
    Args:
        input_path: Path to input image file
        output_path: Path to save corrected image
        progress_callback: Optional callback for progress updates (timepoints processed)
        method: Registration method ('phase_cross_correlation' or 'stackreg')
        reference_channel: Channel index to use for registration (0-based)
        reference: Reference frame strategy ('first', 'previous', 'median')
        no_gpu: Disable GPU acceleration
        crop_fraction: Fraction of image to use for registration (0.0-1.0)
        upsample_factor: Subpixel accuracy for phase_cross_correlation
        max_shift: Maximum expected shift in pixels
        no_save_tmats: Do not save transformation matrices
        
    Returns:
        Dict with 'success' boolean and optional 'error' string
    """
    try:
        # Load image
        img = rp.load_tczyx_image(input_path)
        
        # Load the appropriate registration function
        if method == "phase_cross_correlation":
            from drift_correction_utils.phase_cross_correlation import register_image_xy
        elif method == "stackreg":
            from drift_correction_utils.translation_pystackreg import register_image_xy
        else:
            return {'success': False, 'error': f"Unknown method: {method}"}
        
        # Register image with progress callback
        registered_img, tmats = register_image_xy(
            img,
            channel=reference_channel,
            show_progress=False,  # We use progress_callback instead
            no_gpu=no_gpu,
            reference=reference,
            crop_fraction=crop_fraction,
            upsample_factor=upsample_factor,
            max_shift=max_shift,
            progress_callback=progress_callback
        )
        
        # Save corrected image
        registered_img.save(output_path)
        
        # Save transformation matrices if requested
        if not no_save_tmats:
            tmats_file = output_path.replace('.tif', '_tmats.npy')
            np.save(tmats_file, tmats)
            logger.debug(f"Saved transformation matrices to: {tmats_file}")
        
        return {'success': True}
        
    except Exception as e:
        logger.error(f"Failed to process {input_path}: {e}")
        return {'success': False, 'error': str(e)}


def estimate_timepoints_from_file(file_path: str) -> int:
    """
    Estimate total processing steps for progress tracking.
    
    Total = T (detection phase) + T*C (application phase for all channels)
    
    Args:
        file_path: Path to image file
        
    Returns:
        Total number of processing steps (T + T*C)
    """
    try:
        img = rp.load_tczyx_image(file_path)
        T = img.dims.T
        C = img.dims.C
        # Detection phase processes T frames, application phase processes T*C frames
        return T + (T * C)
    except Exception as e:
        logger.warning(f"Failed to estimate timepoints for {file_path}: {e}")
        return 1


def process_files(args):
    """
    Process multiple files with drift correction using parallel or sequential processing.
    
    Args:
        args: Argparse namespace with configuration parameters
    """
    import os
    from pathlib import Path
    import re
    
    # Determine if recursive search is requested
    search_subfolders = '**' in args.input_search_pattern
    
    # Expand glob pattern using standardized helper function
    input_files = rp.get_files_to_process2(args.input_search_pattern, search_subfolders=search_subfolders)
    if not input_files:
        logger.error(f"No files found matching pattern: {args.input_search_pattern}")
        exit(1)
    
    logger.info(f"Found {len(input_files)} file(s) to process")
    
    # Determine base_folder for path collapsing
    # If pattern contains '**', use the part before '**' as base
    # Otherwise, use the parent directory of the pattern
    if '**' in args.input_search_pattern:
        # Extract base path before '**'
        base_folder = args.input_search_pattern.split('**')[0].rstrip('/\\')
        if not base_folder:  # If pattern starts with '**', use current directory
            base_folder = os.getcwd()
        # Normalize path separators and resolve to absolute path
        base_folder = os.path.abspath(base_folder)
        logger.info(f"Using base folder for path collapsing: {base_folder}")
        logger.info(f"Subfolders after '**' will be collapsed with delimiter '{args.collapse_delimiter}'")
    else:
        # For non-recursive patterns, use parent of first file
        base_folder = str(Path(input_files[0]).parent)
    
    # Create output folder
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output folder: {output_folder}")
    
    # Define custom output path function to use the suffix from args
    def create_output_path(input_path: str, base_folder: str, output_folder: str, collapse_delimiter: str) -> str:
        """Custom output path with user-specified suffix and collapsed folder structure."""
        # Use collapse_filename to flatten the folder structure
        collapsed = rp.collapse_filename(input_path, base_folder, collapse_delimiter)
        # Remove original extension and add suffix + .tif
        base_name = os.path.splitext(collapsed)[0]
        filename = base_name + args.output_suffix + '.tif'
        return os.path.join(output_folder, filename)
    
    # Use progress_manager for unified parallel/sequential processing
    results = process_folder_unified(
        input_files=input_files,
        output_folder=str(output_folder),
        base_folder=base_folder,
        file_processor=process_single_file_drift_correction,
        collapse_delimiter=args.collapse_delimiter,
        output_extension="",  # We handle extension in create_output_path
        parallel=not args.no_parallel,
        n_jobs=None,  # Auto-detect number of workers
        use_processes=False,  # Use threads for better memory sharing
        estimate_timepoints_func=estimate_timepoints_from_file,
        create_output_path_func=create_output_path,
        # Pass processing parameters
        method=args.method,
        reference_channel=args.reference_channel,
        reference=args.reference,
        no_gpu=args.no_gpu,
        crop_fraction=args.crop_fraction,
        upsample_factor=args.upsample_factor,
        max_shift=args.max_shift,
        no_save_tmats=args.no_save_tmats
    )
    
    # Summary
    successful = sum(1 for r in results if r.get('success', True))
    failed = len(results) - successful
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing complete! {successful}/{len(input_files)} file(s) successful")
    if failed > 0:
        logger.warning(f"{failed} file(s) failed")
        for r in results:
            if not r.get('success', True):
                logger.error(f"  Failed: {r['input_path']} - {r.get('error', 'Unknown error')}")
    logger.info(f"{'='*60}")



if __name__ == "__main__":
    import argparse
    import os
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        description="Apply translation-only drift correction to TCZYX images"
    )
    parser.add_argument(
        "--input-search-pattern",
        required=True,
        help="Input file pattern (supports wildcards, e.g., 'data/*.tif' or 'data/**/*.tif' for recursive search). Use '**' to search subfolders recursively."
    )
    parser.add_argument(
        "--output-folder",
        required=True,
        help="Output folder for corrected images"
    )
    parser.add_argument(
        "--output-suffix",
        default="_drift_corrected",
        help="Suffix to add to output filenames (default: _drift_corrected). Example: '_cor' or '_phase_cor'"
    )
    parser.add_argument(
        "--collapse-delimiter",
        default="__",
        help="Delimiter for collapsing subfolder structure in output filenames when using '**' (default: __). Example: 'folder1/folder2/image.tif' becomes 'folder1__folder2__image.tif'"
    )
    parser.add_argument(
        "--method",
        default="phase_cross_correlation",
        choices=["phase_cross_correlation", "stackreg"],
        help="Registration method (default: phase_cross_correlation)"
    )

    parser.add_argument(
        "--reference-channel",
        type=int,
        default=0,
        help="Channel index to use for registration (0-based, default: 0)"
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="first",
        choices=["first", "previous", "median"],
        help="Reference frame for registration (default: first). 'first' = register all to first frame, 'previous' = register each to previous frame, 'median' = register to median projection"
    )
    parser.add_argument(
        "--no-save-tmats",
        action="store_true",
        help="Do not save transformation matrices (default: False)"
    )
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing"
    )
    parser.add_argument(
        "--crop-fraction",
        type=float,
        default=1.0,
        help="Fraction of image to use for registration (0.0-1.0, default: 1.0). Values < 1.0 crop edges to speed up registration (e.g., 0.8 uses center 80%)"
    )
    
    parser.add_argument(
        "--upsample-factor",
        type=int,
        default=10,
        help="Subpixel accuracy for phase_cross_correlation (10 = 0.1 pixel, default: 10)"
    )
    parser.add_argument(
        "--max-shift",
        type=float,
        default=50.0,
        help="Maximum expected shift in pixels. Warning issued if exceeded (default: 50.0)"
    )
    # parser.add_argument(
    #     "--bandpass-low-sigma",
    #     type=float,
    #     default=None,
    #     help="Lower sigma for DoG bandpass filter (suppresses structures smaller than this, e.g., 20 for vesicles). Must be used with --bandpass-high-sigma"
    # )
    # parser.add_argument(
    #     "--bandpass-high-sigma",
    #     type=float,
    #     default=None,
    #     help="Upper sigma for DoG bandpass filter (preserves structures larger than this, e.g., 100 for cells). Must be used with --bandpass-low-sigma"
    # )


    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    process_files(args)
    
    
