"""
Drift Correction Entry Point - Translation-Only Registration

⚠️ TRANSLATION ONLY: This module only corrects for XYZ shifts.
No rotation, scaling, or non-linear transformations are applied.

MIT License
Copyright (c) 2024 BIPHUB - Bioimage Informatics Hub, University of Oslo
"""

import logging
from typing import Literal, Optional, List, Union
import numpy as np

from bioio import BioImage
import bioimage_pipeline_utils as rp

logger = logging.getLogger(__name__)


def drift_correct(
    img: BioImage,
    method: str = "phase_cross_correlation",
    reference: Literal["first", "previous"] = "first",
    use_gpu: bool = False,
    channels_to_register: Optional[List[int]] = None,
    upsample_factor: int = 10,
    max_shift: float = 50.0,
    bandpass_low_sigma: Optional[float] = None,
    bandpass_high_sigma: Optional[float] = None,
) -> np.ndarray:
    """
    Apply TRANSLATION-ONLY drift correction to TCZYX image.
    
    ⚠️ TRANSLATION ONLY: This function only corrects for XYZ shifts.
    No rotation, scaling, or non-linear transformations are applied.
    
    Parameters
    ----------
    img : BioImage
        Input image (will be converted to TCZYX format internally)
    method : str
        Algorithm to use:
        - "phase_cross_correlation" (default, recommended, fast, subpixel)
        - "stackreg_translation" (ImageJ TurboReg translation mode)
    reference : Literal["first", "previous"]
        Reference frame strategy:
        - "first": Register all timepoints to T=0 (most common)
        - "previous": Register each frame to previous frame (sequential)
    use_gpu : bool
        Use GPU acceleration when available (cuCIM for phase_cross_correlation)
    channels_to_register : Optional[List[int]]
        Channels to use for registration. If None, use first channel.
        Registration is computed once and applied to all channels.
    upsample_factor : int
        Subpixel registration accuracy for phase_cross_correlation 
        (10 = 0.1 pixel precision, 100 = 0.01 pixel precision)
    max_shift : float
        Maximum expected shift in pixels. Warning issued if exceeded (default: 50.0)
    bandpass_low_sigma : Optional[float]
        Lower sigma for Difference of Gaussians (DoG) bandpass filter.
        Suppresses structures smaller than this value (e.g., 20 for vesicles).
        Must be specified with bandpass_high_sigma. Default: None (no filtering)
    bandpass_high_sigma : Optional[float]
        Upper sigma for Difference of Gaussians (DoG) bandpass filter.
        Preserves structures larger than this value (e.g., 100 for cells).
        Must be specified with bandpass_low_sigma. Default: None (no filtering)
    
    Returns
    -------
    np.ndarray
        Drift-corrected image data in TCZYX format
        Can be saved directly with rp.save_tczyx_image()
        
    Raises
    ------
    ValueError
        If reference is not "first" or "previous"
    ValueError
        If method is not supported
        
    Examples
    --------
    >>> img = rp.load_tczyx_image("timelapse.tif")
    >>> corrected = drift_correct(img, method="phase_cross_correlation")
    >>> rp.save_tczyx_image(corrected, "corrected.tif")
    
    >>> # With DoG bandpass filter to suppress vesicles and preserve cells
    >>> corrected = drift_correct(img, bandpass_low_sigma=20, bandpass_high_sigma=100)
    >>> rp.save_tczyx_image(corrected, "corrected.tif")
    """
    # Validate inputs
    if reference not in ["first", "previous"]:
        raise ValueError(
            f"Invalid reference '{reference}'. Entry point only supports 'first' or 'previous'. "
            f"For advanced reference strategies ('mean', 'median', specific timepoint), "
            f"use algorithm-specific functions in drift_correction/ folder"
        )
    
    if method not in ["phase_cross_correlation", "stackreg_translation"]:
        raise ValueError(
            f"Unknown method '{method}'. Only translation-only methods are supported: "
            "'phase_cross_correlation', 'stackreg_translation'"
        )
    
    # Get data shape
    T, C, Z, Y, X = img.shape
    
    if T == 1:
        logger.warning("Only 1 timepoint, no drift correction needed. Returning original image.")
        return img
    
    logger.info(f"Applying translation-only drift correction using {method}")
    logger.info(f"Image shape: T={T}, C={C}, Z={Z}, Y={Y}, X={X}")
    logger.info(f"Reference strategy: {reference}")
    
    # Determine which channels to use for registration
    if channels_to_register is None:
        channels_to_register = [0]
        logger.info(f"Using channel 0 for registration")
    else:
        logger.info(f"Using channels {channels_to_register} for registration")
    
    # Extract registration channel(s) and average if multiple
    reg_data = img.data[:, channels_to_register, :, :, :]  # (T, C_reg, Z, Y, X)
    if len(channels_to_register) > 1:
        reg_data = np.mean(reg_data, axis=1, keepdims=False)  # (T, Z, Y, X)
    else:
        reg_data = reg_data[:, 0, :, :, :]  # (T, Z, Y, X)
    
    # Compute shifts using selected method
    shifts: np.ndarray
    if method == "phase_cross_correlation":
        from standard_code.python.drift_correction_utils.phase_cross_correlation import (
            phase_cross_correlation_cpu,
            phase_cross_correlation_gpu
        )
        
        if use_gpu:
            logger.info("Using GPU-accelerated phase cross-correlation (cuCIM)")
            shifts, _ = phase_cross_correlation_gpu(
                reg_data, 
                reference=reference, 
                upsample_factor=upsample_factor,
                bandpass_low_sigma=bandpass_low_sigma,
                bandpass_high_sigma=bandpass_high_sigma
            )
        else:
            logger.info("Using CPU phase cross-correlation (scikit-image)")
            shifts, _ = phase_cross_correlation_cpu(
                reg_data, 
                reference=reference, 
                upsample_factor=upsample_factor,
                bandpass_low_sigma=bandpass_low_sigma,
                bandpass_high_sigma=bandpass_high_sigma
            )
    
    elif method == "stackreg_translation":
        from standard_code.python.drift_correction_utils.pystackreg import stackreg_register
        
        logger.info("Using StackReg TRANSLATION mode")
        shifts, _ = stackreg_register(
            reg_data, 
            reference=reference,
            bandpass_low_sigma=bandpass_low_sigma,
            bandpass_high_sigma=bandpass_high_sigma
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Log shift statistics
    logger.info(f"Computed shifts for {T} timepoints")
    
    # Handle both 2D (T, 2) and 3D (T, Z, 2) shift arrays
    if shifts.ndim == 3:
        # (T, Z, 2) - flatten to (T*Z, 2) for statistics
        shifts_flat = shifts.reshape(-1, shifts.shape[-1])
        logger.info(f"Mean shift: {np.mean(shifts_flat, axis=0)}")
        logger.info(f"Std shift: {np.std(shifts_flat, axis=0)}")
    else:
        # (T, 2) or (T, 3)
        logger.info(f"Mean shift: {np.mean(shifts, axis=0)}")
        logger.info(f"Std shift: {np.std(shifts, axis=0)}")
    
    logger.info(f"Max shift: {np.max(np.abs(shifts)):.2f} pixels")
    
    if np.max(np.abs(shifts)) > max_shift:
        logger.warning(f"Large drift detected (max={np.max(np.abs(shifts)):.2f} pixels > {max_shift} pixels threshold), check sample stability")
    
    # Apply shifts to all channels
    from scipy.ndimage import shift as scipy_shift
    
    corrected_data = np.zeros_like(img.data)
    
    logger.info("Applying shifts to all channels...")
    for t in range(T):
        for c in range(C):
            for z in range(Z):
                # Get shift for this timepoint and z-slice
                if Z == 1:
                    # 2D case: shifts might be (T, 2), (T, 3), or (T, 1, 2)
                    if shifts.ndim == 3:
                        # Per-slice format from StackReg: (T, Z, 2)
                        shift_yx = shifts[t, z, :]
                    elif shifts.shape[1] == 3:
                        # If shifts include Z dimension: (T, 3) for ZYX
                        shift_yx = shifts[t, 1:]  # Skip Z, use Y and X
                    else:
                        # Already YX format: (T, 2)
                        shift_yx = shifts[t]
                    corrected_data[t, c, z, :, :] = scipy_shift(
                        img.data[t, c, z, :, :],
                        shift=shift_yx,
                        order=1,  # Linear interpolation
                        mode='constant',
                        cval=0
                    )
                else:
                    # 3D case: shifts are (T, 3) for ZYX or (T, Z, 2) for per-slice XY
                    if shifts.ndim == 2 and shifts.shape[1] == 3:
                        # Full 3D shift (ZYX)
                        shift_zyx = shifts[t]
                        corrected_data[t, c, z, :, :] = scipy_shift(
                            img.data[t, c, z, :, :],
                            shift=shift_zyx[1:],  # Use YX components
                            order=1,
                            mode='constant',
                            cval=0
                        )
                    elif shifts.ndim == 3:
                        # Per-slice XY shift (T, Z, 2)
                        shift_xy = shifts[t, z]
                        corrected_data[t, c, z, :, :] = scipy_shift(
                            img.data[t, c, z, :, :],
                            shift=shift_xy,
                            order=1,
                            mode='constant',
                            cval=0
                        )
    
    logger.info("Drift correction complete")
    
    # Return corrected data as numpy array - save_tczyx_image can handle this
    return corrected_data


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
        help="Input file pattern (supports wildcards, e.g., 'data/*.tif')"
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
        "--method",
        default="phase_cross_correlation",
        choices=["phase_cross_correlation", "stackreg_translation"],
        help="Registration method (default: phase_cross_correlation)"
    )
    parser.add_argument(
        "--reference",
        default="first",
        choices=["first", "previous"],
        help="Reference frame strategy (default: first)"
    )
    parser.add_argument(
        "--reference-channel",
        type=int,
        default=0,
        help="Channel index to use for registration (0-based, default: 0)"
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")
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
    parser.add_argument(
        "--bandpass-low-sigma",
        type=float,
        default=None,
        help="Lower sigma for DoG bandpass filter (suppresses structures smaller than this, e.g., 20 for vesicles). Must be used with --bandpass-high-sigma"
    )
    parser.add_argument(
        "--bandpass-high-sigma",
        type=float,
        default=None,
        help="Upper sigma for DoG bandpass filter (preserves structures larger than this, e.g., 100 for cells). Must be used with --bandpass-low-sigma"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Expand glob pattern using standardized helper function
    input_files = rp.get_files_to_process2(args.input_search_pattern, search_subfolders=False)
    if not input_files:
        logger.error(f"No files found matching pattern: {args.input_search_pattern}")
        exit(1)
    
    logger.info(f"Found {len(input_files)} file(s) to process")
    
    # Create output folder
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output folder: {output_folder}")
    
    # Create output folder
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output folder: {output_folder}")
    
    # Process each file
    for i, input_path in enumerate(input_files, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing file {i}/{len(input_files)}: {input_path}")
        logger.info(f"{'='*60}")
        
        # Determine output path
        input_filename = Path(input_path).stem
        output_filename = f"{input_filename}{args.output_suffix}.tif"
        output_path = str(output_folder / output_filename)
        
        try:
            # Load image
            logger.info(f"Loading {input_path}")
            img = rp.load_tczyx_image(input_path)
            
            # Apply drift correction
            corrected = drift_correct(
                img,
                method=args.method,
                reference=args.reference,
                use_gpu=args.gpu,
                channels_to_register=[args.reference_channel],
                upsample_factor=args.upsample_factor,
                max_shift=args.max_shift,
                bandpass_low_sigma=args.bandpass_low_sigma,
                bandpass_high_sigma=args.bandpass_high_sigma
            )
            
            # Save result
            logger.info(f"Saving to {output_path}")
            rp.save_tczyx_image(corrected, str(output_path))
            logger.info(f"✓ Successfully processed {input_path}")
            
        except Exception as e:
            logger.error(f"✗ Failed to process {input_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing complete! {len(input_files)} file(s) processed")
    logger.info(f"{'='*60}")
