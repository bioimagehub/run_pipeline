"""
PyStackReg Drift Correction - Translation Only

Wrapper for pystackreg (ImageJ TurboReg) using TRANSLATION mode only.

⚠️ TRANSLATION ONLY: Uses StackReg.TRANSLATION mode exclusively.

MIT License
Copyright (c) 2024 BIPHUB - Bioimage Informatics Hub, University of Oslo
"""

import logging
from typing import Tuple, Literal, Optional
import numpy as np
from scipy.ndimage import shift as scipy_shift

logger = logging.getLogger(__name__)


def stackreg_register(
    zyx_stack: np.ndarray,
    reference: Literal["first", "previous", "mean"] = "first",
    bandpass_low_sigma: Optional[float] = None,
    bandpass_high_sigma: Optional[float] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    StackReg TRANSLATION-ONLY registration using pystackreg.
    
    ⚠️ TRANSLATION ONLY: Uses StackReg.TRANSLATION mode exclusively.
    Other transformation modes (rigid_body, affine, etc.) are NOT available
    in this implementation to ensure translation-only constraint.
    
    Translation mode: TX/TY shifts only (2 DOF per slice)
    
    For Z>1: Performs 2D translation registration on each Z-slice independently.
    
    Parameters
    ----------
    zyx_stack : np.ndarray
        Input stack with shape (T, Z, Y, X) or (T, Y, X) for 2D
    reference : Literal["first", "previous", "mean"]
        Reference frame strategy (all natively supported by pystackreg):
        - "first": Register all frames to T=0
        - "previous": Register each frame to previous frame
        - "mean": Register to mean projection (computed by StackReg)
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
    shifts : np.ndarray
        Translation vectors for each timepoint
        - 2D case: Shape (T, 2) for XY shifts
        - 3D case: Shape (T, Z, 2) for per-slice XY shifts
    corrected : Optional[np.ndarray]
        Always None - shifts are returned to be applied by caller
        This ensures consistent behavior across all registration methods
    corrected : np.ndarray
        Drift-corrected stack with same shape as input
        
    Notes
    -----
    - StackReg natively handles "first", "previous", and "mean" references
    - For 3D data (Z>1), registration is applied independently to each Z-slice
    - "mean" reference is computed by StackReg internally for stability
    - pystackreg.StackReg.register_transform_stack() is used with 
      transformation=StackReg.TRANSLATION
    - DoG bandpass filtering is applied BEFORE registration to suppress
      unwanted features (e.g., bright vesicles in cell tracking)
      
    Raises
    ------
    ImportError
        If pystackreg is not installed
    ValueError
        If reference is not supported
    """
    try:
        from pystackreg import StackReg
    except ImportError:
        raise ImportError(
            "pystackreg is not installed. Install with: pip install pystackreg"
        )
    
    T = zyx_stack.shape[0]
    is_2d = zyx_stack.ndim == 3  # (T, Y, X)
    
    logger.info(f"StackReg TRANSLATION mode - {'2D' if is_2d else '3D'} registration")
    logger.info(f"Reference: {reference}")
    
    # Apply DoG bandpass filter if requested
    if bandpass_low_sigma is not None and bandpass_high_sigma is not None:
        from skimage.filters import difference_of_gaussians
        logger.info(f"Applying DoG bandpass filter (low={bandpass_low_sigma}, high={bandpass_high_sigma})")
        filtered_stack = np.zeros_like(zyx_stack, dtype=np.float32)
        for t in range(T):
            if is_2d:
                filtered_stack[t] = difference_of_gaussians(
                    zyx_stack[t].astype(np.float32),
                    bandpass_low_sigma,
                    bandpass_high_sigma
                )
            else:
                # Apply DoG to each Z-slice for 3D data
                for z in range(zyx_stack.shape[1]):
                    filtered_stack[t, z] = difference_of_gaussians(
                        zyx_stack[t, z].astype(np.float32),
                        bandpass_low_sigma,
                        bandpass_high_sigma
                    )
        zyx_stack = filtered_stack
        logger.info("DoG filtering complete")
    elif (bandpass_low_sigma is None) != (bandpass_high_sigma is None):
        logger.warning("Both bandpass_low_sigma and bandpass_high_sigma must be specified for filtering. Skipping DoG filter.")
    
    # Validate reference
    if reference not in ["first", "previous", "mean"]:
        raise ValueError(
            f"Invalid reference '{reference}'. StackReg supports: 'first', 'previous', 'mean'"
        )
    
    # Initialize StackReg with TRANSLATION mode only
    sr = StackReg(StackReg.TRANSLATION)
    
    if is_2d:
        # 2D registration (T, Y, X)
        logger.info(f"Registering {T} frames (2D)")
        
        # Register to get transformation matrices (returns all matrices)
        tmats = sr.register_stack(
            zyx_stack,
            reference=reference
        )
        
        # Extract shifts from transformation matrices
        # Translation matrix format:
        # [[1, 0, TX],
        #  [0, 1, TY],
        #  [0, 0, 1]]
        # Where TX = X shift (horizontal), TY = Y shift (vertical)
        # 
        # IMPORTANT: StackReg transformation matrices represent the transform
        # FROM reference TO moving image. To align moving TO reference, we need
        # the OPPOSITE (negative) of these values.
        # We store them as [Y, X] to match scipy.ndimage.shift order
        shifts = np.zeros((T, 2), dtype=np.float64)
        
        # tmats shape is (T, 3, 3) - one matrix per frame
        if tmats.ndim == 2:
            # Single transformation matrix (shouldn't happen for T>1)
            for t in range(T):
                shifts[t, 0] = -tmats[1, 2]  # -TY (Y shift) - vertical
                shifts[t, 1] = -tmats[0, 2]  # -TX (X shift) - horizontal
        else:
            # (T, 3, 3) - one matrix per frame
            for t in range(T):
                shifts[t, 0] = -tmats[t, 1, 2]  # -TY (Y shift) - vertical
                shifts[t, 1] = -tmats[t, 0, 2]  # -TX (X shift) - horizontal
        
        logger.info(f"Computed {T} shifts - Mean: {np.mean(shifts, axis=0)}, Max: {np.max(np.abs(shifts)):.2f} px")
        
        # Return shifts and None for corrected (we'll apply shifts manually in main function)
        return shifts, None
    
    else:
        # 3D registration (T, Z, Y, X) - register each Z-slice independently
        T, Z, Y, X = zyx_stack.shape
        logger.info(f"Registering {T} frames with {Z} Z-slices (per-slice 2D)")
        
        shifts = np.zeros((T, Z, 2), dtype=np.float64)
        
        for z in range(Z):
            logger.info(f"Processing Z-slice {z+1}/{Z}")
            
            # Extract 2D stack for this Z-slice
            slice_stack = zyx_stack[:, z, :, :]  # (T, Y, X)
            
            # Register to get transformation matrices (returns all matrices)
            tmats = sr.register_stack(
                slice_stack,
                reference=reference
            )
            
            # Extract shifts for this Z-slice
            # tmats shape is (T, 3, 3) - one matrix per frame
            # Store as [Y, X] to match scipy.ndimage.shift order
            # Negate because StackReg matrices are FROM reference TO moving
            if tmats.ndim == 2:
                # Single transformation matrix (shouldn't happen for T>1)
                for t in range(T):
                    shifts[t, z, 0] = -tmats[1, 2]  # -TY (Y shift) - vertical
                    shifts[t, z, 1] = -tmats[0, 2]  # -TX (X shift) - horizontal
            else:
                # (T, 3, 3) - one matrix per frame
                for t in range(T):
                    shifts[t, z, 0] = -tmats[t, 1, 2]  # -TY (Y shift) - vertical
                    shifts[t, z, 1] = -tmats[t, 0, 2]  # -TX (X shift) - horizontal
        
        # Log statistics across all Z-slices
        mean_shift_per_z = np.mean(shifts, axis=0)  # (Z, 2)
        logger.info(f"Computed shifts for {T} frames x {Z} slices")
        logger.info(f"Mean shift across Z: {np.mean(mean_shift_per_z, axis=0)}")
        logger.info(f"Max shift: {np.max(np.abs(shifts)):.2f} px")
        
        # Return shifts and None for corrected (we'll apply shifts manually in main function)
        return shifts, None


def extract_translation_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Extract translation components from 2D affine transformation matrix.
    
    Parameters
    ----------
    matrix : np.ndarray
        Affine transformation matrix (3, 3) or (T, 3, 3)
    
    Returns
    -------
    np.ndarray
        Translation vector(s) [TX, TY]
        Shape (2,) for single matrix or (T, 2) for batch
    """
    if matrix.ndim == 2:
        # Single matrix
        return np.array([matrix[0, 2], matrix[1, 2]])
    else:
        # Batch of matrices
        T = matrix.shape[0]
        shifts = np.zeros((T, 2))
        shifts[:, 0] = matrix[:, 0, 2]  # TX
        shifts[:, 1] = matrix[:, 1, 2]  # TY
        return shifts
