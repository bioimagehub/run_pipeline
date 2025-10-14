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
    reference: Literal["first", "previous", "mean"] = "first"
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
        
        # Register to get transformation matrices (don't transform yet)
        sr.register_stack(
            zyx_stack,
            reference=reference
        )
        
        # Get the transformation matrices
        tmats = sr.get_matrix()  # Shape: (T, 3, 3) or (3, 3)
        
        # Extract shifts from transformation matrices
        # Translation matrix format:
        # [[1, 0, TX],
        #  [0, 1, TY],
        #  [0, 0, 1]]
        # 
        # These shifts tell us how to transform each frame to align with reference
        shifts = np.zeros((T, 2), dtype=np.float64)
        
        # Handle both (T, 3, 3) and (3, 3) matrix shapes
        if tmats.ndim == 2:
            # Single transformation matrix
            for t in range(T):
                shifts[t, 0] = tmats[0, 2]  # TX (Y shift)
                shifts[t, 1] = tmats[1, 2]  # TY (X shift)
        else:
            # (T, 3, 3) - one matrix per frame
            for t in range(T):
                shifts[t, 0] = tmats[t, 0, 2]  # TX (Y shift)
                shifts[t, 1] = tmats[t, 1, 2]  # TY (X shift)
        
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
            
            # Register to get transformation matrices (don't transform yet)
            sr.register_stack(
                slice_stack,
                reference=reference
            )
            
            # Extract shifts for this Z-slice
            tmats = sr.get_matrix()
            
            # Handle both (T, 3, 3) and (3, 3) matrix shapes
            if tmats.ndim == 2:
                # Single transformation matrix
                for t in range(T):
                    shifts[t, z, 0] = tmats[0, 2]  # TX (Y shift)
                    shifts[t, z, 1] = tmats[1, 2]  # TY (X shift)
            else:
                # (T, 3, 3) - one matrix per frame
                for t in range(T):
                    shifts[t, z, 0] = tmats[t, 0, 2]  # TX (Y shift)
                    shifts[t, z, 1] = tmats[t, 1, 2]  # TY (X shift)
        
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
