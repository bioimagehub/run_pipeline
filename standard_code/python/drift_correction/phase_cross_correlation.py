"""
Phase Cross-Correlation Drift Correction - Translation Only

Uses scikit-image (CPU) and cuCIM (GPU) for fast, subpixel-accurate translation detection.

⚠️ TRANSLATION ONLY: Mathematically guaranteed via Fourier phase shift theorem.

MIT License
Copyright (c) 2024 BIPHUB - Bioimage Informatics Hub, University of Oslo
"""

import logging
from typing import Tuple, Union, Literal
import numpy as np
from scipy.ndimage import shift as scipy_shift

logger = logging.getLogger(__name__)


def phase_cross_correlation_cpu(
    zyx_stack: np.ndarray,
    reference: Union[Literal["first", "previous", "mean", "median"], int] = "first",
    upsample_factor: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CPU-based phase cross-correlation using scikit-image.
    
    ⚠️ TRANSLATION ONLY: Returns XYZ shifts only, no rotation or scaling.
    
    Uses scikit-image's phase_cross_correlation which is mathematically
    guaranteed to only compute translation shifts via Fourier phase shift theorem.
    
    Parameters
    ----------
    zyx_stack : np.ndarray
        Input stack with shape (T, Z, Y, X) or (T, Y, X) for 2D
    reference : Union[Literal["first", "previous", "mean", "median"], int]
        Reference frame strategy:
        - "first": Register all frames to T=0 (most common)
        - "previous": Register each frame to previous frame (sequential/cumulative)
        - "mean": Register to mean projection across all timepoints
        - "median": Register to median projection (robust to outliers)
        - int: Register all frames to specific timepoint T=N
    upsample_factor : int
        Subpixel accuracy (10 = 0.1 pixel, 100 = 0.01 pixel precision)
    
    Returns
    -------
    shifts : np.ndarray
        Translation shifts in pixels (subpixel accuracy)
        - 2D case: Shape (T, 2) for YX shifts
        - 3D case: Shape (T, 3) for ZYX shifts
        For reference="previous", returns cumulative shifts relative to T=0
    corrected : np.ndarray
        Drift-corrected stack with same shape as input
        
    Notes
    -----
    - "previous" mode accumulates shifts: each frame is registered to the 
      previous frame, and cumulative shifts are tracked relative to T=0
    - "mean"/"median" compute a reference projection first, then register 
      all frames to this stable reference (good for noisy data)
    - Integer reference allows registering to any stable timepoint
    """
    from skimage.registration import phase_cross_correlation
    
    T = zyx_stack.shape[0]
    is_2d = zyx_stack.ndim == 3  # (T, Y, X)
    
    logger.info(f"Phase cross-correlation (CPU) - {'2D' if is_2d else '3D'} registration")
    logger.info(f"Reference: {reference}, Upsample factor: {upsample_factor}")
    
    # Prepare reference frame
    if reference == "first":
        ref_frame = zyx_stack[0]
    elif reference == "mean":
        ref_frame = np.mean(zyx_stack, axis=0)
        logger.info("Using mean projection as reference")
    elif reference == "median":
        ref_frame = np.median(zyx_stack, axis=0)
        logger.info("Using median projection as reference")
    elif isinstance(reference, int):
        if reference < 0 or reference >= T:
            raise ValueError(f"Reference timepoint {reference} out of range [0, {T-1}]")
        ref_frame = zyx_stack[reference]
        logger.info(f"Using timepoint {reference} as reference")
    elif reference == "previous":
        ref_frame = None  # Will be updated in loop
    else:
        raise ValueError(f"Unknown reference mode: {reference}")
    
    # Compute shifts
    shift_dim = 2 if is_2d else 3
    shifts = np.zeros((T, shift_dim), dtype=np.float64)
    cumulative_shift = np.zeros(shift_dim, dtype=np.float64)
    
    for t in range(T):
        if t == 0:
            # First frame has zero shift
            shifts[t] = 0
            if reference == "previous":
                ref_frame = zyx_stack[0]
            continue
        
        # Update reference for "previous" mode
        if reference == "previous":
            ref_frame = zyx_stack[t - 1]
        
        # Compute shift
        shift, error, phasediff = phase_cross_correlation(
            ref_frame,
            zyx_stack[t],
            upsample_factor=upsample_factor
        )
        
        if reference == "previous":
            # Accumulate shifts for "previous" mode
            cumulative_shift += shift
            shifts[t] = cumulative_shift
        else:
            shifts[t] = shift
    
    logger.info(f"Computed {T} shifts - Mean: {np.mean(shifts, axis=0)}, Max: {np.max(np.abs(shifts)):.2f} px")
    
    # Apply shifts
    corrected = apply_shifts(zyx_stack, shifts)
    
    return shifts, corrected


def phase_cross_correlation_gpu(
    zyx_stack: np.ndarray,
    reference: Union[Literal["first", "previous", "mean", "median"], int] = "first",
    upsample_factor: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated phase cross-correlation using cuCIM.
    
    ⚠️ TRANSLATION ONLY: Returns XYZ shifts only, no rotation or scaling.
    
    Falls back to CPU if CUDA not available.
    
    Parameters
    ----------
    Same as phase_cross_correlation_cpu
    
    Returns
    -------
    Same as phase_cross_correlation_cpu
    """
    try:
        import cupy as cp
        from cucim.skimage.registration import phase_cross_correlation as phase_cross_correlation_gpu_impl
        
        logger.info("GPU (cuCIM) available, using GPU acceleration")
        
        T = zyx_stack.shape[0]
        is_2d = zyx_stack.ndim == 3
        
        logger.info(f"Phase cross-correlation (GPU) - {'2D' if is_2d else '3D'} registration")
        logger.info(f"Reference: {reference}, Upsample factor: {upsample_factor}")
        
        # Prepare reference frame
        if reference == "first":
            ref_frame = cp.asarray(zyx_stack[0])
        elif reference == "mean":
            ref_frame = cp.mean(cp.asarray(zyx_stack), axis=0)
            logger.info("Using mean projection as reference")
        elif reference == "median":
            ref_frame = cp.median(cp.asarray(zyx_stack), axis=0)
            logger.info("Using median projection as reference")
        elif isinstance(reference, int):
            if reference < 0 or reference >= T:
                raise ValueError(f"Reference timepoint {reference} out of range [0, {T-1}]")
            ref_frame = cp.asarray(zyx_stack[reference])
            logger.info(f"Using timepoint {reference} as reference")
        elif reference == "previous":
            ref_frame = None
        else:
            raise ValueError(f"Unknown reference mode: {reference}")
        
        # Compute shifts
        shift_dim = 2 if is_2d else 3
        shifts = np.zeros((T, shift_dim), dtype=np.float64)
        cumulative_shift = np.zeros(shift_dim, dtype=np.float64)
        
        for t in range(T):
            if t == 0:
                shifts[t] = 0
                if reference == "previous":
                    ref_frame = cp.asarray(zyx_stack[0])
                continue
            
            # Update reference for "previous" mode
            if reference == "previous":
                ref_frame = cp.asarray(zyx_stack[t - 1])
            
            # Compute shift on GPU
            current_frame = cp.asarray(zyx_stack[t])
            shift, error, phasediff = phase_cross_correlation_gpu_impl(
                ref_frame,
                current_frame,
                upsample_factor=upsample_factor
            )
            
            # Convert back to CPU
            shift = cp.asnumpy(shift)
            
            if reference == "previous":
                cumulative_shift += shift
                shifts[t] = cumulative_shift
            else:
                shifts[t] = shift
        
        logger.info(f"Computed {T} shifts - Mean: {np.mean(shifts, axis=0)}, Max: {np.max(np.abs(shifts)):.2f} px")
        
        # Apply shifts (on CPU for now)
        corrected = apply_shifts(zyx_stack, shifts)
        
        return shifts, corrected
    
    except ImportError as e:
        logger.warning(f"GPU (cuCIM/CuPy) not available: {e}")
        logger.warning("Falling back to CPU implementation")
        return phase_cross_correlation_cpu(zyx_stack, reference, upsample_factor)
    except Exception as e:
        logger.error(f"GPU processing failed: {e}")
        logger.warning("Falling back to CPU implementation")
        return phase_cross_correlation_cpu(zyx_stack, reference, upsample_factor)


def apply_shifts(stack: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """
    Apply translation shifts to a stack.
    
    Parameters
    ----------
    stack : np.ndarray
        Input stack (T, Z, Y, X) or (T, Y, X)
    shifts : np.ndarray
        Translation shifts (T, 3) for ZYX or (T, 2) for YX
    
    Returns
    -------
    np.ndarray
        Corrected stack with same shape as input
    """
    corrected = np.zeros_like(stack)
    T = stack.shape[0]
    is_2d = stack.ndim == 3
    
    for t in range(T):
        if is_2d:
            # 2D: shift YX
            corrected[t] = scipy_shift(
                stack[t],
                shift=shifts[t],
                order=1,  # Linear interpolation
                mode='constant',
                cval=0
            )
        else:
            # 3D: shift ZYX
            corrected[t] = scipy_shift(
                stack[t],
                shift=shifts[t],
                order=1,
                mode='constant',
                cval=0
            )
    
    return corrected
