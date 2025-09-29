import numpy as np
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
    
def apply_shift(image: np.ndarray, shift: np.ndarray, mode: str = 'constant', order: int = 3) -> np.ndarray:
    """
    Apply translation shift to an image.
    
    Args:
        image: Input image (2D or 3D)
        shift: Translation vector [dy, dx] for 2D or [dz, dy, dx] for 3D
        mode: Boundary handling mode (default 'constant' = zero-fill)
        order: Interpolation order (0=nearest, 1=linear, 3=cubic)
        
    Returns:
        Shifted image with same shape and dtype as input
    """
    try:
        import cupy as cp
        from cupyx.scipy import ndimage as cndi
        return _apply_shift_cupy(image, shift, mode, order)
    except ImportError:
        return _apply_shift_numpy(image, shift, mode, order)

def _apply_shift_cupy(image: np.ndarray, shift: np.ndarray, mode: str = 'constant', order: int = 3) -> np.ndarray:
    """CuPy-accelerated shift application."""
    import cupy as cp
    from cupyx.scipy import ndimage as cndi
    
    # Convert to CuPy array
    image_cp = cp.asarray(image, dtype=cp.float32)
    shift_cp = cp.asarray(shift, dtype=cp.float32)
    
    # Apply shift
    shifted_cp = cndi.shift(image_cp, shift=shift_cp, order=order, mode=mode, cval=0.0, prefilter=True)
    shifted_np = cp.asnumpy(shifted_cp)
    
    # Convert back to original dtype
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        shifted_np = np.clip(shifted_np, info.min, info.max).astype(image.dtype)
    else:
        shifted_np = shifted_np.astype(image.dtype)
        
    return shifted_np

def _apply_shift_numpy(image: np.ndarray, shift: np.ndarray, mode: str = 'constant', order: int = 3) -> np.ndarray:
    """NumPy/SciPy fallback shift application."""
    try:
        from scipy import ndimage
        
        shifted = ndimage.shift(image.astype(np.float32), shift=shift, order=order, mode=mode, cval=0.0, prefilter=True)
        
        # Convert back to original dtype
        if np.issubdtype(image.dtype, np.integer):
            info = np.iinfo(image.dtype)
            shifted = np.clip(shifted, info.min, info.max).astype(image.dtype)
        else:
            shifted = shifted.astype(image.dtype)
            
        return shifted
    except ImportError:
        logger.error("SciPy not available for shift application")
        return image  # Return unchanged if no shift capability


def apply_shifts_to_tczyx_stack(stack: np.ndarray, shifts: np.ndarray, mode: str = 'constant', order: int = 3) -> np.ndarray:
    """
    Apply a series of shifts to a 5D TCZYX image stack.
    
    Args:
        stack: Input image stack (T, C, Z, Y, X)
        shifts: Array of shifts (T, 2) with [dy, dx] for each timepoint or (T, 3) with [dz, dy, dx] for 3D
        mode: Boundary handling mode (default 'constant' = zero-fill)
        order: Interpolation order (0=nearest, 1=linear, 3=cubic)
        
    Returns:
        Shifted image stack with same shape and dtype as input
    """
    T, C, Z, Y, X = stack.shape
    shifted_stack = np.empty_like(stack)
    
    # Determine if we have 2D or 3D shifts
    shift_dims = shifts.shape[1]
    
    if shift_dims == 2:
        # 2D shifts: [dy, dx] - apply to each Z slice separately
        for t in range(T):
            shift_yx = shifts[t]  # [dy, dx]
            for c in range(C):
                for z in range(Z):
                    shifted_stack[t, c, z] = apply_shift(stack[t, c, z], shift_yx, mode=mode, order=order)
    elif shift_dims == 3:
        # 3D shifts: [dz, dy, dx] - apply to entire 3D volume
        for t in range(T):
            shift_zyx = shifts[t]  # [dz, dy, dx]
            for c in range(C):
                shifted_stack[t, c] = apply_shift(stack[t, c], shift_zyx, mode=mode, order=order)
    else:
        raise ValueError(f"Shifts must have shape (T, 2) or (T, 3), got shape {shifts.shape}")
    
    return shifted_stack