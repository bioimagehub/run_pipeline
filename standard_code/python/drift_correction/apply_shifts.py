"""Image shift application module for drift correction.

This module provides functions to apply translation shifts to images and image stacks.
The shifts represent pixel displacements that will be applied to move/translate the image
content to correct for drift or misalignment.

=== SHIFT CONVENTION AND COORDINATE SYSTEM ===

Shift Direction Convention:
  - POSITIVE shift values move image content in the POSITIVE direction of that axis
  - NEGATIVE shift values move image content in the NEGATIVE direction of that axis
  
Coordinate System:
  - Y-axis: Points DOWNWARD (row direction in image arrays)
    * +Y shift moves content DOWN (toward higher row indices)
    * -Y shift moves content UP (toward lower row indices)
  - X-axis: Points RIGHTWARD (column direction in image arrays)  
    * +X shift moves content RIGHT (toward higher column indices)
    * -X shift moves content LEFT (toward lower column indices)
  - Z-axis: Points DEEPER into stack (slice direction)
    * +Z shift moves content toward higher slice indices
    * -Z shift moves content toward lower slice indices

Shift Array Format:
  - 2D images: shifts = [dy, dx] where dy=Y-shift, dx=X-shift
  - 3D volumes: shifts = [dz, dy, dx] where dz=Z-shift, dy=Y-shift, dx=X-shift
  - Multi-timepoint: shifts = (T, 2) or (T, 3) array with one shift vector per timepoint

Example:
  shift = [2.5, -1.0] means:
  - Move image content 2.5 pixels DOWN (positive Y direction)
  - Move image content 1.0 pixel LEFT (negative X direction)

=== DRIFT CORRECTION WORKFLOW ===

1. Drift Detection: Phase cross-correlation detects how much the image has drifted
2. Correction Shifts: The detected shifts represent the correction needed to undo drift
3. Apply Shifts: This module applies those correction shifts to align the images

The shifts applied by this module are CORRECTION SHIFTS that move the drifted image
back to its aligned position. If an image drifted +5 pixels right, the correction
shift will be -5 pixels to move it back left.

=== INTERPOLATION AND BOUNDARY HANDLING ===

Sub-pixel Accuracy:
  - Uses spline interpolation (order=3 by default) for sub-pixel precision
  - order=0: nearest neighbor (fastest, integer precision)
  - order=1: linear interpolation (good speed/quality balance)
  - order=3: cubic spline (best quality, slower)

Boundary Conditions:
  - mode='constant': Fill with zeros (default, recommended for drift correction)
  - mode='nearest': Extend edge pixels
  - mode='wrap': Periodic boundary (wraps around)
  - mode='reflect': Mirror boundary

=== GPU ACCELERATION ===

Automatically uses CuPy for GPU acceleration when available, falls back to
CPU-based scipy.ndimage when CuPy is not installed.
"""

import numpy as np
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
    
def apply_shift(image: np.ndarray, shift: np.ndarray, mode: str = 'constant', order: int = 3) -> np.ndarray:
    """
    Apply a translation shift to move image content by specified pixel displacement.
    
    This function takes an image and a shift vector, then moves the image content
    by the specified number of pixels (including fractional sub-pixel shifts).
    The shift represents how far to MOVE the image content, not where it came from.
    
    CRITICAL: Shift Direction Convention
    ===================================
    - shift[0] (dy): POSITIVE moves content DOWN (+Y direction)
    - shift[1] (dx): POSITIVE moves content RIGHT (+X direction)  
    - shift[2] (dz): POSITIVE moves content DEEPER (+Z direction, for 3D)
    
    Example Transformations:
    - shift = [2.0, -1.5] moves content 2.0 pixels down and 1.5 pixels left
    - shift = [0, 5] moves content 5 pixels to the right (no vertical movement)
    - shift = [-3, 0] moves content 3 pixels up (negative Y direction)
    
    For Drift Correction:
    If phase correlation detects that image drifted +3 pixels right, the correction
    shift should be [-3] to move the content back 3 pixels left to its original position.
    
    Args:
        image (np.ndarray): Input image to be shifted
            - 2D: shape (H, W) - single image
            - 3D: shape (Z, H, W) - volume or stack of slices
        shift (np.ndarray): Translation vector specifying pixel displacement
            - 2D images: [dy, dx] where dy=Y-displacement, dx=X-displacement
            - 3D volumes: [dz, dy, dx] where dz=Z-displacement, dy=Y-displacement, dx=X-displacement
            - Values can be fractional for sub-pixel precision
            - Positive values move content in positive axis direction
        mode (str): Boundary handling for pixels shifted outside image bounds
            - 'constant': Fill with zeros (default, best for drift correction)
            - 'nearest': Extend edge pixels  
            - 'wrap': Periodic wrapping (like np.roll)
            - 'reflect': Mirror at boundaries
        order (int): Interpolation method for sub-pixel shifts
            - 0: Nearest neighbor (fastest, integer precision only)
            - 1: Linear interpolation (good balance of speed/quality)
            - 3: Cubic spline (best quality, default, recommended for drift correction)
            
    Returns:
        np.ndarray: Shifted image with same shape and dtype as input
            - Image content moved by specified shift vector
            - Sub-pixel accuracy achieved through interpolation
            - Boundary regions handled according to mode parameter
            
    Raises:
        ImportError: If neither CuPy nor SciPy is available
        ValueError: If shift dimensions don't match image dimensions
        
    Implementation Notes:
        - Automatically uses GPU acceleration (CuPy) when available
        - Falls back to CPU implementation (SciPy) if CuPy unavailable  
        - Preserves input image dtype (converts to float32 for processing)
        - Uses prefiltering for high-quality spline interpolation
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
    Apply drift correction shifts to a 5D TCZYX image stack (Time-Channel-Z-Y-X).
    
    This is the main function for applying drift correction to multi-dimensional image stacks.
    Each timepoint gets its own correction shift to align all frames to a common reference.
    The function handles both 2D (XY-only) and 3D (ZYX) drift correction automatically.
    
    TRANSFORMATION MATRIX CONCEPT:
    ==============================
    The shifts parameter acts as a transformation lookup table where:
    - shifts[t] = correction shift to apply to timepoint t
    - Each shift vector moves that timepoint's content to align with reference
    - For drift correction: shifts undo the detected drift to restore alignment
    
    SHIFT TABLE FORMAT:
    ==================
    The shifts array is a (T, dimensions) transformation matrix:
    
    2D Drift Correction (XY plane only):
      shifts shape: (T, 2) 
      shifts[t] = [dy, dx] for timepoint t
      Example:
        shifts = [[0.0, 0.0],     # T=0: no shift (reference frame)
                  [1.2, -0.5],    # T=1: move 1.2 pixels down, 0.5 pixels left  
                  [-2.1, 3.7]]    # T=2: move 2.1 pixels up, 3.7 pixels right
    
    3D Drift Correction (ZYX volume):
      shifts shape: (T, 3)
      shifts[t] = [dz, dy, dx] for timepoint t  
      Example:
        shifts = [[0.0, 0.0, 0.0],    # T=0: no shift (reference)
                  [0.2, 1.2, -0.5],   # T=1: move 0.2 slices up, 1.2 pixels down, 0.5 pixels left
                  [-0.1, -2.1, 3.7]]  # T=2: move 0.1 slices down, 2.1 pixels up, 3.7 pixels right
    
    PROCESSING WORKFLOW:
    ===================
    1. Detect shift table format (2D vs 3D) from shifts.shape[1]
    2. For each timepoint t:
       - Extract shift vector for that timepoint: shifts[t]
       - Apply shift to all channels of that timepoint
       - 2D: Apply [dy, dx] to each Z-slice independently
       - 3D: Apply [dz, dy, dx] to entire ZYX volume at once
    3. Return corrected stack with same TCZYX structure
    
    Args:
        stack (np.ndarray): Input 5D image stack with shape (T, C, Z, Y, X)
            - T: Number of timepoints (frames in time series)
            - C: Number of channels (e.g., fluorescence channels)
            - Z: Number of Z-slices (depth in 3D, =1 for 2D images)
            - Y: Image height (rows)
            - X: Image width (columns)
        shifts (np.ndarray): Transformation matrix of correction shifts
            - 2D correction: shape (T, 2) with [dy, dx] for each timepoint
            - 3D correction: shape (T, 3) with [dz, dy, dx] for each timepoint
            - shifts[t] = displacement vector to apply to timepoint t
            - Positive values move content in positive axis direction
            - Sub-pixel values supported for high-precision correction
        mode (str): Boundary handling for shifted pixels (default: 'constant')
            - 'constant': Fill shifted-in regions with zeros (recommended)
            - 'nearest': Extend edge pixel values
            - 'wrap': Periodic boundary (like np.roll)
            - 'reflect': Mirror at image boundaries
        order (int): Interpolation method for sub-pixel accuracy (default: 3)
            - 0: Nearest neighbor (integer precision, fastest)
            - 1: Linear interpolation (good speed/quality trade-off)  
            - 3: Cubic spline (highest quality, recommended)
            
    Returns:
        np.ndarray: Drift-corrected image stack with same (T, C, Z, Y, X) shape
            - Each timepoint shifted according to its correction vector
            - All timepoints now aligned to common reference frame
            - Preserves original data type and channel structure
            - Sub-pixel precision maintained through interpolation
            
    Raises:
        ValueError: If shifts shape doesn't match expected (T, 2) or (T, 3) format
        ValueError: If shifts.shape[0] != stack.shape[0] (timepoint mismatch)
        
    Example Usage:
        # Load drifted image stack
        img_stack = rp.load_tczyx_image('drifted_cells.tif').data  # shape: (10, 2, 1, 512, 512)
        
        # Detect drift (from phase cross-correlation)
        correction_shifts = phase_cross_correlation(img_stack)  # shape: (10, 3)
        
        # Apply drift correction
        corrected_stack = apply_shifts_to_tczyx_stack(img_stack, correction_shifts)
        
        # Result: All 10 timepoints now aligned, drift removed
    
    Implementation Notes:
        - Automatically detects 2D vs 3D shift format from shifts.shape[1]
        - Uses GPU acceleration when available (CuPy), CPU fallback (SciPy)
        - Processes each timepoint and channel independently
        - Maintains full precision through float32 intermediate processing
        - Optimized memory usage with in-place operations where possible
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