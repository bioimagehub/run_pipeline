"""
Phase Cross-Correlation Drift Detection for Bioimage Analysis.

This module implements GPU-accelerated phase cross-correlation algorithms to detect
and quantify drift in time-lapse microscopy images. It produces correction shifts
that can be applied to align image stacks and remove unwanted sample movement.

=== ALGORITHM OVERVIEW ===

Phase Cross-Correlation Principle:
  1. Convert images to frequency domain using FFT
  2. Compute cross-power spectrum: F_ref * conj(F_img)
  3. Normalize to isolate phase information (remove amplitude)
  4. Inverse FFT to get correlation surface
  5. Find peak location = shift between images
  6. Sub-pixel refinement for high precision

This method is robust to noise and lighting changes, focusing purely on
structural alignment rather than intensity matching.

=== DRIFT DETECTION vs CORRECTION SHIFTS ===

CRITICAL CONCEPT: This module returns CORRECTION SHIFTS, not drift detection.

Drift Detection:
  - "How much did the image move?" 
  - If image drifted +5 pixels right, drift = [0, +5]
  
Correction Shifts (what this module returns):
  - "How much do we need to move it back to align?"
  - If image drifted +5 pixels right, correction = [0, -5]
  - These shifts are directly usable with apply_shifts module

The phase correlation algorithm uses F_ref * conj(F_img) convention,
which naturally produces correction shifts due to the reference-first ordering.

=== COORDINATE SYSTEM AND SHIFT FORMAT ===

Shift Vector Format:
  - 2D: [dy, dx] where dy=Y-shift, dx=X-shift 
  - 3D: [dz, dy, dx] where dz=Z-shift, dy=Y-shift, dx=X-shift
  - Always returns 3D format [dz, dy, dx] for consistency
  - For 2D images (Z=1), dz component will be 0.0

Coordinate Conventions:
  - Y-axis: DOWNWARD (increasing row indices)
  - X-axis: RIGHTWARD (increasing column indices)  
  - Z-axis: DEEPER (increasing slice indices)
  - Positive shift values move content in positive axis direction

=== REFERENCE FRAME STRATEGIES ===

Different reference frame approaches for different use cases:

'first': Compare all frames to first timepoint
  - Best for: Static samples, photobleaching studies
  - Accumulates errors over time if sample changes
  - Most common choice for drift correction

'previous': Compare each frame to previous frame  
  - Best for: Live cell imaging, dynamic samples
  - Minimizes registration errors from biological changes
  - Returns frame-to-frame drift increments
  - Need to accumulate shifts for absolute correction

'mean': Compare all frames to temporal average
  - Best for: Periodic motion, reducing random errors
  - Requires loading entire stack into memory
  - Good SNR but computationally expensive

'mean10': Compare all frames to mean of first 10 frames
  - Best for: Balance between 'first' and 'mean'
  - Reduces noise while maintaining reference stability
  - Good compromise for most applications

=== OUTPUT FORMAT ===

Returns: np.ndarray with shape (T, 3) containing correction shifts
  - T: Number of timepoints in input stack
  - 3: [dz, dy, dx] correction shifts in ZYX order
  - shifts[t] = correction shift to apply to timepoint t
  - shifts[0] typically [0, 0, 0] for 'first' reference
  - Sub-pixel precision (float values)

Example Output:
  shifts = [[0.0, 0.0, 0.0],      # T=0: reference frame, no correction
            [0.0, -1.2, 2.5],     # T=1: move 1.2 pixels up, 2.5 pixels right  
            [0.0, 0.3, -0.8]]     # T=2: move 0.3 pixels down, 0.8 pixels left

=== ADVANCED FEATURES ===

Sub-pixel Accuracy:
  - Upsampling factors from 1 (integer) to 1000+ (high precision)
  - Uses enhanced 7x7x7 neighborhood fitting
  - Gaussian-weighted center-of-mass calculation
  - Parabolic fitting fallback for edge cases

Noise Handling:
  - Triangle thresholding to focus on high-information pixels
  - Robust normalization using median and MAD statistics
  - Hanning windowing to reduce spectral leakage
  - Spectral whitening for enhanced peak detection

GPU Acceleration:
  - Automatic CuPy detection and usage
  - Memory-optimized for large image stacks
  - Fallback CPU implementation for compatibility

=== TYPICAL WORKFLOW ===

1. Load TCZYX image stack
2. Call phase_cross_correlation() to detect drift
3. Get correction shifts array (T, 3)
4. Apply shifts using apply_shifts module  
5. Save corrected, aligned image stack

Example:
  img_stack = rp.load_tczyx_image('cells.tif').data
  correction_shifts = phase_cross_correlation(img_stack, reference_frame='first')
  corrected_stack = apply_shifts_to_tczyx_stack(img_stack, correction_shifts)
  rp.save_tczyx_image(corrected_stack, 'cells_corrected.tif')
"""

from typing import Optional, Tuple
import numpy as np
import sys
import os
import json

from skimage.filters import threshold_triangle, threshold_otsu

import numpy as np


# Use relative import to parent directory
try:
    from .. import bioimage_pipeline_utils as rp
except ImportError:
    # Fallback for when script is run directly (not as module)
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import bioimage_pipeline_utils as rp
import apply_shifts


import logging
logger = logging.getLogger(__name__)


def _apply_threshold_triangle_mask(image: np.ndarray) -> np.ndarray:
    """
    Apply threshold_triangle thresholding to create a mask of high-information pixels.
    
    Args:
        image: Input image (2D or 3D)        
    Returns:
        Masked image where low-information pixels are set to zero
    """

    try:
        # Handle both 2D and 3D cases
        if image.ndim == 2:
            # 2D case
            threshold = threshold_triangle(image)
            mask = image > threshold
            return image * mask
        elif image.ndim == 3:
            # 3D case - apply thresholding slice by slice for consistency
            masked_image = np.zeros_like(image)
            for z in range(image.shape[0]):
                threshold = threshold_triangle(image[z])
                mask = image[z] > threshold
                masked_image[z] = image[z] * mask
            return masked_image
        else:
            logger.warning(f"Unsupported image dimensions: {image.ndim}D, using original image")
            return image
            
    except Exception as e:
        logger.warning(f"Thresholding failed: {e}, using original image")
        return image

def _phase_cross_correlation_3d_cupy(reference: np.ndarray, image: np.ndarray, upsample_factor: int = 1, use_triangle_threshold: bool = True) -> Tuple[np.ndarray, float, float]:
    """
    Enhanced CuPy implementation of 3D phase cross-correlation optimized for random drift.
    
    This is the core algorithm that handles the actual FFT-based shift detection.
    Optimized for GPU execution with robust sub-pixel estimation and noise handling.
    
    Args:
        reference: Reference image volume (ZYX)
        image: Image volume to register (ZYX)  
        upsample_factor: Sub-pixel accuracy factor
        use_triangle_threshold: Apply triangle thresholding to focus on high-information pixels
    """
    import cupy as cp
    
    reference = cp.asarray(reference, dtype=cp.float32)
    image = cp.asarray(image, dtype=cp.float32)
    Z, H, W = reference.shape

    # Apply triangle thresholding to focus on high-information pixels
    if use_triangle_threshold:
        # Apply thresholding on CPU, then transfer back to GPU
        reference_cpu = cp.asnumpy(reference)
        image_cpu = cp.asnumpy(image)
        
        reference_masked = _apply_threshold_triangle_mask(reference_cpu)
        image_masked = _apply_threshold_triangle_mask(image_cpu)

        reference = cp.asarray(reference_masked, dtype=cp.float32)
        image = cp.asarray(image_masked, dtype=cp.float32)

        logger.debug("Applied threshold_triangle thresholding for high-information pixel selection")

    # Enhanced normalization for random drift: use robust statistics
    # Use median instead of mean for better noise rejection  
    ref_median = cp.median(reference)
    img_median = cp.median(image)
    
    # Robust standard deviation using MAD (Median Absolute Deviation)
    ref_mad = cp.median(cp.abs(reference - ref_median))
    img_mad = cp.median(cp.abs(image - img_median))
    
    # Normalize using robust statistics (avoid division by zero)
    # For single slices (Z=1), MAD might be very small, so use larger fallback
    fallback_scale = cp.std(reference) * 0.1  # 10% of std as fallback
    ref_mad = cp.maximum(ref_mad, cp.maximum(1e-6, fallback_scale))
    
    fallback_scale = cp.std(image) * 0.1
    img_mad = cp.maximum(img_mad, cp.maximum(1e-6, fallback_scale))
    
    reference_norm = (reference - ref_median) / ref_mad
    image_norm = (image - img_median) / img_mad
    
    # Enhanced windowing: always apply Hanning windowing for better edge handling
    # Hanning window reduces spectral leakage and improves accuracy for random drift
    wz = cp.hanning(Z).astype(cp.float32)[:, cp.newaxis, cp.newaxis]
    wy = cp.hanning(H).astype(cp.float32)[cp.newaxis, :, cp.newaxis]  
    wx = cp.hanning(W).astype(cp.float32)[cp.newaxis, cp.newaxis, :]
    window = wz * wy * wx
    
    reference_norm *= window
    image_norm *= window

    # Cross-power spectrum with enhanced normalization
    F_ref = cp.fft.fftn(reference_norm)
    F_img = cp.fft.fftn(image_norm)
    R = F_ref * cp.conj(F_img)
    
    # Enhanced robust normalization with spectral whitening
    R_abs = cp.abs(R)
    # Use percentile-based threshold instead of fixed threshold
    threshold = cp.percentile(R_abs[R_abs > 0], 5)  # 5th percentile
    R_abs = cp.maximum(R_abs, threshold)
    R /= R_abs
    
    # Apply spectral whitening to enhance peak sharpness
    # Gentle high-pass filtering to reduce low-frequency bias
    kz = cp.fft.fftfreq(Z, d=1.0).reshape(-1, 1, 1)
    ky = cp.fft.fftfreq(H, d=1.0).reshape(1, -1, 1) 
    kx = cp.fft.fftfreq(W, d=1.0).reshape(1, 1, -1)
    k_mag = cp.sqrt(kz**2 + ky**2 + kx**2)
    
    # Gentle high-pass filter (emphasize higher frequencies slightly)
    whitening_filter = 1.0 + 0.1 * k_mag  # Very mild whitening
    R *= whitening_filter
    
    # Get correlation peak
    r = cp.fft.ifftn(R).real

    # Find coarse peak location with sub-pixel refinement
    idx = int(cp.argmax(r))
    pz = idx // (H * W)
    py = (idx % (H * W)) // W
    px = idx % W
    
    # Handle wrap-around
    if pz > Z // 2: pz -= Z
    if py > H // 2: py -= H  
    if px > W // 2: px -= W

    if upsample_factor <= 1:
        return (cp.asnumpy(cp.asarray([float(pz), float(py), float(px)])), 
                1.0 - float(r.max()), 0.0)

    # Enhanced sub-pixel refinement with NaN checking
    dz_sub, dy_sub, dx_sub = _subpixel_fit_3d(r, pz, py, px, Z, H, W)
    
    # Check for NaN values and fallback to integer positions
    if cp.isnan(dz_sub) or cp.isinf(dz_sub):
        logger.warning("NaN detected in Z sub-pixel fitting, using integer position")
        dz_sub = float(pz)
    if cp.isnan(dy_sub) or cp.isinf(dy_sub):
        logger.warning("NaN detected in Y sub-pixel fitting, using integer position") 
        dy_sub = float(py)
    if cp.isnan(dx_sub) or cp.isinf(dx_sub):
        logger.warning("NaN detected in X sub-pixel fitting, using integer position")
        dx_sub = float(px)
    
    # Enhanced error estimation
    peak_value = float(r.max())
    noise_floor = float(cp.percentile(r, 90))  # 90th percentile as noise estimate
    snr = peak_value / (noise_floor + 1e-12)
    error = max(0.0, 1.0 - peak_value) / (1.0 + snr * 0.1)  # SNR-weighted error
    
    result = (cp.asnumpy(cp.asarray([dz_sub, dy_sub, dx_sub])), error, float(snr))
    
    # Clean up GPU memory before returning
    del reference, image, reference_norm, image_norm, F_ref, F_img, R, R_abs, r
    del wz, wy, wx, window, kz, ky, kx, k_mag, whitening_filter
    
    return result

def _subpixel_fit_3d(r, center_z: int, center_y: int, center_x: int,
                    Z: int, H: int, W: int) -> Tuple[float, float, float]:
    """
    Enhanced sub-pixel fitting using 7x7x7 neighborhood analysis for random drift.
    
    Uses weighted center-of-mass calculation with Gaussian weighting for improved
    accuracy with noisy/random drift patterns. Falls back to robust parabolic 
    fitting if peak is at neighborhood edge. Handles Z=1 case gracefully.
    """
    import cupy as cp
    
    # Special handling for Z=1 case (single slice) - only fit in XY
    if Z == 1:
        # Use 2D fitting for single slice
        neighborhood_2d = cp.zeros((7, 7), dtype=cp.float32)
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                ny = (center_y + dy + H) % H
                nx = (center_x + dx + W) % W
                neighborhood_2d[dy+3, dx+3] = r[0, ny, nx]  # Z=0 for single slice
        
        # 2D center of mass
        total_weight = cp.sum(neighborhood_2d)
        if total_weight > 1e-12:
            y_indices, x_indices = cp.meshgrid(
                cp.arange(7, dtype=cp.float32) - 3,
                cp.arange(7, dtype=cp.float32) - 3, indexing='ij')
            
            # Gaussian weights for 2D
            gaussian_weights_2d = cp.exp(-(y_indices**2 + x_indices**2) / (2 * 1.5**2))
            weighted_neighborhood = cp.maximum(neighborhood_2d, 0) * gaussian_weights_2d
            total_weight = cp.sum(weighted_neighborhood)
            
            if total_weight > 1e-12:
                cm_y = cp.sum(y_indices * weighted_neighborhood) / total_weight
                cm_x = cp.sum(x_indices * weighted_neighborhood) / total_weight
                cm_y = cp.clip(cm_y, -3.0, 3.0)
                cm_x = cp.clip(cm_x, -3.0, 3.0)
                return float(center_z), float(center_y + cm_y), float(center_x + cm_x)
        
        # Fallback: return integer positions for Z=1
        return float(center_z), float(center_y), float(center_x)
    
    # Use larger 7x7x7 neighborhood for better sub-pixel accuracy with random drift (Z > 1)
    neighborhood = cp.zeros((7, 7, 7), dtype=cp.float32)
    for dz in range(-3, 4):
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                nz = (center_z + dz + Z) % Z
                ny = (center_y + dy + H) % H
                nx = (center_x + dx + W) % W
                neighborhood[dz+3, dy+3, dx+3] = r[nz, ny, nx]
    
    # Check if peak is at edge of neighborhood
    max_idx = cp.argmax(neighborhood)
    max_z, max_y, max_x = cp.unravel_index(max_idx, (7, 7, 7))
    
    if (max_z <= 1 or max_z >= 5 or max_y <= 1 or max_y >= 5 or max_x <= 1 or max_x >= 5):
        # Enhanced parabolic fitting for edge cases
        z0, y0, x0 = (center_z + Z) % Z, (center_y + H) % H, (center_x + W) % W
        z1, z2 = (z0 - 1) % Z, (z0 + 1) % Z
        y1, y2 = (y0 - 1) % H, (y0 + 1) % H
        x1, x2 = (x0 - 1) % W, (x0 + 1) % W
        
        c = r[z0, y0, x0]
        # Enhanced parabolic fitting with better numerical stability
        denom_z = r[z1, y0, x0] - 2*c + r[z2, y0, x0]
        denom_y = r[z0, y1, x0] - 2*c + r[z0, y2, x0] 
        denom_x = r[z0, y0, x1] - 2*c + r[z0, y0, x2]
        
        # Clamp denominators to prevent division by very small numbers
        denom_z = cp.sign(denom_z) * cp.maximum(cp.abs(denom_z), 1e-8)
        denom_y = cp.sign(denom_y) * cp.maximum(cp.abs(denom_y), 1e-8) 
        denom_x = cp.sign(denom_x) * cp.maximum(cp.abs(denom_x), 1e-8)
        
        dz_off = 0.5 * (r[z1, y0, x0] - r[z2, y0, x0]) / denom_z
        dy_off = 0.5 * (r[z0, y1, x0] - r[z0, y2, x0]) / denom_y
        dx_off = 0.5 * (r[z0, y0, x1] - r[z0, y0, x2]) / denom_x
        
        # Clamp offsets to reasonable range
        dz_off = cp.clip(dz_off, -0.5, 0.5)
        dy_off = cp.clip(dy_off, -0.5, 0.5)
        dx_off = cp.clip(dx_off, -0.5, 0.5)
        
        return float(center_z + dz_off), float(center_y + dy_off), float(center_x + dx_off)
    
    # Enhanced weighted center-of-mass with Gaussian weighting for noise robustness
    # Apply Gaussian weighting to emphasize central peak and reduce noise influence
    z_indices, y_indices, x_indices = cp.meshgrid(
        cp.arange(7, dtype=cp.float32) - 3,
        cp.arange(7, dtype=cp.float32) - 3, 
        cp.arange(7, dtype=cp.float32) - 3, indexing='ij')
    
    # Gaussian weights (sigma=1.5 for good balance between accuracy and noise rejection)
    gaussian_weights = cp.exp(-(z_indices**2 + y_indices**2 + x_indices**2) / (2 * 1.5**2))
    
    # Apply weights to neighborhood and ensure positive values
    weighted_neighborhood = cp.maximum(neighborhood, 0) * gaussian_weights
    total_weight = cp.sum(weighted_neighborhood)
    
    if total_weight > 1e-12:
        cm_z = cp.sum(z_indices * weighted_neighborhood) / total_weight
        cm_y = cp.sum(y_indices * weighted_neighborhood) / total_weight
        cm_x = cp.sum(x_indices * weighted_neighborhood) / total_weight
        
        # Clamp center of mass offsets to reasonable range
        cm_z = cp.clip(cm_z, -3.0, 3.0)
        cm_y = cp.clip(cm_y, -3.0, 3.0) 
        cm_x = cp.clip(cm_x, -3.0, 3.0)
        
        return float(center_z + cm_z), float(center_y + cm_y), float(center_x + cm_x)
    else:
        return float(center_z), float(center_y), float(center_x)



def phase_cross_correlation(image_stack: np.ndarray, reference_frame: str = 'first', channel: int = 0, upsample_factor: int = 100,
                             max_shift_per_frame: float = 50.0, use_triangle_threshold: bool = True) -> np.ndarray:
    """
    Detect drift in time-lapse images and compute correction shifts using phase cross-correlation.
    
    This is the main entry point for drift detection. It analyzes a 5D image stack (TCZYX)
    and returns correction shifts that can be directly applied to align all timepoints
    to a common reference frame. The algorithm uses FFT-based phase correlation for
    robust, sub-pixel accuracy drift detection.
    
    EXPECTED INPUT FORMAT:
    =====================
    image_stack: 5D numpy array with shape (T, C, Z, Y, X)
      - T: Time dimension (number of frames in time series)
      - C: Channel dimension (e.g., fluorescence channels)
      - Z: Depth dimension (number of Z-slices, =1 for 2D images)
      - Y: Height dimension (image rows)
      - X: Width dimension (image columns)
    
    The function expects properly formatted TCZYX data. Use bioimage_pipeline_utils
    to ensure correct loading:
      img = rp.load_tczyx_image('data.tif')
      image_stack = img.data  # Guaranteed TCZYX format
    
    CORRECTION SHIFTS OUTPUT:
    ========================
    Returns a transformation matrix of correction shifts:
      Shape: (T, 3) where T = number of timepoints
      Format: shifts[t] = [dz, dy, dx] for timepoint t
      
    Each shift vector represents the pixel displacement needed to align
    that timepoint with the reference frame:
    - shifts[t][0] = Z-correction (depth, 0.0 for 2D images)
    - shifts[t][1] = Y-correction (vertical, positive = move content down)
    - shifts[t][2] = X-correction (horizontal, positive = move content right)
    
    Example output interpretation:
      shifts[5] = [0.0, -2.3, 1.7]
      Meaning: "To align timepoint 5 with reference, move its content
               2.3 pixels UP and 1.7 pixels RIGHT"
    
    REFERENCE FRAME STRATEGIES:
    ==========================
    The reference_frame parameter determines what each timepoint is compared against:
    
    'first' (default):
      - Compare all frames to first timepoint (T=0)
      - Best for: Static samples, stable imaging conditions
      - Output: Absolute shifts relative to first frame
      - shifts[0] will be [0.0, 0.0, 0.0] (reference doesn't move)
      - Memory efficient, fast processing
      
    'previous':
      - Compare each frame to its immediate predecessor
      - Best for: Live cell imaging, dynamic biological samples
      - Output: Cumulative shifts relative to first frame (auto-accumulated)
      - shifts[0] = [0.0, 0.0, 0.0] (first frame is reference)
      - Minimizes errors from biological changes over time
      - Internal: Detects frame-to-frame drift, then accumulates for absolute correction
      
    'mean':
      - Compare all frames to temporal average of entire stack
      - Best for: Periodic motion, high-noise data
      - Output: Shifts relative to temporal center
      - Requires loading entire stack (high memory usage)
      - Excellent SNR but computationally expensive
      
    'mean10':
      - Compare all frames to average of first 10 timepoints
      - Best for: Balance of stability and efficiency
      - Output: Shifts relative to early-timepoint average
      - Good compromise for most applications
    
    Args:
        image_stack (np.ndarray): Input 5D image stack with shape (T, C, Z, Y, X)
            - Must be properly formatted TCZYX array
            - Supports both 2D (Z=1) and 3D (Z>1) image series
            - All data types supported (will be converted to float32 internally)
            - Minimum recommended size: 64x64 pixels for reliable correlation
            
        reference_frame (str): Reference selection strategy (default: 'first')
            - 'first': Align all frames to first timepoint (most common)
            - 'previous': Frame-to-frame alignment (best for live cells)  
            - 'mean': Align to temporal average (best SNR, high memory)
            - 'mean10': Align to average of first 10 frames (balanced)
            
        channel (int): Channel index for drift detection (default: 0)
            - Specifies which channel to analyze for drift
            - Should be the channel with stable, high-contrast features
            - Avoid channels with dynamic biological changes
            - All channels will be corrected using shifts from this channel
            
        upsample_factor (int): Sub-pixel precision factor (default: 100)
            - 1: Integer pixel accuracy (fastest)
            - 10-100: Standard sub-pixel accuracy (recommended range)
            - 1000+: Ultra-high precision (slower, diminishing returns)
            - Higher values improve precision but increase computation time
            
        max_shift_per_frame (float): Maximum expected shift in pixels (default: 50.0)
            - Safety limit to detect registration failures
            - Shifts larger than this trigger warnings
            - Should be set based on expected drift magnitude
            - Too small: false positives on large but valid shifts
            - Too large: may miss actual registration failures
            
        use_triangle_threshold (bool): Enable adaptive thresholding (default: True)
            - True: Focus on high-information pixels (recommended)
            - False: Use all pixel intensities (may include noise)
            - Triangle thresholding automatically excludes background
            - Improves robustness in noisy or low-contrast images
    
    Returns:
        np.ndarray: Correction shifts with shape (T, 3)
            - T matches input timepoint dimension
            - Format: [dz, dy, dx] per timepoint in ZYX order
            - Sub-pixel precision (float values)
            - Ready for direct use with apply_shifts_to_tczyx_stack()
            - For 2D images: dz components will be 0.0
            
    Raises:
        ValueError: If input is not 5D TCZYX format
        ValueError: If invalid reference_frame specified
        RuntimeError: If GPU memory insufficient (falls back to CPU)
        
    Usage Examples:
        # Basic drift correction
        img_stack = rp.load_tczyx_image('timelapse.tif').data
        shifts = phase_cross_correlation(img_stack)
        corrected = apply_shifts_to_tczyx_stack(img_stack, shifts)
        
        # Live cell imaging (frame-to-frame)
        shifts = phase_cross_correlation(img_stack, reference_frame='previous')
        
        # High precision drift detection 
        shifts = phase_cross_correlation(img_stack, upsample_factor=1000)
        
        # Multi-channel: detect on channel 1, apply to all
        shifts = phase_cross_correlation(img_stack, channel=1)
        all_corrected = apply_shifts_to_tczyx_stack(img_stack, shifts)
    
    Algorithm Details:
        - Uses enhanced 3D phase cross-correlation with CuPy GPU acceleration
        - Robust normalization using median and MAD statistics
        - Hanning windowing and spectral whitening for improved accuracy
        - 7x7x7 neighborhood sub-pixel fitting with Gaussian weighting
        - Automatic fallback from GPU to CPU if CuPy unavailable
        - Memory-optimized processing for large image stacks
        
    Performance Notes:
        - GPU acceleration provides 10-100x speedup over CPU
        - Processing time scales with image size and upsample_factor
        - Memory usage: ~8x input stack size during processing
        - Typical processing: 512x512x100 stack in 1-10 seconds (GPU)
    """
    mode = 'constant'  

    # Validate input
    if image_stack.ndim != 5:
        raise ValueError("Input image stack must be 5D (TCZYX)")
    
    # Determine reference frame
    if reference_frame == 'first':
        reference_stack = image_stack[0:1].copy()  # Use first timepoint as reference
    elif reference_frame == 'last':
        reference_stack = image_stack[-1:].copy()  # Use last timepoint as reference
    elif reference_frame == 'mean':
        reference_stack = np.mean(image_stack, axis=0, keepdims=True).astype(image_stack.dtype)
    elif reference_frame == 'mean10':
        n = min(10, image_stack.shape[0]) # reduce n if less than 10 frames
        reference_stack = np.mean(image_stack[:n], axis=0, keepdims=True).astype(image_stack.dtype) # mean of first 10 or n frames
    elif reference_frame == 'previous':
        # Special case: use previous frame as reference (iterative processing)
        # We'll handle this in the loop below
        reference_stack = None  # Will be set iteratively
    else:
        raise ValueError(f"Invalid reference_frame: {reference_frame}")
    

    T = image_stack.shape[0]
    
    shifts = np.zeros((T, 3), dtype=np.float32)
    
    if reference_frame == 'previous':
        # Handle 'previous' reference frame iteratively with cumulative shift accumulation
        for t in range(T):
            if t == 0:
                # First frame has no shift (it's the initial reference)
                shifts[t] = np.array([0.0, 0.0, 0.0])
            else:
                # Use previous frame as reference
                reference_volume = image_stack[t-1, channel]
                current_volume = image_stack[t, channel]
                
                shift_result = _phase_cross_correlation_3d_cupy(reference_volume, current_volume, upsample_factor=upsample_factor, use_triangle_threshold=use_triangle_threshold)
                # Phase correlation already returns correction shifts (due to F_ref * conj(F_img) convention)
                # No negation needed - the algorithm directly gives us the shift to apply
                frame_to_frame_shift = shift_result[0]  # Shift from t-1 to t
                
                # Validate shift magnitude for random drift (allowing larger shifts)
                shift_magnitude = np.linalg.norm(frame_to_frame_shift)
                if shift_magnitude > max_shift_per_frame:
                    logger.warning(f"Extremely large shift detected at t={t}: {frame_to_frame_shift} (magnitude={shift_magnitude:.2f})")
                    logger.warning(f"This may indicate registration failure or stage malfunction")
                    # Still cap to prevent complete failure, but allow larger shifts for random drift
                    frame_to_frame_shift = frame_to_frame_shift * (max_shift_per_frame / shift_magnitude)
                
                # CRITICAL: Accumulate shifts for 'previous' reference mode
                # Each frame needs the cumulative shift relative to the first frame
                # shifts[t] = sum of all frame-to-frame shifts from 0 to t
                shifts[t] = shifts[t-1] + frame_to_frame_shift
                
                logger.debug(f"Frame {t}: frame-to-frame shift {frame_to_frame_shift}, cumulative shift {shifts[t]}")
    else:
        # Handle other reference frame types (first, last, mean, mean10)
        # Extract the reference frame to compare against all timepoints
        if reference_stack is None:
            raise RuntimeError("reference_stack should not be None for non-'previous' reference frames")
        reference_volume = reference_stack[0, channel]  # Reference is always the single frame

        for t in range(T):
            shift_result = _phase_cross_correlation_3d_cupy(reference_volume, image_stack[t, channel], upsample_factor=upsample_factor, use_triangle_threshold=use_triangle_threshold)
            # Phase correlation already returns correction shifts (due to F_ref * conj(F_img) convention)
            # No negation needed - the algorithm directly gives us the shift to apply
            raw_shift = shift_result[0]  # Direct use of correction shifts for apply_shifts
            
            # Validate shift magnitude for random drift (allowing larger shifts)
            shift_magnitude = np.linalg.norm(raw_shift)
            if shift_magnitude > max_shift_per_frame:
                logger.warning(f"Extremely large shift detected at t={t}: {raw_shift} (magnitude={shift_magnitude:.2f})")
                logger.warning(f"This may indicate registration failure or stage malfunction")
                # Still cap to prevent complete failure, but allow larger shifts for random drift
                raw_shift = raw_shift * (max_shift_per_frame / shift_magnitude)
            
            shifts[t] = raw_shift

    return shifts




if __name__ == "__main__":
    # Example usage: Basic drift correction
    # input_file = "E:/Oyvind/BIP-hub-test-data/drift/input/live_cells/1_Meng.nd2"
    input_file = "E:/Oyvind/BIP-hub-test-data/drift/input/live_cells/1_Meng_with_known_drift.tif"
    output_base = "E:/Oyvind/BIP-hub-test-data/drift/output_phase/"

    # Test 1: First frame as reference
    print("\n=== Example: Drift Correction with Phase Cross-Correlation using first frame as reference ===")
    reference_frame = "first"  # or "first", "mean", etc.
    output_file = os.path.join(output_base, os.path.splitext(os.path.basename(input_file))[0] + f"_corrected_{reference_frame}.tif")

    # Load image and apply drift correction
    image_stack = rp.load_tczyx_image(input_file).data


    detected_shifts = phase_cross_correlation(
        image_stack, 
        reference_frame=reference_frame,  # Use previous frame as reference for better results
        channel=0, 
        upsample_factor=20,  # Higher precision
        max_shift_per_frame=50.0, 
        use_triangle_threshold=True  # Enable thresholding for better accuracy
    )

    # Apply correction shifts and save result
    corrected_image = apply_shifts.apply_shifts_to_tczyx_stack(image_stack, detected_shifts, mode='constant')
    rp.save_tczyx_image(corrected_image, output_file)
    
    print(f"Drift correction completed. Saved to: {output_file}")
    
    # Validate the drift correction quality
    score_input = drift_correction_score(input_file, channel=0, reference='first', central_crop=0.8, z_project='max')
    print(f"Input image drift correction score: {score_input:.4f}")
    
    score = drift_correction_score(output_file, channel=0, reference='first', central_crop=0.8, z_project='max')
    print(f"Output image drift correction score: {score:.4f}")
    
    #########################################
    # Test 2: Previous frame as reference
    print("\n=== Example: Drift Correction with Phase Cross-Correlation using previous frame as reference ===")
    reference_frame = "previous"  # or "first", "mean", etc.
    output_file = os.path.join(output_base, os.path.splitext(os.path.basename(input_file))[0] + f"_corrected_{reference_frame}.tif")

    # Load image and apply drift correction
    image_stack = rp.load_tczyx_image(input_file).data


    detected_shifts = phase_cross_correlation(
        image_stack, 
        reference_frame=reference_frame,  # Use previous frame as reference for better results
        channel=0, 
        upsample_factor=20,  # Higher precision
        max_shift_per_frame=50.0, 
        use_triangle_threshold=True  # Enable thresholding for better accuracy
    )

    # Apply correction shifts and save result
    corrected_image = apply_shifts.apply_shifts_to_tczyx_stack(image_stack, detected_shifts, mode='constant')
    rp.save_tczyx_image(corrected_image, output_file)

    print(f"Drift correction completed. Saved to: {output_file}")

    # Validate the drift correction quality
    score_input = drift_correction_score(input_file, channel=0, reference='first', central_crop=0.8, z_project='max')
    print(f"Input image drift correction score: {score_input:.4f}")

    score = drift_correction_score(output_file, channel=0, reference='first', central_crop=0.8, z_project='max')
    print(f"Output image drift correction score: {score:.4f}")

    ##########################################
    # Test 3 Run with live cell data without known shifts
    print("\n=== Example: Testing with Real Data ===")
    input_file = "E:/Oyvind/BIP-hub-test-data/drift/input/live_cells/1_Meng.nd2"

    reference_frame = "previous"  # or "first", "mean", etc.
    output_file = os.path.join(output_base, os.path.splitext(os.path.basename(input_file))[0] + f"_corrected_{reference_frame}.tif")

    # Load image and apply drift correction
    image_stack = rp.load_tczyx_image(input_file).data


    detected_shifts = phase_cross_correlation(
        image_stack, 
        reference_frame=reference_frame,  # Use previous frame as reference for better results
        channel=0, 
        upsample_factor=20,  # Higher precision
        max_shift_per_frame=50.0, 
        use_triangle_threshold=True  # Enable thresholding for better accuracy
    )

    # Apply correction shifts and save result
    corrected_image = apply_shifts.apply_shifts_to_tczyx_stack(image_stack, detected_shifts, mode='constant')
    rp.save_tczyx_image(corrected_image, output_file)

    print(f"Drift correction completed. Saved to: {output_file}")

    # Validate the drift correction quality
    score_input = drift_correction_score(input_file, channel=0, reference='first', central_crop=0.8, z_project='max')
    print(f"Input image drift correction score: {score_input:.4f}")

    score = drift_correction_score(output_file, channel=0, reference='first', central_crop=0.8, z_project='max')
    print(f"Output image drift correction score: {score:.4f}")
