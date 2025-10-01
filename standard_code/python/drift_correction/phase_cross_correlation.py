from typing import Optional, Tuple
import sys
import os

from skimage.filters import threshold_triangle#, threshold_otsu

import numpy as np
import cupy as cp

# Universal import fix - add parent directories to sys.path
import sys
import os
from pathlib import Path

# Add standard_code directory to path
current_dir = Path(__file__).parent
standard_code_dir = current_dir.parent
project_root = standard_code_dir.parent
for path in [str(project_root), str(standard_code_dir)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import bioimage_pipeline_utils as rp
from drift_correction.drift_correct_utils import drift_correction_score, gaussian_blur_2d_gpu, apply_shifts_to_tczyx_stack

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # Only show WARNING and above (ERROR, CRITICAL)

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


def phase_cross_correlation(reference: np.ndarray, 
                            image: np.ndarray, 
                            upsample_factor: int = 1, 
                            use_triangle_threshold: bool = True, 
                            gaussian_blur_sigma: float = 0.0) -> Tuple[np.ndarray, float, float]:
    """
    Enhanced CuPy implementation of 3D phase cross-correlation optimized for random drift.
    
    This is the core algorithm that handles the actual FFT-based shift detection.
    Optimized for GPU execution with robust sub-pixel estimation and noise handling.
    
    Args:
        reference: Reference image volume (ZYX or YX)
        image: Image volume to register (ZYX or YX) 
        upsample_factor: Sub-pixel accuracy factor
        use_triangle_threshold: Apply triangle thresholding to focus on high-information pixels
        gaussian_blur_sigma: Gaussian blur preprocessing sigma (0.0 = disabled)
        
    Returns:
        Tuple containing:
        - shift: np.ndarray with [dz, dy, dx] for 3D or [dy, dx] for 2D input
        - error: float, registration error estimate  
        - snr: float, signal-to-noise ratio
    """
    # Validate input
    if reference.ndim != image.ndim:
        raise ValueError("Reference and image must have the same number of dimensions")
    # Handle both 2D and 3D cases
    if reference.ndim == 2:
        # Handle 2D case by adding singleton Z dimension
        reference = reference[np.newaxis, :, :]
        image = image[np.newaxis, :, :]
        is2d = True # When True this dimension will be removed before returning results
    elif reference.ndim == 3:
        is2d = False
    else:
        raise ValueError(f"Input images must be 2D or 3D arrays, got {reference.ndim} and {image.ndim}")

    # ...existing code for processing...
    reference = cp.asarray(reference, dtype=cp.float32)
    image = cp.asarray(image, dtype=cp.float32)
    Z, H, W = reference.shape

    # Apply Gaussian blur preprocessing if requested
    if gaussian_blur_sigma > 0.0:
        logger.debug(f"Applying Gaussian blur preprocessing with sigma={gaussian_blur_sigma}")
        # Convert to CPU, apply blur, then back to GPU for memory efficiency
        reference_cpu = cp.asnumpy(reference)
        image_cpu = cp.asnumpy(image)
        
        # Create temporary 5D arrays for blur function (T=1, C=1, Z, Y, X)
        ref_5d = reference_cpu[np.newaxis, np.newaxis, :, :, :]
        img_5d = image_cpu[np.newaxis, np.newaxis, :, :, :]
        
        # Apply blur using the GPU function from drift_correct_utils
        # Specify channel=0 since we have only one channel (C=1) in the 5D arrays
        ref_blurred_5d = gaussian_blur_2d_gpu(ref_5d, sigma=gaussian_blur_sigma, channel=0)
        img_blurred_5d = gaussian_blur_2d_gpu(img_5d, sigma=gaussian_blur_sigma, channel=0)
        
        # Extract back to 3D and transfer to GPU
        reference = cp.asarray(ref_blurred_5d[0, 0, :, :, :], dtype=cp.float32)
        image = cp.asarray(img_blurred_5d[0, 0, :, :, :], dtype=cp.float32)
        
        logger.debug("Gaussian blur preprocessing completed")

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
        # Handle dimension reduction for 2D case
        if is2d:
            return (cp.asnumpy(cp.asarray([float(py), float(px)])), 
                    1.0 - float(r.max()), 0.0)
        else:
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
    
    # Handle dimension reduction for 2D case
    if is2d:
        result = (cp.asnumpy(cp.asarray([dy_sub, dx_sub])), error, float(snr))
    else:
        result = (cp.asnumpy(cp.asarray([dz_sub, dy_sub, dx_sub])), error, float(snr))
    
    # Clean up GPU memory before returning
    del reference, image, reference_norm, image_norm, F_ref, F_img, R, R_abs, r
    del wz, wy, wx, window, kz, ky, kx, k_mag, whitening_filter
    
    return result

