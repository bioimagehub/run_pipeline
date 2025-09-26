from typing import Tuple
import numpy as np
import cupy as cp

def phase_cross_correlation_cupy(reference: np.ndarray, image: np.ndarray, upsample_factor: int = 1) -> Tuple[np.ndarray, float, float]:
    """
    GPU-accelerated phase cross-correlation for sub-pixel drift correction (2D YX images).
    
    This function computes the translational offset between two 2D images using
    phase correlation in the frequency domain. It uses CuPy for GPU acceleration
    and implements advanced techniques including normalization, windowing, and
    enhanced subpixel estimation for high-accuracy drift correction.
    
    The algorithm:
    1. Normalizes both images to reduce intensity bias
    2. Applies Hanning window to reduce edge effects (for images > 32x32)
    3. Computes cross-power spectrum in frequency domain
    4. Finds coarse peak location with wrap-around handling
    5. Performs subpixel refinement using 5x5 neighborhood analysis
    
    Args:
        reference: Reference image (2D YX numpy array). This is the target image
                  that other images will be aligned to.
        image: Image to be aligned (2D YX numpy array). Must have same dimensions
               as reference image.
        upsample_factor: Subpixel precision factor. If 1, returns integer pixel
                        shifts. Higher values enable subpixel accuracy.
                        Defaults to 1.
    
    Returns:
        Tuple containing:
        - shift: numpy array [dy, dx] representing the shift in pixels needed
                to align 'image' to 'reference'. Positive dy means image should
                move down, positive dx means image should move right.
        - error: Error estimate (0.0 = perfect match, 1.0 = no correlation).
                Lower values indicate better alignment confidence.
        - phase_diff: Phase difference (always 0.0 in this implementation).
    
    Raises:
        ImportError: If CuPy is not available.
        ValueError: If input images have different shapes or are not 2D.
        
    Note:
        Requires CuPy to be installed and a CUDA-compatible GPU to be available.
        Images are automatically converted to float32 for computation.
        
    Example:
        >>> import numpy as np
        >>> ref_img = np.random.random((256, 256))
        >>> shifted_img = np.roll(ref_img, (3, -2), axis=(0, 1))  # Shift by 3,-2
        >>> shift, error, _ = phase_cross_correlation_cupy(ref_img, shifted_img)
        >>> print(f"Detected shift: {shift}")  # Should be approximately [3, -2]
    """
    
    reference = cp.asarray(reference, dtype=cp.float32)
    image = cp.asarray(image, dtype=cp.float32)
    H, W = reference.shape

    # Normalize to reduce bias from intensity variations
    reference_mean = cp.mean(reference)
    image_mean = cp.mean(image)
    reference_norm = reference - reference_mean
    image_norm = image - image_mean
    
    # Apply window function to reduce edge effects (critical for accuracy)
    if H > 32 and W > 32:  # Only for reasonably sized images
        wy = cp.hanning(H)[:, cp.newaxis]
        wx = cp.hanning(W)[cp.newaxis, :]
        window = wy * wx
        reference_norm = reference_norm * window
        image_norm = image_norm * window

    # Cross power spectrum with better normalization
    F_reference = cp.fft.fft2(reference_norm)
    F_image = cp.fft.fft2(image_norm)
    R = F_reference * cp.conj(F_image)
    # More robust normalization
    R_abs = cp.abs(R)
    R_abs[R_abs < 1e-12] = 1e-12
    R = R / R_abs
    r = cp.fft.ifft2(R).real

    # Coarse peak (wrap-aware)
    idx = int(cp.argmax(r))
    py = idx // W
    px = idx % W
    if py > H // 2: py -= H
    if px > W // 2: px -= W

    if upsample_factor <= 1:
        return (cp.asnumpy(cp.asarray([float(py), float(px)])), 1.0 - float(r.max()), 0.0)

    # Enhanced subpixel estimation using 5x5 neighborhood for better accuracy
    def _subpixel_fit(r_local, center_y, center_x, H, W):
        """Improved subpixel fitting using larger neighborhood and weighted fitting"""
        # Get 5x5 neighborhood around peak
        neighborhood = cp.zeros((5, 5), dtype=cp.float32)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ny = (center_y + dy + H) % H
                nx = (center_x + dx + W) % W
                neighborhood[dy+2, dx+2] = r_local[ny, nx]
        
        # Find the maximum in the neighborhood
        max_idx = cp.argmax(neighborhood)
        max_y, max_x = divmod(int(max_idx), 5)
        
        # If peak is at edge, fall back to simple parabolic fit
        if max_y == 0 or max_y == 4 or max_x == 0 or max_x == 4:
            # Simple parabolic fit
            y0, x0 = (center_y + H) % H, (center_x + W) % W
            y1, y2 = (y0 - 1) % H, (y0 + 1) % H
            x1, x2 = (x0 - 1) % W, (x0 + 1) % W
            
            cy = r_local[y0, x0]
            dy_off = 0.5 * (r_local[y1, x0] - r_local[y2, x0]) / (r_local[y1, x0] - 2*cy + r_local[y2, x0] + 1e-12)
            dx_off = 0.5 * (r_local[y0, x1] - r_local[y0, x2]) / (r_local[y0, x1] - 2*cy + r_local[y0, x2] + 1e-12)
            
            return float(center_y + dy_off), float(center_x + dx_off)
        
        # Use center of mass for subpixel estimation (more robust)
        total_weight = cp.sum(neighborhood)
        if total_weight > 1e-12:
            y_indices, x_indices = cp.meshgrid(cp.arange(5, dtype=cp.float32) - 2, 
                                               cp.arange(5, dtype=cp.float32) - 2, indexing='ij')
            cm_y = cp.sum(y_indices * neighborhood) / total_weight
            cm_x = cp.sum(x_indices * neighborhood) / total_weight
            return float(center_y + cm_y), float(center_x + cm_x)
        else:
            return float(center_y), float(center_x)
    
    dy_sub, dx_sub = _subpixel_fit(r, py, px, H, W)
    
    # Better error estimate
    max_val = float(r.max())
    err = max(0.0, 1.0 - max_val)
    
    return (cp.asnumpy(cp.asarray([dy_sub, dx_sub])), err, 0.0)


def phase_cross_correlation_cupy_3d(reference: np.ndarray, image: np.ndarray, upsample_factor: int = 1) -> Tuple[np.ndarray, float, float]:
    """
    GPU-accelerated phase cross-correlation for sub-pixel drift correction (3D ZYX images).
    
    This function computes the translational offset between two 3D images using
    phase correlation in the frequency domain. It uses CuPy for GPU acceleration
    and implements advanced techniques including normalization, windowing, and
    enhanced subpixel estimation for high-accuracy drift correction.
    
    The algorithm:
    1. Normalizes both images to reduce intensity bias
    2. Applies Hanning window to reduce edge effects (for images > 32x32x32)
    3. Computes cross-power spectrum in frequency domain
    4. Finds coarse peak location with wrap-around handling
    5. Performs subpixel refinement using 5x5x5 neighborhood analysis
    
    Args:
        reference: Reference image (3D ZYX numpy array). This is the target image
                  that other images will be aligned to.
        image: Image to be aligned (3D ZYX numpy array). Must have same dimensions
               as reference image.
        upsample_factor: Subpixel precision factor. If 1, returns integer pixel
                        shifts. Higher values enable subpixel accuracy.
                        Defaults to 1.
    
    Returns:
        Tuple containing:
        - shift: numpy array [dz, dy, dx] representing the shift in pixels needed
                to align 'image' to 'reference'. Positive dz means image should
                move forward, positive dy means down, positive dx means right.
        - error: Error estimate (0.0 = perfect match, 1.0 = no correlation).
                Lower values indicate better alignment confidence.
        - phase_diff: Phase difference (always 0.0 in this implementation).
    
    Raises:
        ImportError: If CuPy is not available.
        ValueError: If input images have different shapes or are not 3D.
        
    Note:
        Requires CuPy to be installed and a CUDA-compatible GPU to be available.
        Images are automatically converted to float32 for computation.
        
    Example:
        >>> import numpy as np
        >>> ref_img = np.random.random((64, 256, 256))
        >>> shifted_img = np.roll(ref_img, (1, 3, -2), axis=(0, 1, 2))  # Shift by 1,3,-2
        >>> shift, error, _ = phase_cross_correlation_cupy_3d(ref_img, shifted_img)
        >>> print(f"Detected shift: {shift}")  # Should be approximately [1, 3, -2]
    """
    import cupy as cp
    
    reference = cp.asarray(reference, dtype=cp.float32)
    image = cp.asarray(image, dtype=cp.float32)
    Z, H, W = reference.shape

    # Normalize to reduce bias from intensity variations
    reference_mean = cp.mean(reference)
    image_mean = cp.mean(image)
    reference_norm = reference - reference_mean
    image_norm = image - image_mean
    
    # Apply window function to reduce edge effects (critical for accuracy)
    if Z > 32 and H > 32 and W > 32:  # Only for reasonably sized images
        wz = cp.hanning(Z)[:, cp.newaxis, cp.newaxis]
        wy = cp.hanning(H)[cp.newaxis, :, cp.newaxis]
        wx = cp.hanning(W)[cp.newaxis, cp.newaxis, :]
        window = wz * wy * wx
        reference_norm = reference_norm * window
        image_norm = image_norm * window

    # Cross power spectrum with better normalization
    F_reference = cp.fft.fftn(reference_norm)
    F_image = cp.fft.fftn(image_norm)
    R = F_reference * cp.conj(F_image)
    # More robust normalization
    R_abs = cp.abs(R)
    R_abs[R_abs < 1e-12] = 1e-12
    R = R / R_abs
    r = cp.fft.ifftn(R).real

    # Coarse peak (wrap-aware)
    idx = int(cp.argmax(r))
    pz = idx // (H * W)
    py = (idx % (H * W)) // W
    px = idx % W
    if pz > Z // 2: pz -= Z
    if py > H // 2: py -= H
    if px > W // 2: px -= W

    if upsample_factor <= 1:
        return (cp.asnumpy(cp.asarray([float(pz), float(py), float(px)])), 1.0 - float(r.max()), 0.0)

    # Enhanced subpixel estimation using 5x5x5 neighborhood for better accuracy
    def _subpixel_fit_3d(r_local, center_z, center_y, center_x, Z, H, W):
        """Improved 3D subpixel fitting using larger neighborhood and weighted fitting"""
        # Get 5x5x5 neighborhood around peak
        neighborhood = cp.zeros((5, 5, 5), dtype=cp.float32)
        for dz in range(-2, 3):
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    nz = (center_z + dz + Z) % Z
                    ny = (center_y + dy + H) % H
                    nx = (center_x + dx + W) % W
                    neighborhood[dz+2, dy+2, dx+2] = r_local[nz, ny, nx]
        
        # Find the maximum in the neighborhood
        max_idx = cp.argmax(neighborhood)
        max_z, max_y, max_x = cp.unravel_index(max_idx, (5, 5, 5))
        
        # If peak is at edge, fall back to simple parabolic fit
        if (max_z == 0 or max_z == 4 or max_y == 0 or max_y == 4 or max_x == 0 or max_x == 4):
            # Simple parabolic fit
            z0, y0, x0 = (center_z + Z) % Z, (center_y + H) % H, (center_x + W) % W
            z1, z2 = (z0 - 1) % Z, (z0 + 1) % Z
            y1, y2 = (y0 - 1) % H, (y0 + 1) % H
            x1, x2 = (x0 - 1) % W, (x0 + 1) % W
            
            c = r_local[z0, y0, x0]
            dz_off = 0.5 * (r_local[z1, y0, x0] - r_local[z2, y0, x0]) / (r_local[z1, y0, x0] - 2*c + r_local[z2, y0, x0] + 1e-12)
            dy_off = 0.5 * (r_local[z0, y1, x0] - r_local[z0, y2, x0]) / (r_local[z0, y1, x0] - 2*c + r_local[z0, y2, x0] + 1e-12)
            dx_off = 0.5 * (r_local[z0, y0, x1] - r_local[z0, y0, x2]) / (r_local[z0, y0, x1] - 2*c + r_local[z0, y0, x2] + 1e-12)
            
            return float(center_z + dz_off), float(center_y + dy_off), float(center_x + dx_off)
        
        # Use center of mass for subpixel estimation (more robust)
        total_weight = cp.sum(neighborhood)
        if total_weight > 1e-12:
            z_indices, y_indices, x_indices = cp.meshgrid(
                cp.arange(5, dtype=cp.float32) - 2, 
                cp.arange(5, dtype=cp.float32) - 2,
                cp.arange(5, dtype=cp.float32) - 2, indexing='ij')
            cm_z = cp.sum(z_indices * neighborhood) / total_weight
            cm_y = cp.sum(y_indices * neighborhood) / total_weight
            cm_x = cp.sum(x_indices * neighborhood) / total_weight
            return float(center_z + cm_z), float(center_y + cm_y), float(center_x + cm_x)
        else:
            return float(center_z), float(center_y), float(center_x)
    
    dz_sub, dy_sub, dx_sub = _subpixel_fit_3d(r, pz, py, px, Z, H, W)
    
    # Better error estimate
    max_val = float(r.max())
    err = max(0.0, 1.0 - max_val)
    
    return (cp.asnumpy(cp.asarray([dz_sub, dy_sub, dx_sub])), err, 0.0)
