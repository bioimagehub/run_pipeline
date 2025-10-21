"""
GPU-Optimized Phase Cross-Correlation v3 - Hybrid Implementation

Combines the mathematical approach from v2 (pure Python port) with custom GPU
optimization from v1 for maximum performance while maintaining accuracy.

MIT License
Copyright (c) 2024 BIPHUB - Bioimage Informatics Hub, University of Oslo
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any

import bioimage_pipeline_utils as rp

from scipy.ndimage import shift as scipy_shift
from skimage.registration import phase_cross_correlation

try:
    import cupy as cp
    from cupyx.scipy.ndimage import shift as cupy_shift
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)


def _phase_cross_correlation_cupy(a: np.ndarray, b: np.ndarray, upsample_factor: int = 1) -> Tuple[np.ndarray, float, float]:
    """
    CuPy-accelerated phase correlation - matches scikit-image algorithm.
    Optimized for GPU with proper normalization and subpixel refinement.
    """
    import cupy as cp
    
    # Transfer to GPU
    a_gpu = cp.asarray(a, dtype=cp.float32)
    b_gpu = cp.asarray(b, dtype=cp.float32)
    
    # Get shape
    shape = a_gpu.shape
    if len(shape) == 2:
        H, W = shape
    else:
        D, H, W = shape
    
    # Mean subtraction (matches scikit-image normalization)
    a_norm = a_gpu - cp.mean(a_gpu)
    b_norm = b_gpu - cp.mean(b_gpu)
    
    # FFT-based cross-correlation
    if len(shape) == 2:
        Fa = cp.fft.fft2(a_norm)
        Fb = cp.fft.fft2(b_norm)
    else:
        Fa = cp.fft.fftn(a_norm)
        Fb = cp.fft.fftn(b_norm)
    
    # Cross power spectrum (phase-only correlation)
    R = Fa * cp.conj(Fb)
    R_abs = cp.abs(R)
    R_abs[R_abs < 1e-12] = 1e-12
    R = R / R_abs
    
    # Inverse FFT to get correlation
    if len(shape) == 2:
        r = cp.fft.ifft2(R).real
    else:
        r = cp.fft.ifftn(R).real
    
    # Find coarse peak (integer shift)
    idx = int(cp.argmax(r))
    
    if len(shape) == 2:
        py = idx // W
        px = idx % W
        # Handle wrap-around
        if py > H // 2: py -= H
        if px > W // 2: px -= W
        shift_coarse = [py, px]
    else:
        pz = idx // (H * W)
        py = (idx % (H * W)) // W
        px = idx % W
        if pz > D // 2: pz -= D
        if py > H // 2: py -= H
        if px > W // 2: px -= W
        shift_coarse = [pz, py, px]
    
    if upsample_factor <= 1:
        result = cp.asnumpy(cp.asarray(shift_coarse, dtype=cp.float64))
        return (result, 1.0 - float(r.max()), 0.0)
    
    # For subpixel refinement, use DFT upsampling (matches scikit-image)
    # This is more accurate than center-of-mass
    from skimage.registration._phase_cross_correlation import _upsampled_dft
    
    # Transfer back to CPU for DFT upsampling (scikit-image function)
    r_cpu = cp.asnumpy(r)
    R_cpu = cp.asnumpy(R)
    
    # Use scikit-image's upsampled DFT method for subpixel refinement
    if len(shape) == 2:
        # Create upsampled region around peak
        shifts = np.array(shift_coarse, dtype=np.float64)
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        dftshift = np.fix(upsampled_region_size / 2.0)
        
        sample_region_offset = dftshift - shifts * upsample_factor
        
        # Compute upsampled DFT
        cross_correlation = _upsampled_dft(
            R_cpu.conj(),
            upsampled_region_size,
            upsample_factor,
            sample_region_offset
        ).conj()
        
        # Find maximum in upsampled region
        maxima = np.unravel_index(np.argmax(np.abs(cross_correlation)), cross_correlation.shape)
        maxima = np.array(maxima, dtype=np.float64)
        
        # Compute shift
        shifts = shifts + (maxima - dftshift) / upsample_factor
        
        result = shifts
    else:
        # For 3D, use simpler refinement
        result = np.array(shift_coarse, dtype=np.float64)
    
    max_val = float(r_cpu.max())
    err = max(0.0, 1.0 - max_val)
    
    return (result, err, 0.0)


def compute_shift(img1: np.ndarray, img2: np.ndarray, upsample_factor: int = 10, use_gpu: bool = False) -> np.ndarray:
    """
    Compute translation shift between two images using phase cross-correlation.
    Returns shift as (dy, dx) for 2D or (dz, dy, dx) for 3D.
    """
    if use_gpu and GPU_AVAILABLE:
        shift, error, phasediff = _phase_cross_correlation_cupy(img1, img2, upsample_factor)
        return shift
    else:
        shift, error, phasediff = phase_cross_correlation(img1, img2, upsample_factor=upsample_factor)
        return shift


def extract_frame(img: np.ndarray, t: int, c: int, z_min: int, z_max: int) -> np.ndarray:
    """Extract frame (timepoint, channel, z-range) from TCZYX image."""
    return img[t, c, z_min:z_max+1, :, :]


def add_Point3f(p1, p2):
    return [p1[0]+p2[0], p1[1]+p2[1], p1[2]+p2[2]]


def subtract_Point3f(p1, p2):
    return [p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2]]


def compute_and_update_frame_translations_dt(
    img: np.ndarray,
    dt: int,
    options: Dict[str, Any],
    shifts: Optional[List[List[float]]] = None
) -> List[List[float]]:
    """
    Compute translations between frames separated by dt.
    Matches v2 logic exactly.
    """
    nt = img.shape[0]
    c = options.get('channel', 0)
    z_min = options.get('z_min', 0)
    z_max = options.get('z_max', img.shape[2]-1)
    upsample_factor = options.get('upsample_factor', 10)
    max_shifts = options.get('max_shifts', [20, 20, 5])
    correct_only_xy = options.get('correct_only_xy', True)
    use_gpu = options.get('use_gpu', False)
    
    if shifts is None:
        shifts = [[0.0, 0.0, 0.0] for _ in range(nt)]
    
    for t in range(dt, nt+dt, dt):
        if t > nt-1:
            t = nt-1
        
        frame1 = extract_frame(img, t-dt, c, z_min, z_max)
        frame2 = extract_frame(img, t, c, z_min, z_max)
        
        # Project to 2D if XY-only correction
        if correct_only_xy or frame1.shape[0] == 1:
            frame1_proj = np.max(frame1, axis=0) if frame1.shape[0] > 1 else frame1[0]
            frame2_proj = np.max(frame2, axis=0) if frame2.shape[0] > 1 else frame2[0]
            shift = compute_shift(frame2_proj, frame1_proj, upsample_factor, use_gpu)
            shift3 = [shift[1], shift[0], 0.0]  # (dy, dx) -> (x, y, z)
        else:
            shift = compute_shift(frame2, frame1, upsample_factor, use_gpu)
            shift3 = [shift[2], shift[1], shift[0]]
        
        # Limit shifts
        for d in range(3):
            if shift3[d] > max_shifts[d]:
                shift3[d] = max_shifts[d]
            if shift3[d] < -max_shifts[d]:
                shift3[d] = -max_shifts[d]
        
        # Update shifts (interpolation across time range)
        local_shift = subtract_Point3f(shifts[t], shifts[t-dt])
        add_shift = subtract_Point3f(shift3, local_shift)
        for i, tt in enumerate(range(t-dt, nt)):
            for d in range(3):
                shifts[tt][d] += (i/dt) * add_shift[d]
    
    return shifts


def invert_shifts(shifts: List[List[float]]) -> List[List[float]]:
    """Invert shifts for correction application."""
    return [[-s[0], -s[1], -s[2]] for s in shifts]


def compute_min_max(shifts):
    """Compute min/max shifts in each dimension."""
    arr = np.array(shifts)
    minx, miny, minz = np.min(arr, axis=0)
    maxx, maxy, maxz = np.max(arr, axis=0)
    return int(minx), int(miny), int(minz), int(maxx), int(maxy), int(maxz)


def register_hyperstack_subpixel(
    img: np.ndarray,
    shifts: List[List[float]],
    fill_value: float = 0.0,
    use_gpu: bool = False
) -> np.ndarray:
    """
    Apply subpixel shifts to TCZYX image.
    GPU-accelerated version with proper canvas expansion.
    """
    T, C, Z, Y, X = img.shape
    minx, miny, minz, maxx, maxy, maxz = compute_min_max(shifts)
    width = X + int(round(maxx - minx))
    height = Y + int(round(maxy - miny))
    slices = Z + int(round(maxz - minz))
    
    out = np.full((T, C, slices, height, width), fill_value, dtype=img.dtype)
    
    use_cupy = use_gpu and GPU_AVAILABLE
    
    if use_cupy:
        logger.info("Using GPU for image transformation...")
    
    for t in range(T):
        sx, sy, sz = shifts[t]
        sx -= minx
        sy -= miny
        sz -= minz
        
        sx_int = int(sx)
        sy_int = int(sy)
        sx_frac = sx - sx_int
        sy_frac = sy - sy_int
        
        for c in range(C):
            for z in range(Z):
                z_out = int(round(z + sz))
                if 0 <= z_out < slices:
                    if use_cupy:
                        arr = cp.asarray(img[t, c, z, :, :])
                        shifted = cupy_shift(arr, shift=[sy_frac, sx_frac], order=1, mode='constant', cval=fill_value)
                        shifted_np = cp.asnumpy(shifted)
                    else:
                        shifted_np = scipy_shift(img[t, c, z, :, :], shift=[sy_frac, sx_frac], order=1, mode='constant', cval=fill_value)
                    
                    out[t, c, z_out, sy_int:sy_int+Y, sx_int:sx_int+X] = shifted_np
    
    return out


def register_image_xy(
    img,
    reference: str = 'first',
    channel: int = 0,
    show_progress: bool = True,
    no_gpu: bool = False,
    crop_fraction: float = 1.0,
    upsample_factor: int = 10,
    max_shift: float = 50.0
) -> Tuple[Any, np.ndarray]:
    """
    Register TCZYX image using translation in XY - GPU-optimized v3.
    
    Compatible wrapper for drift_correction.py integration.
    Combines v2 mathematical approach with v1 GPU optimization.
    """
    img_data = img.data
    T = img_data.shape[0]
    
    options = dict(
        channel=channel,
        z_min=0,
        z_max=img_data.shape[2] - 1,
        upsample_factor=upsample_factor,
        max_shifts=[max_shift, max_shift, 5],
        correct_only_xy=True,
        use_gpu=not no_gpu
    )
    
    logger.info(f"Computing drift shifts (v3 GPU-optimized) using {reference} reference...")
    
    # Compute shifts
    dt = 1
    shifts = compute_and_update_frame_translations_dt(img_data, dt, options)
    
    # Invert for correction
    shifts = invert_shifts(shifts)
    
    logger.info("Applying shifts to image stack...")
    registered_data = register_hyperstack_subpixel(img_data, shifts, fill_value=0.0, use_gpu=not no_gpu)
    
    registered_img = rp.BioImage(
        registered_data,
        physical_pixel_sizes=img.physical_pixel_sizes,
        channel_names=img.channel_names,
        metadata=img.metadata
    )
    
    tmats = np.array(shifts, dtype=np.float32)
    
    return registered_img, tmats


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    parser = argparse.ArgumentParser(description="Drift correction v3 - GPU optimized")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("-o", "--output", help="Output image path", default=None)
    parser.add_argument("-c", "--channel", type=int, default=0, help="Channel index")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")
    args = parser.parse_args()
    
    img = rp.load_tczyx_image(args.input)
    registered, shifts = register_image_xy(img, reference='first', channel=args.channel, no_gpu=not args.gpu)
    
    if args.output:
        registered.save(args.output)
        logger.info(f"Saved to: {args.output}")
