"""
Phase Cross-Correlation Drift Correction - Translation Only

Simple phase cross-correlation implementation for drift correction using scipy.

MIT License
Copyright (c) 2024 BIPHUB - Bioimage Informatics Hub, University of Oslo
"""

import logging
from typing import Tuple, Literal
import numpy as np
from bioio import BioImage
from scipy.ndimage import shift as scipy_shift
from skimage.registration import phase_cross_correlation
from tqdm import tqdm
import sys
import os


# Use relative import to parent directory
try:
    from .. import bioimage_pipeline_utils as rp
except ImportError:
    # Fallback for when script is run directly (not as module)

    # Go up to standard_code/python directory to find bioimage_pipeline_utils
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    import bioimage_pipeline_utils as rp

logger = logging.getLogger(__name__)


def _crop_center(data: np.ndarray, crop_fraction: float) -> np.ndarray:
    """Crop edges of image stack, keeping center pixels.
    
    Args:
        data: Input array with shape (T, Y, X)
        crop_fraction: Fraction to keep (e.g., 0.8 keeps center 80%)
        
    Returns:
        Cropped array centered on original
    """
    if crop_fraction >= 1.0:
        return data
    
    T, H, W = data.shape
    crop_h = int(H * crop_fraction)
    crop_w = int(W * crop_fraction)
    
    # Calculate crop boundaries centered on image
    start_h = (H - crop_h) // 2
    start_w = (W - crop_w) // 2
    end_h = start_h + crop_h
    end_w = start_w + crop_w
    
    cropped = data[:, start_h:end_h, start_w:end_w]
    logger.info(f"Cropped from ({H}, {W}) to ({crop_h}, {crop_w}) for faster registration")
    
    return cropped


def _register_image_xy_cpu(
        img: BioImage,
        reference: Literal['first', 'previous', 'median'] = 'first',
        channel: int = 0,
        show_progress: bool = True,
        crop_fraction: float = 1.0,
        upsample_factor: int = 10
        ) -> Tuple[BioImage, np.ndarray]: 
    '''Register a TCZYX image using translation in XY dimensions only.
    
    Performs drift correction by computing transformations using phase cross-correlation
    from a max-projected reference channel, then applying those transformations to all 
    channels in the full 3D stack. Uses subpixel-accurate shifts via upsampling.
    
    The Z dimension is max-projected for registration calculation only, but the
    full 3D volume is transformed and returned.
    
    Args:
        img: BioImage object containing TCZYX image data. Must have at least
            2 timepoints for registration to be meaningful.
        reference: Registration reference strategy:
            - 'first': Register all frames to the first timepoint
            - 'previous': Register each frame to its previous frame (default)
            - 'median': Register all frames to the temporal median image
        channel: Zero-indexed channel to use for computing transformations.
            Typically choose the brightest or most stable channel.
        show_progress: Whether to display progress bars (default: True).
        no_gpu: Placeholder for API compatibility (not used in this implementation).
        crop_fraction: Fraction of image to use for registration (default: 1.0).
            Values < 1.0 crop edges to speed up registration (e.g., 0.8 uses
            center 80% of image). Cropping preserves center pixel alignment,
            ensuring shift values remain accurate. Only applied to registration
            calculation, not to output image.
        upsample_factor: Subpixel precision factor (default: 10). Higher values
            give better accuracy but slower computation. 1 = integer pixels only,
            10 = 0.1 pixel precision, 100 = 0.01 pixel precision.
    
    Returns:
        Tuple containing:
            - registered (BioImage): Drift-corrected BioImage object with 5D 
              TCZYX data and preserved metadata
            - shifts (np.ndarray): Shift vectors for each timepoint,
              shape (T, 2) for Y and X dimensions. Compatible with PyStackReg
              transformation matrix format for drop-in replacement.
    
    Raises:
        ValueError: If image has insufficient dimensions or invalid channel index
        
    Example:
        >>> img = rp.load_tczyx_image("timelapse.tif")
        >>> registered, shifts = register_image_xy(img, reference='first', channel=0)
        >>> registered.save("corrected.tif")
        
    Note:
        For images with only 1 timepoint, returns original BioImage unchanged
        with zero shift vectors.
    '''
    logger.info(f"Starting XY drift correction (phase cross-correlation) with reference='{reference}', channel={channel}")
    
    # Extract reference channel as 4D TZYX array
    ref_channel_data = img.get_image_data("TZYX", C=channel)

    # Max projection over Z (axis=1) to reduce to 2D for registration
    ref_channel_data = np.max(ref_channel_data, axis=1, keepdims=False)  # Shape: (T, Y, X)

    # Crop edges for faster registration if requested
    ref_channel_data = _crop_center(ref_channel_data, crop_fraction)

    # Verify we have multiple timepoints
    if ref_channel_data.shape[0] <= 1:
        logger.warning("Image has only 1 timepoint, returning original data")
        return img, np.zeros((1, 2))
    
    n_frames = ref_channel_data.shape[0]
    
    logger.info(f"Computing shifts from max-projected stack with shape {ref_channel_data.shape} using reference '{reference}'")
    
    # Compute reference image based on strategy
    if reference == 'first':
        ref_frame = ref_channel_data[0]
    elif reference == 'median':
        ref_frame = np.median(ref_channel_data, axis=0)
        logger.info("Using median projection as reference")
    elif reference == 'previous':
        ref_frame = None  # Will be updated in loop
    else:
        raise ValueError(f"Unknown reference strategy: {reference}")
    
    # Compute shifts for each frame
    if show_progress:
        iterator = tqdm(range(n_frames), desc="Finding shifts", unit="frame")
    else:
        iterator = range(n_frames)
    
    shifts = np.zeros((n_frames, 2), dtype=np.float64)
    cumulative_shift = np.zeros(2, dtype=np.float64)
    
    for t in iterator:
        if t == 0:
            # First frame has zero shift
            shifts[t] = 0
            if reference == 'previous':
                ref_frame = ref_channel_data[0]
            continue
        
        # Update reference for "previous" mode
        if reference == 'previous':
            ref_frame = ref_channel_data[t - 1]
        
        # Compute shift with subpixel accuracy
        shift, error, phasediff = phase_cross_correlation(
            ref_frame,
            ref_channel_data[t],
            upsample_factor=upsample_factor,
            space="real",
            normalization="phase"
        )
        
        if reference == 'previous':
            # Accumulate shifts for "previous" mode
            cumulative_shift += shift
            shifts[t] = cumulative_shift
        else:
            shifts[t] = shift

    logger.info(f"Computed {n_frames} shifts - Mean: {np.mean(shifts, axis=0)}, Max: {np.max(np.abs(shifts)):.2f} px")

    # Release reference channel memory
    del ref_channel_data

    # Load entire image dataset (5D TCZYX)
    img_data = img.data

    # Apply the transformations to all channels and Z-slices
    logger.info(f"Applying transformations to full stack with shape {img_data.shape}")
    registered_data = np.zeros_like(img_data)
    
    # Setup progress tracking
    if show_progress:
        pbar = tqdm(total=img_data.shape[1] * img_data.shape[0], desc="Applying shifts", unit="frame")
    
    for c in range(img_data.shape[1]):  # Loop over channels
        if show_progress:
            pbar.set_description(f"Applying shifts C={c}") # pyright: ignore[reportPossiblyUnboundVariable] # 
        else:
            logger.info(f"Transforming channel {c}/{img_data.shape[1]-1}")
        
        channel_data = img.get_image_data("TZYX", C=c)
        
        # Apply transformation to each Z-slice and timepoint
        for z in range(channel_data.shape[1]):  # Loop over Z
            for t in range(channel_data.shape[0]):  # Loop over time
                registered_data[t, c, z, :, :] = scipy_shift(
                    channel_data[t, z, :, :],
                    shift=[shifts[t, 0], shifts[t, 1]],
                    order=1,
                    mode='constant',
                    cval=0
                )
                if show_progress and z == 0:  # Update once per frame (only on first Z-slice)
                    pbar.update(1) # pyright: ignore[reportPossiblyUnboundVariable] # 
    
    if show_progress:
        pbar.close() # pyright: ignore[reportPossiblyUnboundVariable] # 

    logger.info("XY drift correction completed successfully")

    # Create BioImage with registered data and preserved metadata
    img_registered = BioImage(
        registered_data,
        physical_pixel_sizes=img.physical_pixel_sizes,
        channel_names=img.channel_names,
        metadata=img.metadata
    )

    return img_registered, shifts


def _phase_cross_correlation_cupy(a: np.ndarray, b: np.ndarray, upsample_factor: int = 1) -> Tuple[np.ndarray, float, float]:
    """CuPy-accelerated phase correlation implementation (accepts numpy or cupy arrays)."""
    import cupy as cp
    
    # Check if already on GPU (optimization: avoid redundant transfers)
    if isinstance(a, cp.ndarray):
        a_gpu = a.astype(cp.float32) if a.dtype != cp.float32 else a
    else:
        a_gpu = cp.asarray(a, dtype=cp.float32)
    
    if isinstance(b, cp.ndarray):
        b_gpu = b.astype(cp.float32) if b.dtype != cp.float32 else b
    else:
        b_gpu = cp.asarray(b, dtype=cp.float32)
    
    H, W = a_gpu.shape[-2:]  # Support both 2D and 3D

    # Normalize to reduce bias from intensity variations
    a_mean = cp.mean(a_gpu)
    b_mean = cp.mean(b_gpu)
    a_norm = a_gpu - a_mean
    b_norm = b_gpu - b_mean
    
    # Apply window function to reduce edge effects (critical for accuracy)
    if H > 32 and W > 32:  # Only for reasonably sized images
        wy = cp.hanning(H)
        wx = cp.hanning(W)
        if a_gpu.ndim == 2:
            wy = wy[:, cp.newaxis]
            wx = wx[cp.newaxis, :]
            window = wy * wx
            a_norm = a_norm * window
            b_norm = b_norm * window
        else:  # 3D case
            D = a_gpu.shape[0]
            wz = cp.hanning(D)
            # Create 3D window
            window = wz[:, cp.newaxis, cp.newaxis] * wy[cp.newaxis, :, cp.newaxis] * wx[cp.newaxis, cp.newaxis, :]
            a_norm = a_norm * window
            b_norm = b_norm * window

    # Cross power spectrum with better normalization
    if a_gpu.ndim == 2:
        Fa = cp.fft.fft2(a_norm)
        Fb = cp.fft.fft2(b_norm)
    else:  # 3D
        Fa = cp.fft.fftn(a_norm)
        Fb = cp.fft.fftn(b_norm)
    
    R = Fa * cp.conj(Fb)
    # More robust normalization
    R_abs = cp.abs(R)
    R_abs[R_abs < 1e-12] = 1e-12
    R = R / R_abs
    
    if a.ndim == 2:
        r = cp.fft.ifft2(R).real
    else:
        r = cp.fft.ifftn(R).real

    # Coarse peak (wrap-aware)
    idx = int(cp.argmax(r))
    shape = r.shape
    if a_gpu.ndim == 2:
        py = idx // W
        px = idx % W
        if py > H // 2: py -= H
        if px > W // 2: px -= W
        shift_coarse = [py, px]
    else:  # 3D
        D, H, W = shape
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

    # Enhanced subpixel estimation using 5x5 neighborhood for better accuracy
    def _improved_subpixel_fit_2d(r_local, center_y, center_x, H, W):
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
    
    if a_gpu.ndim == 2:
        py, px = shift_coarse
        dy_sub, dx_sub = _improved_subpixel_fit_2d(r, py, px, H, W)
        shift_result = [dy_sub, dx_sub]
    else:  # 3D - use simple parabolic fit for Z
        pz, py, px = shift_coarse
        dy_sub, dx_sub = _improved_subpixel_fit_2d(r[pz], py, px, H, W)
        # Simple parabolic fit for Z
        D = r.shape[0]
        z0 = (pz + D) % D
        z1, z2 = (z0 - 1) % D, (z0 + 1) % D
        cz = cp.max(r[z0])
        dz_off = 0.5 * (cp.max(r[z1]) - cp.max(r[z2])) / (cp.max(r[z1]) - 2*cz + cp.max(r[z2]) + 1e-12)
        dz_sub = float(pz + dz_off)
        shift_result = [dz_sub, dy_sub, dx_sub]
    
    # Better error estimate
    max_val = float(r.max())
    err = max(0.0, 1.0 - max_val)
    
    result = cp.asnumpy(cp.asarray(shift_result, dtype=cp.float64))
    return (result, err, 0.0)


def _register_image_xy_gpu(
        img: BioImage,
        reference: Literal['first', 'previous', 'median'] = 'first',
        channel: int = 0,
        show_progress: bool = True,
        crop_fraction: float = 1.0,
        upsample_factor: int = 10
        ) -> Tuple[BioImage, np.ndarray]:
    '''
    GPU-accelerated registration using CuPy with optimized memory management.
    
    Optimization Strategy:
        1. Transfer entire stack to GPU once at start
        2. Compute all shifts on GPU (data never leaves GPU)
        3. Apply all shifts on GPU (data never leaves GPU)
        4. Transfer corrected stack back to CPU once at end
        Result: Minimal CPU↔GPU transfers = Maximum performance
    '''
    import cupy as cp
    from cupyx.scipy.ndimage import shift as cupy_shift
    
    logger.info(f"Starting XY drift correction (GPU/CuPy) with reference='{reference}', channel={channel}")
    logger.info("Using CuPy GPU acceleration for phase cross-correlation")
    
    # Extract reference channel as 4D TZYX array
    ref_channel_data = img.get_image_data("TZYX", C=channel)

    # Max projection over Z (axis=1) to reduce to 2D for registration
    ref_channel_data = np.max(ref_channel_data, axis=1, keepdims=False)  # Shape: (T, Y, X)

    # Crop edges for faster registration if requested (on CPU before GPU transfer)
    ref_channel_data = _crop_center(ref_channel_data, crop_fraction)

    # Verify we have multiple timepoints
    if ref_channel_data.shape[0] <= 1:
        logger.warning("Image has only 1 timepoint, returning original data")
        return img, np.zeros((1, 2))
    
    n_frames = ref_channel_data.shape[0]
    
    logger.info(f"Computing shifts from max-projected stack with shape {ref_channel_data.shape} using reference '{reference}'")
    
    # Transfer reference stack to GPU once (major optimization!)
    logger.info(f"Transferring {ref_channel_data.nbytes / 1e9:.2f} GB reference stack to GPU...")
    ref_channel_gpu = cp.asarray(ref_channel_data, dtype=cp.float32)
    logger.info("GPU transfer complete")
    
    # Compute reference image based on strategy (on GPU!)
    if reference == 'first':
        ref_frame_gpu = ref_channel_gpu[0]
    elif reference == 'median':
        ref_frame_gpu = cp.median(ref_channel_gpu, axis=0)
        logger.info("Using median projection as reference (computed on GPU)")
    elif reference == 'previous':
        ref_frame_gpu = ref_channel_gpu[0]  # Initialize with first frame
    else:
        raise ValueError(f"Unknown reference strategy: {reference}")
    
    # Compute shifts
    if show_progress:
        iterator = tqdm(range(n_frames), desc="Finding shifts (GPU)", unit="frame")
    else:
        iterator = range(n_frames)
    
    shifts = np.zeros((n_frames, 2), dtype=np.float64)
    cumulative_shift = np.zeros(2, dtype=np.float64)
    
    for t in iterator:
        if t == 0:
            # First frame has zero shift
            shifts[t] = 0
            continue
        
        # Update reference for "previous" mode (on GPU!)
        if reference == 'previous':
            ref_frame_gpu = ref_channel_gpu[t - 1]
        
        # Compute shift on GPU (data stays on GPU - no transfer!)
        current_frame_gpu = ref_channel_gpu[t]
        
        # Pass GPU arrays directly (optimized!) with subpixel accuracy
        shift, error, phasediff = _phase_cross_correlation_cupy(
            ref_frame_gpu,
            current_frame_gpu,
            upsample_factor=upsample_factor
        )
        
        if reference == 'previous':
            # Accumulate shifts for "previous" mode
            cumulative_shift += shift
            shifts[t] = cumulative_shift
        else:
            shifts[t] = shift

    logger.info(f"Computed {n_frames} shifts - Mean: {np.mean(shifts, axis=0)}, Max: {np.max(np.abs(shifts)):.2f} px")

    # Clean up GPU memory after shift computation
    del ref_channel_gpu, ref_frame_gpu
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    # Load entire image dataset (5D TCZYX)
    img_data = img.data

    # Apply the transformations to all channels and Z-slices
    logger.info(f"Applying transformations to full stack with shape {img_data.shape}")
    logger.info(f"Transferring {img_data.nbytes / 1e9:.2f} GB full stack to GPU...")
    
    # Transfer full stack to GPU once
    img_data_gpu = cp.asarray(img_data, dtype=cp.float32)
    logger.info("GPU transfer complete")
    
    # Preallocate output on GPU
    registered_data_gpu = cp.zeros_like(img_data_gpu)
    
    # Setup progress tracking
    if show_progress:
        pbar = tqdm(total=img_data.shape[1] * img_data.shape[0], desc="Applying shifts (GPU)", unit="frame")
    
    for c in range(img_data.shape[1]):  # Loop over channels
        if show_progress:
            pbar.set_description(f"Applying shifts (GPU) C={c}")  # pyright: ignore[reportPossiblyUnboundVariable]
        else:
            logger.info(f"Transforming channel {c}/{img_data.shape[1]-1}")
        
        # Apply transformation to each Z-slice and timepoint (all on GPU!)
        for z in range(img_data.shape[2]):  # Loop over Z
            for t in range(img_data.shape[0]):  # Loop over time
                registered_data_gpu[t, c, z, :, :] = cupy_shift(
                    img_data_gpu[t, c, z, :, :],
                    shift=[shifts[t, 0], shifts[t, 1]],
                    order=1,  # Cubic interpolation for better quality
                    mode='constant',
                    cval=0.0
                )
                if show_progress and z == 0:  # Update once per frame (only on first Z-slice)
                    pbar.update(1)  # pyright: ignore[reportPossiblyUnboundVariable]
            
            # Periodic cleanup to prevent memory fragmentation
            if z % 5 == 0 and z > 0:
                cp.get_default_memory_pool().free_all_blocks()
    
    if show_progress:
        pbar.close()  # pyright: ignore[reportPossiblyUnboundVariable]

    logger.info("Transferring corrected stack from GPU to CPU...")
    # Transfer final result back to CPU
    registered_data = cp.asnumpy(registered_data_gpu).astype(img_data.dtype)
    
    # Final GPU memory cleanup
    del img_data_gpu, registered_data_gpu
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    
    logger.info("XY drift correction completed successfully")

    # Create BioImage with registered data and preserved metadata
    img_registered = BioImage(
        registered_data,
        physical_pixel_sizes=img.physical_pixel_sizes,
        channel_names=img.channel_names,
        metadata=img.metadata
    )

    return img_registered, shifts


def register_image_xy(
        img: BioImage,
        reference: Literal['first', 'previous', 'median'] = 'first',
        channel: int = 0,
        show_progress: bool = True,
        no_gpu: bool = False,
        crop_fraction: float = 1.0,
        upsample_factor: int = 10
        ) -> Tuple[BioImage, np.ndarray]:
    '''Register a TCZYX image using translation in XY dimensions only.
    
    Performs drift correction by computing transformations using phase cross-correlation
    from a max-projected reference channel, then applying those transformations to all 
    channels in the full 3D stack. Automatically uses GPU acceleration if available.
    
    Args:
        img: BioImage object containing TCZYX image data.
        reference: Registration reference strategy ('first', 'previous', or 'median').
        channel: Zero-indexed channel to use for computing transformations.
        show_progress: Whether to display progress bars (default: True).
        no_gpu: Force CPU execution even if GPU is available (default: False).
        crop_fraction: Fraction of image to use for registration (default: 1.0).
        upsample_factor: Subpixel precision factor (default: 10). Higher = more accurate
            but slower. 1 = integer pixels, 10 = 0.1px precision, 100 = 0.01px precision.
    
    Returns:
        Tuple containing registered BioImage and shift vectors (T, 2) for Y and X.
    '''
    if no_gpu:
        return _register_image_xy_cpu(img, reference, channel, show_progress, crop_fraction, upsample_factor)
    else:
        try:
            import cupy as cp
            # Verify GPU is available
            _ = cp.cuda.Device(0)
            return _register_image_xy_gpu(img, reference, channel, show_progress, crop_fraction, upsample_factor)
        except (ImportError, RuntimeError) as e:
            logger.warning(f"GPU not available ({e}), falling back to CPU")
            return _register_image_xy_cpu(img, reference, channel, show_progress, crop_fraction, upsample_factor)


def test_code():
    # Configure logging to show INFO messages
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    path = r"E:\Oyvind\BIP-hub-test-data\drift\input\test_2\1_Meng_timecrop_template_rolled-t15.tif"
    img = rp.load_tczyx_image(path)
    registered, shifts = register_image_xy(img, reference='previous', channel=0, no_gpu=False, crop_fraction=0.8)

    #np.save(r"E:\Oyvind\BIP-hub-test-data\drift\input\test_2\1_Meng_timecrop_template_rolled-t15_PCC_shifts.npy", shifts)
    outpath = r"E:\Oyvind\BIP-hub-test-data\drift\input\test_2\1_Meng_timecrop_template_rolled-t15_PCC_corrected.tif"
     # delete previous output files if they exist
    
    if os.path.exists(outpath):
        os.remove(outpath)
    registered.save(outpath)


if __name__ == "__main__":
    test_code()
