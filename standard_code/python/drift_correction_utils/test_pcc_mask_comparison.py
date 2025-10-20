"""
Test comparing different masking strategies for phase cross-correlation.

This script tests whether intensity-based masking improves PCC accuracy compared to:
1. No masking
2. Hanning window (standard approach)
3. Soft intensity mask (percentile-based with Gaussian smoothing)

MIT License
Copyright (c) 2024 BIPHUB - Bioimage Informatics Hub, University of Oslo
"""

import logging
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation

try:
    from .. import bioimage_pipeline_utils as rp
except ImportError:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    import bioimage_pipeline_utils as rp

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_hanning_window(shape: Tuple[int, int]) -> np.ndarray:
    """Create 2D Hanning window (standard PCC approach)."""
    H, W = shape
    wy = np.hanning(H)[:, np.newaxis]
    wx = np.hanning(W)[np.newaxis, :]
    return wy * wx


def create_intensity_soft_mask(
    image: np.ndarray,
    lower_percentile: float = 25.0,
    upper_percentile: float = 99.0,
    sigma: float = 20.0
) -> np.ndarray:
    """
    Create soft intensity-based mask with smooth edges.
    
    Args:
        image: Input image
        lower_percentile: Exclude pixels below this percentile
        upper_percentile: Exclude pixels above this percentile
        sigma: Gaussian smoothing sigma for soft edges (larger = smoother)
    
    Returns:
        Smooth mask with values 0-1
    """
    # Create binary mask
    lower_thresh = np.percentile(image, lower_percentile)
    upper_thresh = np.percentile(image, upper_percentile)
    binary_mask = ((image >= lower_thresh) & (image <= upper_thresh)).astype(np.float32)
    
    # Smooth the mask to avoid hard edges
    soft_mask = gaussian_filter(binary_mask, sigma=sigma)
    
    # Normalize to 0-1
    if soft_mask.max() > 0:
        soft_mask = soft_mask / soft_mask.max()
    
    return soft_mask


def create_intensity_hanning_combined_mask(
    image: np.ndarray,
    lower_percentile: float = 25.0,
    upper_percentile: float = 99.0,
    sigma: float = 20.0
) -> np.ndarray:
    """
    Combine intensity-based masking with Hanning window.
    
    This applies both edge tapering (Hanning) and intensity filtering.
    """
    # Get intensity mask
    intensity_mask = create_intensity_soft_mask(image, lower_percentile, upper_percentile, sigma)
    
    # Get Hanning window
    hanning = create_hanning_window(image.shape)
    
    # Multiply them together
    combined = intensity_mask * hanning
    
    return combined


def test_pcc_with_known_shift(
    reference: np.ndarray,
    shifted: np.ndarray,
    true_shift: Tuple[float, float],
    mask: Optional[np.ndarray] = None,
    mask_name: str = "No mask"
) -> Tuple[np.ndarray, float, float]:
    """
    Test PCC with a known ground truth shift.
    
    Returns:
        (estimated_shift, error_magnitude, correlation_quality)
    """
    # Apply mask if provided
    if mask is not None:
        reference_masked = reference * mask
        shifted_masked = shifted * mask
    else:
        reference_masked = reference
        shifted_masked = shifted
    
    # Compute phase cross-correlation
    shift_estimate, error, phasediff = phase_cross_correlation(
        reference_masked,
        shifted_masked,
        upsample_factor=10,  # High precision for testing
        normalization="phase"
    )
    
    # Calculate error
    error_y = shift_estimate[0] - true_shift[0]
    error_x = shift_estimate[1] - true_shift[1]
    error_magnitude = np.sqrt(error_y**2 + error_x**2)
    
    correlation_quality = 1.0 - error
    
    logger.info(f"{mask_name}:")
    logger.info(f"  True shift: ({true_shift[0]:.3f}, {true_shift[1]:.3f})")
    logger.info(f"  Estimated: ({shift_estimate[0]:.3f}, {shift_estimate[1]:.3f})")
    logger.info(f"  Error: {error_magnitude:.4f} pixels")
    logger.info(f"  Correlation quality: {correlation_quality:.4f}")
    
    return shift_estimate, error_magnitude, correlation_quality


def visualize_masks(image: np.ndarray, masks: dict):
    """Visualize different mask strategies."""
    n_masks = len(masks) + 1
    fig, axes = plt.subplots(2, (n_masks + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()
    
    # Original image
    im0 = axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    
    # Show each mask
    for i, (name, mask) in enumerate(masks.items(), start=1):
        masked_image = image * mask
        im = axes[i].imshow(masked_image, cmap='gray')
        axes[i].set_title(f'{name}\n(mean: {np.mean(mask):.3f})')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046)
    
    # Hide unused subplots
    for i in range(len(masks) + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main test comparing masking strategies."""
    
    # Load test image
    path = r"E:\Oyvind\BIP-hub-test-data\drift\input\test_3\1_Meng_timecrop.tif"
    logger.info(f"Loading image from: {path}")
    
    img = rp.load_tczyx_image(path)
    
    # Get two consecutive frames for realistic test
    frame0 = img.get_image_data("YX", T=0, C=0, Z=0).astype(np.float32)
    frame1 = img.get_image_data("YX", T=1, C=0, Z=0).astype(np.float32)
    
    logger.info(f"Frame shape: {frame0.shape}")
    logger.info(f"Intensity range: [{frame0.min():.1f}, {frame0.max():.1f}]")
    
    # First, compute ground truth shift between consecutive frames
    logger.info("\n" + "="*60)
    logger.info("Computing ground truth shift (high-precision PCC)...")
    logger.info("="*60)
    true_shift, _, _ = phase_cross_correlation(
        frame0, frame1, upsample_factor=100, normalization="phase"
    )
    logger.info(f"Ground truth shift: ({true_shift[0]:.3f}, {true_shift[1]:.3f}) pixels")
    
    # Create different masks
    logger.info("\n" + "="*60)
    logger.info("Creating masks...")
    logger.info("="*60)
    
    masks = {
        'Hanning Window': create_hanning_window(frame0.shape),
        'Soft Intensity (σ=20)': create_intensity_soft_mask(frame0, sigma=20),
        'Soft Intensity (σ=50)': create_intensity_soft_mask(frame0, sigma=50),
        'Hanning + Intensity': create_intensity_hanning_combined_mask(frame0, sigma=30),
    }
    
    # Visualize masks
    visualize_masks(frame0, masks)
    
    # Test each masking strategy
    logger.info("\n" + "="*60)
    logger.info("Testing PCC accuracy with different masks...")
    logger.info("="*60 + "\n")
    
    results = {}
    
    # Test without mask
    shift_est, error, quality = test_pcc_with_known_shift(
        frame0, frame1, true_shift, mask=None, mask_name="No Mask (Baseline)"
    )
    results['No Mask'] = {'shift': shift_est, 'error': error, 'quality': quality}
    logger.info("")
    
    # Test with each mask
    for name, mask in masks.items():
        shift_est, error, quality = test_pcc_with_known_shift(
            frame0, frame1, true_shift, mask=mask, mask_name=name
        )
        results[name] = {'shift': shift_est, 'error': error, 'quality': quality}
        logger.info("")
    
    # Summary comparison
    logger.info("="*60)
    logger.info("SUMMARY: Error Comparison (pixels)")
    logger.info("="*60)
    
    # Sort by error
    sorted_results = sorted(results.items(), key=lambda x: x[1]['error'])
    
    for i, (name, data) in enumerate(sorted_results):
        rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
        logger.info(f"{rank} {name:30s} Error: {data['error']:.4f}px  Quality: {data['quality']:.4f}")
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    names = list(results.keys())
    errors = [results[n]['error'] for n in names]
    qualities = [results[n]['quality'] for n in names]
    
    # Error comparison
    colors = ['red' if n == 'No Mask' else 'green' if i == np.argmin(errors) else 'steelblue' 
              for i, n in enumerate(names)]
    ax1.bar(range(len(names)), errors, color=colors, alpha=0.7)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.set_ylabel('Position Error (pixels)')
    ax1.set_title('PCC Accuracy: Lower is Better')
    ax1.axhline(y=errors[0], color='red', linestyle='--', alpha=0.5, label='Baseline (no mask)')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Quality comparison
    colors = ['red' if n == 'No Mask' else 'green' if i == np.argmax(qualities) else 'steelblue' 
              for i, n in enumerate(names)]
    ax2.bar(range(len(names)), qualities, color=colors, alpha=0.7)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel('Correlation Quality (0-1)')
    ax2.set_title('Correlation Quality: Higher is Better')
    ax2.axhline(y=qualities[0], color='red', linestyle='--', alpha=0.5, label='Baseline (no mask)')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Final recommendation
    logger.info("\n" + "="*60)
    best_name = sorted_results[0][0]
    best_error = sorted_results[0][1]['error']
    baseline_error = results['No Mask']['error']
    improvement = (baseline_error - best_error) / baseline_error * 100
    
    if improvement > 1:
        logger.info(f"✅ RECOMMENDATION: Use '{best_name}'")
        logger.info(f"   Improves accuracy by {improvement:.1f}% over baseline")
    elif improvement < -1:
        logger.info(f"⚠️  WARNING: All masks performed worse than baseline!")
        logger.info(f"   Stick with no mask or Hanning window only")
    else:
        logger.info(f"ℹ️  RESULT: Minimal difference between methods (<1% improvement)")
        logger.info(f"   Use standard Hanning window for computational efficiency")
    logger.info("="*60)


if __name__ == "__main__":
    main()
