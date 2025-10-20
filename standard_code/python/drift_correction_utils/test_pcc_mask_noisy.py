"""
Test masking strategies with added noise to simulate challenging conditions.

Tests whether intensity masking helps PCC when images have:
- High background noise
- Saturated bright pixels
- Low SNR conditions

MIT License
Copyright (c) 2024 BIPHUB - Bioimage Informatics Hub, University of Oslo
"""

import logging
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from scipy.ndimage import shift as scipy_shift, gaussian_filter
from skimage.registration import phase_cross_correlation

try:
    from .. import bioimage_pipeline_utils as rp
except ImportError:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    import bioimage_pipeline_utils as rp

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_soft_intensity_mask(
    image: np.ndarray,
    lower_percentile: float = 25.0,
    upper_percentile: float = 99.0,
    sigma: float = 20.0
) -> np.ndarray:
    """Create soft intensity-based mask."""
    lower_thresh = np.percentile(image, lower_percentile)
    upper_thresh = np.percentile(image, upper_percentile)
    binary_mask = ((image >= lower_thresh) & (image <= upper_thresh)).astype(np.float32)
    soft_mask = gaussian_filter(binary_mask, sigma=sigma)
    if soft_mask.max() > 0:
        soft_mask = soft_mask / soft_mask.max()
    return soft_mask


def add_realistic_noise(
    image: np.ndarray,
    background_noise_level: float = 0.1,
    saturated_pixel_fraction: float = 0.01
) -> np.ndarray:
    """
    Add realistic noise to image.
    
    Args:
        image: Input image
        background_noise_level: Std of Gaussian noise relative to image std
        saturated_pixel_fraction: Fraction of pixels to saturate (bright artifacts)
    
    Returns:
        Noisy image
    """
    noisy = image.copy().astype(np.float32)
    
    # Add background Gaussian noise
    noise_std = np.std(image) * background_noise_level
    noise = np.random.normal(0, noise_std, image.shape)
    noisy += noise
    
    # Add saturated pixels (hot pixels, cosmic rays, etc.)
    n_saturated = int(image.size * saturated_pixel_fraction)
    if n_saturated > 0:
        saturated_coords = np.random.choice(image.size, size=n_saturated, replace=False)
        saturated_mask = np.zeros(image.size, dtype=bool)
        saturated_mask[saturated_coords] = True
        saturated_mask = saturated_mask.reshape(image.shape)
        noisy[saturated_mask] = image.max() * 1.5  # 50% oversaturated
    
    # Clip to valid range
    noisy = np.clip(noisy, 0, None)
    
    return noisy


def test_with_noise_levels():
    """Test masking effectiveness under different noise conditions."""
    
    # Load clean image
    path = r"E:\Oyvind\BIP-hub-test-data\drift\input\test_3\1_Meng_timecrop.tif"
    logger.info(f"Loading image from: {path}")
    
    img = rp.load_tczyx_image(path)
    frame0_clean = img.get_image_data("YX", T=0, C=0, Z=0).astype(np.float32)
    
    # Create known shift
    true_shift = (5.5, -3.2)  # Known shift
    frame1_clean = scipy_shift(frame0_clean, shift=true_shift, order=3, mode='constant', cval=0)
    
    logger.info(f"Frame shape: {frame0_clean.shape}")
    logger.info(f"True shift: {true_shift}")
    
    # Test different noise levels
    noise_levels = [
        (0.0, 0.0, "Clean (no noise)"),
        (0.05, 0.005, "Low noise (5% background, 0.5% saturated)"),
        (0.15, 0.02, "Medium noise (15% background, 2% saturated)"),
        (0.30, 0.05, "High noise (30% background, 5% saturated)"),
    ]
    
    results_no_mask = []
    results_with_mask = []
    
    fig, axes = plt.subplots(len(noise_levels), 3, figsize=(15, 4*len(noise_levels)))
    
    for i, (bg_noise, sat_frac, description) in enumerate(noise_levels):
        logger.info("\n" + "="*70)
        logger.info(f"Testing: {description}")
        logger.info("="*70)
        
        # Add noise
        if bg_noise > 0 or sat_frac > 0:
            np.random.seed(42)  # Reproducible
            frame0_noisy = add_realistic_noise(frame0_clean, bg_noise, sat_frac)
            frame1_noisy = add_realistic_noise(frame1_clean, bg_noise, sat_frac)
        else:
            frame0_noisy = frame0_clean
            frame1_noisy = frame1_clean
        
        # Create mask
        mask = create_soft_intensity_mask(frame0_noisy, lower_percentile=30, upper_percentile=98, sigma=30)
        
        # Test without mask
        shift_no_mask, error_no_mask, _ = phase_cross_correlation(
            frame0_noisy, frame1_noisy, upsample_factor=10, normalization="phase"
        )
        error_mag_no_mask = np.sqrt((shift_no_mask[0] - true_shift[0])**2 + 
                                     (shift_no_mask[1] - true_shift[1])**2)
        
        # Test with mask
        shift_with_mask, error_with_mask, _ = phase_cross_correlation(
            frame0_noisy * mask, frame1_noisy * mask, upsample_factor=10, normalization="phase"
        )
        error_mag_with_mask = np.sqrt((shift_with_mask[0] - true_shift[0])**2 + 
                                       (shift_with_mask[1] - true_shift[1])**2)
        
        results_no_mask.append(error_mag_no_mask)
        results_with_mask.append(error_mag_with_mask)
        
        improvement = (error_mag_no_mask - error_mag_with_mask) / error_mag_no_mask * 100 if error_mag_no_mask > 0 else 0
        
        logger.info(f"Without mask: Error = {error_mag_no_mask:.4f} px")
        logger.info(f"With mask:    Error = {error_mag_with_mask:.4f} px")
        logger.info(f"Improvement: {improvement:+.1f}%")
        
        # Visualize
        axes[i, 0].imshow(frame0_noisy, cmap='gray')
        axes[i, 0].set_title(f'{description}\nNoisy Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask, cmap='viridis')
        axes[i, 1].set_title(f'Soft Intensity Mask\n(mean={np.mean(mask):.3f})')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(frame0_noisy * mask, cmap='gray')
        axes[i, 2].set_title(f'Masked Image\nError: {error_mag_with_mask:.3f}px')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('pcc_mask_noise_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    descriptions = [desc for _, _, desc in noise_levels]
    x = np.arange(len(descriptions))
    width = 0.35
    
    # Error comparison
    bars1 = ax1.bar(x - width/2, results_no_mask, width, label='Without Mask', color='coral', alpha=0.8)
    bars2 = ax1.bar(x + width/2, results_with_mask, width, label='With Intensity Mask', color='steelblue', alpha=0.8)
    ax1.set_ylabel('Position Error (pixels)')
    ax1.set_title('PCC Accuracy Under Noisy Conditions')
    ax1.set_xticks(x)
    ax1.set_xticklabels(descriptions, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Improvement plot
    improvements = [(results_no_mask[i] - results_with_mask[i]) / results_no_mask[i] * 100 
                    if results_no_mask[i] > 0 else 0 
                    for i in range(len(results_no_mask))]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax2.bar(x, improvements, color=colors, alpha=0.7)
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Masking Improvement Over Baseline\n(Positive = Mask is Better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(descriptions, rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pcc_mask_improvement.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("FINAL SUMMARY")
    logger.info("="*70)
    for i, desc in enumerate(descriptions):
        imp = improvements[i]
        symbol = "✅" if imp > 5 else "⚠️" if imp < -5 else "ℹ️"
        logger.info(f"{symbol} {desc:50s} Improvement: {imp:+.1f}%")
    logger.info("="*70)
    
    # Overall recommendation
    avg_improvement = np.mean(improvements[1:])  # Exclude clean case
    if avg_improvement > 5:
        logger.info("\n✅ RECOMMENDATION: Use intensity masking!")
        logger.info(f"   Average improvement in noisy conditions: {avg_improvement:.1f}%")
    elif avg_improvement < -5:
        logger.info("\n❌ RECOMMENDATION: Do NOT use intensity masking!")
        logger.info(f"   Masking degrades accuracy by {-avg_improvement:.1f}% on average")
    else:
        logger.info("\nℹ️  RECOMMENDATION: Masking has minimal impact (<5%)")
        logger.info("   Use standard Hanning window for simplicity")


if __name__ == "__main__":
    test_with_noise_levels()
