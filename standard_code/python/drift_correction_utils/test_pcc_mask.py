
import logging
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

try:
    from .. import bioimage_pipeline_utils as rp
except ImportError:
    # Fallback for when script is run directly (not as module)

    # Go up to standard_code/python directory to find bioimage_pipeline_utils
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    import bioimage_pipeline_utils as rp

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


def create_intensity_mask(
    image: np.ndarray, 
    lower_percentile: float = 25.0, 
    upper_percentile: float = 99.0
) -> Tuple[np.ndarray, dict]:
    """
    Create a mask that excludes weak background and brightest pixels.
    
    Args:
        image: Input image array (any shape)
        lower_percentile: Percentile threshold for excluding background (0-100)
        upper_percentile: Percentile threshold for excluding brightest pixels (0-100)
    
    Returns:
        Tuple of (binary_mask, stats_dict) where:
            - binary_mask: Boolean array, True for pixels to include
            - stats_dict: Dictionary with threshold statistics
    """
    # Calculate percentile thresholds
    lower_threshold = np.percentile(image, lower_percentile)
    upper_threshold = np.percentile(image, upper_percentile)
    
    # Create mask: include pixels between thresholds
    mask = (image >= lower_threshold) & (image <= upper_threshold)
    
    stats = {
        'lower_percentile': lower_percentile,
        'upper_percentile': upper_percentile,
        'lower_threshold': lower_threshold,
        'upper_threshold': upper_threshold,
        'pixels_included': np.sum(mask),
        'pixels_excluded': np.sum(~mask),
        'fraction_included': np.sum(mask) / mask.size,
        'image_min': np.min(image),
        'image_max': np.max(image),
        'image_mean': np.mean(image),
        'masked_mean': np.mean(image[mask]) if np.any(mask) else 0
    }
    
    logger.info(f"Mask statistics:")
    logger.info(f"  Image range: [{stats['image_min']:.2f}, {stats['image_max']:.2f}]")
    logger.info(f"  Lower threshold ({lower_percentile}th percentile): {lower_threshold:.2f}")
    logger.info(f"  Upper threshold ({upper_percentile}th percentile): {upper_threshold:.2f}")
    logger.info(f"  Pixels included: {stats['pixels_included']:,} ({stats['fraction_included']*100:.1f}%)")
    logger.info(f"  Pixels excluded: {stats['pixels_excluded']:,}")
    logger.info(f"  Mean intensity (all): {stats['image_mean']:.2f}")
    logger.info(f"  Mean intensity (masked): {stats['masked_mean']:.2f}")
    
    return mask, stats


def visualize_mask_overlay(
    image: np.ndarray, 
    mask: np.ndarray, 
    title: str = "Image with Excluded Regions (Red Overlay)"
) -> None:
    """
    Visualize the image with excluded regions shown as red overlay.
    
    Args:
        image: Input image (2D array)
        mask: Binary mask (True = include, False = exclude)
        title: Plot title
    """
    # Normalize image to 0-1 for visualization
    img_norm = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-10)
    
    # Create RGB image
    rgb_image = np.stack([img_norm, img_norm, img_norm], axis=-1)
    
    # Apply red overlay to excluded regions
    excluded_mask = ~mask
    rgb_image[excluded_mask, 0] = 1.0  # Red channel
    rgb_image[excluded_mask, 1] *= 0.3  # Dim green
    rgb_image[excluded_mask, 2] *= 0.3  # Dim blue
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    im0 = axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    
    # Mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title(f'Mask (Included Pixels)\n{np.sum(mask):,} pixels')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(rgb_image)
    axes[2].set_title('Image with Red Overlay on Excluded')
    axes[2].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def main():
    """Main function to test intensity masking with visualization."""
    
    # Load image
    path = r"E:\Oyvind\BIP-hub-test-data\drift\input\test_3\1_Meng_timecrop.tif"
    logger.info(f"Loading image from: {path}")
    
    img = rp.load_tczyx_image(path)
    logger.info(f"Image shape: {img.shape}")
    logger.info(f"Image dimensions: {img.dims.order}")
    
    # Get first timepoint, first channel as 2D image for visualization
    img_2d = img.get_image_data("YX", T=0, C=0, Z=0)
    logger.info(f"Extracted 2D slice shape: {img_2d.shape}")
    
    # Create intensity mask with automatic thresholds
    # Lower percentile removes background, upper percentile removes brightest artifacts
    mask, stats = create_intensity_mask(
        img_2d, 
        lower_percentile=25.0,  # Exclude bottom 25% (background)
        upper_percentile=99.0   # Exclude top 1% (brightest pixels)
    )
    
    # Visualize the result
    visualize_mask_overlay(img_2d, mask)
    
    # Demonstrate applying mask to image
    masked_image = img_2d.copy()
    masked_image[~mask] = 0  # Set excluded pixels to 0
    
    # Show histogram comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(img_2d.ravel(), bins=100, alpha=0.7, label='All pixels', color='blue')
    axes[0].hist(img_2d[mask].ravel(), bins=100, alpha=0.7, label='Included pixels', color='green')
    axes[0].axvline(stats['lower_threshold'], color='red', linestyle='--', label=f'Lower ({stats["lower_percentile"]}%ile)')
    axes[0].axvline(stats['upper_threshold'], color='orange', linestyle='--', label=f'Upper ({stats["upper_percentile"]}%ile)')
    axes[0].set_xlabel('Intensity')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Intensity Distribution')
    axes[0].legend()
    axes[0].set_yscale('log')
    
    axes[1].imshow(masked_image, cmap='gray')
    axes[1].set_title('Masked Image (Excluded = 0)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    logger.info("Test completed successfully!")


if __name__ == "__main__":
    main()



