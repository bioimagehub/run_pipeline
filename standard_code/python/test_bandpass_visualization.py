"""
Bandpass Filter Visualization Test Script

This script helps you visualize the effect of different preprocessing filters
on your drift correction quality. Specifically designed to suppress bright
vesicles while preserving cell body structures for registration.

Usage:
    python test_bandpass_visualization.py

MIT License
Copyright (c) 2024 BIPHUB - Bioimage Informatics Hub, University of Oslo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.filters import difference_of_gaussians
import bioimage_pipeline_utils as rp
from pathlib import Path

# Test data path
TEST_FILE = r"E:\Oyvind\BIP-hub-test-data\drift\input\test_3\1_Meng_timecrop.tif"


def apply_preprocessing(image_2d: np.ndarray, method: str, **kwargs) -> np.ndarray:
    """
    Apply preprocessing filter to 2D image.
    
    Parameters
    ----------
    image_2d : np.ndarray
        Input 2D image (Y, X)
    method : str
        Filter method:
        - 'original': No filtering
        - 'gaussian': Simple Gaussian blur
        - 'bandpass_dog': Difference of Gaussians (bandpass)
        - 'highpass': Remove large-scale variations
        - 'laplacian': Edge detection
    **kwargs : dict
        Method-specific parameters
    
    Returns
    -------
    np.ndarray
        Filtered 2D image
    """
    if method == 'original':
        return image_2d
    
    elif method == 'gaussian':
        sigma = kwargs.get('sigma', 2.0)
        return gaussian_filter(image_2d.astype(np.float32), sigma=sigma)
    
    elif method == 'bandpass_dog':
        low_sigma = kwargs.get('low_sigma', 1.0)
        high_sigma = kwargs.get('high_sigma', 10.0)
        # Difference of Gaussians: suppresses both high and low frequencies
        return difference_of_gaussians(image_2d.astype(np.float32), low_sigma, high_sigma)
    
    elif method == 'highpass':
        sigma = kwargs.get('sigma', 20.0)
        # Subtract smoothed version to remove large-scale variations
        smoothed = gaussian_filter(image_2d.astype(np.float32), sigma=sigma)
        return image_2d.astype(np.float32) - smoothed
    
    elif method == 'laplacian':
        # Laplacian of Gaussian (edge detection)
        from scipy.ndimage import gaussian_laplace
        sigma = kwargs.get('sigma', 2.0)
        return -gaussian_laplace(image_2d.astype(np.float32), sigma=sigma)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def normalize_for_display(img: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    """Normalize image to [0, 1] using percentile clipping for better visualization."""
    if img.size == 0:
        return img
    
    # Handle negative values (from highpass filters)
    vmin = np.percentile(img, 100 - percentile)
    vmax = np.percentile(img, percentile)
    
    if vmax - vmin < 1e-10:
        return np.zeros_like(img)
    
    img_norm = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    return img_norm


def visualize_filters(img: rp.BioImage, timepoint: int = 0, z_slice: int = 0, channel: int = 0):
    """
    Create comparison visualization of different preprocessing filters.
    
    Parameters
    ----------
    img : BioImage
        Input image (TCZYX format)
    timepoint : int
        Which timepoint to visualize
    z_slice : int
        Which Z-slice to visualize (or 0 for 2D)
    channel : int
        Which channel to visualize
    """
    # Extract 2D slice
    T, C, Z, Y, X = img.shape
    
    if timepoint >= T:
        print(f"Warning: timepoint {timepoint} >= {T}, using T=0")
        timepoint = 0
    
    if Z > 1 and z_slice >= Z:
        print(f"Warning: z_slice {z_slice} >= {Z}, using Z=0")
        z_slice = 0
    
    image_2d = img.data[timepoint, channel, z_slice, :, :]
    
    print(f"\nImage info:")
    print(f"  Shape: T={T}, C={C}, Z={Z}, Y={Y}, X={X}")
    print(f"  Dtype: {image_2d.dtype}")
    print(f"  Intensity range: [{image_2d.min():.1f}, {image_2d.max():.1f}]")
    print(f"  Mean: {image_2d.mean():.1f}, Std: {image_2d.std():.1f}")
    
    # Define filter configurations to test
    # Focused comparison: DoG filters with high_sigma=100, varying low_sigma
    filters = [
        {'name': 'Original\n(No filter)', 'method': 'original'},
        {'name': 'DoG (10→100)\nlow_sigma=10', 'method': 'bandpass_dog', 'low_sigma': 10.0, 'high_sigma': 100.0},
        {'name': 'DoG (20→100)\nlow_sigma=20', 'method': 'bandpass_dog', 'low_sigma': 20.0, 'high_sigma': 100.0},
        {'name': 'DoG (30→100)\nlow_sigma=30', 'method': 'bandpass_dog', 'low_sigma': 30.0, 'high_sigma': 100.0},
    ]
    
    # Create figure
    n_filters = len(filters)
    n_cols = 3
    n_rows = int(np.ceil(n_filters / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()
    
    print(f"\nApplying {n_filters} filters...")
    
    for idx, filter_config in enumerate(filters):
        print(f"  {idx+1}/{n_filters}: {filter_config['name'].replace(chr(10), ' ')}")
        
        # Apply filter
        method = filter_config.pop('method')
        name = filter_config.pop('name')
        filtered = apply_preprocessing(image_2d, method, **filter_config)
        
        # Normalize for display
        filtered_norm = normalize_for_display(filtered)
        
        # Display
        axes[idx].imshow(filtered_norm, cmap='gray', interpolation='nearest')
        axes[idx].set_title(name, fontsize=10)
        axes[idx].axis('off')
        
        # Add statistics as text
        stats_text = f"Range: [{filtered.min():.1f}, {filtered.max():.1f}]\nMean: {filtered.mean():.1f}"
        axes[idx].text(5, 25, stats_text, color='yellow', fontsize=8, 
                      bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    # Hide unused subplots
    for idx in range(n_filters, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(__file__).parent / "bandpass_filter_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison to: {output_path}")
    
    plt.show()
    
    return fig


def test_registration_quality(img: rp.BioImage, filter_config: dict, 
                               t1: int = 0, t2: int = 10, channel: int = 0):
    """
    Test how well a filter improves registration between two timepoints.
    
    Shows before/after overlay to visualize drift reduction.
    
    Parameters
    ----------
    img : BioImage
        Input image
    filter_config : dict
        Filter configuration {'method': str, **kwargs}
    t1, t2 : int
        Timepoints to compare
    channel : int
        Channel to use
    """
    from skimage.registration import phase_cross_correlation
    
    T, C, Z, Y, X = img.shape
    
    if t2 >= T:
        print(f"Warning: t2={t2} >= T={T}, using T={T-1}")
        t2 = T - 1
    
    # Get Z-projection (max projection for clarity)
    if Z > 1:
        img1_2d = np.max(img.data[t1, channel, :, :, :], axis=0)
        img2_2d = np.max(img.data[t2, channel, :, :, :], axis=0)
    else:
        img1_2d = img.data[t1, channel, 0, :, :]
        img2_2d = img.data[t2, channel, 0, :, :]
    
    # Apply filter
    method = filter_config.pop('method')
    img1_filtered = apply_preprocessing(img1_2d, method, **filter_config)
    img2_filtered = apply_preprocessing(img2_2d, method, **filter_config)
    
    # Compute shift on filtered images
    shift, error, phasediff = phase_cross_correlation(
        img1_filtered,
        img2_filtered,
        upsample_factor=10
    )
    
    print(f"\n{'='*60}")
    print(f"Registration Quality Test (T={t1} → T={t2})")
    print(f"{'='*60}")
    print(f"Filter: {method}")
    print(f"Detected shift: {shift}")
    print(f"Registration error: {error:.4f} (lower is better)")
    print(f"{'='*60}")
    
    # Create before/after visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original images
    axes[0, 0].imshow(normalize_for_display(img1_2d), cmap='gray')
    axes[0, 0].set_title(f'Original T={t1}')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(normalize_for_display(img2_2d), cmap='gray')
    axes[0, 1].set_title(f'Original T={t2}')
    axes[0, 1].axis('off')
    
    # Overlay (magenta/green)
    overlay_orig = np.zeros((Y, X, 3))
    overlay_orig[:, :, 0] = normalize_for_display(img1_2d)  # Red channel
    overlay_orig[:, :, 1] = normalize_for_display(img2_2d)  # Green channel
    axes[0, 2].imshow(overlay_orig)
    axes[0, 2].set_title('Overlay: T1=Red, T2=Green\n(No registration)')
    axes[0, 2].axis('off')
    
    # Filtered images
    axes[1, 0].imshow(normalize_for_display(img1_filtered), cmap='gray')
    axes[1, 0].set_title(f'Filtered T={t1}')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(normalize_for_display(img2_filtered), cmap='gray')
    axes[1, 1].set_title(f'Filtered T={t2}')
    axes[1, 1].axis('off')
    
    # Overlay after filtering (shows what registration sees)
    overlay_filt = np.zeros((Y, X, 3))
    overlay_filt[:, :, 0] = normalize_for_display(img1_filtered)
    overlay_filt[:, :, 1] = normalize_for_display(img2_filtered)
    axes[1, 2].imshow(overlay_filt)
    axes[1, 2].set_title(f'Filtered Overlay\nShift: {shift}\nError: {error:.4f}')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    output_path = Path(__file__).parent / "registration_quality_test.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved quality test to: {output_path}")
    
    plt.show()
    
    return shift, error


def main():
    """Run the bandpass filter visualization test."""
    print("="*60)
    print("Bandpass Filter Visualization Test")
    print("="*60)
    print(f"Test file: {TEST_FILE}")
    
    # Check if file exists
    if not Path(TEST_FILE).exists():
        print(f"\n✗ Error: File not found: {TEST_FILE}")
        print("Please update TEST_FILE path in the script.")
        return
    
    # Load image
    print("\nLoading image...")
    img = rp.load_tczyx_image(TEST_FILE)
    
    print(f"Loaded: T={img.shape[0]}, C={img.shape[1]}, Z={img.shape[2]}, "
          f"Y={img.shape[3]}, X={img.shape[4]}")
    
    # 1. Visualize different filters
    print("\n" + "="*60)
    print("PART 1: Filter Comparison")
    print("="*60)
    visualize_filters(img, timepoint=5, z_slice=0, channel=0)
    
    # 2. Test registration quality with recommended filter
    print("\n" + "="*60)
    print("PART 2: Registration Quality Test")
    print("="*60)
    print("\nTesting recommended filter (DoG 10→100 for cell tracking)...")
    
    test_registration_quality(
        img,
        filter_config={'method': 'bandpass_dog', 'low_sigma': 10.0, 'high_sigma': 100.0},
        t1=0,
        t2=10,
        channel=0
    )
    
    print("\n" + "="*60)
    print("✓ Test Complete!")
    print("="*60)
    print("\nReview the generated images:")
    print("  1. bandpass_filter_comparison.png - Compare all filter types")
    print("  2. registration_quality_test.png - See registration improvement")
    print("\nBased on the results, choose the filter that:")
    print("  - Suppresses bright vesicles (small spots)")
    print("  - Preserves cell boundaries (medium structures)")
    print("  - Has LOW registration error value")
    print("\nRecommended starting points for your data:")
    print("  - DoG (10→100): Suppress vesicles (~20px), preserve cells (200-500px)")
    print("  - DoG (10→50): More aggressive vesicle suppression")
    print("  - DoG (20→150): Focus on large cells only")
    print("  - Highpass σ=50-100: Remove large-scale intensity gradients")


if __name__ == "__main__":
    import sys
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
