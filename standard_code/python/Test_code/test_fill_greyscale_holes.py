"""
Test script for greyscale hole filling functionality.

Creates synthetic test images to demonstrate hole filling:
1. Nucleus with nucleolus (2D)
2. Nucleus Z-stack with nucleoli (3D)

Author: BIPHUB - Bioimage Informatics Hub, University of Oslo
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add standard_code to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from python.fill_greyscale_holes import greyscale_fill_holes_2d, greyscale_fill_holes_3d


def create_test_nucleus_2d(size=256, nucleus_intensity=200, nucleolus_intensity=50):
    """Create synthetic nucleus with nucleolus."""
    image = np.zeros((size, size), dtype=np.uint8)
    
    # Create nucleus (circular region)
    y, x = np.ogrid[:size, :size]
    center_y, center_x = size // 2, size // 2
    radius = size // 3
    
    # Nucleus mask
    nucleus_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    image[nucleus_mask] = nucleus_intensity
    
    # Add nucleolus (smaller dark circle inside)
    nucleolus_radius = radius // 4
    nucleolus_offset = radius // 3
    nucleolus_center_y = center_y - nucleolus_offset
    nucleolus_center_x = center_x
    nucleolus_mask = (x - nucleolus_center_x)**2 + (y - nucleolus_center_y)**2 <= nucleolus_radius**2
    image[nucleolus_mask] = nucleolus_intensity
    
    # Add some noise for realism
    noise = np.random.normal(0, 10, image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    return image


def create_test_nucleus_3d(size=128, depth=20, nucleus_intensity=200, nucleolus_intensity=50):
    """Create synthetic 3D nucleus Z-stack with nucleoli."""
    image = np.zeros((depth, size, size), dtype=np.uint8)
    
    for z in range(depth):
        # Vary nucleus size through Z to create 3D sphere
        z_factor = 1 - ((z - depth/2) / (depth/2))**2
        if z_factor < 0:
            z_factor = 0
        
        current_radius = int((size // 3) * np.sqrt(z_factor))
        
        if current_radius > 0:
            y, x = np.ogrid[:size, :size]
            center_y, center_x = size // 2, size // 2
            
            # Nucleus
            nucleus_mask = (x - center_x)**2 + (y - center_y)**2 <= current_radius**2
            image[z, nucleus_mask] = nucleus_intensity
            
            # Nucleolus (smaller sphere inside)
            nucleolus_radius = max(1, current_radius // 4)
            nucleolus_offset = current_radius // 3
            nucleolus_center_y = center_y - nucleolus_offset
            nucleolus_mask = (x - center_x)**2 + (y - nucleolus_center_y)**2 <= nucleolus_radius**2
            image[z, nucleolus_mask] = nucleolus_intensity
            
            # Add noise
            noise = np.random.normal(0, 10, (size, size))
            image[z] = np.clip(image[z] + noise, 0, 255).astype(np.uint8)
    
    return image


def test_2d_filling():
    """Test 2D hole filling and visualize results."""
    print("Testing 2D greyscale hole filling...")
    
    # Create test image
    nucleus = create_test_nucleus_2d()
    
    # Fill holes
    filled = greyscale_fill_holes_2d(nucleus)
    
    # Calculate difference
    difference = filled.astype(int) - nucleus.astype(int)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(nucleus, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original (with nucleolus hole)')
    axes[0].axis('off')
    
    axes[1].imshow(filled, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('Filled (nucleolus filled)')
    axes[1].axis('off')
    
    im = axes[2].imshow(difference, cmap='RdBu_r', vmin=-50, vmax=50)
    axes[2].set_title('Difference (what was filled)')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('test_2d_hole_filling.png', dpi=150, bbox_inches='tight')
    print("  Saved: test_2d_hole_filling.png")
    plt.close()
    
    # Print statistics
    filled_pixels = np.sum(difference > 10)
    print(f"  Filled {filled_pixels} pixels in nucleolus region")
    print(f"  Original nucleolus mean intensity: {nucleus[difference > 10].mean():.1f}")
    print(f"  Filled nucleolus mean intensity: {filled[difference > 10].mean():.1f}")


def test_3d_filling():
    """Test 3D hole filling and visualize results."""
    print("\nTesting 3D greyscale hole filling...")
    
    # Create test 3D image
    nucleus_stack = create_test_nucleus_3d()
    
    # Fill holes
    filled_stack = greyscale_fill_holes_3d(nucleus_stack)
    
    # Calculate difference
    difference = filled_stack.astype(int) - nucleus_stack.astype(int)
    
    # Visualize middle slice
    mid_z = nucleus_stack.shape[0] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(nucleus_stack[mid_z], cmap='gray', vmin=0, vmax=255)
    axes[0].set_title(f'Original (Z={mid_z})')
    axes[0].axis('off')
    
    axes[1].imshow(filled_stack[mid_z], cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(f'Filled (Z={mid_z})')
    axes[1].axis('off')
    
    im = axes[2].imshow(difference[mid_z], cmap='RdBu_r', vmin=-50, vmax=50)
    axes[2].set_title(f'Difference (Z={mid_z})')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('test_3d_hole_filling.png', dpi=150, bbox_inches='tight')
    print("  Saved: test_3d_hole_filling.png")
    plt.close()
    
    # Print statistics
    filled_pixels = np.sum(difference > 10)
    print(f"  Filled {filled_pixels} voxels in 3D nucleolus region")
    print(f"  Original nucleolus mean intensity: {nucleus_stack[difference > 10].mean():.1f}")
    print(f"  Filled nucleolus mean intensity: {filled_stack[difference > 10].mean():.1f}")


def test_comparison_with_binary():
    """Compare greyscale fill with binary fill to show the difference."""
    print("\nComparing greyscale fill vs binary fill...")
    
    from scipy.ndimage import binary_fill_holes
    
    # Create test image
    nucleus = create_test_nucleus_2d()
    
    # Greyscale fill
    greyscale_filled = greyscale_fill_holes_2d(nucleus)
    
    # Binary fill (what happens if you use binary_fill_holes on greyscale)
    binary_mask = nucleus > 0  # Convert to binary
    binary_filled = binary_fill_holes(binary_mask)
    binary_filled_uint8 = (binary_filled * 255).astype(np.uint8)
    
    # Visualize comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(nucleus, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Original Greyscale')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(greyscale_filled, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title('Greyscale Fill (preserves intensity)')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(binary_mask, cmap='gray')
    axes[1, 0].set_title('Binary Conversion (loses intensity info)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(binary_filled_uint8, cmap='gray', vmin=0, vmax=255)
    axes[1, 1].set_title('Binary Fill (all white, no intensity info)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_greyscale_vs_binary_fill.png', dpi=150, bbox_inches='tight')
    print("  Saved: test_greyscale_vs_binary_fill.png")
    plt.close()
    
    # Print key insight
    print("\n  KEY DIFFERENCE:")
    print(f"    Greyscale fill: Mean intensity in filled region = {greyscale_filled[greyscale_filled > 0].mean():.1f}")
    print(f"    Binary fill: All pixels = 255 (loses original intensity distribution)")
    print("    → Use greyscale fill for quantitative analysis!")


if __name__ == "__main__":
    print("="*70)
    print("Testing Greyscale Hole Filling Module")
    print("="*70)
    
    test_2d_filling()
    test_3d_filling()
    test_comparison_with_binary()
    
    print("\n" + "="*70)
    print("All tests complete! Check the generated PNG files.")
    print("="*70)
