"""
Test comparing integer vs subpixel precision drift correction.

This demonstrates how upsample_factor affects correction quality.

MIT License
Copyright (c) 2024 BIPHUB - Bioimage Informatics Hub, University of Oslo
"""

import numpy as np
from scipy.ndimage import shift as scipy_shift
from skimage.registration import phase_cross_correlation
import matplotlib.pyplot as plt


def test_subpixel_accuracy():
    """Test how upsample_factor affects accuracy for subpixel drifts."""
    
    # Create test image
    img = np.zeros((100, 100), dtype=np.float32)
    img[40:60, 40:60] = 255
    
    # Test different true shifts (including subpixel)
    true_shifts = [
        (0.3, 0.2),   # Small subpixel
        (0.7, 0.8),   # Large subpixel (almost 1 pixel)
        (1.3, -0.5),  # Mixed integer + subpixel
        (2.1, 1.9),   # Larger shift with subpixel
    ]
    
    # Test different upsample factors
    upsample_factors = [1, 10, 100]
    
    print("="*80)
    print("SUBPIXEL ACCURACY TEST")
    print("="*80)
    print("\nTesting how upsample_factor affects drift correction accuracy\n")
    
    results = {}
    
    for true_shift_y, true_shift_x in true_shifts:
        print(f"\n{'-'*80}")
        print(f"True shift: ({true_shift_y:.2f}, {true_shift_x:.2f}) pixels")
        print(f"{'-'*80}")
        
        # Create shifted image with subpixel precision
        shifted_img = scipy_shift(img, shift=[true_shift_y, true_shift_x], order=3, mode='constant')
        
        for upsample in upsample_factors:
            # Detect shift
            detected_shift, error, phasediff = phase_cross_correlation(
                img, shifted_img, 
                upsample_factor=upsample,
                normalization="phase"
            )
            
            # Calculate error
            error_y = abs(detected_shift[0] - true_shift_y)
            error_x = abs(detected_shift[1] - true_shift_x)
            total_error = np.sqrt(error_y**2 + error_x**2)
            
            # Correct the image
            corrected = scipy_shift(shifted_img, shift=detected_shift, order=3, mode='constant')
            alignment_error = np.sum(np.abs(img - corrected))
            
            print(f"  upsample={upsample:3d}:  " +
                  f"Detected: ({detected_shift[0]:6.3f}, {detected_shift[1]:6.3f})  " +
                  f"Error: {total_error:6.4f}px  " +
                  f"Alignment: {alignment_error:8.1f}")
            
            results[(true_shift_y, true_shift_x, upsample)] = {
                'detected': detected_shift,
                'error': total_error,
                'alignment': alignment_error
            }
    
    # Visualize results
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Group by upsample factor
    for upsample in upsample_factors:
        errors = [results[(sy, sx, upsample)]['error'] 
                  for sy, sx in true_shifts]
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        print(f"upsample={upsample:3d}:  Mean error={mean_error:.4f}px  Max error={max_error:.4f}px")
    
    # Plot comparison
    fig, axes = plt.subplots(1, len(true_shifts), figsize=(16, 4))
    
    for i, (true_shift_y, true_shift_x) in enumerate(true_shifts):
        ax = axes[i]
        
        # Get errors for this shift across all upsample factors
        errors = [results[(true_shift_y, true_shift_x, u)]['error'] 
                  for u in upsample_factors]
        
        colors = ['red', 'orange', 'green']
        bars = ax.bar(range(len(upsample_factors)), errors, color=colors, alpha=0.7)
        
        ax.set_xticks(range(len(upsample_factors)))
        ax.set_xticklabels([str(u) for u in upsample_factors])
        ax.set_xlabel('Upsample Factor')
        ax.set_ylabel('Position Error (pixels)')
        ax.set_title(f'True shift: ({true_shift_y:.2f}, {true_shift_x:.2f})px')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, err in zip(bars, errors):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{err:.3f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('subpixel_accuracy_test.png', dpi=150, bbox_inches='tight')
    print(f"\n[CHART] Visualization saved to: subpixel_accuracy_test.png")
    plt.show()
    
    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print("For microscopy drift correction:")
    print("  - upsample_factor=1   : [BAD] Integer pixels only, visible jumping")
    print("  - upsample_factor=10  : [GOOD] 0.1px precision, good balance (DEFAULT)")
    print("  - upsample_factor=100 : [BEST] 0.01px precision, slower but smoother")
    print("\nThe current fix sets default upsample_factor=10 for smooth correction!")
    print("="*80)


if __name__ == "__main__":
    test_subpixel_accuracy()
