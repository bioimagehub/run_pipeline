"""
Final comparison: StackReg vs PCC (with and without upsampling).

Tests on synthetic data with known subpixel drifts to compare:
1. StackReg (no upsampling needed - optimization-based)
2. PCC with upsample_factor=1 (integer only - causes jumping)
3. PCC with upsample_factor=10 (subpixel - should match StackReg)

MIT License
Copyright (c) 2024 BIPHUB - Bioimage Informatics Hub, University of Oslo
"""

import numpy as np
from scipy.ndimage import shift as scipy_shift
from pystackreg import StackReg
from skimage.registration import phase_cross_correlation
import matplotlib.pyplot as plt


def create_synthetic_drift_sequence():
    """Create a time series with known subpixel drift."""
    # Create base image
    img = np.zeros((128, 128), dtype=np.float32)
    
    # Add some features
    img[50:70, 50:70] = 200  # Square
    img[30:40, 80:90] = 150  # Small square
    img[90:100, 30:45] = 180  # Rectangle
    
    # Add some noise
    img += np.random.normal(0, 5, img.shape)
    img = np.clip(img, 0, 255)
    
    # Create drift sequence with SUBPIXEL drifts
    n_frames = 10
    frames = np.zeros((n_frames, 128, 128), dtype=np.float32)
    
    # Gradual drift (realistic subpixel motion)
    true_drifts = []
    cumulative_y = 0.0
    cumulative_x = 0.0
    
    for t in range(n_frames):
        # Small random drift per frame (subpixel)
        drift_y = np.random.uniform(-0.3, 0.5)
        drift_x = np.random.uniform(-0.4, 0.4)
        
        cumulative_y += drift_y
        cumulative_x += drift_x
        
        true_drifts.append((cumulative_y, cumulative_x))
        
        # Apply drift
        frames[t] = scipy_shift(img, shift=[cumulative_y, cumulative_x],
                               order=3, mode='constant', cval=0)
    
    return frames, true_drifts, img


def test_correction_methods():
    """Compare StackReg vs PCC with different upsample factors."""
    
    print("="*80)
    print("DRIFT CORRECTION METHOD COMPARISON")
    print("="*80)
    print("\nGenerating synthetic data with known subpixel drifts...")
    
    frames, true_drifts, reference = create_synthetic_drift_sequence()
    n_frames = frames.shape[0]
    
    print(f"Created {n_frames} frames with realistic subpixel drift")
    print("\nTrue cumulative drifts:")
    for t, (dy, dx) in enumerate(true_drifts):
        print(f"  Frame {t}: ({dy:6.3f}, {dx:6.3f}) pixels")
    
    # Method 1: StackReg
    print(f"\n{'-'*80}")
    print("METHOD 1: StackReg (optimization-based, automatic subpixel)")
    print("-"*80)
    
    sr = StackReg(StackReg.TRANSLATION)
    tmats_stackreg = sr.register_stack(frames, reference='first')
    corrected_stackreg = sr.transform_stack(frames, tmats=tmats_stackreg)
    
    # Extract shift vectors from transformation matrices
    shifts_stackreg = []
    for t in range(n_frames):
        # StackReg tmat represents forward transform, so negate for actual drift
        dy = -tmats_stackreg[t, 1, 2]
        dx = -tmats_stackreg[t, 0, 2]
        shifts_stackreg.append((dy, dx))
        print(f"  Frame {t}: detected ({dy:6.3f}, {dx:6.3f})")
    
    # Calculate error
    errors_stackreg = []
    for t, ((true_y, true_x), (det_y, det_x)) in enumerate(zip(true_drifts, shifts_stackreg)):
        error = np.sqrt((true_y - det_y)**2 + (true_x - det_x)**2)
        errors_stackreg.append(error)
    
    mean_error_stackreg = np.mean(errors_stackreg)
    print(f"\nMean detection error: {mean_error_stackreg:.4f} pixels")
    
    # Method 2: PCC with upsample_factor=1 (integer only)
    print(f"\n{'-'*80}")
    print("METHOD 2: PCC with upsample_factor=1 (INTEGER PIXELS ONLY)")
    print("-"*80)
    
    shifts_pcc_int = []
    corrected_pcc_int = np.zeros_like(frames)
    
    for t in range(n_frames):
        shift, _, _ = phase_cross_correlation(
            reference, frames[t], upsample_factor=1, normalization="phase"
        )
        shifts_pcc_int.append(shift)
        corrected_pcc_int[t] = scipy_shift(frames[t], shift=shift, 
                                           order=3, mode='constant', cval=0)
        print(f"  Frame {t}: detected ({shift[0]:6.3f}, {shift[1]:6.3f})")
    
    errors_pcc_int = []
    for t, ((true_y, true_x), shift) in enumerate(zip(true_drifts, shifts_pcc_int)):
        det_y, det_x = -shift[0], -shift[1]  # Negate because PCC shift is opposite
        error = np.sqrt((true_y - det_y)**2 + (true_x - det_x)**2)
        errors_pcc_int.append(error)
    
    mean_error_pcc_int = np.mean(errors_pcc_int)
    print(f"\nMean detection error: {mean_error_pcc_int:.4f} pixels")
    
    # Method 3: PCC with upsample_factor=10 (subpixel)
    print(f"\n{'-'*80}")
    print("METHOD 3: PCC with upsample_factor=10 (SUBPIXEL - 0.1px precision)")
    print("-"*80)
    
    shifts_pcc_sub = []
    corrected_pcc_sub = np.zeros_like(frames)
    
    for t in range(n_frames):
        shift, _, _ = phase_cross_correlation(
            reference, frames[t], upsample_factor=10, normalization="phase"
        )
        shifts_pcc_sub.append(shift)
        corrected_pcc_sub[t] = scipy_shift(frames[t], shift=shift,
                                           order=3, mode='constant', cval=0)
        print(f"  Frame {t}: detected ({shift[0]:6.3f}, {shift[1]:6.3f})")
    
    errors_pcc_sub = []
    for t, ((true_y, true_x), shift) in enumerate(zip(true_drifts, shifts_pcc_sub)):
        det_y, det_x = -shift[0], -shift[1]
        error = np.sqrt((true_y - det_y)**2 + (true_x - det_x)**2)
        errors_pcc_sub.append(error)
    
    mean_error_pcc_sub = np.mean(errors_pcc_sub)
    print(f"\nMean detection error: {mean_error_pcc_sub:.4f} pixels")
    
    # Calculate alignment quality for each method
    print(f"\n{'-'*80}")
    print("ALIGNMENT QUALITY (how well frames match reference)")
    print("-"*80)
    
    alignment_stackreg = np.mean([np.sum(np.abs(reference - corrected_stackreg[t])) 
                                  for t in range(n_frames)])
    alignment_pcc_int = np.mean([np.sum(np.abs(reference - corrected_pcc_int[t])) 
                                for t in range(n_frames)])
    alignment_pcc_sub = np.mean([np.sum(np.abs(reference - corrected_pcc_sub[t])) 
                                for t in range(n_frames)])
    
    print(f"StackReg:            {alignment_stackreg:10.1f}")
    print(f"PCC (upsample=1):    {alignment_pcc_int:10.1f}  [{(alignment_pcc_int/alignment_stackreg*100):.1f}% of StackReg]")
    print(f"PCC (upsample=10):   {alignment_pcc_sub:10.1f}  [{(alignment_pcc_sub/alignment_stackreg*100):.1f}% of StackReg]")
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Detected drifts comparison
    ax = axes[0, 0]
    true_y = [d[0] for d in true_drifts]
    true_x = [d[1] for d in true_drifts]
    
    sr_y = [d[0] for d in shifts_stackreg]
    sr_x = [d[1] for d in shifts_stackreg]
    
    pcc_int_y = [-s[0] for s in shifts_pcc_int]
    pcc_int_x = [-s[1] for s in shifts_pcc_int]
    
    pcc_sub_y = [-s[0] for s in shifts_pcc_sub]
    pcc_sub_x = [-s[1] for s in shifts_pcc_sub]
    
    frames_idx = range(n_frames)
    ax.plot(frames_idx, true_y, 'k-o', label='True drift (Y)', linewidth=2, markersize=8)
    ax.plot(frames_idx, sr_y, 'g--s', label='StackReg (Y)', markersize=6)
    ax.plot(frames_idx, pcc_int_y, 'r:^', label='PCC upsample=1 (Y)', markersize=6)
    ax.plot(frames_idx, pcc_sub_y, 'b-.d', label='PCC upsample=10 (Y)', markersize=6)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Drift (pixels)')
    ax.set_title('Detected Drift - Y Dimension')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Error comparison
    ax = axes[0, 1]
    ax.bar(['StackReg', 'PCC\nupsample=1', 'PCC\nupsample=10'],
           [mean_error_stackreg, mean_error_pcc_int, mean_error_pcc_sub],
           color=['green', 'red', 'blue'], alpha=0.7)
    ax.set_ylabel('Mean Detection Error (pixels)')
    ax.set_title('Detection Accuracy Comparison')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate([mean_error_stackreg, mean_error_pcc_int, mean_error_pcc_sub]):
        ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Example corrected frames
    ax = axes[1, 0]
    example_frame = 5
    ax.imshow(np.hstack([
        corrected_stackreg[example_frame],
        corrected_pcc_int[example_frame],
        corrected_pcc_sub[example_frame]
    ]), cmap='gray')
    ax.set_title(f'Corrected Frame {example_frame}\nStackReg | PCC(1) | PCC(10)')
    ax.axis('off')
    
    # Plot 4: Alignment quality
    ax = axes[1, 1]
    ax.bar(['StackReg', 'PCC\nupsample=1', 'PCC\nupsample=10'],
           [alignment_stackreg, alignment_pcc_int, alignment_pcc_sub],
           color=['green', 'red', 'blue'], alpha=0.7)
    ax.set_ylabel('Mean Alignment Error (intensity sum)')
    ax.set_title('Correction Quality (Lower = Better)')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stackreg_vs_pcc_final_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n[CHART] Saved: stackreg_vs_pcc_final_comparison.png")
    plt.show()
    
    # Final conclusions
    print(f"\n{'='*80}")
    print("CONCLUSIONS")
    print("="*80)
    print("\n1. StackReg uses OPTIMIZATION-based registration:")
    print("   - Automatically achieves subpixel precision")
    print("   - No explicit upsampling parameter needed")
    print(f"   - Detection error: {mean_error_stackreg:.4f} pixels")
    
    print("\n2. PCC with upsample_factor=1 (integer only):")
    print("   - Rounds all drifts to nearest pixel")
    print("   - Causes 'jumping' artifacts in corrected images")
    print(f"   - Detection error: {mean_error_pcc_int:.4f} pixels [{mean_error_pcc_int/mean_error_stackreg:.1f}x worse]")
    
    print("\n3. PCC with upsample_factor=10 (subpixel):")
    print("   - Achieves 0.1 pixel precision via FFT upsampling")
    print("   - Comparable accuracy to StackReg!")
    print(f"   - Detection error: {mean_error_pcc_sub:.4f} pixels [{mean_error_pcc_sub/mean_error_stackreg:.1f}x StackReg]")
    
    print("\n" + "="*80)
    print("RECOMMENDATION:")
    print("="*80)
    print("For PCC drift correction, ALWAYS use upsample_factor >= 10")
    print("This gives comparable results to StackReg without jumping artifacts!")
    print("="*80)


if __name__ == "__main__":
    test_correction_methods()
