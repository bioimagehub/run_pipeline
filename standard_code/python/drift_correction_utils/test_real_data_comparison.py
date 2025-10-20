"""
Compare StackReg vs PCC methods on REAL microscopy data.

Uses actual microscopy data from test_3 to compare correction quality.

MIT License
Copyright (c) 2024 BIPHUB - Bioimage Informatics Hub, University of Oslo
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import sys
from pathlib import Path

try:
    from .. import bioimage_pipeline_utils as rp
except ImportError:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    import bioimage_pipeline_utils as rp

from pystackreg import StackReg
from skimage.registration import phase_cross_correlation

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def measure_frame_similarity(frame1: np.ndarray, frame2: np.ndarray) -> dict:
    """Measure similarity between two frames using multiple metrics."""
    from scipy.stats import pearsonr
    
    # Flatten for correlation
    f1_flat = frame1.flatten()
    f2_flat = frame2.flatten()
    
    # Pearson correlation
    correlation, _ = pearsonr(f1_flat, f2_flat)
    
    # Mean squared error
    mse = np.mean((frame1 - frame2) ** 2)
    
    # Mean absolute error
    mae = np.mean(np.abs(frame1 - frame2))
    
    # Structural similarity (simplified)
    ssim_approx = 1.0 - (mse / (np.std(f1_flat) * np.std(f2_flat) + 1e-10))
    
    return {
        'correlation': correlation,
        'mse': mse,
        'mae': mae,
        'ssim_approx': ssim_approx
    }


def measure_temporal_stability(corrected_stack: np.ndarray) -> dict:
    """Measure how stable the corrected stack is over time."""
    n_frames = corrected_stack.shape[0]
    
    # Measure frame-to-frame differences
    frame_diffs = []
    for t in range(n_frames - 1):
        diff = np.mean(np.abs(corrected_stack[t + 1] - corrected_stack[t]))
        frame_diffs.append(diff)
    
    # Measure correlation with first frame
    first_frame = corrected_stack[0]
    correlations = []
    for t in range(1, n_frames):
        metrics = measure_frame_similarity(first_frame, corrected_stack[t])
        correlations.append(metrics['correlation'])
    
    return {
        'mean_frame_diff': np.mean(frame_diffs),
        'std_frame_diff': np.std(frame_diffs),
        'mean_correlation_to_first': np.mean(correlations),
        'std_correlation_to_first': np.std(correlations),
        'frame_diffs': frame_diffs,
        'correlations': correlations
    }


def test_real_data():
    """Compare correction methods on real microscopy data."""
    
    print("="*80)
    print("REAL DATA DRIFT CORRECTION COMPARISON")
    print("="*80)
    
    # Load original data
    input_path = r"E:\Oyvind\BIP-hub-test-data\drift\input\test_3\1_Meng_timecrop.tif"
    
    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        return
    
    print(f"\nLoading original data from: {input_path}")
    img_original = rp.load_tczyx_image(input_path)
    
    # Get 2D max projection for visualization and analysis
    original_data = img_original.get_image_data("TZYX", C=0)
    original_2d = np.max(original_data, axis=1)  # Max project Z
    
    n_frames = original_2d.shape[0]
    print(f"Loaded {n_frames} frames, shape: {original_2d.shape}")
    
    # Load PCC corrected result
    pcc_path = r"E:\Oyvind\BIP-hub-test-data\drift\output\test_3\1_Meng_timecrop_phase_cor_gpu_first.tif"
    
    if not os.path.exists(pcc_path):
        print(f"ERROR: PCC corrected file not found: {pcc_path}")
        print("Please run drift correction first!")
        return
    
    print(f"\nLoading PCC corrected data from: {pcc_path}")
    img_pcc = rp.load_tczyx_image(pcc_path)
    pcc_data = img_pcc.get_image_data("TZYX", C=0)
    pcc_2d = np.max(pcc_data, axis=1)
    
    # Run StackReg correction for comparison
    print(f"\n{'-'*80}")
    print("Running StackReg correction for comparison...")
    print("-"*80)
    
    sr = StackReg(StackReg.TRANSLATION)
    from tqdm import tqdm
    
    print("Computing transformations...")
    tmats = sr.register_stack(original_2d, reference='first')
    
    print("Applying transformations...")
    stackreg_2d = sr.transform_stack(original_2d, tmats=tmats)
    
    # Extract shifts from both methods
    print(f"\n{'-'*80}")
    print("DETECTED SHIFTS")
    print("-"*80)
    
    # StackReg shifts
    stackreg_shifts = []
    print("\nStackReg detected shifts:")
    for t in range(n_frames):
        dy = -tmats[t, 1, 2]
        dx = -tmats[t, 0, 2]
        stackreg_shifts.append((dy, dx))
        if t < 5 or t >= n_frames - 2:  # Show first 5 and last 2
            print(f"  Frame {t:2d}: dy={dy:7.3f}, dx={dx:7.3f}")
        elif t == 5:
            print("  ...")
    
    # PCC shifts (recalculate for comparison)
    pcc_shifts = []
    print("\nPCC detected shifts (upsample_factor=10):")
    reference_frame = original_2d[0]
    for t in range(n_frames):
        if t == 0:
            pcc_shifts.append((0.0, 0.0))
            print(f"  Frame {t:2d}: dy={0.0:7.3f}, dx={0.0:7.3f}")
        else:
            shift, _, _ = phase_cross_correlation(
                reference_frame, original_2d[t],
                upsample_factor=10, normalization="phase"
            )
            pcc_shifts.append((-shift[0], -shift[1]))  # Negate for actual drift
            if t < 5 or t >= n_frames - 2:
                print(f"  Frame {t:2d}: dy={-shift[0]:7.3f}, dx={-shift[1]:7.3f}")
            elif t == 5:
                print("  ...")
    
    # Compare shift agreement
    print(f"\n{'-'*80}")
    print("SHIFT AGREEMENT")
    print("-"*80)
    
    shift_differences = []
    for t in range(n_frames):
        sr_dy, sr_dx = stackreg_shifts[t]
        pcc_dy, pcc_dx = pcc_shifts[t]
        diff = np.sqrt((sr_dy - pcc_dy)**2 + (sr_dx - pcc_dx)**2)
        shift_differences.append(diff)
    
    print(f"Mean shift difference: {np.mean(shift_differences):.4f} pixels")
    print(f"Max shift difference:  {np.max(shift_differences):.4f} pixels")
    print(f"Std shift difference:  {np.std(shift_differences):.4f} pixels")
    
    # Measure temporal stability
    print(f"\n{'-'*80}")
    print("TEMPORAL STABILITY (Lower = Better)")
    print("-"*80)
    
    stability_original = measure_temporal_stability(original_2d)
    stability_pcc = measure_temporal_stability(pcc_2d)
    stability_stackreg = measure_temporal_stability(stackreg_2d)
    
    print("\nFrame-to-frame differences (mean absolute difference):")
    print(f"  Original (uncorrected): {stability_original['mean_frame_diff']:8.2f} ± {stability_original['std_frame_diff']:.2f}")
    print(f"  PCC (upsample=10):      {stability_pcc['mean_frame_diff']:8.2f} ± {stability_pcc['std_frame_diff']:.2f}")
    print(f"  StackReg:               {stability_stackreg['mean_frame_diff']:8.2f} ± {stability_stackreg['std_frame_diff']:.2f}")
    
    print("\nCorrelation with first frame:")
    print(f"  Original (uncorrected): {stability_original['mean_correlation_to_first']:.4f} ± {stability_original['std_correlation_to_first']:.4f}")
    print(f"  PCC (upsample=10):      {stability_pcc['mean_correlation_to_first']:.4f} ± {stability_pcc['std_correlation_to_first']:.4f}")
    print(f"  StackReg:               {stability_stackreg['mean_correlation_to_first']:.4f} ± {stability_stackreg['std_correlation_to_first']:.4f}")
    
    # Visual comparison
    print(f"\n{'-'*80}")
    print("Generating visual comparison...")
    print("-"*80)
    
    fig = plt.figure(figsize=(18, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: Example frames
    example_frame = n_frames // 2
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_2d[example_frame], cmap='gray', vmin=np.percentile(original_2d, 1), 
               vmax=np.percentile(original_2d, 99))
    ax1.set_title(f'Original\nFrame {example_frame}')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(pcc_2d[example_frame], cmap='gray', vmin=np.percentile(pcc_2d, 1), 
               vmax=np.percentile(pcc_2d, 99))
    ax2.set_title(f'PCC Corrected\nFrame {example_frame}')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(stackreg_2d[example_frame], cmap='gray', vmin=np.percentile(stackreg_2d, 1), 
               vmax=np.percentile(stackreg_2d, 99))
    ax3.set_title(f'StackReg Corrected\nFrame {example_frame}')
    ax3.axis('off')
    
    # Row 2: Shift comparison
    ax4 = fig.add_subplot(gs[1, :])
    
    frames_idx = range(n_frames)
    sr_y = [s[0] for s in stackreg_shifts]
    sr_x = [s[1] for s in stackreg_shifts]
    pcc_y = [s[0] for s in pcc_shifts]
    pcc_x = [s[1] for s in pcc_shifts]
    
    ax4.plot(frames_idx, sr_y, 'g-o', label='StackReg Y', markersize=4, linewidth=2)
    ax4.plot(frames_idx, sr_x, 'g--s', label='StackReg X', markersize=4, linewidth=2)
    ax4.plot(frames_idx, pcc_y, 'b-^', label='PCC Y', markersize=4, linewidth=1.5, alpha=0.7)
    ax4.plot(frames_idx, pcc_x, 'b--d', label='PCC X', markersize=4, linewidth=1.5, alpha=0.7)
    
    ax4.set_xlabel('Frame')
    ax4.set_ylabel('Detected Drift (pixels)')
    ax4.set_title('Detected Drift - StackReg vs PCC')
    ax4.legend(loc='best')
    ax4.grid(alpha=0.3)
    
    # Row 3: Temporal stability metrics
    ax5 = fig.add_subplot(gs[2, 0])
    methods = ['Original', 'PCC', 'StackReg']
    frame_diffs = [stability_original['mean_frame_diff'], 
                   stability_pcc['mean_frame_diff'],
                   stability_stackreg['mean_frame_diff']]
    colors = ['red', 'blue', 'green']
    
    bars = ax5.bar(methods, frame_diffs, color=colors, alpha=0.7)
    ax5.set_ylabel('Mean Frame Difference')
    ax5.set_title('Temporal Stability\n(Lower = More Stable)')
    ax5.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, frame_diffs):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    ax6 = fig.add_subplot(gs[2, 1])
    correlations = [stability_original['mean_correlation_to_first'],
                   stability_pcc['mean_correlation_to_first'],
                   stability_stackreg['mean_correlation_to_first']]
    
    bars = ax6.bar(methods, correlations, color=colors, alpha=0.7)
    ax6.set_ylabel('Mean Correlation to First Frame')
    ax6.set_title('Alignment Quality\n(Higher = Better)')
    ax6.set_ylim([min(correlations) - 0.01, 1.0])
    ax6.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, correlations):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.plot(frames_idx[1:], stability_original['frame_diffs'], 'r-', label='Original', linewidth=2, alpha=0.7)
    ax7.plot(frames_idx[1:], stability_pcc['frame_diffs'], 'b-', label='PCC', linewidth=2, alpha=0.7)
    ax7.plot(frames_idx[1:], stability_stackreg['frame_diffs'], 'g-', label='StackReg', linewidth=2, alpha=0.7)
    ax7.set_xlabel('Frame')
    ax7.set_ylabel('Frame-to-Frame Difference')
    ax7.set_title('Temporal Smoothness')
    ax7.legend()
    ax7.grid(alpha=0.3)
    
    plt.savefig('real_data_comparison.png', dpi=150, bbox_inches='tight')
    print("\n[CHART] Saved: real_data_comparison.png")
    plt.show()
    
    # Final summary
    print(f"\n{'='*80}")
    print("SUMMARY - REAL DATA COMPARISON")
    print("="*80)
    
    print(f"\nShift Detection Agreement:")
    print(f"  Mean difference: {np.mean(shift_differences):.4f} pixels")
    if np.mean(shift_differences) < 0.5:
        print(f"  -> Excellent agreement between methods!")
    elif np.mean(shift_differences) < 1.0:
        print(f"  -> Good agreement between methods")
    else:
        print(f"  -> Methods detect somewhat different drifts")
    
    print(f"\nTemporal Stability (frame-to-frame difference):")
    improvement_pcc = (1 - stability_pcc['mean_frame_diff'] / stability_original['mean_frame_diff']) * 100
    improvement_sr = (1 - stability_stackreg['mean_frame_diff'] / stability_original['mean_frame_diff']) * 100
    
    print(f"  PCC improvement:      {improvement_pcc:+.1f}%")
    print(f"  StackReg improvement: {improvement_sr:+.1f}%")
    
    if stability_pcc['mean_frame_diff'] < stability_stackreg['mean_frame_diff']:
        ratio = stability_pcc['mean_frame_diff'] / stability_stackreg['mean_frame_diff']
        print(f"  -> PCC is {(1-ratio)*100:.1f}% more stable than StackReg")
    else:
        ratio = stability_stackreg['mean_frame_diff'] / stability_pcc['mean_frame_diff']
        print(f"  -> StackReg is {(1-ratio)*100:.1f}% more stable than PCC")
    
    print(f"\nCorrelation with Reference:")
    print(f"  Original:  {stability_original['mean_correlation_to_first']:.4f}")
    print(f"  PCC:       {stability_pcc['mean_correlation_to_first']:.4f}")
    print(f"  StackReg:  {stability_stackreg['mean_correlation_to_first']:.4f}")
    
    if stability_pcc['mean_correlation_to_first'] > stability_stackreg['mean_correlation_to_first']:
        print(f"  -> PCC maintains better alignment consistency")
    elif stability_stackreg['mean_correlation_to_first'] > stability_pcc['mean_correlation_to_first']:
        print(f"  -> StackReg maintains better alignment consistency")
    else:
        print(f"  -> Both methods perform similarly")
    
    print("="*80)


if __name__ == "__main__":
    test_real_data()
