"""
Comprehensive test comparing Phase Cross-Correlation vs StackReg drift correction.

Tests real microscopy data to validate accuracy and performance.
"""

import numpy as np
import logging
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
try:
    from .. import bioimage_pipeline_utils as rp
except ImportError:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    import bioimage_pipeline_utils as rp

from phase_cross_correlation import register_image_xy
from pystackreg import StackReg
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def register_with_stackreg(img, reference='first', channel=0):
    """Run StackReg registration for comparison."""
    logger.info("Running StackReg registration...")
    
    # Extract reference channel as 4D TZYX array
    ref_channel_data = img.get_image_data("TZYX", C=channel)
    
    # Max projection over Z to get 2D for registration
    ref_channel_2d = np.max(ref_channel_data, axis=1, keepdims=False)  # Shape: (T, Y, X)
    
    # Initialize StackReg
    sr = StackReg(StackReg.TRANSLATION)
    
    # Register stack
    logger.info("Computing transformations...")
    tmats = sr.register_stack(ref_channel_2d, reference=reference)
    
    # Extract shifts from transformation matrices
    n_frames = ref_channel_2d.shape[0]
    shifts = np.zeros((n_frames, 2), dtype=np.float64)
    
    for t in range(n_frames):
        # StackReg tmat represents forward transform, negate for actual drift
        shifts[t, 0] = -tmats[t, 1, 2]  # Y shift
        shifts[t, 1] = -tmats[t, 0, 2]  # X shift
    
    # Apply transformations to full stack
    logger.info("Applying transformations to full stack...")
    img_data = img.data
    registered_data = np.zeros_like(img_data)
    
    pbar = tqdm(total=img_data.shape[1] * img_data.shape[0], desc="Applying shifts", unit="frame")
    
    for c in range(img_data.shape[1]):  # Loop over channels
        pbar.set_description(f"Applying shifts C={c}")
        channel_data = img.get_image_data("TZYX", C=c)
        
        for z in range(channel_data.shape[1]):  # Loop over Z
            z_slice = channel_data[:, z, :, :]  # Extract TYX slice
            registered_data[:, c, z, :, :] = sr.transform_stack(z_slice, tmats=tmats)
            if z == 0:
                pbar.update(z_slice.shape[0])
    
    pbar.close()
    
    # Create BioImage with registered data
    from bioio import BioImage
    img_registered = BioImage(
        registered_data,
        physical_pixel_sizes=img.physical_pixel_sizes,
        channel_names=img.channel_names,
        metadata=img.metadata
    )
    
    return img_registered, shifts


def compare_methods_real_data():
    """Test: Compare methods on real microscopy data"""
    logger.info("\n" + "="*80)
    logger.info("TEST: Real Microscopy Data Comparison")
    logger.info("="*80)
    
    # Load real data (original uncorrected data)
    data_path = Path(r"E:\Oyvind\BIP-hub-test-data\drift\input\test_3\1_Meng_timecrop.tif")
    
    if not data_path.exists():
        logger.error(f"Test data not found: {data_path}")
        logger.info("Please check the file path and try again.")
        return
    
    logger.info(f"Loading real data from: {data_path}")
    img = rp.load_tczyx_image(str(data_path))
    logger.info(f"Loaded image shape: {img.shape}")
    
    # Test PCC with different upsample factors
    logger.info("\n" + "-"*80)
    logger.info("METHOD 1: Phase Cross-Correlation (upsample=1 - INTEGER ONLY)")
    logger.info("-"*80)
    img_pcc_int, shifts_pcc_int = register_image_xy(
        img, reference='first', channel=0, 
        show_progress=True, no_gpu=False, 
        crop_fraction=0.8, upsample_factor=1
    )
    
    logger.info("\n" + "-"*80)
    logger.info("METHOD 2: Phase Cross-Correlation (upsample=10 - SUBPIXEL)")
    logger.info("-"*80)
    img_pcc_sub, shifts_pcc_sub = register_image_xy(
        img, reference='first', channel=0,
        show_progress=True, no_gpu=False,
        crop_fraction=0.8, upsample_factor=10
    )
    
    logger.info("\n" + "-"*80)
    logger.info("METHOD 3: StackReg Translation")
    logger.info("-"*80)
    img_stackreg, shifts_stackreg = register_with_stackreg(
        img, reference='first', channel=0
    )
    
    # Analyze shift patterns
    logger.info("\n" + "="*80)
    logger.info("SHIFT ANALYSIS")
    logger.info("="*80)
    
    logger.info(f"\nPCC Integer (upsample=1) shifts:")
    logger.info(f"  Mean shift:     ({np.mean(shifts_pcc_int[:, 0]):7.3f}, {np.mean(shifts_pcc_int[:, 1]):7.3f}) px")
    logger.info(f"  Std shift:      ({np.std(shifts_pcc_int[:, 0]):7.3f}, {np.std(shifts_pcc_int[:, 1]):7.3f}) px")
    logger.info(f"  Max abs shift:  {np.max(np.abs(shifts_pcc_int)):.2f} px")
    logger.info(f"  Total drift:    {np.linalg.norm(shifts_pcc_int[-1]):.2f} px")
    
    logger.info(f"\nPCC Subpixel (upsample=10) shifts:")
    logger.info(f"  Mean shift:     ({np.mean(shifts_pcc_sub[:, 0]):7.3f}, {np.mean(shifts_pcc_sub[:, 1]):7.3f}) px")
    logger.info(f"  Std shift:      ({np.std(shifts_pcc_sub[:, 0]):7.3f}, {np.std(shifts_pcc_sub[:, 1]):7.3f}) px")
    logger.info(f"  Max abs shift:  {np.max(np.abs(shifts_pcc_sub)):.2f} px")
    logger.info(f"  Total drift:    {np.linalg.norm(shifts_pcc_sub[-1]):.2f} px")
    
    logger.info(f"\nStackReg shifts:")
    logger.info(f"  Mean shift:     ({np.mean(shifts_stackreg[:, 0]):7.3f}, {np.mean(shifts_stackreg[:, 1]):7.3f}) px")
    logger.info(f"  Std shift:      ({np.std(shifts_stackreg[:, 0]):7.3f}, {np.std(shifts_stackreg[:, 1]):7.3f}) px")
    logger.info(f"  Max abs shift:  {np.max(np.abs(shifts_stackreg)):.2f} px")
    logger.info(f"  Total drift:    {np.linalg.norm(shifts_stackreg[-1]):.2f} px")
    
    # Compare shift smoothness (standard deviation of frame-to-frame differences)
    logger.info("\n" + "="*80)
    logger.info("SHIFT SMOOTHNESS - Lower = Smoother (less 'jumping')")
    logger.info("="*80)
    
    diff_pcc_int = np.diff(shifts_pcc_int, axis=0)
    diff_pcc_sub = np.diff(shifts_pcc_sub, axis=0)
    diff_stackreg = np.diff(shifts_stackreg, axis=0)
    
    smoothness_pcc_int = np.std(np.linalg.norm(diff_pcc_int, axis=1))
    smoothness_pcc_sub = np.std(np.linalg.norm(diff_pcc_sub, axis=1))
    smoothness_stackreg = np.std(np.linalg.norm(diff_stackreg, axis=1))
    
    logger.info(f"\nPCC Integer:  {smoothness_pcc_int:.4f} px/frame")
    logger.info(f"PCC Subpixel: {smoothness_pcc_sub:.4f} px/frame")
    logger.info(f"StackReg:     {smoothness_stackreg:.4f} px/frame")
    
    # Relative comparison
    logger.info("\nRelative smoothness (StackReg = 100%):")
    
    pcc_int_pct = (smoothness_pcc_int/smoothness_stackreg)*100
    if smoothness_pcc_int > smoothness_stackreg * 1.2:
        logger.info(f"  PCC Integer:  {pcc_int_pct:.1f}%  [WORSE - More jumping]")
    elif smoothness_pcc_int > smoothness_stackreg * 1.05:
        logger.info(f"  PCC Integer:  {pcc_int_pct:.1f}%  [Slightly worse]")
    else:
        logger.info(f"  PCC Integer:  {pcc_int_pct:.1f}%  [Good]")
    
    pcc_sub_pct = (smoothness_pcc_sub/smoothness_stackreg)*100
    if smoothness_pcc_sub < smoothness_stackreg * 0.95:
        logger.info(f"  PCC Subpixel: {pcc_sub_pct:.1f}%  [BETTER - Smoother!]")
    elif smoothness_pcc_sub < smoothness_stackreg * 1.05:
        logger.info(f"  PCC Subpixel: {pcc_sub_pct:.1f}%  [Similar]")
    else:
        logger.info(f"  PCC Subpixel: {pcc_sub_pct:.1f}%  [Worse]")
    
    # Agreement between methods
    logger.info("\n" + "="*80)
    logger.info("SHIFT AGREEMENT - How similar are the detected drifts?")
    logger.info("="*80)
    
    diff_int_vs_sr = np.mean(np.linalg.norm(shifts_pcc_int - shifts_stackreg, axis=1))
    diff_sub_vs_sr = np.mean(np.linalg.norm(shifts_pcc_sub - shifts_stackreg, axis=1))
    diff_int_vs_sub = np.mean(np.linalg.norm(shifts_pcc_int - shifts_pcc_sub, axis=1))
    
    logger.info(f"\nPCC Integer vs StackReg:  {diff_int_vs_sr:.4f} px mean difference")
    logger.info(f"PCC Subpixel vs StackReg: {diff_sub_vs_sr:.4f} px mean difference")
    logger.info(f"PCC Integer vs Subpixel:  {diff_int_vs_sub:.4f} px mean difference")
    
    if diff_sub_vs_sr < diff_int_vs_sr:
        logger.info("\n-> PCC Subpixel agrees better with StackReg than Integer version")
    
    # Save outputs for visual inspection
    output_dir = Path(r"E:\Oyvind\BIP-hub-test-data\drift\output\test_3_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n" + "="*80)
    logger.info(f"Saving outputs to: {output_dir}")
    logger.info("="*80)
    
    img_pcc_int.save(str(output_dir / "corrected_pcc_integer.tif"))
    logger.info("Saved: corrected_pcc_integer.tif")
    
    img_pcc_sub.save(str(output_dir / "corrected_pcc_subpixel.tif"))
    logger.info("Saved: corrected_pcc_subpixel.tif")
    
    img_stackreg.save(str(output_dir / "corrected_stackreg.tif"))
    logger.info("Saved: corrected_stackreg.tif")
    
    np.save(output_dir / "shifts_pcc_integer.npy", shifts_pcc_int)
    np.save(output_dir / "shifts_pcc_subpixel.npy", shifts_pcc_sub)
    np.save(output_dir / "shifts_stackreg.npy", shifts_stackreg)
    logger.info("Saved: shift arrays (.npy)")
    
    # Create visualization comparing shifts
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        frames = np.arange(shifts_pcc_int.shape[0])
        
        # Y shifts
        ax = axes[0, 0]
        ax.plot(frames, shifts_pcc_int[:, 0], 'r-o', label='PCC Integer', markersize=3, linewidth=1.5)
        ax.plot(frames, shifts_pcc_sub[:, 0], 'b-s', label='PCC Subpixel', markersize=3, linewidth=1.5)
        ax.plot(frames, shifts_stackreg[:, 0], 'g-^', label='StackReg', markersize=3, linewidth=1.5)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Y Shift (pixels)')
        ax.set_title('Detected Drift - Y Dimension')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # X shifts
        ax = axes[0, 1]
        ax.plot(frames, shifts_pcc_int[:, 1], 'r-o', label='PCC Integer', markersize=3, linewidth=1.5)
        ax.plot(frames, shifts_pcc_sub[:, 1], 'b-s', label='PCC Subpixel', markersize=3, linewidth=1.5)
        ax.plot(frames, shifts_stackreg[:, 1], 'g-^', label='StackReg', markersize=3, linewidth=1.5)
        ax.set_xlabel('Frame')
        ax.set_ylabel('X Shift (pixels)')
        ax.set_title('Detected Drift - X Dimension')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Frame-to-frame differences (smoothness)
        ax = axes[1, 0]
        diff_mag_int = np.linalg.norm(diff_pcc_int, axis=1)
        diff_mag_sub = np.linalg.norm(diff_pcc_sub, axis=1)
        diff_mag_sr = np.linalg.norm(diff_stackreg, axis=1)
        
        ax.plot(frames[1:], diff_mag_int, 'r-o', label='PCC Integer', markersize=3, linewidth=1.5)
        ax.plot(frames[1:], diff_mag_sub, 'b-s', label='PCC Subpixel', markersize=3, linewidth=1.5)
        ax.plot(frames[1:], diff_mag_sr, 'g-^', label='StackReg', markersize=3, linewidth=1.5)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Frame-to-Frame Shift Change (pixels)')
        ax.set_title('Shift Smoothness (Lower = Less Jumping)')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Summary bar chart
        ax = axes[1, 1]
        methods = ['PCC\nInteger', 'PCC\nSubpixel', 'StackReg']
        smoothness = [smoothness_pcc_int, smoothness_pcc_sub, smoothness_stackreg]
        colors = ['red', 'blue', 'green']
        
        bars = ax.bar(methods, smoothness, color=colors, alpha=0.7)
        ax.set_ylabel('Smoothness (std of frame-to-frame changes)')
        ax.set_title('Overall Smoothness Comparison\n(Lower = Better)')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, smoothness):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'shift_comparison.png', dpi=150, bbox_inches='tight')
        logger.info("Saved: shift_comparison.png")
        plt.close()
        
    except ImportError:
        logger.warning("Matplotlib not available, skipping visualization")
    
    logger.info("\n" + "="*80)
    logger.info("TEST COMPLETE")
    logger.info("="*80)
    
    logger.info("\nRECOMMENDATION:")
    if smoothness_pcc_sub < smoothness_stackreg:
        logger.info("✓ PCC with subpixel precision (upsample=10) is SMOOTHER than StackReg!")
        logger.info("  Use PCC with upsample_factor=10 for best results.")
    elif smoothness_pcc_sub < smoothness_pcc_int * 0.8:
        logger.info("✓ PCC subpixel is much smoother than integer version!")
        logger.info("  The fix is working correctly.")
    else:
        logger.info("! Review the outputs to compare quality.")


if __name__ == "__main__":
    compare_methods_real_data()
