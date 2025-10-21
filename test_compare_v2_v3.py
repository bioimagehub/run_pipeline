"""
Compare shifts between v2 (CPU) and v3 (GPU) implementations.
"""
import sys
sys.path.insert(0, r'e:\Oyvind\OF_git\run_pipeline\standard_code\python')

import numpy as np
import logging
import bioimage_pipeline_utils as rp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compare_implementations():
    """Compare v2 and v3 on the same file."""
    
    path = r"Z:\Schink\Meng\driftcor\input\1.nd2"
    
    logger.info("="*80)
    logger.info("TESTING PHASE CROSS-CORRELATION V2 (CPU)")
    logger.info("="*80)
    
    # Test v2
    from drift_correction_utils.phase_cross_correlation_v2 import register_image_xy as register_v2
    
    img = rp.load_tczyx_image(path)
    registered_v2, shifts_v2 = register_v2(img, reference='first', channel=0, no_gpu=True)
    
    logger.info("")
    logger.info("="*80)
    logger.info("TESTING PHASE CROSS-CORRELATION V3 (GPU)")
    logger.info("="*80)
    
    # Test v3
    from drift_correction_utils.phase_cross_correlation_v3 import register_image_xy as register_v3
    
    img = rp.load_tczyx_image(path)
    registered_v3, shifts_v3 = register_v3(img, reference='first', channel=0, no_gpu=False)
    
    # Compare shifts
    logger.info("")
    logger.info("="*80)
    logger.info("SHIFT COMPARISON")
    logger.info("="*80)
    
    diff = np.abs(shifts_v2 - shifts_v3)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    logger.info(f"V2 shifts shape: {shifts_v2.shape}")
    logger.info(f"V3 shifts shape: {shifts_v3.shape}")
    logger.info(f"")
    logger.info(f"Shift statistics:")
    logger.info(f"  Max difference: {max_diff:.6f} pixels")
    logger.info(f"  Mean difference: {mean_diff:.6f} pixels")
    logger.info(f"  Per-dimension max diff X/Y/Z: [{diff[:,0].max():.6f}, {diff[:,1].max():.6f}, {diff[:,2].max():.6f}]")
    logger.info(f"")
    
    # Show first 10 frames
    logger.info(f"First 10 frames comparison:")
    logger.info(f"{'Frame':<6} {'V2_X':<10} {'V2_Y':<10} {'V3_X':<10} {'V3_Y':<10} {'Diff_X':<10} {'Diff_Y':<10}")
    logger.info("-"*70)
    for t in range(min(10, shifts_v2.shape[0])):
        logger.info(f"{t:<6} {shifts_v2[t,0]:<10.4f} {shifts_v2[t,1]:<10.4f} "
                   f"{shifts_v3[t,0]:<10.4f} {shifts_v3[t,1]:<10.4f} "
                   f"{diff[t,0]:<10.6f} {diff[t,1]:<10.6f}")
    
    logger.info("")
    # Check success criterion
    if max_diff < 1.0:
        logger.info(f"✅ SUCCESS: Maximum difference ({max_diff:.6f} px) is < 1 pixel threshold!")
    else:
        logger.warning(f"⚠️  NEEDS IMPROVEMENT: Maximum difference ({max_diff:.6f} px) exceeds 1 pixel threshold")
    
    logger.info("="*80)
    
    return max_diff, mean_diff, shifts_v2, shifts_v3

if __name__ == "__main__":
    max_diff, mean_diff, shifts_v2, shifts_v3 = compare_implementations()
