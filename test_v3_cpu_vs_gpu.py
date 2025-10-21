"""
Compare CPU vs GPU mode within v3 implementation.
Tests mathematical equivalence of GPU-accelerated phase correlation vs CPU.
"""
import sys
sys.path.insert(0, r'e:\Oyvind\OF_git\run_pipeline\standard_code\python')

import numpy as np
import logging
import bioimage_pipeline_utils as rp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compare_cpu_gpu_v3():
    """Compare v3 CPU mode vs GPU mode."""
    
    path = r"Z:\Schink\Meng\driftcor\input\1.nd2"
    
    logger.info("="*80)
    logger.info("TESTING V3 CPU MODE")
    logger.info("="*80)
    
    from drift_correction_utils.phase_cross_correlation_v3 import register_image_xy
    
    img = rp.load_tczyx_image(path)
    registered_cpu, shifts_cpu = register_image_xy(img, reference='first', channel=0, no_gpu=True)
    
    logger.info("")
    logger.info("="*80)
    logger.info("TESTING V3 GPU MODE")
    logger.info("="*80)
    
    img = rp.load_tczyx_image(path)
    registered_gpu, shifts_gpu = register_image_xy(img, reference='first', channel=0, no_gpu=False)
    
    # Compare shifts
    logger.info("")
    logger.info("="*80)
    logger.info("CPU vs GPU SHIFT COMPARISON (within V3)")
    logger.info("="*80)
    
    diff = np.abs(shifts_cpu - shifts_gpu)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    logger.info(f"CPU shifts shape: {shifts_cpu.shape}")
    logger.info(f"GPU shifts shape: {shifts_gpu.shape}")
    logger.info(f"")
    logger.info(f"Shift statistics:")
    logger.info(f"  Max difference: {max_diff:.6f} pixels")
    logger.info(f"  Mean difference: {mean_diff:.6f} pixels")
    logger.info(f"  Per-dimension max diff X/Y/Z: [{diff[:,0].max():.6f}, {diff[:,1].max():.6f}, {diff[:,2].max():.6f}]")
    logger.info(f"")
    
    # Show first 10 and last 10 frames
    logger.info(f"First 10 frames comparison:")
    logger.info(f"{'Frame':<6} {'CPU_X':<10} {'CPU_Y':<10} {'GPU_X':<10} {'GPU_Y':<10} {'Diff_X':<10} {'Diff_Y':<10}")
    logger.info("-"*70)
    for t in range(min(10, shifts_cpu.shape[0])):
        logger.info(f"{t:<6} {shifts_cpu[t,0]:<10.4f} {shifts_cpu[t,1]:<10.4f} "
                   f"{shifts_gpu[t,0]:<10.4f} {shifts_gpu[t,1]:<10.4f} "
                   f"{diff[t,0]:<10.6f} {diff[t,1]:<10.6f}")
    
    logger.info(f"")
    logger.info(f"Last 10 frames comparison:")
    logger.info(f"{'Frame':<6} {'CPU_X':<10} {'CPU_Y':<10} {'GPU_X':<10} {'GPU_Y':<10} {'Diff_X':<10} {'Diff_Y':<10}")
    logger.info("-"*70)
    for t in range(max(0, shifts_cpu.shape[0]-10), shifts_cpu.shape[0]):
        logger.info(f"{t:<6} {shifts_cpu[t,0]:<10.4f} {shifts_cpu[t,1]:<10.4f} "
                   f"{shifts_gpu[t,0]:<10.4f} {shifts_gpu[t,1]:<10.4f} "
                   f"{diff[t,0]:<10.6f} {diff[t,1]:<10.6f}")
    
    logger.info("")
    # Check success criterion
    if max_diff < 0.01:
        logger.info(f"✅ EXCELLENT: Maximum difference ({max_diff:.6f} px) is < 0.01 pixel (near machine precision)!")
    elif max_diff < 0.1:
        logger.info(f"✅ GOOD: Maximum difference ({max_diff:.6f} px) is < 0.1 pixel threshold!")
    elif max_diff < 1.0:
        logger.info(f"⚠️  ACCEPTABLE: Maximum difference ({max_diff:.6f} px) is < 1 pixel but >0.1 pixel")
    else:
        logger.warning(f"❌ NEEDS IMPROVEMENT: Maximum difference ({max_diff:.6f} px) exceeds 1 pixel threshold")
    
    logger.info("="*80)
    
    return max_diff, mean_diff, shifts_cpu, shifts_gpu

if __name__ == "__main__":
    max_diff, mean_diff, shifts_cpu, shifts_gpu = compare_cpu_gpu_v3()
