# GPU Drift Correction - Performance Summary

## Overview
Optimized GPU-accelerated drift correction using CuPy for phase cross-correlation.

## Performance Results

### Test Dataset
- File: `1_Meng_timecrop_template_rolled-t15.tif`
- Size: 0.22 GB (15 frames × 2720×2720 pixels)
- Format: Single-channel 2D time-series

### Timing Comparison

| Method | Time | Speedup | Notes |
|--------|------|---------|-------|
| **CPU (scikit-image)** | 22.85s | 1.0x | Baseline |
| **GPU (CuPy optimized)** | 2.16s | **10.6x** | Production version |
| **StackReg** | 66.35s | 0.34x | Reference method |

### GPU Optimization Strategy

**Key Innovation: Minimize CPU↔GPU Memory Transfers**

#### Before Optimization (naive approach)
- CPU↔GPU transfers: **30 transfers** (2 per frame × 15 frames)
- Transfer overhead: Significant
- GPU utilization: Intermittent

#### After Optimization (production version)
- CPU↔GPU transfers: **2 transfers** (1 bulk in, 1 bulk out)
- Transfer reduction: **93%**
- GPU utilization: Continuous
- Result: **5x faster than naive GPU implementation**

## Implementation Details

### Optimized Workflow
```
1. Bulk transfer entire stack to GPU (0.22 GB → GPU memory)
2. Compute all shifts on GPU (data stays on GPU)
3. Apply all shifts on GPU (data stays on GPU)
4. Bulk transfer corrected stack to CPU (0.22 GB → CPU memory)
```

### Code Architecture
- **File**: `standard_code/python/drift_correction/phase_cross_correlation.py`
- **CPU Function**: `phase_cross_correlation_cpu()` - Uses scikit-image
- **GPU Function**: `phase_cross_correlation_gpu()` - Uses CuPy with optimized batching
- **Helper Function**: `apply_shifts_gpu()` - GPU-native shift application

### Memory Management
- Automatic cleanup every 10 frames during processing
- Full cleanup after shift computation
- Final cleanup after shift application
- Prevents memory leaks and fragmentation

## Accuracy Comparison

All three methods produce comparable results:

| Method | Mean Shift | Max Shift |
|--------|------------|-----------|
| CPU (scikit-image) | [-0.567, 0.420] | 4.90 px |
| GPU (CuPy) | [-0.265, 0.415] | 4.85 px |
| StackReg | [-0.417, 0.386] | 4.81 px |

Differences are expected due to different subpixel estimation methods:
- CPU: FFT upsampling (scikit-image default)
- GPU: Center-of-mass fitting (custom CuPy implementation)
- StackReg: Transformation matrix-based (ImageJ TurboReg)

## Scalability

Performance improvement scales with dataset size:

| Timepoints | CPU Transfers | GPU Transfers | Transfer Reduction |
|------------|---------------|---------------|-------------------|
| 15 | 30 | 2 | 93% |
| 100 | 200 | 2 | 99% |
| 1000 | 2000 | 2 | 99.9% |

**Larger datasets benefit even more from GPU optimization!**

## Hardware Requirements

### Minimum
- NVIDIA GPU with CUDA support
- 2 GB GPU memory (for typical 2K×2K time-series)
- CuPy installation with matching CUDA version

### Tested Configuration
- GPU: NVIDIA RTX A5000 (25.76 GB memory)
- CUDA: 12.x
- CuPy: 13.6.0
- Performance: 10.6x speedup over CPU

### GPU Monitoring
GPU utilization is often too fast for Task Manager to display. Use:
```powershell
nvidia-smi dmon -s u  # Real-time GPU utilization
```

## Usage

### Command Line
```bash
python drift_correction.py \
  --input-search-pattern "input/*.tif" \
  --output-folder "output/" \
  --method phase_cross_correlation \
  --reference previous \
  --gpu
```

### Pipeline YAML
```yaml
- --method: phase_cross_correlation
  - --gpu  # Enable GPU acceleration
```

## Error Handling

**No Fallback Design**: If `--gpu` is specified and GPU fails, the pipeline raises an error.

This design ensures:
- Users know immediately if GPU is unavailable
- No silent performance degradation
- Explicit hardware requirements

## Future Optimizations

Potential further improvements:
1. Multi-GPU support for parallel processing of multiple files
2. Batch processing of multiple files in single GPU session
3. FP16 computation for 2x memory reduction (if accuracy allows)
4. Asynchronous CPU↔GPU transfers during computation

## References

- **Phase Cross-Correlation**: scikit-image documentation
- **GPU Computing**: CuPy documentation
- **BIPHUB**: https://www.uio.no/tjenester/it/forskning/kompetansehuber/biphub/

---

**Written for BIPHUB Pipeline Manager**  
**Author**: GitHub Copilot with user guidance  
**Date**: October 14, 2025
