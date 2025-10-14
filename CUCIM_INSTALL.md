# Installing cuCIM for GPU-Accelerated Drift Correction

## Overview

cuCIM (CUDA-accelerated image processing) provides GPU acceleration for phase cross-correlation drift correction. However, cuCIM installation is **platform-specific** and **optional**.

## Platform Support

### ✅ Linux (Recommended for GPU acceleration)
cuCIM is fully supported on Linux with NVIDIA GPUs.

**Installation:**
```bash
# After activating your environment
pip install cucim-cu12 --extra-index-url https://pypi.nvidia.com
```

Or add to your UV environment:
```bash
uv add --group drift-correct cucim-cu12 --index https://pypi.nvidia.com
```

### ⚠️ Windows
cuCIM does **NOT have official Windows wheels** as of October 2025. 

**Workaround options:**
1. **Use CPU mode** (default) - scikit-image phase_cross_correlation is fast and accurate
2. **Use WSL2** - Install Linux subsystem and run pipeline from WSL2
3. **Use Docker** - Run pipeline in Linux container with GPU passthrough
4. **Wait for conda-forge** - cuCIM may become available via conda-forge in future

**Current behavior on Windows:**
- `--gpu` flag will attempt to import cuCIM
- If not available, automatically falls back to CPU implementation
- **No error**, just a warning message in logs

### ❓ macOS
cuCIM requires NVIDIA CUDA, which is not available on macOS (Apple Silicon or Intel).
CPU mode only.

## Testing GPU Availability

Run this Python snippet to check if cuCIM is available:

```python
try:
    from cucim.core.operations.morphology import distance_transform_edt
    from cucim.skimage.registration import phase_cross_correlation
    print("✅ cuCIM is available and working!")
except ImportError as e:
    print(f"❌ cuCIM not available: {e}")
    print("   Falling back to CPU (scikit-image)")
```

## Performance Comparison

For a 50-frame timelapse (2720x2720 pixels):

- **CPU (scikit-image)**: ~60 seconds
- **GPU (cuCIM)**: ~5-10 seconds (6-12x faster)

For most use cases, CPU performance is acceptable. GPU acceleration is most beneficial for:
- Very large images (>4K resolution)
- Long time series (>100 frames)
- High-throughput batch processing
- Real-time processing requirements

## Recommendation

**For most users:** Stick with CPU mode (default). It's fast enough and works everywhere.

**For power users with Linux + NVIDIA GPU:** Install cuCIM for maximum speed.

## See Also

- [cuCIM Documentation](https://docs.rapids.ai/api/cucim/stable/)
- [RAPIDS Installation Guide](https://rapids.ai/start.html)
