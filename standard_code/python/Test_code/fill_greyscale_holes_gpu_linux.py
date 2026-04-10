from __future__ import annotations

import os
import sysconfig
import time
from pathlib import Path
import sys

cuda_path = Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cuda_runtime"
os.environ.setdefault("CUDA_PATH", str(cuda_path))

import cupy as cp
import numpy as np
import tifffile

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_DIR = SCRIPT_DIR.parent
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import fill_greyscale_holes as fgh


def _get_env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return int(raw_value)


def _benchmark_case(name: str, image: np.ndarray, kernel_size: int | None) -> None:
    use_kernel = kernel_size is not None
    label = f"{name} kernel={kernel_size}" if use_kernel else f"{name} full-image"
    print(f"running {label}...")

    start = time.perf_counter()
    if image.ndim == 2:
        if use_kernel:
            fgh.greyscale_fill_holes_kernel_2d(image, kernel_size=kernel_size, kernel_overlap=kernel_size // 2, use_gpu=False)
        else:
            fgh.greyscale_fill_holes_2d(image, use_gpu=False)
    else:
        for frame in image:
            if use_kernel:
                fgh.greyscale_fill_holes_kernel_2d(frame, kernel_size=kernel_size, kernel_overlap=kernel_size // 2, use_gpu=False)
            else:
                fgh.greyscale_fill_holes_2d(frame, use_gpu=False)
    cpu_time = time.perf_counter() - start

    warmup_frame = image[0] if image.ndim == 3 else image
    if use_kernel:
        fgh.greyscale_fill_holes_kernel_2d(warmup_frame, kernel_size=kernel_size, kernel_overlap=kernel_size // 2, use_gpu=True)
    else:
        fgh.greyscale_fill_holes_2d(warmup_frame, use_gpu=True)
    cp.cuda.Stream.null.synchronize()

    start = time.perf_counter()
    if image.ndim == 2:
        if use_kernel:
            fgh.greyscale_fill_holes_kernel_2d(image, kernel_size=kernel_size, kernel_overlap=kernel_size // 2, use_gpu=True)
        else:
            fgh.greyscale_fill_holes_2d(image, use_gpu=True)
    else:
        for frame in image:
            if use_kernel:
                fgh.greyscale_fill_holes_kernel_2d(frame, kernel_size=kernel_size, kernel_overlap=kernel_size // 2, use_gpu=True)
            else:
                fgh.greyscale_fill_holes_2d(frame, use_gpu=True)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - start

    print(f"{label}: cpu={cpu_time:.6f}s gpu={gpu_time:.6f}s speedup={cpu_time / gpu_time:.2f}x")


def _load_frame_stack(path: Path, max_frames: int | None = None) -> np.ndarray:
    image = tifffile.imread(path)
    if image.ndim == 2:
        return image
    if image.ndim != 3:
        raise ValueError(f"Expected 2D image or 3D stack, got shape {image.shape}")
    if max_frames is not None:
        return image[:max_frames]
    return image


def main() -> None:
    benchmark_frames = _get_env_int("RP_BENCHMARK_MAX_FRAMES", 3)
    single_frame = _load_frame_stack(SCRIPT_DIR / "input_frame.tif")
    multi_frame = _load_frame_stack(SCRIPT_DIR / "input_frames.tif", max_frames=benchmark_frames)

    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    print(f"GPU memory free={free_bytes / (1024 ** 3):.2f} GiB total={total_bytes / (1024 ** 3):.2f} GiB")
    print(
        "Batch tuning:",
        f"fraction={os.environ.get('RP_GPU_KERNEL_TARGET_FREE_FRACTION', 'default')}",
        f"max_windows={os.environ.get('RP_GPU_KERNEL_MAX_BATCH_WINDOWS', 'default')}",
        f"window_bytes_factor={os.environ.get('RP_GPU_KERNEL_WINDOW_BYTES_FACTOR', 'default')}",
        f"benchmark_frames={benchmark_frames}",
    )

    for kernel_size in (None, 40):
        _benchmark_case("single-frame", single_frame, kernel_size)
        _benchmark_case(f"multi-frame-{benchmark_frames}", multi_frame, kernel_size)


if __name__ == "__main__":
    main()
