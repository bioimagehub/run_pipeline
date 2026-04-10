from __future__ import annotations

from pathlib import Path
import json
import sys
import time

import numpy as np
import tifffile

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_DIR = SCRIPT_DIR.parent
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import fill_greyscale_holes as fgh


def main() -> None:
    image = tifffile.imread(SCRIPT_DIR / "input_frame.tif")

    start = time.perf_counter()
    cpu = fgh.greyscale_fill_holes_kernel_2d(
        image,
        kernel_size=40,
        kernel_overlap=20,
        use_gpu=False,
    )
    cpu_time = time.perf_counter() - start

    start = time.perf_counter()
    gpu = fgh.greyscale_fill_holes_kernel_2d(
        image,
        kernel_size=40,
        kernel_overlap=20,
        use_gpu=True,
    )
    fgh.cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - start

    result = {
        "cpu_time": cpu_time,
        "gpu_time": gpu_time,
        "speedup": cpu_time / gpu_time,
        "equal": bool(np.array_equal(cpu, gpu)),
        "cpu_sum": int(cpu.astype(np.int64).sum()),
        "gpu_sum": int(gpu.astype(np.int64).sum()),
    }
    output_path = SCRIPT_DIR / "benchmark_gpu_kernel_full_frame.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()