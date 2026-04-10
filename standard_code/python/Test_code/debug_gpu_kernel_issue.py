from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import tifffile

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_DIR = SCRIPT_DIR.parent
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import fill_greyscale_holes as fgh


def main() -> None:
    image = tifffile.imread(SCRIPT_DIR / "input_frame.tif")
    kernel_size = 40
    overlap = 20
    stride = kernel_size - overlap

    y_starts = fgh._kernel_window_starts(image.shape[0], kernel_size, stride)
    x_starts = fgh._kernel_window_starts(image.shape[1], kernel_size, stride)
    windows = [
        (y0, x0, y_index, x_index)
        for y_index, y0 in enumerate(y_starts[:2])
        for x_index, x0 in enumerate(x_starts[:2])
    ]

    image_gpu = fgh.cp.asarray(image)
    batch_gpu = fgh._greyscale_fill_holes_gpu_batch(image_gpu, windows, (kernel_size, kernel_size))
    batch = fgh.cp.asnumpy(batch_gpu)

    print(f"window count: {len(windows)}")
    for idx, (y0, x0, _, _) in enumerate(windows):
        tile = image[y0:y0 + kernel_size, x0:x0 + kernel_size]
        cpu = fgh.greyscale_fill_holes_2d(tile, use_gpu=False)
        gpu_single = fgh._greyscale_fill_holes_2d_gpu(tile)
        gpu_batch = batch[idx]

        cpu_delta = np.maximum(cpu.astype(np.int64) - tile.astype(np.int64), 0)
        gpu_single_delta = np.maximum(gpu_single.astype(np.int64) - tile.astype(np.int64), 0)
        gpu_batch_delta = np.maximum(gpu_batch.astype(np.int64) - tile.astype(np.int64), 0)

        print(
            f"tile {idx}: cpu_sum={int(cpu_delta.sum())} gpu_single_sum={int(gpu_single_delta.sum())} "
            f"gpu_batch_sum={int(gpu_batch_delta.sum())} cpu_eq_single={np.array_equal(cpu, gpu_single)} "
            f"cpu_eq_batch={np.array_equal(cpu, gpu_batch)} batch_changed={bool(np.any(gpu_batch != tile))}"
        )


if __name__ == "__main__":
    main()
