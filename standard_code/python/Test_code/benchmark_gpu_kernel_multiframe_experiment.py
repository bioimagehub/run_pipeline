from __future__ import annotations

from pathlib import Path
import json
import os
import sys
import time

import numpy as np
import tifffile

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_DIR = SCRIPT_DIR.parent
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import fill_greyscale_holes as fgh


LOG_PATH = SCRIPT_DIR / "benchmark_gpu_kernel_multiframe_experiment.log"


def _log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
    print(message, flush=True)


def _get_env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return int(raw_value)


def _get_env_int_list(name: str, default: list[int]) -> list[int]:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    values = [int(part.strip()) for part in raw_value.split(',') if part.strip()]
    return values or default


def _get_env_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _batch_windows_across_frames(
    stack_gpu,
    windows: list[tuple[int, int, int]],
    tile_shape: tuple[int, int],
):
    if fgh.cp is None or fgh.gpu_reconstruction is None:
        raise ImportError("cupy/cucim GPU reconstruction is not available")
    if not windows:
        return fgh.cp.empty((0,) + tile_shape, dtype=stack_gpu.dtype)

    tile_height, tile_width = tile_shape
    tile_count = len(windows)
    gap = 1
    tiles_per_row = max(1, int(np.ceil(np.sqrt(tile_count))))
    tile_rows = int(np.ceil(tile_count / tiles_per_row))
    packed_height = tile_rows * tile_height + (tile_rows + 1) * gap
    packed_width = tiles_per_row * tile_width + (tiles_per_row + 1) * gap

    filled_tiles = fgh.cp.empty((tile_count, tile_height, tile_width), dtype=stack_gpu.dtype)
    tile_maxima = fgh.cp.empty((tile_count,), dtype=stack_gpu.dtype)

    packed_mask = fgh.cp.empty((packed_height, packed_width), dtype=stack_gpu.dtype)
    separator_value = stack_gpu.min()
    packed_mask.fill(separator_value)

    for tile_index, (frame_index, y0, x0) in enumerate(windows):
        y1 = y0 + tile_height
        x1 = x0 + tile_width
        tile_gpu = stack_gpu[frame_index, y0:y1, x0:x1]
        tile_row = tile_index // tiles_per_row
        tile_col = tile_index % tiles_per_row
        packed_y0 = gap + tile_row * (tile_height + gap)
        packed_x0 = gap + tile_col * (tile_width + gap)
        packed_mask[packed_y0:packed_y0 + tile_height, packed_x0:packed_x0 + tile_width] = tile_gpu
        tile_maxima[tile_index] = tile_gpu.max()

    image_max = packed_mask.max()
    packed_inverted = image_max - packed_mask
    packed_seed = packed_inverted.copy()

    for tile_index in range(tile_count):
        tile_row = tile_index // tiles_per_row
        tile_col = tile_index % tiles_per_row
        packed_y0 = gap + tile_row * (tile_height + gap)
        packed_x0 = gap + tile_col * (tile_width + gap)
        packed_seed[
            packed_y0 + 1:packed_y0 + tile_height - 1,
            packed_x0 + 1:packed_x0 + tile_width - 1,
        ] = image_max - tile_maxima[tile_index]

    packed_filled = image_max - fgh.gpu_reconstruction(packed_seed, packed_inverted, method='dilation')
    for tile_index in range(tile_count):
        tile_row = tile_index // tiles_per_row
        tile_col = tile_index % tiles_per_row
        packed_y0 = gap + tile_row * (tile_height + gap)
        packed_x0 = gap + tile_col * (tile_width + gap)
        filled_tiles[tile_index] = packed_filled[
            packed_y0:packed_y0 + tile_height,
            packed_x0:packed_x0 + tile_width,
        ]

    return filled_tiles


def _greyscale_fill_holes_kernel_multiframe_gpu(
    stack: np.ndarray,
    kernel_size: int,
    kernel_overlap: int,
    frame_group_size: int,
) -> np.ndarray:
    if stack.ndim != 3:
        raise ValueError(f"Expected stack shape (T, Y, X), got {stack.shape}")

    frame_count, height, width = stack.shape
    stride = kernel_size - kernel_overlap
    y_starts = fgh._kernel_window_starts(height, kernel_size, stride)
    x_starts = fgh._kernel_window_starts(width, kernel_size, stride)
    tile_shape = (min(kernel_size, height), min(kernel_size, width))

    stack_gpu = fgh.cp.asarray(stack)
    merged_gpu = stack_gpu.astype(fgh.cp.float64, copy=True)

    windows_per_frame = len(y_starts) * len(x_starts)
    window_batch_size, free_bytes, bytes_per_window = fgh._estimate_gpu_kernel_batch_size(
        tile_shape,
        stack_gpu.dtype,
        windows_per_frame * frame_group_size,
    )
    effective_frame_group_size = max(1, min(frame_group_size, frame_count))

    print(
        f"multiframe kernel batching: frames={frame_count} group={effective_frame_group_size} windows/frame={windows_per_frame} "
        f"tile={tile_shape[0]}x{tile_shape[1]} free_gib={free_bytes / (1024 ** 3):.2f} "
        f"target_group_mib={(window_batch_size * bytes_per_window) / (1024 ** 2):.2f}"
    )

    positions_per_batch = max(1, window_batch_size // effective_frame_group_size)
    window_positions = [(y0, x0) for y0 in y_starts for x0 in x_starts]

    for frame_start in range(0, frame_count, effective_frame_group_size):
        frame_stop = min(frame_count, frame_start + effective_frame_group_size)
        group_frame_count = frame_stop - frame_start
        group_frame_slice = slice(frame_start, frame_stop)

        for position_start in range(0, len(window_positions), positions_per_batch):
            batch_positions = window_positions[position_start:position_start + positions_per_batch]
            batch_windows = [
                (frame_index, y0, x0)
                for y0, x0 in batch_positions
                for frame_index in range(frame_start, frame_stop)
            ]
            filled_tiles_gpu = _batch_windows_across_frames(stack_gpu, batch_windows, tile_shape)
            filled_tiles_gpu = filled_tiles_gpu.astype(merged_gpu.dtype, copy=False)
            filled_tiles_gpu = filled_tiles_gpu.reshape(len(batch_positions), group_frame_count, tile_shape[0], tile_shape[1])

            for position_index, (y0, x0) in enumerate(batch_positions):
                y1 = y0 + tile_shape[0]
                x1 = x0 + tile_shape[1]
                fgh.cp.maximum(
                    merged_gpu[group_frame_slice, y0:y1, x0:x1],
                    filled_tiles_gpu[position_index],
                    out=merged_gpu[group_frame_slice, y0:y1, x0:x1],
                )

    merged = fgh.cp.asnumpy(merged_gpu)
    return fgh._cast_to_dtype(merged, stack.dtype)


def _baseline_gpu_kernel(stack: np.ndarray, kernel_size: int, kernel_overlap: int) -> np.ndarray:
    return np.stack(
        [
            fgh.greyscale_fill_holes_kernel_2d(
                frame,
                kernel_size=kernel_size,
                kernel_overlap=kernel_overlap,
                use_gpu=True,
            )
            for frame in stack
        ],
        axis=0,
    )


def _baseline_cpu_kernel(stack: np.ndarray, kernel_size: int, kernel_overlap: int) -> np.ndarray:
    return np.stack(
        [
            fgh.greyscale_fill_holes_kernel_2d(
                frame,
                kernel_size=kernel_size,
                kernel_overlap=kernel_overlap,
                use_gpu=False,
            )
            for frame in stack
        ],
        axis=0,
    )


def main() -> None:
    kernel_size = _get_env_int("RP_EXPERIMENT_KERNEL_SIZE", 40)
    kernel_overlap = _get_env_int("RP_EXPERIMENT_KERNEL_OVERLAP", kernel_size // 2)
    benchmark_frames = _get_env_int("RP_BENCHMARK_MAX_FRAMES", 4)
    frame_group_sizes = _get_env_int_list("RP_EXPERIMENT_FRAME_GROUPS", [2, 4])
    skip_cpu_baseline = _get_env_bool("RP_EXPERIMENT_SKIP_CPU_BASELINE", False)

    stack = tifffile.imread(SCRIPT_DIR / "input_frames.tif")
    if stack.ndim != 3:
        raise ValueError(f"Expected input_frames.tif to be 3D (T,Y,X), got {stack.shape}")
    stack = stack[:benchmark_frames]

    results: dict[str, object] = {
        "frames": int(stack.shape[0]),
        "kernel_size": kernel_size,
        "kernel_overlap": kernel_overlap,
        "frame_group_sizes": frame_group_sizes,
        "skip_cpu_baseline": skip_cpu_baseline,
    }

    _log(
        f"starting experiment frames={stack.shape[0]} kernel={kernel_size} overlap={kernel_overlap} groups={frame_group_sizes}"
    )

    start = time.perf_counter()
    _log("running baseline gpu")
    baseline_gpu = _baseline_gpu_kernel(stack, kernel_size=kernel_size, kernel_overlap=kernel_overlap)
    fgh.cp.cuda.Stream.null.synchronize()
    baseline_gpu_time = time.perf_counter() - start
    results["baseline_gpu_time"] = baseline_gpu_time
    _log(f"baseline gpu done in {baseline_gpu_time:.6f}s")

    baseline_cpu: np.ndarray | None = None
    baseline_cpu_time: float | None = None
    if not skip_cpu_baseline:
        start = time.perf_counter()
        _log("running baseline cpu")
        baseline_cpu = _baseline_cpu_kernel(stack, kernel_size=kernel_size, kernel_overlap=kernel_overlap)
        baseline_cpu_time = time.perf_counter() - start
        results["baseline_cpu_time"] = baseline_cpu_time
        results["baseline_cpu_eq_gpu"] = bool(np.array_equal(baseline_cpu, baseline_gpu))
        _log(f"baseline cpu done in {baseline_cpu_time:.6f}s")

    experiments: list[dict[str, object]] = []
    for frame_group_size in frame_group_sizes:
        _log(f"running experiment frame_group_size={frame_group_size}")
        start = time.perf_counter()
        experimental = _greyscale_fill_holes_kernel_multiframe_gpu(
            stack,
            kernel_size=kernel_size,
            kernel_overlap=kernel_overlap,
            frame_group_size=frame_group_size,
        )
        fgh.cp.cuda.Stream.null.synchronize()
        experimental_time = time.perf_counter() - start

        experiments.append(
            {
                "frame_group_size": frame_group_size,
                "experimental_time": experimental_time,
                "speedup_vs_baseline_gpu": baseline_gpu_time / experimental_time,
                "speedup_vs_cpu": (
                    None if baseline_cpu_time is None else baseline_cpu_time / experimental_time
                ),
                "equal_to_baseline_gpu": bool(np.array_equal(experimental, baseline_gpu)),
                "equal_to_cpu": (
                    None if baseline_cpu is None else bool(np.array_equal(experimental, baseline_cpu))
                ),
                "output_sum": int(experimental.astype(np.int64).sum()),
            }
        )
        _log(f"experiment frame_group_size={frame_group_size} done in {experimental_time:.6f}s")

    results["experiments"] = experiments

    output_path = SCRIPT_DIR / "benchmark_gpu_kernel_multiframe_experiment.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _log(f"wrote results to {output_path.name}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()