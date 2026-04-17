"""
Batch Cellpose segmentation for microscopy images using TCZYX-safe I/O.

This wrapper is intended for pipeline use when direct Cellpose CLI handling of
multi-page TIFF/OME-TIFF stacks is ambiguous or too memory-hungry. It loads
images through bioimage pipeline utilities, iterates over timepoints and/or Z
planes explicitly, and saves mask outputs as OME-TIFF.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

import bioimage_pipeline_utils as rp

logger = logging.getLogger(__name__)


def _default_maxcores() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def _find_input_files(pattern: str) -> List[Path]:
    matches = sorted(Path(p) for p in glob.glob(pattern, recursive=True))
    files = [p for p in matches if p.is_file()]
    if not files:
        raise FileNotFoundError(f"No input files matched pattern: {pattern}")
    return files


def _build_model(model_type: str, use_gpu: bool):
    from cellpose import models

    logger.info("Loading Cellpose model '%s' (GPU=%s)", model_type, use_gpu)

    if hasattr(models, "Cellpose"):
        return models.Cellpose(model_type=model_type, gpu=use_gpu)

    pretrained_model = model_type if model_type else "cpsam"
    return models.CellposeModel(gpu=use_gpu, pretrained_model=pretrained_model)


def _segment_plane_batch(
    model,
    planes: np.ndarray,
    diameter: float | None,
    cellpose_batch_size: int,
) -> np.ndarray:
    plane_array = np.asarray(planes)
    if plane_array.ndim == 2:
        plane_array = plane_array[np.newaxis, ...]
    if plane_array.ndim != 3:
        raise ValueError(f"Expected 2D plane batch with shape (N, Y, X), got {plane_array.shape}")

    plane_list = [np.asarray(plane_array[i]) for i in range(plane_array.shape[0])]
    result = model.eval(
        plane_list,
        diameter=diameter,
        batch_size=cellpose_batch_size,
        do_3D=False,
        z_axis=None,
        channel_axis=None,
    )
    masks = result[0] if isinstance(result, tuple) else result
    return np.stack([np.asarray(mask, dtype=np.uint16) for mask in masks], axis=0)


def process_file(
    input_path: Path,
    output_folder: Path,
    output_suffix: str,
    model,
    channel: int,
    diameter: float | None,
    z_mode: str,
    max_timepoints: int | None,
    time_batch_size: int,
    cellpose_batch_size: int,
) -> Path:
    logger.warning("Processing %s", input_path.name)
    img = rp.load_tczyx_image(str(input_path))
    data = np.asarray(img.data)

    if data.ndim != 5:
        raise ValueError(f"Expected TCZYX 5D image, got shape {data.shape}")

    t_size, c_size, z_size, y_size, x_size = data.shape
    if channel < 0 or channel >= c_size:
        raise ValueError(f"Channel index {channel} out of range for image with {c_size} channel(s)")

    logger.info("Loaded image with shape TCZYX=%s", data.shape)

    if max_timepoints is not None:
        t_size = min(t_size, max(1, max_timepoints))
        data = data[:t_size]
        logger.warning("Limiting processing to first %d timepoint(s)", t_size)

    mask = np.zeros((t_size, 1, z_size, y_size, x_size), dtype=np.uint16)
    time_batch_size = max(1, int(time_batch_size))
    cellpose_batch_size = max(1, int(cellpose_batch_size))

    logger.info(
        "Running with time_batch_size=%d and cellpose_batch_size=%d",
        time_batch_size,
        cellpose_batch_size,
    )

    progress = tqdm(
        total=t_size,
        desc=f"Frames {input_path.name[:40]}",
        unit="frame",
        dynamic_ncols=True,
    )
    try:
        for t_start in range(0, t_size, time_batch_size):
            t_end = min(t_size, t_start + time_batch_size)
            logger.info("Segmenting T=%d-%d/%d for %s", t_start + 1, t_end, t_size, input_path.name)
            batch_stack = data[t_start:t_end, channel]

            if z_mode == "max-project" and z_size > 1:
                projected_batch = np.max(batch_stack, axis=1)
                batch_masks = _segment_plane_batch(model, projected_batch, diameter, cellpose_batch_size)
                mask[t_start:t_end, 0, 0] = batch_masks
            else:
                plane_batch = batch_stack.reshape((-1, y_size, x_size))
                batch_masks = _segment_plane_batch(model, plane_batch, diameter, cellpose_batch_size)
                batch_masks = batch_masks.reshape((t_end - t_start, z_size, y_size, x_size))
                mask[t_start:t_end, 0] = batch_masks

            progress.update(t_end - t_start)
    finally:
        progress.close()

    output_path = output_folder / f"{rp.strip_tiff_suffix(input_path.name)}{output_suffix}.ome.tif"
    rp.save_tczyx_image(mask, str(output_path))
    logger.warning("Saved mask to %s", output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="segment_cellpose_batch",
        description="Run Cellpose segmentation over batch microscopy inputs using TCZYX-safe loading.",
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Segment cells in all drift-corrected files
  environment: uv@3.11:cellpose4
  commands:
  - python
  - '%REPO%/standard_code/python/segment_cellpose_batch.py'
  - --input-search-pattern: '%YAML%/input_drift_corrected/*_drift.ome.tif'
  - --output-folder: '%YAML%/cellpose_output'
  - --output-suffix: '_cellpose'
  - --channel: 0
  - --model-type: cpsam
  - --use-gpu

- name: Segment after Z max projection
  environment: uv@3.11:cellpose4
  commands:
  - python
  - '%REPO%/standard_code/python/segment_cellpose_batch.py'
  - --input-search-pattern: '%YAML%/input_drift_corrected/*_drift.ome.tif'
  - --output-folder: '%YAML%/cellpose_output'
  - --z-mode: max-project
  - --model-type: cpsam
  - --use-gpu
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--input-search-pattern', type=str, required=True, help='Glob pattern for input image files.')
    parser.add_argument('--output-folder', type=str, required=True, help='Destination folder for output masks.')
    parser.add_argument('--output-suffix', type=str, default='_cellpose', help='Suffix appended to the input stem before saving (default: _cellpose)')
    parser.add_argument('--no-parallel', action='store_true', help='Present for CLI consistency; processing is currently sequential.')
    parser.add_argument('--maxcores', type=int, default=_default_maxcores(), help='Present for CLI consistency; ignored by the current implementation.')
    parser.add_argument('--log-level', type=str, default='WARNING', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level (default: WARNING)')
    parser.add_argument('--channel', type=int, default=0, help='Channel index to segment (default: 0)')
    parser.add_argument('--model-type', type=str, default='cpsam', help='Cellpose model type, e.g. cyto3, nuclei, cpsam (default: cpsam)')
    parser.add_argument('--diameter', type=float, default=None, help='Optional target object diameter in pixels')
    parser.add_argument('--z-mode', type=str, default='slice-by-slice', choices=['slice-by-slice', 'max-project'], help='How to handle Z when Z>1 (default: slice-by-slice)')
    parser.add_argument('--max-timepoints', type=int, default=None, help='Optional limit for number of timepoints processed per file')
    parser.add_argument('--time-batch-size', type=int, default=4, help='Number of timepoints to send to Cellpose together while keeping per-frame segmentation (default: 4)')
    parser.add_argument('--cellpose-batch-size', type=int, default=8, help='Cellpose internal tile batch size for GPU inference (default: 8)')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU inference when available')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    input_files = _find_input_files(args.input_search_pattern)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    logger.warning("Found %d input file(s)", len(input_files))
    model = _build_model(args.model_type, args.use_gpu)

    for input_path in input_files:
        process_file(
            input_path=input_path,
            output_folder=output_folder,
            output_suffix=args.output_suffix,
            model=model,
            channel=args.channel,
            diameter=args.diameter,
            z_mode=args.z_mode,
            max_timepoints=args.max_timepoints,
            time_batch_size=args.time_batch_size,
            cellpose_batch_size=args.cellpose_batch_size,
        )

    logger.warning("Cellpose batch segmentation finished")


if __name__ == '__main__':
    main()
