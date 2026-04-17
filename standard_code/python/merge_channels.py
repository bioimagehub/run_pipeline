from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from contextlib import contextmanager
from typing import Any, Optional

import h5py
import numpy as np
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

import bioimage_pipeline_utils as rp

logger = logging.getLogger(__name__)


def _ensure_5d(array: np.ndarray) -> np.ndarray:
    """Ensure an array is in TCZYX-compatible 5D form."""
    data = np.asarray(array)
    while data.ndim < 5:
        data = data[np.newaxis, ...]
    return data


def _parse_input_patterns(input_search_pattern: str) -> dict[str, str]:
    """Parse input patterns into an ordered name -> glob mapping.

    Accepts either a plain glob string or JSON content such as:
    - '{"image1":"folder_a/*_a.tif","image2":"folder_b/*_b.tif"}'
    - '["folder_a/*.tif", "folder_b/*.tif"]'
    """
    stripped = input_search_pattern.strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = stripped

    if isinstance(parsed, str):
        return {"image1": parsed}

    if isinstance(parsed, list):
        if not parsed:
            raise ValueError("--input-search-pattern list cannot be empty")
        return {f"image{i + 1}": str(pattern) for i, pattern in enumerate(parsed)}

    if isinstance(parsed, dict):
        if not parsed:
            raise ValueError("--input-search-pattern mapping cannot be empty")
        return {str(name): str(pattern) for name, pattern in parsed.items()}

    raise ValueError("--input-search-pattern must be a glob string, JSON list, or JSON object")


def _parse_channels_argument(channels: Optional[str]) -> Optional[dict[str, Any]]:
    """Parse optional channel selection JSON.

    Supported values:
    - omitted / null / all -> use all channels from each image
    - '0,1,2' -> keep those channels from every image
    - '[0, 2]' -> same as above
    - '[[0,1], 2]' -> sum channels 0+1, then keep channel 2, for every image
    - '{"image1": [0, 1], "image2": "all"}' -> per-image selection
    """
    if channels is None:
        return None

    stripped = channels.strip()
    if not stripped or stripped.lower() in {"all", "none", "null"}:
        return None

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = [int(part.strip()) for part in stripped.split(",") if part.strip()]

    if isinstance(parsed, dict):
        return {str(key): value for key, value in parsed.items()}

    return {"__all__": parsed}


def _normalize_channel_groups(spec: Any, channel_count: int, image_name: str) -> list[list[int]]:
    """Convert a channel selection spec into validated channel groups."""
    if spec is None or (isinstance(spec, str) and spec.lower() == "all"):
        return [[channel_index] for channel_index in range(channel_count)]

    if isinstance(spec, int):
        groups = [[spec]]
    elif isinstance(spec, (list, tuple)):
        groups: list[list[int]] = []
        for item in spec:
            if isinstance(item, int):
                groups.append([item])
            elif isinstance(item, (list, tuple)):
                if not item:
                    raise ValueError(f"Empty channel group is not allowed for {image_name}")
                if not all(isinstance(idx, int) for idx in item):
                    raise ValueError(f"All channel indices must be integers for {image_name}")
                groups.append(list(item))
            else:
                raise ValueError(f"Unsupported channel selection item for {image_name}: {item!r}")
    else:
        raise ValueError(f"Unsupported channel selection for {image_name}: {spec!r}")

    for group in groups:
        for channel_index in group:
            if channel_index < 0 or channel_index >= channel_count:
                raise ValueError(
                    f"Invalid channel index {channel_index} for {image_name}; valid range is 0-{channel_count - 1}"
                )

    return groups


def _build_output_path(group_name: str, output_folder: str, output_suffix: str, output_format: str) -> str:
    """Create the output path for a grouped merge result."""
    normalized_format = rp.normalize_output_format(output_format)
    extension = rp.output_extension_for_format(normalized_format)
    safe_name = group_name if group_name else "merged_group"
    return os.path.join(output_folder, f"{safe_name}{output_suffix}{extension}")


def _save_output(data: np.ndarray, output_path: str, output_format: str, physical_pixel_sizes: Any) -> None:
    """Save a TCZYX array-like object using project-standard output helpers."""
    normalized_format = rp.normalize_output_format(output_format)
    save_kwargs: dict[str, Any] = {}
    if physical_pixel_sizes is not None and normalized_format in {"tif", "ome.tif"}:
        save_kwargs["physical_pixel_sizes"] = physical_pixel_sizes
        save_kwargs["dim_order"] = "TCZYX"
    rp.save_with_output_format(data, output_path, normalized_format, **save_kwargs)


def _get_lazy_data(img: Any) -> Any:
    """Return a lazily readable TCZYX-compatible array when available."""
    lazy_data = getattr(img, "dask_data", None)
    if lazy_data is not None:
        data = lazy_data
        while len(data.shape) < 5:
            data = data[None, ...]
        return data
    return _ensure_5d(np.asarray(getattr(img, "data", img)))


def _create_output_sink(
    output_path: str,
    output_format: str,
    output_dim_order: str,
    output_shape: tuple[int, int, int, int, int],
    output_dtype: np.dtype,
) -> tuple[Any, Any]:
    """Create a low-memory sink that writes merged output incrementally."""
    normalized_format = rp.normalize_output_format(output_format)
    t, c, z, y, x = output_shape

    if normalized_format == "npy":
        if output_dim_order == "TZYXC":
            array_sink = np.lib.format.open_memmap(
                output_path,
                mode="w+",
                dtype=output_dtype,
                shape=(t, z, y, x, c),
            )

            def write_channel(
                channel_index: int,
                t_start: int,
                t_end: int,
                z_start: int,
                z_end: int,
                block: np.ndarray,
            ) -> None:
                array_sink[t_start:t_end, z_start:z_end, :, :, channel_index:channel_index + 1] = np.transpose(
                    block.astype(output_dtype, copy=False),
                    (0, 2, 3, 4, 1),
                )

            def finalize(physical_pixel_sizes: Any) -> None:
                array_sink.flush()

            return write_channel, finalize

        array_sink = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=output_dtype,
            shape=output_shape,
        )

        def write_channel(
            channel_index: int,
            t_start: int,
            t_end: int,
            z_start: int,
            z_end: int,
            block: np.ndarray,
        ) -> None:
            array_sink[t_start:t_end, channel_index:channel_index + 1, z_start:z_end, :, :] = block.astype(output_dtype, copy=False)

        def finalize(physical_pixel_sizes: Any) -> None:
            array_sink.flush()

        return write_channel, finalize

    if normalized_format == "ilastik-h5":
        h5_file = h5py.File(output_path, "w")
        dataset = h5_file.create_dataset(
            "exported_data",
            shape=(t, z, y, x, c),
            dtype=output_dtype,
            compression="gzip",
            compression_opts=4,
        )
        axis_configs = [
            {"key": "t", "typeFlags": 8, "resolution": 0, "description": ""},
            {"key": "z", "typeFlags": 2, "resolution": 0, "description": ""},
            {"key": "y", "typeFlags": 2, "resolution": 0, "description": ""},
            {"key": "x", "typeFlags": 2, "resolution": 0, "description": ""},
            {"key": "c", "typeFlags": 1, "resolution": 0, "description": ""},
        ]
        dataset.attrs["axistags"] = json.dumps({"axes": axis_configs})

        def write_channel(
            channel_index: int,
            t_start: int,
            t_end: int,
            z_start: int,
            z_end: int,
            block: np.ndarray,
        ) -> None:
            dataset[t_start:t_end, z_start:z_end, :, :, channel_index:channel_index + 1] = np.transpose(
                block.astype(output_dtype, copy=False),
                (0, 2, 3, 4, 1),
            )

        def finalize(physical_pixel_sizes: Any) -> None:
            h5_file.close()

        return write_channel, finalize

    temp_handle, temp_path = tempfile.mkstemp(prefix="merge_channels_", suffix=".npy")
    os.close(temp_handle)
    array_sink = np.lib.format.open_memmap(
        temp_path,
        mode="w+",
        dtype=output_dtype,
        shape=output_shape,
    )

    def write_channel(
        channel_index: int,
        t_start: int,
        t_end: int,
        z_start: int,
        z_end: int,
        block: np.ndarray,
    ) -> None:
        array_sink[t_start:t_end, channel_index:channel_index + 1, z_start:z_end, :, :] = block.astype(output_dtype, copy=False)

    def finalize(physical_pixel_sizes: Any) -> None:
        try:
            array_sink.flush()
            _save_output(array_sink, output_path, normalized_format, physical_pixel_sizes)
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    return write_channel, finalize


def merge_grouped_images(
    group_name: str,
    grouped_files: dict[str, str],
    pattern_order: list[str],
    channel_selection: Optional[dict[str, Any]],
    output_folder: str,
    output_suffix: str,
    output_format: str,
    output_dim_order: str,
    z_chunk_size: int,
) -> str:
    """Merge channels from one grouped file set into a single output stack.

    Output channel order is always:
    image1 selected channels, then image2 selected channels, and so on.
    """
    reference_shape: Optional[tuple[int, int, int, int]] = None
    physical_pixel_sizes: Any = None
    metadata_inputs: list[dict[str, Any]] = []
    prepared_sources: list[dict[str, Any]] = []
    total_output_channels = 0
    output_dtype: Optional[np.dtype] = None

    for image_name in pattern_order:
        if image_name not in grouped_files:
            logger.warning("Group '%s' is missing input '%s'; skipping it.", group_name, image_name)
            continue

        input_path = grouped_files[image_name]
        img = rp.load_tczyx_image(input_path)
        data = _get_lazy_data(img)
        shape = tuple(int(value) for value in data.shape)

        current_shape = (shape[0], shape[2], shape[3], shape[4])
        if reference_shape is None:
            reference_shape = current_shape
            physical_pixel_sizes = getattr(img, "physical_pixel_sizes", None)
        elif current_shape != reference_shape:
            raise ValueError(
                f"Grouped images must have matching TZYX dimensions. "
                f"Group '{group_name}', input '{image_name}' had {current_shape}, expected {reference_shape}."
            )

        raw_spec = None if channel_selection is None else channel_selection.get(image_name, channel_selection.get("__all__"))
        channel_groups = _normalize_channel_groups(raw_spec, shape[1], image_name)
        total_output_channels += len(channel_groups)

        current_dtype = np.dtype(data.dtype)
        output_dtype = current_dtype if output_dtype is None else np.result_type(output_dtype, current_dtype)
        if any(len(group) > 1 for group in channel_groups) and np.issubdtype(current_dtype, np.integer):
            output_dtype = np.result_type(output_dtype, np.uint32)

        prepared_sources.append(
            {
                "input_name": image_name,
                "source_path": input_path.replace("\\", "/"),
                "data": data,
                "channel_groups": channel_groups,
            }
        )

    if not prepared_sources or reference_shape is None or output_dtype is None:
        raise ValueError(f"No input channels were available for group '{group_name}'")

    output_shape = (
        int(reference_shape[0]),
        int(total_output_channels),
        int(reference_shape[1]),
        int(reference_shape[2]),
        int(reference_shape[3]),
    )
    output_path = _build_output_path(group_name, output_folder, output_suffix, output_format)
    write_channel, finalize_output = _create_output_sink(
        output_path=output_path,
        output_format=output_format,
        output_dim_order=output_dim_order,
        output_shape=output_shape,
        output_dtype=np.dtype(output_dtype),
    )

    output_channel_index = 0
    z_chunk_size = max(1, int(z_chunk_size))
    total_chunks = output_shape[0] * total_output_channels * ((output_shape[2] + z_chunk_size - 1) // z_chunk_size)
    with tqdm(
        total=total_chunks,
        desc=f"Merging {group_name}",
        unit="chunk",
        leave=False,
    ) as progress:
        for source in prepared_sources:
            data = source["data"]
            for channel_group in source["channel_groups"]:
                metadata_inputs.append(
                    {
                        "input_name": source["input_name"],
                        "source_path": source["source_path"],
                        "source_channels": list(channel_group),
                    }
                )
                for time_index in range(output_shape[0]):
                    for z_start in range(0, output_shape[2], z_chunk_size):
                        z_end = min(output_shape[2], z_start + z_chunk_size)
                        block = data[time_index:time_index + 1, :, z_start:z_end, :, :]
                        if hasattr(block, "compute"):
                            block = block.compute()
                        block_np = _ensure_5d(np.asarray(block))
                        if len(channel_group) == 1:
                            channel_index = channel_group[0]
                            block_np = block_np[:, channel_index:channel_index + 1, :, :, :]
                        else:
                            block_np = np.take(block_np, channel_group, axis=1).sum(
                                axis=1,
                                keepdims=True,
                                dtype=np.dtype(output_dtype),
                            )
                        write_channel(output_channel_index, time_index, time_index + 1, z_start, z_end, block_np)
                        progress.update(1)
                output_channel_index += 1

    finalize_output(physical_pixel_sizes)

    metadata_path = f"{rp.strip_tiff_suffix(output_path)}_metadata.yaml"
    metadata = {
        "Merge channels": {
            "group_name": group_name,
            "output_channel_order": metadata_inputs,
            "output_format": rp.normalize_output_format(output_format),
            "output_dim_order": output_dim_order,
            "output_shape_tczyx": [int(value) for value in output_shape],
            "memory_strategy": "lazy dask-backed chunked processing",
        }
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)

    logger.info("Saved merged group '%s' to %s", group_name, output_path)
    return output_path


@contextmanager
def _tqdm_joblib(tqdm_object: tqdm):
    """Patch joblib so tqdm updates during parallel processing."""
    import joblib

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def _collect_search_patterns_from_args(args: argparse.Namespace) -> dict[str, str]:
    """Collect input search patterns from the CLI arguments.

    Preferred usage is --input-search-pattern1, --input-search-pattern2, etc.
    The legacy --input-search-pattern argument is still accepted for compatibility.
    """
    search_patterns: dict[str, str] = {}

    legacy_pattern = getattr(args, "input_search_pattern", None)
    if legacy_pattern:
        search_patterns.update(_parse_input_patterns(legacy_pattern))

    for index in range(1, 10):
        value = getattr(args, f"input_search_pattern{index}", None)
        if value:
            search_patterns[f"image{index}"] = value

    if not search_patterns:
        raise ValueError(
            "Provide at least one input pattern using --input-search-pattern1 "
            "(and optionally --input-search-pattern2, --input-search-pattern3, ...)."
        )

    return search_patterns


def process_folder(args: argparse.Namespace) -> None:
    """Process all grouped inputs matched by the provided patterns."""
    search_patterns = _collect_search_patterns_from_args(args)
    grouped = rp.get_grouped_files_to_process(search_patterns, args.search_subfolders)

    if not grouped:
        raise FileNotFoundError(f"No files matched the provided input search patterns: {search_patterns}")

    os.makedirs(args.output_folder, exist_ok=True)
    channel_selection = _parse_channels_argument(args.channels)
    pattern_order = list(search_patterns.keys())
    items = list(grouped.items())

    logger.info("Found %d grouped item(s) using inputs: %s", len(items), ", ".join(pattern_order))

    if args.no_parallel or len(items) <= 1:
        for group_name, grouped_files in tqdm(items, desc="Merging groups", unit="group"):
            merge_grouped_images(
                group_name=group_name,
                grouped_files=grouped_files,
                pattern_order=pattern_order,
                channel_selection=channel_selection,
                output_folder=args.output_folder,
                output_suffix=args.output_suffix,
                output_format=args.output_format,
                output_dim_order=args.output_dim_order,
                z_chunk_size=args.z_chunk_size,
            )
        return

    with tqdm(total=len(items), desc="Merging groups", unit="group") as progress:
        with _tqdm_joblib(progress):
            Parallel(n_jobs=rp.resolve_maxcores(args.maxcores, len(items)))(
                delayed(merge_grouped_images)(
                    group_name=group_name,
                    grouped_files=grouped_files,
                    pattern_order=pattern_order,
                    channel_selection=channel_selection,
                    output_folder=args.output_folder,
                    output_suffix=args.output_suffix,
                    output_format=args.output_format,
                    output_dim_order=args.output_dim_order,
                    z_chunk_size=args.z_chunk_size,
                )
                for group_name, grouped_files in items
            )


def build_parser() -> argparse.ArgumentParser:
    """Build the project-standard CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Merge channels from grouped images into one output stack. "
            "If --channels is omitted, all channels from all grouped images are kept."
        ),
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Merge all channels from grouped inputs
  environment: uv@3.11:default
  commands:
  - python
  - '%REPO%/standard_code/python/merge_channels.py'
  - --input-search-pattern1: '%YAML%/input_a/*_a.ome.tif'
  - --input-search-pattern2: '%YAML%/input_b/*_b.ome.tif'
  - --output-folder: '%YAML%/output_data'

- name: Merge selected channels per image
  environment: uv@3.11:default
  commands:
  - python
  - '%REPO%/standard_code/python/merge_channels.py'
  - --input-search-pattern1: '%YAML%/input_a/*_a.ome.tif'
  - --input-search-pattern2: '%YAML%/input_b/*_b.ome.tif'
  - --output-folder: '%YAML%/output_data'
  - --channels: '{"image1":[0,1],"image2":[2]}'
  - --output-format: ome.tif

- name: Merge grouped inputs and export Ilastik H5
  environment: uv@3.11:default
  commands:
  - python
  - '%REPO%/standard_code/python/merge_channels.py'
  - --input-search-pattern1: '%YAML%/input_a/*_a.ome.tif'
  - --input-search-pattern2: '%YAML%/input_b/*_b.ome.tif'
  - --output-folder: '%YAML%/output_data'
  - --channels: '{"image1":[[0,1],2],"image2":"all"}'
  - --output-format: ilastik-h5
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-search-pattern",
        type=str,
        required=False,
        help="Legacy single pattern or JSON mapping. Prefer --input-search-pattern1, --input-search-pattern2, etc.",
    )
    for index in range(1, 10):
        parser.add_argument(
            f"--input-search-pattern{index}",
            type=str,
            required=False,
            help=f"Input glob pattern for image {index}.",
        )
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Destination folder for merged outputs.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_merged",
        help="Suffix appended to the grouped basename before the extension.",
    )
    parser.add_argument(
        "--channels",
        "--merge-channels",
        dest="channels",
        type=str,
        default=None,
        help=(
            "Optional JSON channel selection. Omit to use all channels from every input image. "
            "Examples: '[0,1]', '[[0,1],2]', or '{\"image1\":[0],\"image2\":\"all\"}'."
        ),
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["tif", "ome.tif", "npy", "ilastik-h5"],
        default="ome.tif",
        help="Output format (default: ome.tif)",
    )
    parser.add_argument(
        "--output-dim-order",
        type=str,
        choices=["TCZYX", "TZYXC"],
        default="TCZYX",
        help="Output dimension order for NumPy exports (default: TCZYX)",
    )
    parser.add_argument(
        "--search-subfolders",
        action="store_true",
        help="Search recursively if your patterns do not already include **.",
    )
    parser.add_argument(
        "--z-chunk-size",
        type=int,
        default=1,
        help="Number of Z slices to compute at once during lazy merging (default: 1).",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing.",
    )
    parser.add_argument(
        "--maxcores",
        type=int,
        default=None,
        help="Maximum number of CPU cores to use for parallel processing.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: WARNING)",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    process_folder(args)


if __name__ == "__main__":
    main()

