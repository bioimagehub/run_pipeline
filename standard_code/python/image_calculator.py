from __future__ import annotations

import argparse
import ast
import logging
import os
import tempfile
from contextlib import contextmanager
from functools import reduce
from typing import Any, Optional

import h5py
import numpy as np
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

import bioimage_pipeline_utils as rp

logger = logging.getLogger(__name__)


def _mask_values(image: Any, mask_source: Any) -> np.ndarray:
    """Keep image values only where the mask source is larger than zero."""
    return np.where(np.asarray(mask_source) > 0, image, 0)


_ALLOWED_FUNCTIONS: dict[str, Any] = {
    "abs": np.abs,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "clip": np.clip,
    "where": np.where,
    "mask": _mask_values,
}


def _ensure_5d(array: np.ndarray) -> np.ndarray:
    """Ensure an array is in TCZYX-compatible 5D form."""
    data = np.asarray(array)
    while data.ndim < 5:
        data = data[np.newaxis, ...]
    return data


def _parse_input_patterns(input_search_pattern: str) -> dict[str, str]:
    """Parse a legacy input-search-pattern string into an image-name mapping."""
    stripped = input_search_pattern.strip()
    if not stripped:
        raise ValueError("--input-search-pattern cannot be empty")
    return {"image1": stripped}


def _collect_search_patterns_from_args(args: argparse.Namespace) -> dict[str, str]:
    """Collect numbered image input patterns from the CLI arguments."""
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


def _get_lazy_data(img: Any) -> Any:
    """Return a lazily readable TCZYX-compatible array when available."""
    lazy_data = getattr(img, "dask_data", None)
    if lazy_data is not None:
        data = lazy_data
        while len(data.shape) < 5:
            data = data[None, ...]
        return data
    return _ensure_5d(np.asarray(getattr(img, "data", img)))


def _build_output_path(group_name: str, output_folder: str, output_suffix: str, output_format: str) -> str:
    """Create the output path for a grouped calculator result."""
    normalized_format = rp.normalize_output_format(output_format)
    extension = rp.output_extension_for_format(normalized_format)
    safe_name = group_name if group_name else "calculated_group"
    return os.path.join(output_folder, f"{safe_name}{output_suffix}{extension}")


def _save_output(data: np.ndarray, output_path: str, output_format: str, physical_pixel_sizes: Any) -> None:
    """Save a TCZYX array-like object using project-standard output helpers."""
    normalized_format = rp.normalize_output_format(output_format)
    save_kwargs: dict[str, Any] = {}
    if physical_pixel_sizes is not None and normalized_format in {"tif", "ome.tif"}:
        save_kwargs["physical_pixel_sizes"] = physical_pixel_sizes
        save_kwargs["dim_order"] = "TCZYX"
    rp.save_with_output_format(data, output_path, normalized_format, **save_kwargs)


def _create_output_sink(
    output_path: str,
    output_format: str,
    output_dim_order: str,
    output_shape: tuple[int, int, int, int, int],
    output_dtype: np.dtype,
) -> tuple[Any, Any]:
    """Create a low-memory sink that writes output incrementally."""
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

            def write_block(t_start: int, t_end: int, z_start: int, z_end: int, block: np.ndarray) -> None:
                array_sink[t_start:t_end, z_start:z_end, :, :, :] = np.transpose(
                    block.astype(output_dtype, copy=False),
                    (0, 2, 3, 4, 1),
                )

            def finalize(physical_pixel_sizes: Any) -> None:
                array_sink.flush()

            return write_block, finalize

        array_sink = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=output_dtype,
            shape=output_shape,
        )

        def write_block(t_start: int, t_end: int, z_start: int, z_end: int, block: np.ndarray) -> None:
            array_sink[t_start:t_end, :, z_start:z_end, :, :] = block.astype(output_dtype, copy=False)

        def finalize(physical_pixel_sizes: Any) -> None:
            array_sink.flush()

        return write_block, finalize

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
        dataset.attrs["axistags"] = yaml.safe_dump({"axes": axis_configs})

        def write_block(t_start: int, t_end: int, z_start: int, z_end: int, block: np.ndarray) -> None:
            dataset[t_start:t_end, z_start:z_end, :, :, :] = np.transpose(
                block.astype(output_dtype, copy=False),
                (0, 2, 3, 4, 1),
            )

        def finalize(physical_pixel_sizes: Any) -> None:
            h5_file.close()

        return write_block, finalize

    temp_handle, temp_path = tempfile.mkstemp(prefix="image_calculator_", suffix=".npy")
    os.close(temp_handle)
    array_sink = np.lib.format.open_memmap(
        temp_path,
        mode="w+",
        dtype=output_dtype,
        shape=output_shape,
    )

    def write_block(t_start: int, t_end: int, z_start: int, z_end: int, block: np.ndarray) -> None:
        array_sink[t_start:t_end, :, z_start:z_end, :, :] = block.astype(output_dtype, copy=False)

    def finalize(physical_pixel_sizes: Any) -> None:
        try:
            array_sink.flush()
            _save_output(array_sink, output_path, normalized_format, physical_pixel_sizes)
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    return write_block, finalize


class _SafeExpressionEvaluator(ast.NodeVisitor):
    """Evaluate a restricted image expression safely on NumPy arrays."""

    def __init__(self, variables: dict[str, np.ndarray]):
        self.variables = variables

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in self.variables:
            return self.variables[node.id]
        if node.id in _ALLOWED_FUNCTIONS:
            return _ALLOWED_FUNCTIONS[node.id]
        raise ValueError(f"Unknown name in expression: {node.id}")

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (int, float, bool)):
            return node.value
        raise ValueError(f"Unsupported constant in expression: {node.value!r}")

    def visit_Call(self, node: ast.Call) -> Any:
        func = self.visit(node.func)
        if func not in _ALLOWED_FUNCTIONS.values():
            raise ValueError("Only simple math helper functions are allowed in --expression")
        args = [self.visit(arg) for arg in node.args]
        kwargs: dict[str, Any] = {}
        for kw in node.keywords:
            if kw.arg is None:
                raise ValueError("Keyword unpacking is not supported in --expression")
            kwargs[kw.arg] = self.visit(kw.value)
        return func(*args, **kwargs)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)

        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            return left ** right
        if isinstance(node.op, ast.BitAnd):
            return left & right
        if isinstance(node.op, ast.BitOr):
            return left | right
        if isinstance(node.op, ast.BitXor):
            return left ^ right
        raise ValueError(f"Unsupported operator in expression: {type(node.op).__name__}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.Invert):
            return ~operand
        if isinstance(node.op, ast.Not):
            return np.logical_not(operand)
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        values = [self.visit(value) for value in node.values]
        if isinstance(node.op, ast.And):
            return reduce(np.logical_and, values)
        if isinstance(node.op, ast.Or):
            return reduce(np.logical_or, values)
        raise ValueError(f"Unsupported boolean operator: {type(node.op).__name__}")

    def visit_Compare(self, node: ast.Compare) -> Any:
        left = self.visit(node.left)
        result = None
        for op, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            if isinstance(op, ast.Gt):
                current = left > right
            elif isinstance(op, ast.GtE):
                current = left >= right
            elif isinstance(op, ast.Lt):
                current = left < right
            elif isinstance(op, ast.LtE):
                current = left <= right
            elif isinstance(op, ast.Eq):
                current = left == right
            elif isinstance(op, ast.NotEq):
                current = left != right
            else:
                raise ValueError(f"Unsupported comparison operator: {type(op).__name__}")
            result = current if result is None else np.logical_and(result, current)
            left = right
        return result

    def generic_visit(self, node: ast.AST) -> Any:
        allowed_nodes = (
            ast.Expression,
            ast.Name,
            ast.Load,
            ast.Constant,
            ast.Call,
            ast.BinOp,
            ast.UnaryOp,
            ast.BoolOp,
            ast.Compare,
            ast.And,
            ast.Or,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.Pow,
            ast.BitAnd,
            ast.BitOr,
            ast.BitXor,
            ast.UAdd,
            ast.USub,
            ast.Invert,
            ast.Not,
            ast.Gt,
            ast.GtE,
            ast.Lt,
            ast.LtE,
            ast.Eq,
            ast.NotEq,
        )
        if isinstance(node, allowed_nodes):
            return super().generic_visit(node)
        raise ValueError(f"Unsupported syntax in --expression: {type(node).__name__}")


def _extract_image_names(expression: str) -> list[str]:
    """Extract image variable names referenced in the expression."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid --expression syntax: {exc}") from exc

    image_names = sorted(
        {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id.startswith("image") and node.id[5:].isdigit()
        },
        key=lambda name: int(name[5:]),
    )
    if not image_names:
        raise ValueError("--expression must reference at least one grouped input such as image1 or image2")
    return image_names


def _evaluate_expression(expression: str, variables: dict[str, np.ndarray]) -> np.ndarray:
    """Evaluate the expression against a block of TCZYX arrays."""
    tree = ast.parse(expression, mode="eval")
    result = _SafeExpressionEvaluator(variables).visit(tree)
    first_shape = next(iter(variables.values())).shape

    if np.isscalar(result):
        filled = np.full(first_shape, result)
        return _ensure_5d(np.asarray(filled))

    result_array = _ensure_5d(np.asarray(result))
    if result_array.shape != first_shape:
        try:
            result_array = np.broadcast_to(result_array, first_shape)
        except ValueError as exc:
            raise ValueError(
                f"Expression output shape {result_array.shape} does not match the input shape {first_shape}"
            ) from exc
    return _ensure_5d(np.asarray(result_array))


def calculate_grouped_images(
    group_name: str,
    grouped_files: dict[str, str],
    expression: str,
    image_names: list[str],
    output_folder: str,
    output_suffix: str,
    output_format: str,
    output_dim_order: str,
    z_chunk_size: int,
) -> str:
    """Apply an elementwise calculator expression to one grouped file set."""
    prepared_sources: dict[str, dict[str, Any]] = {}
    reference_shape: Optional[tuple[int, int, int, int, int]] = None
    physical_pixel_sizes: Any = None

    for image_name in image_names:
        if image_name not in grouped_files:
            raise ValueError(
                f"Group '{group_name}' is missing required input '{image_name}' for expression: {expression}"
            )

        input_path = grouped_files[image_name]
        img = rp.load_tczyx_image(input_path)
        data = _get_lazy_data(img)
        raw_shape = tuple(int(value) for value in data.shape)
        if len(raw_shape) != 5:
            raise ValueError(f"Expected TCZYX-compatible 5D data for {input_path}, got shape {raw_shape}")
        shape = (raw_shape[0], raw_shape[1], raw_shape[2], raw_shape[3], raw_shape[4])

        if reference_shape is None:
            reference_shape = shape
            physical_pixel_sizes = getattr(img, "physical_pixel_sizes", None)
        elif shape != reference_shape:
            raise ValueError(
                f"Grouped images must have matching TCZYX dimensions. "
                f"Group '{group_name}', input '{image_name}' had {shape}, expected {reference_shape}."
            )

        prepared_sources[image_name] = {
            "source_path": input_path.replace("\\", "/"),
            "data": data,
        }

    if reference_shape is None:
        raise ValueError(f"No input images were available for group '{group_name}'")

    output_shape = reference_shape
    output_path = _build_output_path(group_name, output_folder, output_suffix, output_format)
    write_block = None
    finalize_output = None
    result_dtype: Optional[np.dtype] = None

    z_chunk_size = max(1, int(z_chunk_size))
    total_chunks = output_shape[0] * ((output_shape[2] + z_chunk_size - 1) // z_chunk_size)

    with tqdm(total=total_chunks, desc=f"Calculating {group_name}", unit="chunk", leave=False) as progress:
        for time_index in range(output_shape[0]):
            for z_start in range(0, output_shape[2], z_chunk_size):
                z_end = min(output_shape[2], z_start + z_chunk_size)
                block_variables: dict[str, np.ndarray] = {}

                for image_name, source in prepared_sources.items():
                    block = source["data"][time_index:time_index + 1, :, z_start:z_end, :, :]
                    if hasattr(block, "compute"):
                        block = block.compute()
                    block_variables[image_name] = _ensure_5d(np.asarray(block))

                result_block = _evaluate_expression(expression, block_variables)
                if result_block.dtype == np.bool_:
                    result_block = result_block.astype(np.uint8)
                if result_dtype is None:
                    result_dtype = np.dtype(result_block.dtype)
                    write_block, finalize_output = _create_output_sink(
                        output_path=output_path,
                        output_format=output_format,
                        output_dim_order=output_dim_order,
                        output_shape=output_shape,
                        output_dtype=result_dtype,
                    )
                assert write_block is not None
                write_block(time_index, time_index + 1, z_start, z_end, result_block)
                progress.update(1)

    assert finalize_output is not None
    finalize_output(physical_pixel_sizes)

    metadata_path = f"{rp.strip_tiff_suffix(output_path)}_metadata.yaml"
    metadata = {
        "Image calculator": {
            "group_name": group_name,
            "expression": expression,
            "output_format": rp.normalize_output_format(output_format),
            "output_dim_order": output_dim_order,
            "output_shape_tczyx": [int(value) for value in output_shape],
            "inputs": {
                image_name: prepared_sources[image_name]["source_path"]
                for image_name in image_names
            },
            "memory_strategy": "lazy dask-backed chunked processing",
        }
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)

    logger.info("Saved calculator group '%s' to %s", group_name, output_path)
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


def process_folder(args: argparse.Namespace) -> None:
    """Process all grouped inputs matched by the provided patterns."""
    search_patterns = _collect_search_patterns_from_args(args)
    grouped = rp.get_grouped_files_to_process(search_patterns, args.search_subfolders)

    if not grouped:
        raise FileNotFoundError(f"No files matched the provided input search patterns: {search_patterns}")

    os.makedirs(args.output_folder, exist_ok=True)
    items = list(grouped.items())
    image_names = _extract_image_names(args.expression)

    for image_name in image_names:
        if image_name not in search_patterns:
            raise ValueError(
                f"Expression references {image_name}, but no matching --input-search-patternN argument was provided"
            )

    logger.info(
        "Found %d grouped item(s) using expression '%s' with inputs: %s",
        len(items),
        args.expression,
        ", ".join(image_names),
    )

    if args.no_parallel or len(items) <= 1:
        for group_name, grouped_files in tqdm(items, desc="Calculating groups", unit="group"):
            calculate_grouped_images(
                group_name=group_name,
                grouped_files=grouped_files,
                expression=args.expression,
                image_names=image_names,
                output_folder=args.output_folder,
                output_suffix=args.output_suffix,
                output_format=args.output_format,
                output_dim_order=args.output_dim_order,
                z_chunk_size=args.z_chunk_size,
            )
        return

    with tqdm(total=len(items), desc="Calculating groups", unit="group") as progress:
        with _tqdm_joblib(progress):
            Parallel(n_jobs=rp.resolve_maxcores(args.maxcores, len(items)))(
                delayed(calculate_grouped_images)(
                    group_name=group_name,
                    grouped_files=grouped_files,
                    expression=args.expression,
                    image_names=image_names,
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
            "Apply a grouped image calculation to matching inputs such as image1, image2, image3, and so on. "
            "The expression is evaluated elementwise in TCZYX order. "
            "Use image1 * image2 > 0 for a binary overlap mask, or image1 * (image2 > 0) to keep image1 values only where image2 is positive."
        ),
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Binary overlap from two grouped masks
  environment: uv@3.11:default
  commands:
  - python
  - '%REPO%/standard_code/python/image_calculator.py'
  - --input-search-pattern1: '%YAML%/mask_a/*_a.ome.tif'
  - --input-search-pattern2: '%YAML%/mask_b/*_b.ome.tif'
  - --output-folder: '%YAML%/output_data'
  - --expression: 'image1 * image2 > 0'

- name: Keep image1 intensities where image2 is positive
  environment: uv@3.11:default
  commands:
  - python
  - '%REPO%/standard_code/python/image_calculator.py'
  - --input-search-pattern1: '%YAML%/input_a/*_a.ome.tif'
  - --input-search-pattern2: '%YAML%/input_b/*_b.ome.tif'
  - --output-folder: '%YAML%/output_data'
  - --expression: 'image1 * (image2 > 0)'
  - --output-format: ome.tif

- name: Keep image1 intensities using the mask helper
  environment: uv@3.11:default
  commands:
  - python
  - '%REPO%/standard_code/python/image_calculator.py'
  - --input-search-pattern1: '%YAML%/input_a/*_a.ome.tif'
  - --input-search-pattern2: '%YAML%/input_b/*_b.ome.tif'
  - --output-folder: '%YAML%/output_data'
  - --expression: 'mask(image1, image2)'
  - --output-format: ome.tif
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-search-pattern",
        type=str,
        required=False,
        help="Legacy single pattern. Prefer --input-search-pattern1, --input-search-pattern2, etc.",
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
        help="Destination folder for calculator outputs.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_calc",
        help="Suffix appended to the grouped basename before the extension.",
    )
    parser.add_argument(
        "--expression",
        "--operation",
        dest="expression",
        type=str,
        required=True,
        help=(
            "Elementwise expression using grouped images such as image1, image2, image3. "
            "Examples: 'image1 * image2 > 0' for a binary mask, 'image1 * (image2 > 0)' or 'mask(image1, image2)' to keep image1 values where image2 is positive."
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
        help="Number of Z slices to compute at once during lazy processing (default: 1).",
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
