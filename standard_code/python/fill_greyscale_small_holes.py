from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import time
from typing import Optional

import numpy as np
from scipy import ndimage as ndi
from tqdm import tqdm

# Local imports
import bioimage_pipeline_utils as rp


# Module-level logger
logger = logging.getLogger(__name__)


def strip_image_suffix(filename: str) -> str:
    """Return filename without known image extensions, including compound OME suffixes."""
    lower_name = filename.lower()
    for suffix in (".ome.tif", ".ome.tiff", ".tif", ".tiff", ".npy", ".ome"):
        if lower_name.endswith(suffix):
            return filename[: -len(suffix)]
    return os.path.splitext(filename)[0]


def process_file(
    input_path: str,
    output_path: str,
    size: int = 5,
    mode: str = "reflect",
    output_format: str = "ome-tif",
) -> bool:
    """Fill small dark holes in grayscale images by grey closing and save delta image."""
    try:
        logger.info(f"Processing: {os.path.basename(input_path)}")

        image = rp.load_tczyx_image(input_path)
        data = image.data

        closed = ndi.grey_closing(data, size=(1, 1, 1, size, size), mode=mode)
        delta = closed - data

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if output_format == "ome-tif":
            rp.save_tczyx_image(delta, output_path)
        else:
            np.save(output_path, delta)

        logger.info(f"Saved: {output_path}")
        return True
    except Exception as exc:
        logger.error(f"Failed processing {input_path}: {exc}")
        return False


def process_files(
    input_pattern: str,
    output_folder: Optional[str] = None,
    size: int = 5,
    mode: str = "reflect",
    output_format: str = "ome-tif",
    collapse_delimiter: str = "__",
    no_parallel: bool = False,
    output_extension: str = "_filled",
    dry_run: bool = False,
) -> None:
    """Process multiple files matching a pattern."""
    search_subfolders = "**" in input_pattern
    files = rp.get_files_to_process2(input_pattern, search_subfolders=search_subfolders)

    if not files:
        logger.error(f"No files found matching pattern: {input_pattern}")
        return

    logger.info(f"Found {len(files)} file(s) to process")

    if "**" in input_pattern:
        base_folder = input_pattern.split("**")[0].rstrip("/\\")
        if not base_folder:
            base_folder = os.getcwd()
        base_folder = os.path.abspath(base_folder)
    else:
        base_folder = str(Path(files[0]).parent)

    if output_folder is None:
        output_folder = base_folder + "_filled"

    logger.info(f"Output folder: {output_folder}")

    file_pairs = []
    for src in files:
        collapsed = rp.collapse_filename(src, base_folder, collapse_delimiter)
        if output_format == "ome-tif":
            out_suffix = ".ome.tif"
        else:
            out_suffix = ".npy"
        out_stem = strip_image_suffix(collapsed)
        if output_format == "ome-tif" and out_stem.lower().endswith(".ome"):
            out_stem = out_stem[:-4]
        out_name = out_stem + output_extension + out_suffix
        out_path = os.path.join(output_folder, out_name)
        file_pairs.append((src, out_path))

    if dry_run:
        print(f"[DRY RUN] Would process {len(file_pairs)} files")
        print(f"[DRY RUN] Output folder: {output_folder}")
        print(f"[DRY RUN] Size: {size}, mode: {mode}, output-format: {output_format}")
        for src, dst in file_pairs:
            print(f"[DRY RUN] {src} -> {dst}")
        return

    if no_parallel or len(file_pairs) == 1:
        for src, dst in file_pairs:
            process_file(src, dst, size=size, mode=mode, output_format=output_format)
    else:
        max_workers = min(os.cpu_count() or 4, len(file_pairs))
        logger.info(f"Processing with {max_workers} workers")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_file, src, dst, size, mode, output_format): (src, dst)
                for src, dst in file_pairs
            }
            with tqdm(total=len(futures), desc="Processing files", unit="file") as pbar:
                for future in as_completed(futures):
                    src, _ = futures[future]
                    try:
                        success = future.result()
                        if not success:
                            logger.error(f"Failed: {src}")
                    except Exception as exc:
                        logger.error(f"Exception processing {src}: {exc}")
                    finally:
                        # Advance only when each submitted file has finished.
                        pbar.update(1)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fill small grayscale holes by grey closing and save delta image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Fill grayscale small holes (default settings)
    environment: uv@3.11:segmentation
  commands:
  - python
  - '%REPO%/standard_code/python/fill_greyscale_small_holes.py'
  - --input-search-pattern: '%YAML%/input/**/*.ome.tif'
  - --output-folder: '%YAML%/output'
  - --log-level: INFO

- name: Fill grayscale small holes (custom kernel and mode)
    environment: uv@3.11:segmentation
  commands:
  - python
  - '%REPO%/standard_code/python/fill_greyscale_small_holes.py'
  - --input-search-pattern: '%YAML%/input/**/*.ome.tif'
  - --output-folder: '%YAML%/output'
  - --size: 10
  - --mode: reflect
    - --output-format: npy
  - --no-parallel
  - --log-level: INFO

- name: Fill grayscale small holes (save as npy)
    environment: uv@3.11:segmentation
    commands:
    - python
    - '%REPO%/standard_code/python/fill_greyscale_small_holes.py'
    - --input-search-pattern: '%YAML%/input/**/*.ome.tif'
    - --output-folder: '%YAML%/output'
    - --output-format: npy
    - --log-level: INFO
        """,
    )

    parser.add_argument(
        "--input-search-pattern",
        type=str,
        required=True,
        help="Input file pattern (supports wildcards, use '**' for recursive search)",
    )

    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Output folder (default: input_folder + '_filled')",
    )

    parser.add_argument(
        "--size",
        type=int,
        default=5,
        help="Grey closing kernel size for Y and X dimensions (default: 5)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="reflect",
        choices=["reflect", "nearest", "mirror", "constant", "wrap"],
        help="Border mode passed to scipy.ndimage.grey_closing (default: reflect)",
    )

    parser.add_argument(
        "--output-format",
        type=str,
        default="ome-tif",
        choices=["ome-tif", "npy"],
        help="Output format for delta image (default: ome-tif)",
    )

    parser.add_argument(
        "--collapse-delimiter",
        type=str,
        default="__",
        help="Delimiter for collapsing subfolder paths (default: '__')",
    )

    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing (process files sequentially)",
    )

    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_filled",
        help="Additional suffix to add before output extension (default: _filled)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without executing",
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version and exit",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: WARNING)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if args.version:
        version_file = Path(__file__).parent.parent.parent / "VERSION"
        try:
            version = version_file.read_text(encoding="utf-8").strip()
        except Exception:
            version = "unknown"
        print(f"fill_greyscale_small_holes.py version: {version}")
        return

    if args.size < 1:
        raise ValueError("--size must be >= 1")

    start_time = time()
    process_files(
        input_pattern=args.input_search_pattern,
        output_folder=args.output_folder,
        size=args.size,
        mode=args.mode,
        output_format=args.output_format,
        collapse_delimiter=args.collapse_delimiter,
        no_parallel=args.no_parallel,
        output_extension=args.output_suffix,
        dry_run=args.dry_run,
    )
    logger.info(f"Finished in {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
