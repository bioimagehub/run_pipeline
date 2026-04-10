"""
Intensity clipping / percentile normalization for microscopy images.

Applies robust percentile-based normalization per image or per timepoint.

MIT License - BIPHUB style compatible
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

import bioimage_pipeline_utils as rp

logger = logging.getLogger(__name__)


# ======================================================
# CORE NORMALIZATION
# ======================================================

def percentile_normalize(
        image: np.ndarray,
        pmin: float,
        pmax: float,
        output_mode: str = "uint"
) -> np.ndarray:
    """
    Percentile normalization.

    Args:
        image: numpy array
        pmin: lower percentile
        pmax: upper percentile
        output_mode:
            "uint"  -> rescale to original dtype range
            "float" -> return float32 [0,1]

    Returns:
        normalized image
    """

    dtype = image.dtype

    low = np.percentile(image, pmin)
    high = np.percentile(image, pmax)

    if high <= low:
        return image.copy()

    img = image.astype(np.float32)
    img = (img - low) / (high - low)
    img = np.clip(img, 0, 1)

    if output_mode == "float":
        return img.astype(np.float32)

    # Rescale back to original dtype range
    if np.issubdtype(dtype, np.integer):
        max_val = np.iinfo(dtype).max
        img = (img * max_val).round()
        return img.astype(dtype)
    else:
        return img.astype(dtype)


# ======================================================
# FILE PROCESSING
# ======================================================

def normalize_single_file(
        input_path: str,
        output_path: str,
        percentile_min: float,
        percentile_max: float,
        output_mode: str = "uint",
        per_timepoint: bool = True
) -> bool:

    try:
        logger.info(f"Normalizing: {os.path.basename(input_path)}")

        img = rp.load_tczyx_image(input_path)
        data = img.data  # TCZYX

        T, C, Z, Y, X = data.shape
        corrected = np.zeros_like(data)

        for c in range(C):
            if per_timepoint:
                for t in range(T):
                    corrected[t, c] = percentile_normalize(
                        data[t, c],
                        percentile_min,
                        percentile_max,
                        output_mode
                    )
            else:
                corrected[:, c] = percentile_normalize(
                    data[:, c],
                    percentile_min,
                    percentile_max,
                    output_mode
                )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        rp.save_tczyx_image(
            corrected,
            output_path,
            physical_pixel_sizes=getattr(img, "physical_pixel_sizes", None),
            channel_names=getattr(img, "channel_names", None)
        )

        logger.info(f"Saved: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed: {input_path} -> {e}")
        return False


# ======================================================
# MULTI-FILE PROCESSING
# ======================================================

def process_files(
        input_pattern: str,
        output_folder: Optional[str],
        percentile_min: float,
        percentile_max: float,
        output_mode: str,
        per_timepoint: bool,
        no_parallel: bool,
        output_extension: str,
        dry_run: bool
):

    search_subfolders = '**' in input_pattern
    files = rp.get_files_to_process2(input_pattern, search_subfolders)

    if not files:
        logger.error("No files found.")
        return

    logger.info(f"Found {len(files)} file(s)")

    base_folder = str(Path(files[0]).parent)

    if output_folder is None:
        output_folder = base_folder + "_normalized"

    file_pairs = []
    for src in files:
        name = os.path.basename(src)
        name = os.path.splitext(name)[0] + output_extension + ".tif"
        dst = os.path.join(output_folder, name)
        file_pairs.append((src, dst))

    if dry_run:
        print(f"[DRY RUN] Would process {len(file_pairs)} files")
        return

    if no_parallel or len(file_pairs) == 1:
        for src, dst in file_pairs:
            normalize_single_file(
                src, dst,
                percentile_min,
                percentile_max,
                output_mode,
                per_timepoint
            )
    else:
        max_workers = min(os.cpu_count() or 4, len(file_pairs))
        logger.info(f"Processing with {max_workers} workers")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    normalize_single_file,
                    src, dst,
                    percentile_min,
                    percentile_max,
                    output_mode,
                    per_timepoint
                ): src
                for src, dst in file_pairs
            }

            for future in as_completed(futures):
                src = futures[future]
                try:
                    success = future.result()
                    if not success:
                        logger.error(f"Failed: {src}")
                except Exception as e:
                    logger.error(f"Exception processing {src}: {e}")


# ======================================================
# CLI
# ======================================================

def main():

        parser = argparse.ArgumentParser(
                description="Percentile clipping / normalization for microscopy images.",
                formatter_class=argparse.RawDescriptionHelpFormatter,
                epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Normalize images to original dtype range
    environment: uv@3.11:default
    commands:
    - python
    - '%REPO%/standard_code/python/intensity_clip_normalize.py'
    - --input-search-pattern: '%YAML%/images/**/*.tif'
    - --output-folder: '%YAML%/normalized'

- name: Normalize each timepoint to float output
    environment: uv@3.11:default
    commands:
    - python
    - '%REPO%/standard_code/python/intensity_clip_normalize.py'
    - --input-search-pattern: '%YAML%/images/**/*.tif'
    - --output-folder: '%YAML%/normalized'
    - --percentile-min: 0.5
    - --percentile-max: 99.5
    - --output-mode: float
    - --per-timepoint
                """
        )

    parser.add_argument("--input-search-pattern", required=True)
    parser.add_argument("--output-folder", default=None)

    parser.add_argument("--percentile-min", type=float, default=1.0)
    parser.add_argument("--percentile-max", type=float, default=99.0)

    parser.add_argument(
        "--output-mode",
        choices=["uint", "float"],
        default="uint",
        help="Return uint scaled to original dtype or float32 [0,1]"
    )

    parser.add_argument(
        "--per-timepoint",
        action="store_true",
        help="Normalize each timepoint independently"
    )

    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_normalized"
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
        format='%(asctime)s | %(levelname)s | %(message)s'
    )

    process_files(
        input_pattern=args.input_search_pattern,
        output_folder=args.output_folder,
        percentile_min=args.percentile_min,
        percentile_max=args.percentile_max,
        output_mode=args.output_mode,
        per_timepoint=args.per_timepoint,
        no_parallel=args.no_parallel,
        output_extension=args.output_suffix,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()