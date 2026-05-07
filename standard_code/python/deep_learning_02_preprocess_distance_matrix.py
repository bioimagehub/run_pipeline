"""
Preprocess labelled masks into normalised distance-transform probability maps
for deep-learning training.

For each labelled-mask image the Euclidean distance from each foreground pixel
to the nearest edge of its label region is computed and normalised per label so
that the pixel furthest from the label boundary equals 1.  Background pixels
and pixels outside every labelled region are set to 0.

Output files keep their original names and are written into <output-folder>.

Output folder structure
-----------------------
<output-folder>/   <- distance maps (float32 .tif, original filenames)

MIT License - BIPHUB, University of Oslo
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

import bioimage_pipeline_utils as rp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core distance-transform logic
# ---------------------------------------------------------------------------

def _make_distance_map(label_mask: np.ndarray) -> np.ndarray:
    """Return a normalised distance-transform probability map.

    For each labelled region the Euclidean distance from the label boundary is
    computed and scaled so that the pixel furthest from the edge equals 1.
    Background and pixels outside every labelled region remain 0.

    Args:
        label_mask: 2-D integer array with one unique non-zero integer per
                    object (background = 0).

    Returns:
        Float32 array with values in [0, 1], same shape as *label_mask*.
    """
    result = np.zeros(label_mask.shape, dtype=np.float32)
    for lbl in np.unique(label_mask):
        if lbl == 0:
            continue
        region = label_mask == lbl
        dist = distance_transform_edt(region).astype(np.float32)
        max_dist = float(dist.max())
        if max_dist > 0:
            result[region] = dist[region] / max_dist
    return result


# ---------------------------------------------------------------------------
# Per-file worker  (must be a module-level function for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _process_one(
    mask_path: str,
    output_dir: str,
    force: bool,
) -> str:
    """Process a single labelled-mask file.

    Writes a distance map with the same filename into *output_dir*.

    Args:
        mask_path:  Absolute path to the input labelled-mask file.
        output_dir: Output directory for the distance-transform map.
        force:      When False, skip files whose output already exists.

    Returns:
        Short status string (``"ok: …"`` or ``"skipped: …"``).
    """
    out_path = Path(output_dir) / Path(mask_path).name

    if not force and out_path.exists():
        logger.debug(
            "Skipping %s (output exists). Pass --force to reprocess.", mask_path
        )
        return f"skipped: {mask_path}"

    mask = tifffile.imread(mask_path)

    # Squeeze out trivial dimensions (single-channel / single-Z TIFFs)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(
            f"{mask_path}: expected a 2-D labelled mask, got shape {mask.shape}. "
            "Reduce to 2-D before running this script."
        )

    dist_map = _make_distance_map(mask.astype(np.int32))
    tifffile.imwrite(str(out_path), dist_map)

    logger.debug("Processed %s", mask_path)
    return f"ok: {mask_path}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate distance-transform probability maps from labelled masks "
            "for deep-learning training. "
            "Outputs paired binary-mask (input) and normalised distance-map "
            "(target) subfolders compatible with deep_learning_03_train.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output files keep their original filenames and are written into --output-folder.

Example YAML config for run_pipeline.exe:
---
run:
- name: Generate distance-transform targets
  environment: uv@3.11:deep-learning
  commands:
  - python
  - '%REPO%/standard_code/python/deep_learning_02_preprocess_distance_matrix.py'
  - --input-search-pattern: '%YAML%/masks/**/*.tif'
  - --output-folder: '%YAML%/deep_learning_output/training_set/target-distance'

- name: Pause to inspect outputs
  type: pause
  message: 'Inspect distance maps before training.'
        """,
    )

    parser.add_argument(
        "--input-search-pattern",
        required=True,
        help=(
            "Glob pattern for input labelled-mask files "
            "(e.g. 'masks/**/*.tif'). Files are sorted before indexing."
        ),
    )
    parser.add_argument(
        "--output-folder",
        required=True,
        help="Folder where distance-map .tif files are written (named 0.tif, 1.tif, …).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess files even if outputs already exist.",
    )
    parser.add_argument(
        "--no-force",
        action="store_true",
        help="Do not reprocess if both output files already exist (default behaviour).",
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
        help=(
            "Maximum number of CPU cores for parallel processing "
            "(default: all available minus 1). Ignored when --no-parallel is set."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: WARNING).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # ---------------------------------------------------------------------- #
    # Discover input files                                                     #
    # ---------------------------------------------------------------------- #
    search_subfolders = "**" in args.input_search_pattern
    files = sorted(
        rp.get_files_to_process2(
            args.input_search_pattern,
            search_subfolders=search_subfolders,
        )
    )
    if not files:
        raise SystemExit(f"No files matched: {args.input_search_pattern}")
    logger.info("Found %d mask file(s)", len(files))

    # ---------------------------------------------------------------------- #
    # Create output directory                                                 #
    # ---------------------------------------------------------------------- #
    output_folder = Path(args.output_folder).resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    logger.info("Output folder: %s", output_folder)

    # ---------------------------------------------------------------------- #
    # Process files                                                            #
    # ---------------------------------------------------------------------- #
    # force=True unless --no-force was explicitly passed
    force = not args.no_force

    ok = skipped = errors = 0

    if args.no_parallel or len(files) == 1:
        for mask_path in tqdm(files, desc="Processing", unit="file"):
            try:
                result = _process_one(mask_path, str(output_folder), force)
                if result.startswith("skipped"):
                    skipped += 1
                else:
                    ok += 1
            except Exception as exc:
                logger.error("Failed %s: %s", mask_path, exc)
                errors += 1
    else:
        max_workers = rp.resolve_maxcores(args.maxcores, len(files))
        logger.info("Processing in parallel (workers=%d)", max_workers)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_process_one, mask_path, str(output_folder), force): mask_path
                for mask_path in files
            }
            for fut in tqdm(
                as_completed(futures), total=len(futures), desc="Processing", unit="file"
            ):
                mask_path = futures[fut]
                try:
                    result = fut.result()
                    if result.startswith("skipped"):
                        skipped += 1
                    else:
                        ok += 1
                except Exception as exc:
                    logger.error("Failed %s: %s", mask_path, exc)
                    errors += 1

    logger.info(
        "Done: %d processed, %d skipped, %d errors", ok, skipped, errors
    )
    if errors:
        raise SystemExit(f"{errors} file(s) failed. See log output above.")


if __name__ == "__main__":
    main()
