"""
Preprocess labelled masks into Gaussian-blob centroid maps for deep-learning
training.

For each label in a mask the centroid is located and a normalised 2-D Gaussian
blob (peak = 1, background = 0) is stamped at that position.  The result is
equivalent to what ``deep_learning_01_make_training_set.py`` produces from
manual point clicks, but derived automatically from segmentation masks.

Output files keep their original names and are written into <output-folder>.

Output folder structure
-----------------------
<output-folder>/   <- Gaussian-blob maps (float32 .tif, original filenames)

MIT License - BIPHUB, University of Oslo
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import center_of_mass
from tqdm import tqdm

import bioimage_pipeline_utils as rp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gaussian helper (mirrors deep_learning_01_make_training_set.py)
# ---------------------------------------------------------------------------

def _get_gaussian(radius: int) -> np.ndarray:
    """Return a normalised 2-D Gaussian kernel of the given pixel radius."""
    x = np.linspace(-radius, radius, radius * 2)
    y = np.linspace(-radius, radius, radius * 2)
    xx, yy = np.meshgrid(x, y)
    d = np.sqrt(xx ** 2 + yy ** 2)
    sigma = radius / 2.0
    gauss = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
    gauss = (gauss - gauss.min()) / (gauss.max() - gauss.min())
    return gauss.astype(np.float32)


def _make_gauss_map(label_mask: np.ndarray, gaussian_radius: int) -> np.ndarray:
    """Return a Gaussian-blob centroid map from a labelled mask.

    For each unique non-zero label the centroid is computed and a Gaussian blob
    of *gaussian_radius* is stamped at that position.  Overlapping blobs are
    resolved with ``np.maximum``, matching the behaviour of
    ``centroids2images`` in ``deep_learning_01_make_training_set.py``.

    Args:
        label_mask:      2-D integer array (background = 0).
        gaussian_radius: Pixel radius of the Gaussian kernel.

    Returns:
        Float32 array with values in [0, 1], same shape as *label_mask*.
    """
    h, w = label_mask.shape
    g = gaussian_radius
    circle_mat = _get_gaussian(g)

    # Work on a padded canvas to handle blobs near the border
    canvas = np.zeros((h + g * 2, w + g * 2), dtype=np.float32)

    labels = np.unique(label_mask)
    for lbl in labels:
        if lbl == 0:
            continue
        cy, cx = center_of_mass(label_mask == lbl)
        r, c = int(round(cy)), int(round(cx))
        r0 = g + r - g
        r1 = g + r + g
        c0 = g + c - g
        c1 = g + c + g
        canvas[r0:r1, c0:c1] = np.maximum(canvas[r0:r1, c0:c1], circle_mat)

    return canvas[g:-g, g:-g]


# ---------------------------------------------------------------------------
# Per-file worker  (must be a module-level function for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _process_one(
    mask_path: str,
    output_dir: str,
    gaussian_radius: int,
    force: bool,
) -> str:
    """Process a single labelled-mask file.

    Writes a Gaussian-blob map with the same filename into *output_dir*.

    Args:
        mask_path:       Absolute path to the input labelled-mask file.
        output_dir:      Output directory for the Gaussian map.
        gaussian_radius: Pixel radius of the Gaussian kernel.
        force:           When False, skip files whose output already exists.

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

    gauss_map = _make_gauss_map(mask.astype(np.int32), gaussian_radius)
    tifffile.imwrite(str(out_path), gauss_map)

    logger.debug("Processed %s", mask_path)
    return f"ok: {mask_path}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Gaussian-blob centroid maps from labelled masks "
            "for deep-learning training. "
            "Each label's centroid receives a Gaussian blob (peak=1, background=0), "
            "equivalent to the targets produced by manual click annotation in "
            "deep_learning_01_make_training_set.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output files keep their original filenames and are written into --output-folder.

Example YAML config for run_pipeline.exe:
---
run:
- name: Generate Gaussian-blob targets from masks
  environment: uv@3.11:deep-learning
  commands:
  - python
  - '%REPO%/standard_code/python/deep_learning_02_preprocess_gauss.py'
  - --input-search-pattern: '%YAML%/masks/**/*.tif'
  - --output-folder: '%YAML%/deep_learning_output/training_set/target-gauss'

- name: Generate Gaussian-blob targets (custom radius)
  environment: uv@3.11:deep-learning
  commands:
  - python
  - '%REPO%/standard_code/python/deep_learning_02_preprocess_gauss.py'
  - --input-search-pattern: '%YAML%/masks/**/*.tif'
  - --output-folder: '%YAML%/deep_learning_output/training_set/target-gauss'
  - --gaussian-radius: 15

- name: Train with Gaussian-blob targets
  environment: uv@3.11:deep-learning
  commands:
  - python
  - '%REPO%/standard_code/python/deep_learning_03_train.py'
  - --input-search-pattern: '%YAML%/deep_learning_output/training_set/input/*.tif'
  - --target-search-pattern: '%YAML%/deep_learning_output/training_set/target-gauss/*.tif'

- name: Pause to inspect outputs
  type: pause
  message: 'Inspect Gaussian maps before training.'
        """,
    )

    parser.add_argument(
        "--input-search-pattern",
        required=True,
        help=(
            "Glob pattern for input labelled-mask files "
            "(e.g. 'masks/**/*.tif')."
        ),
    )
    parser.add_argument(
        "--output-folder",
        required=True,
        help="Folder where Gaussian-blob .tif files are written (original filenames).",
    )
    parser.add_argument(
        "--gaussian-radius",
        type=int,
        default=20,
        help="Pixel radius of the Gaussian blob stamped at each label centroid (default: 20).",
    )
    parser.add_argument(
        "--no-force",
        action="store_true",
        help="Do not reprocess if output file already exists (default behaviour).",
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

    logger.info("Output folder   : %s", output_folder)
    logger.info("Gaussian radius : %d px", args.gaussian_radius)

    # ---------------------------------------------------------------------- #
    # Process files                                                            #
    # ---------------------------------------------------------------------- #
    force = not args.no_force

    ok = skipped = errors = 0

    if args.no_parallel or len(files) == 1:
        for mask_path in tqdm(files, desc="Processing", unit="file"):
            try:
                result = _process_one(
                    mask_path, str(output_folder), args.gaussian_radius, force
                )
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
                ex.submit(
                    _process_one,
                    mask_path, str(output_folder), args.gaussian_radius, force,
                ): mask_path
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
