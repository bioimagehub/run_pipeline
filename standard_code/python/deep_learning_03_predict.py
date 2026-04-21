"""
Batch centroid U-Net inference using MONAI sliding-window stitching.

Loads each input timelapse, runs per-timepoint inference on the selected
channel, and saves a float32 prediction density map alongside the inputs.

Model path
----------
Point --model at the .ckpt produced by deep_learning_02_train.py:
  <input-folder>/model_training/checkpoints/last.ckpt
  <input-folder>/model_training/checkpoints/epoch042-val_loss0.0008.ckpt

MIT License - BIPHUB, University of Oslo
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import tifffile
from monai.inferers.utils import sliding_window_inference
from tqdm import tqdm

# Import utilities from companion files in the same directory
sys.path.insert(0, str(Path(__file__).parent))
from deep_learning_00_utils import UNet  # noqa: E402
from deep_learning_02_train import CentroidUNetModule  # noqa: E402

import bioimage_pipeline_utils as rp

# Module-level logger
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _contrast(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    """Percentile clip."""
    return np.clip(arr, np.percentile(arr, low), np.percentile(arr, high))


def _scale(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]."""
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def predict_image(
    model: torch.nn.Module,
    image: np.ndarray,
    patch_size: int = 256,
    overlap: float = 0.25,
    device: torch.device | str = "cuda",
    sw_batch_size: int = 32,
) -> np.ndarray:
    """Run sliding-window inference on a single (H, W) float32 image.

    Stitching is done on CPU; each patch is inferred on *device*.

    Args:
        model: Trained UNet in eval mode.
        image: 2-D float32 array, already contrast-normalised.
        patch_size: Square patch size for sliding window.
        overlap: Fractional overlap between neighbouring patches.
        device: Torch device for patch inference.
        sw_batch_size: Number of patches per GPU forward pass.

    Returns:
        2-D float32 prediction map, same spatial shape as *image*.
    """
    tensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    def _infer(patch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model(patch.to(device)).cpu()

    pred = sliding_window_inference(
        inputs=tensor,
        roi_size=(patch_size, patch_size),
        sw_batch_size=sw_batch_size,
        predictor=_infer,
        overlap=overlap,
        mode="gaussian",
        device=torch.device("cpu"),
    )
    return pred.squeeze().numpy()


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_single_file(
    input_path: str,
    output_path: str,
    model: torch.nn.Module,
    device: torch.device,
    patch_size: int,
    overlap: float,
    sw_batch_size: int,
    channel: int,
    force: bool,
) -> bool:
    """Load one timelapse, predict per timepoint, save density map stack."""
    try:
        if os.path.exists(output_path) and not force:
            logger.info("Output exists, skipping: %s", os.path.basename(output_path))
            return True

        logger.info("Processing: %s", os.path.basename(input_path))

        img = rp.load_tczyx_image(input_path)
        T, C, Z, Y, X = img.dims.T, img.dims.C, img.dims.Z, img.dims.Y, img.dims.X

        if channel >= C:
            logger.warning(
                "Channel %d requested but image has %d channels; using channel 0.",
                channel, C,
            )
            channel = 0

        preds: list[np.ndarray] = []
        for t in tqdm(
            range(T),
            desc=os.path.basename(input_path),
            unit="frame",
            leave=False,
            dynamic_ncols=True,
        ):
            # Get ZYX for this timepoint / channel, max-project Z -> YX
            block = img.get_image_data("ZYX", T=t, C=channel).astype(np.float32)
            frame = np.max(block, axis=0) if Z > 1 else block[0]

            frame = _scale(_contrast(frame, 0.1, 99.9))
            pred = predict_image(model, frame, patch_size, overlap, device, sw_batch_size)
            preds.append(pred)

        # Convert to TCZYX shape (T, C=1, Z=1, Y, X)
        pred_stack = np.stack(preds)  # (T, Y, X)
        pred_stack = pred_stack[:, np.newaxis, np.newaxis, :, :]  # (T, 1, 1, Y, X)

        # Save using rp.save_tczyx_image to preserve OME-TIFF and TCZYX
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        rp.save_tczyx_image(pred_stack, output_path)
        logger.info("Saved: %s  shape=%s", output_path, pred_stack.shape)
        return True

    except Exception as exc:
        logger.error("Failed %s: %s", input_path, exc, exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch centroid U-Net inference with MONAI sliding-window stitching.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Note: --no-parallel / --maxcores control file-level parallelism only.
GPU inference inside each file is always single-process.

Example YAML config for run_pipeline.exe:
---
run:
- name: Predict centroids (last checkpoint)
  environment: uv@3.11:deep-learning
  commands:
  - python
  - '%REPO%/standard_code/python/deep_learning_03_predict.py'
  - --input-search-pattern: '%YAML%/input_drift_corrected/**/*.ome.tif'
  - --output-folder: '%YAML%/deep_learning_output/predictions'
  - --model: '%YAML%/deep_learning_output/model_training/checkpoints/last.ckpt'

- name: Predict centroids (best checkpoint, custom patch size)
  environment: uv@3.11:deep-learning
  commands:
  - python
  - '%REPO%/standard_code/python/deep_learning_03_predict.py'
  - --input-search-pattern: '%YAML%/input_drift_corrected/**/*.ome.tif'
  - --output-folder: '%YAML%/deep_learning_output/predictions'
  - --model: '%YAML%/deep_learning_output/model_training/checkpoints/last.ckpt'
  - --patch-size: 512
  - --overlap: 0.5
  - --channel: 0
  - --output-suffix: _pred

- name: Pause for manual inspection
  type: pause
  message: 'Inspect prediction density maps before continuing.'
        """,
    )

    parser.add_argument(
        "--input-search-pattern",
        required=True,
        help="Glob pattern for input timelapse files (e.g. '**/*.ome.tif').",
    )
    parser.add_argument(
        "--output-folder",
        required=True,
        help="Folder where prediction .tif files are written.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Path to a Lightning .ckpt file produced by deep_learning_02_train.py. "
            "Typically: <deep_learning_output>/model_training/checkpoints/last.ckpt"
        ),
    )
    parser.add_argument(
        "--output-suffix",
        default="_pred",
        help="Suffix appended to the input stem for the output filename (default: _pred).",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Channel index to predict on (0-based, default: 0).",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Square patch size for sliding-window inference (default: 256).",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help="Fractional overlap between neighbouring patches (default: 0.25).",
    )
    parser.add_argument(
        "--sw-batch-size",
        type=int,
        default=32,
        help="Number of patches per GPU forward pass (default: 32).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess even if the output file already exists.",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable file-level parallel processing (recommended when using GPU).",
    )
    parser.add_argument(
        "--maxcores",
        type=int,
        default=None,
        help="Maximum CPU cores for parallel file processing. Ignored if --no-parallel is set.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # ------------------------------------------------------------------ #
    # Validate model                                                       #
    # ------------------------------------------------------------------ #
    checkpoint_path = Path(args.model).resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"Model file not found: {checkpoint_path}")

    # ------------------------------------------------------------------ #
    # Discover input files                                                 #
    # ------------------------------------------------------------------ #
    search_subfolders = "**" in args.input_search_pattern
    files = sorted(
        rp.get_files_to_process2(
            args.input_search_pattern,
            search_subfolders=search_subfolders,
        )
    )
    if not files:
        raise SystemExit(f"No files matched: {args.input_search_pattern}")
    logger.info("Found %d file(s)", len(files))

    # ------------------------------------------------------------------ #
    # Build output paths                                                   #
    # ------------------------------------------------------------------ #
    output_folder = Path(args.output_folder).resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    # Use rp.resolve_output_path to generate correct output filenames
    tasks: list[tuple[str, str]] = []
    for fpath in files:
        # Compose output path using output_folder and resolved filename
        out_name = rp.resolve_output_path(fpath, ".ome.tif", args.output_suffix)
        out_path = str(output_folder / Path(out_name).name)
        tasks.append((fpath, out_path))

    # ------------------------------------------------------------------ #
    # Load model once (shared across files)                                #
    # ------------------------------------------------------------------ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading checkpoint: %s", checkpoint_path)
    logger.info("Device: %s", device)
    lit_model = CentroidUNetModule.load_from_checkpoint(
        str(checkpoint_path), map_location="cpu"
    )
    model = lit_model.model.to(device).eval()

    # ------------------------------------------------------------------ #
    # GPU inference is single-process; default to sequential               #
    # ------------------------------------------------------------------ #
    if device.type == "cuda" and not args.no_parallel:
        logger.info("GPU detected; forcing sequential processing.")
        args.no_parallel = True

    # ------------------------------------------------------------------ #
    # Process files                                                        #
    # ------------------------------------------------------------------ #
    ok = 0
    if args.no_parallel or len(tasks) == 1:
        for inp, out in tqdm(tasks, desc="Files", unit="file", dynamic_ncols=True):
            if process_single_file(
                inp, out, model, device,
                args.patch_size, args.overlap, args.sw_batch_size,
                args.channel, args.force,
            ):
                ok += 1
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        max_workers = rp.resolve_maxcores(args.maxcores, len(tasks))
        logger.info("Processing in parallel (workers=%d)", max_workers)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(
                    process_single_file,
                    inp, out, model, device,
                    args.patch_size, args.overlap, args.sw_batch_size,
                    args.channel, args.force,
                )
                for inp, out in tasks
            ]
            for f in as_completed(futures):
                try:
                    if f.result():
                        ok += 1
                except Exception as exc:
                    logger.error("Task failed: %s", exc)

    logger.info("Done: %d succeeded, %d failed", ok, len(tasks) - ok)


if __name__ == "__main__":
    main()
