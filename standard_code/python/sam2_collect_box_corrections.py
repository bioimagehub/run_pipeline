"""
sam2_collect_box_corrections.py
--------------------------------
Interactive napari tool to collect box-corrected training samples for SAM2
macropinosome fine-tuning.

Mirrors deep_learning_01_make_training_set.py: loads all files matching
--input-search-pattern, extracts patches from one timepoint per file, and
stacks them into a (N, H, W) napari-compatible array so you can browse all
your images in one session.

Workflow
--------
1. All patches from matching files open as a (N, patch_H, patch_W) stack in
   napari.
2. Navigate to the patch containing a macropinosome (use the N-axis slider).
3. Select the "Boxes" shapes layer, choose the rectangle tool, and draw a
   box around one macropinosome.
4. Press P to process:
   - A padded crop is extracted from that patch (box + --margin px).
   - The algorithmic segmentation runs (green overlay = training target).
   - If --model-id is given, SAM2's current prediction appears in blue.
5. Press A to accept and save the pair.  Press D to discard and redraw.

Algorithmic segmentation (Python port of the BIPHUB ImageJ macro)
------------------------------------------------------------------
  1. Grayscale fill-holes via morphological reconstruction from border seeds
     (mirrors ImageJ "Fill Holes (Binary/Gray)").
  2. Sample the pixel value at the box center after filling.
  3. Threshold: keep pixels in [center_val − tol, center_val + tol].
  4. Binary fill-holes.
  5. 4-connected component labeling; keep the component at the box center.

Output — feeds directly into sam2_macropinosome_training.py
------------------------------------------------------------
<output-folder>/
    training_set/
        input/          float32 image crops (.tif)
        target_sam2/    float32 binary algorithmic masks (.tif, 0.0 or 1.0)

Controls
--------
  P  — Process the latest drawn box
  A  — Accept: save crop + mask, clear overlay
  D  — Discard: clear overlay and shapes layer, redraw

MIT License - BIPHUB, University of Oslo
"""

from __future__ import annotations

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import tifffile

import bioimage_pipeline_utils as rp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Algorithmic segmentation (Python equivalent of the ImageJ macro)
# ---------------------------------------------------------------------------

def segment_algorithmic(
    crop_2d: np.ndarray,
    center_y: int,
    center_x: int,
    tolerance: float = 0.0,
) -> np.ndarray:
    """Segment a macropinosome from a 2-D crop using the BIPHUB ImageJ algorithm.

    The algorithm is a pixel-exact Python port of the ImageJ macro used at
    BIPHUB for macropinosome segmentation:

      1. Grayscale fill-holes via morphological reconstruction (seeds = image
         border; fills local intensity minima from the border inward).
      2. Sample the pixel value at (center_y, center_x) — the macropinosome
         interior after filling.
      3. Binary threshold at [center_val − tolerance, center_val + tolerance].
      4. Binary fill-holes to close any interior gaps.
      5. 4-connected component labeling (connectivity=1 in skimage).
      6. Retain only the component that contains the seed point.

    Args:
        crop_2d:   (H, W) float or integer image array (any dtype).
        center_y:  Row index of the seed point within *crop_2d*.
        center_x:  Column index of the seed point within *crop_2d*.
        tolerance: ±tolerance around the center pixel value used in the
                   threshold step.  0 = exact match (default).

    Returns:
        (H, W) float32 binary mask; 1.0 inside the macropinosome, 0.0 outside.
    """
    from scipy.ndimage import binary_fill_holes
    from skimage.measure import label
    from skimage.morphology import reconstruction

    crop_f = crop_2d.astype(np.float32)
    H, W = crop_f.shape

    # Step 1: Grayscale fill-holes (mirrors ImageJ "Fill Holes (Binary/Gray)")
    # Build a seed image that equals the crop on the border and is max elsewhere.
    # Reconstruction by erosion then fills holes (local minima) upward.
    seed = crop_f.copy()
    seed[1:-1, 1:-1] = crop_f.max()
    filled = reconstruction(seed, crop_f, method="erosion").astype(np.float32)

    # Step 2: Sample center value
    cy = int(np.clip(center_y, 0, H - 1))
    cx = int(np.clip(center_x, 0, W - 1))
    center_val = float(filled[cy, cx])

    # Step 3: Threshold (ImageJ: setThreshold(id, id) → Convert to Mask)
    binary = (filled >= center_val - tolerance) & (filled <= center_val + tolerance)

    # Step 4: Binary fill-holes (ImageJ: run("Fill Holes"))
    binary = binary_fill_holes(binary)

    # Step 5: 4-connected component labeling (ImageJ: connectivity=4, 8-bit)
    labeled = label(binary.astype(np.uint8), connectivity=1)  # connectivity=1 → 4-connected

    # Step 6: Keep the component at the seed point
    comp_id = int(labeled[cy, cx])
    if comp_id == 0:
        logger.warning(
            "No component found at seed (%d, %d) after labeling; "
            "the macropinosome may not be at the box center.  "
            "Returning empty mask.",
            cy, cx,
        )
        return np.zeros((H, W), dtype=np.float32)

    return (labeled == comp_id).astype(np.float32)


# ---------------------------------------------------------------------------
# SAM2 preview helper (inference only — no gradient tracking needed here)
# ---------------------------------------------------------------------------

def _predict_sam2_crop(
    predictor,
    crop_2d: np.ndarray,
    box_xyxy_in_crop: np.ndarray,
) -> np.ndarray:
    """Run SAM2 image prediction with a box prompt on a 2-D grayscale crop.

    Args:
        predictor:          SAM2ImagePredictor instance.
        crop_2d:            (H, W) grayscale image.
        box_xyxy_in_crop:   [x1, y1, x2, y2] in crop pixel coords.

    Returns:
        (H, W) bool mask.
    """
    import contextlib
    import torch
    from sam2_utils import _normalize_to_uint8

    image_rgb = np.stack([_normalize_to_uint8(crop_2d)] * 3, axis=-1)  # (H, W, 3) uint8
    device = getattr(predictor, "_device", "cpu")
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast_ctx:
        predictor.set_image(image_rgb)
        masks, _scores, _logits = predictor.predict(
            box=box_xyxy_in_crop.astype(np.float32),
            multimask_output=False,
        )
    return masks[0]  # (H, W) bool


# ---------------------------------------------------------------------------
# Multi-file patch loading (mirrors deep_learning_01_make_training_set.py)
# ---------------------------------------------------------------------------

def _load_patches(
    files: list[str],
    patch_size: int,
    channel: int,
    z_index: int,
    max_patches: int | None,
) -> tuple[np.ndarray, list[tuple[int, int, int, int, int]]]:
    """Load a selected timepoint from each file, extract a grid of patches.

    Args:
        files:       Sorted list of input file paths.
        patch_size:  Square patch side length in pixels.
        channel:     Channel index to use (0-based).
        z_index:     Z-slice index; -1 = middle slice.
        max_patches: Limit total patches; None = use all.

    Returns:
        patches_arr : (N, patch_size, patch_size) float32 array.
        patch_meta  : list of (file_idx, r0, c0, r1, c1) per patch.
    """
    frames: list[np.ndarray] = []
    for fpath in files:
        img = rp.load_tczyx_image(fpath)
        C = img.dims.C
        Z = img.dims.Z
        T = img.dims.T
        ch = int(np.clip(channel, 0, C - 1))
        zi = int(np.clip(z_index if z_index >= 0 else Z // 2, 0, Z - 1))
        t_mid = T // 2
        block = img.get_image_data("ZYX", T=t_mid, C=ch)
        frame = block[zi] if block.shape[0] > 1 else block[0]
        frames.append(frame.astype(np.float32))
        logger.info("Loaded %s  shape=%s  ch=%d  z=%d", Path(fpath).name, frame.shape, ch, zi)

    # Pad all frames to the same spatial shape
    max_h = max(f.shape[0] for f in frames)
    max_w = max(f.shape[1] for f in frames)
    padded: list[np.ndarray] = []
    for f in frames:
        pad = np.zeros((max_h, max_w), dtype=np.float32)
        pad[: f.shape[0], : f.shape[1]] = f
        padded.append(pad)

    # Extract non-overlapping patch grid from each frame
    ps = patch_size
    all_patches: list[np.ndarray] = []
    patch_meta: list[tuple[int, int, int, int, int]] = []  # (file_idx, r0, c0, r1, c1)

    for file_idx, frame in enumerate(padded):
        H, W = frame.shape
        for r0 in range(0, H, ps):
            for c0 in range(0, W, ps):
                r1 = min(r0 + ps, H)
                c1 = min(c0 + ps, W)
                patch = np.zeros((ps, ps), dtype=np.float32)
                patch[: r1 - r0, : c1 - c0] = frame[r0:r1, c0:c1]
                all_patches.append(patch)
                patch_meta.append((file_idx, r0, c0, r1, c1))
                if max_patches is not None and len(all_patches) >= max_patches:
                    break
            if max_patches is not None and len(all_patches) >= max_patches:
                break

    patches_arr = np.stack(all_patches).astype(np.float32)
    logger.info(
        "Patches: %d total  shape=%s  from %d file(s)",
        len(all_patches), patches_arr.shape, len(files),
    )
    return patches_arr, patch_meta


# ---------------------------------------------------------------------------
# Main interactive collection routine
# ---------------------------------------------------------------------------

def run_interactive_collection(
    files: list[str],
    output_folder: Path,
    channel: int,
    z_index: int,
    patch_size: int,
    max_patches: int | None,
    margin: int,
    tolerance: float,
    model_id: str | None,
) -> None:
    """Open napari and start the interactive box-correction collection session.

    Args:
        files:          List of source image paths to load.
        output_folder:  Root folder where training_set/input/ and
                        training_set/target_sam2/ will be written.
        channel:        Channel index (C dimension) to use.
        z_index:        Z-slice index; -1 = middle slice.
        patch_size:     Square patch side length for grid extraction.
        max_patches:    Maximum patches to show; None = all.
        margin:         Extra pixels around the drawn box for the crop.
        tolerance:      ±tolerance for the algorithmic threshold step.
        model_id:       HuggingFace SAM2 model ID for preview, or None to skip.
    """
    import napari

    # ---- Load patches (same approach as deep_learning_01) ------------------
    patches_arr, patch_meta = _load_patches(
        files, patch_size, channel, z_index, max_patches
    )
    N, ps_h, ps_w = patches_arr.shape

    # ---- Optional SAM2 predictor (preview only) ----------------------------
    predictor = None
    if model_id:
        try:
            from sam2_utils import load_sam2_image_predictor
            predictor = load_sam2_image_predictor(model_id)
            logger.info("SAM2 loaded for preview: %s", model_id)
        except Exception as exc:
            logger.warning(
                "SAM2 preview unavailable (%s). Only the algorithmic mask will be shown.",
                exc,
            )

    # ---- Output directories ------------------------------------------------
    input_dir = output_folder / "training_set" / "input"
    target_dir = output_folder / "training_set" / "target_sam2"
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Start counter past any existing files so sessions accumulate cleanly
    existing = sorted(
        [int(p.stem) for p in input_dir.glob("*.tif") if p.stem.isdigit()]
    )
    sample_counter = [existing[-1] + 1 if existing else 0]

    # ---- Mutable pending-sample state (closure-shared dict) ----------------
    pending: dict = {"crop": None, "algo_mask": None, "patch_idx": None}

    # ---- Napari viewer — (N, ps_h, ps_w) stack like dl_01 -----------------
    file_names = ", ".join(Path(f).name for f in files[:3])
    if len(files) > 3:
        file_names += f" … (+{len(files) - 3} more)"
    viewer = napari.Viewer(
        title=f"Box Corrections — {file_names}  [{N} patches]"
    )

    viewer.add_image(patches_arr, name="Patches", colormap="gray")

    shapes_layer = viewer.add_shapes(
        name="Boxes",
        shape_type="rectangle",
        edge_color="yellow",
        face_color="transparent",
        edge_width=2,
    )

    # Single-frame overlay layers (2-D, updated each time P is pressed)
    _H, _W = display_data.shape[-2], display_data.shape[-1]
    # (N, ps_h, ps_w) overlays — one frame per patch, zeros elsewhere
    algo_overlay = np.zeros((N, ps_h, ps_w), dtype=np.float32)
    sam2_overlay = np.zeros((N, ps_h, ps_w), dtype=np.float32)

    algo_layer = viewer.add_image(
        algo_overlay,
        name="Algorithmic mask  [target — press A to save]",
        colormap="green",
        opacity=0.6,
        visible=False,
        blending="additive",
    )
    sam2_layer = viewer.add_image(
        sam2_overlay,
        name="SAM2 current prediction  [preview only]",
        colormap="blue",
        opacity=0.45,
        visible=False,
        blending="additive",
    )

    viewer.status = (
        f"{N} patches loaded. Navigate with the N-slider, draw a rectangle "
        "in the 'Boxes' layer, then press P.  A = accept & save.  D = discard."
    )

    # ---- Helper: extract patch_idx + box coords from last drawn shape ------
    def _last_box_nyx() -> tuple[int, int, int, int, int] | None:
        """Return (patch_idx, y1, x1, y2, x2) from the last shape in Boxes.

        For a 3-D viewer (N, Y, X) napari stores shape coords as (4, 3) where
        column 0 = N (patch index), column 1 = Y, column 2 = X.
        """
        if not shapes_layer.data:
            return None
        pts = np.array(shapes_layer.data[-1])
        patch_idx = int(round(float(pts[:, 0].mean())))
        patch_idx = int(np.clip(patch_idx, 0, N - 1))
        y1 = int(pts[:, 1].min())
        x1 = int(pts[:, 2].min())
        y2 = int(pts[:, 1].max())
        x2 = int(pts[:, 2].max())
        return patch_idx, y1, x1, y2, x2


    # ---- Key binding: P = Process ------------------------------------------
    @viewer.bind_key("p")
    def on_process(viewer: napari.Viewer) -> None:
        result = _last_box_nyx()
        if result is None:
            viewer.status = "No box drawn yet. Draw a rectangle in the 'Boxes' layer first."
            return

        patch_idx, y1, x1, y2, x2 = result
        patch = patches_arr[patch_idx]  # (ps_h, ps_w)

        # Extract padded crop within the patch
        ay1 = max(0, y1 - margin)
        ax1 = max(0, x1 - margin)
        ay2 = min(ps_h, y2 + margin)
        ax2 = min(ps_w, x2 + margin)
        crop = patch[ay1:ay2, ax1:ax2].copy()
        crop_h, crop_w = crop.shape

        # Seed = center of drawn box in crop coordinates
        cy = (y1 + y2) // 2 - ay1
        cx = (x1 + x2) // 2 - ax1

        viewer.status = "Running algorithmic segmentation…"
        try:
            algo_mask = segment_algorithmic(crop, cy, cx, tolerance=tolerance)
        except Exception as exc:
            logger.exception("Algorithmic segmentation failed: %s", exc)
            viewer.status = f"Segmentation error: {exc}"
            return

        # SAM2 preview (optional)
        box_in_crop = np.array(
            [x1 - ax1, y1 - ay1, x2 - ax1, y2 - ay1], dtype=np.float32
        )
        if predictor is not None:
            viewer.status = "Running SAM2 preview…"
            try:
                sam2_mask_crop = _predict_sam2_crop(predictor, crop, box_in_crop)
                sam2_overlay[patch_idx] = 0.0
                sam2_overlay[patch_idx, ay1:ay1 + crop_h, ax1:ax1 + crop_w] = (
                    sam2_mask_crop.astype(np.float32)
                )
                sam2_layer.data = sam2_overlay.copy()
                sam2_layer.visible = True
            except Exception as exc:
                logger.warning("SAM2 preview failed: %s", exc)
                sam2_layer.visible = False
        else:
            sam2_layer.visible = False

        # Show algorithmic mask in the correct patch frame
        algo_overlay[patch_idx] = 0.0
        algo_overlay[patch_idx, ay1:ay1 + crop_h, ax1:ax1 + crop_w] = algo_mask
        algo_layer.data = algo_overlay.copy()
        algo_layer.visible = True

        # Navigate viewer to the processed patch
        viewer.dims.set_current_step(0, patch_idx)

        # Store pending sample
        pending["crop"] = crop
        pending["algo_mask"] = algo_mask
        pending["patch_idx"] = patch_idx

        file_idx = patch_meta[patch_idx][0]
        n_pos = int(algo_mask.sum())
        viewer.status = (
            f"Patch {patch_idx}  file={Path(files[file_idx]).name}  "
            f"crop=[y:{ay1}:{ay1+crop_h}, x:{ax1}:{ax1+crop_w}]  "
            f"algo pixels={n_pos}  —  A = save   D = discard"
        )

    # ---- Key binding: A = Accept and save ----------------------------------
    @viewer.bind_key("a")
    def on_accept(viewer: napari.Viewer) -> None:
        crop = pending.get("crop")
        algo_mask = pending.get("algo_mask")
        if crop is None or algo_mask is None:
            viewer.status = "Nothing pending. Press P to process a box first."
            return

        idx = sample_counter[0]
        fname = f"{idx}.tif"
        tifffile.imwrite(str(input_dir / fname), crop.astype(np.float32))
        tifffile.imwrite(str(target_dir / fname), algo_mask.astype(np.float32))
        logger.info(
            "Saved sample %d  crop shape=%s  mask sum=%d",
            idx, crop.shape, int(algo_mask.sum()),
        )
        sample_counter[0] += 1
        _reset_state()
        viewer.status = (
            f"Saved sample {idx}.tif  |  total saved: {sample_counter[0]}  |  "
            "Draw the next box."
        )

    # ---- Key binding: D = Discard ------------------------------------------
    @viewer.bind_key("d")
    def on_discard(viewer: napari.Viewer) -> None:
        _reset_state()
        viewer.status = "Discarded. Draw a new box."

    # ---- Reset helper ------------------------------------------------------
    def _reset_state() -> None:
        pidx = pending.get("patch_idx")
        if pidx is not None:
            algo_overlay[pidx] = 0.0
            sam2_overlay[pidx] = 0.0
            algo_layer.data = algo_overlay.copy()
            sam2_layer.data = sam2_overlay.copy()
        pending["crop"] = None
        pending["algo_mask"] = None
        pending["patch_idx"] = None
        algo_layer.visible = False
        sam2_layer.visible = False
        shapes_layer.data = []

    napari.run()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive napari tool: draw boxes around macropinosomes across "
            "multiple input files, get algorithmic + SAM2 mask previews, and "
            "save training samples for sam2_macropinosome_training.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Mirrors deep_learning_01_make_training_set.py: patches from all matching files
are stacked into a single (N, H, W) napari array so every image in your
library is accessible in one session.

Algorithmic segmentation (Python port of the BIPHUB ImageJ macro):
  1. Grayscale fill-holes (morphological reconstruction from border seeds).
  2. Threshold at center pixel value ± --tolerance.
  3. Binary fill-holes.
  4. 4-connected components; keep the component at the box center.

Controls
--------
  Draw rectangle  — 'Boxes' shapes layer (rectangle tool)
  P               — process the latest drawn box
  A               — accept and save the algorithmic mask
  D               — discard and clear the overlay

Example YAML config for run_pipeline.exe:
---
run:
- name: Collect box corrections (algorithmic segmentation only)
  environment: uv@3.11:deep-learning
  commands:
  - python
  - '%REPO%/standard_code/python/sam2_collect_box_corrections.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.ome.tif'
  - --output-folder: '%YAML%/deep_learning_output'

- name: Collect box corrections (with SAM2 preview, custom patch size)
  environment: uv@3.11:deep-learning
  commands:
  - python
  - '%REPO%/standard_code/python/sam2_collect_box_corrections.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.ome.tif'
  - --output-folder: '%YAML%/deep_learning_output'
  - --model-id: facebook/sam2-hiera-large
  - --patch-size: 512
  - --channel: 0
  - --margin: 30
  - --tolerance: 1.0

- name: Train SAM2 adapters on collected samples
  environment: uv@3.11:deep-learning
  commands:
  - python
  - '%REPO%/standard_code/python/sam2_macropinosome_training.py'
  - --input-folder: '%YAML%/deep_learning_output'

- name: Pause to inspect TensorBoard logs
  type: pause
  message: 'Run: tensorboard --logdir deep_learning_output/sam2_adapted/sam2_adapter_logs'

- name: Stop intentionally
  type: stop
  message: 'Pipeline stopped intentionally.'
""",
    )

    parser.add_argument(
        "--input-search-pattern",
        required=True,
        help=(
            "Glob pattern for input timelapse files "
            "(e.g. '**/*.ome.tif').  All matching files are loaded and their "
            "patches are stacked in a single napari session."
        ),
    )
    parser.add_argument(
        "--output-folder",
        required=True,
        help=(
            "Root folder for training data.  "
            "Samples are written to <output-folder>/training_set/input/ "
            "and <output-folder>/training_set/target_sam2/."
        ),
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=512,
        help=(
            "Size of square patches extracted from each frame.  "
            "Default: 512.  Matches deep_learning_01_make_training_set.py."
        ),
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help=(
            "Limit the total number of patches shown in napari.  "
            "Default: use all patches."
        ),
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help=(
            "Channel index (C dimension) to use for segmentation and saving "
            "image patches.  Default: 0.  "
            "(The ImageJ macro used channel 1 which is index 0 in 0-based counting.)"
        ),
    )
    parser.add_argument(
        "--z-index",
        type=int,
        default=-1,
        help=(
            "Z-slice index to use.  -1 = middle slice (default).  "
            "For 2-D time-lapses this is always 0."
        ),
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=20,
        help=(
            "Extra pixels added around the drawn box when extracting the crop. "
            "Default: 20."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help=(
            "±tolerance around the center pixel value for the threshold step. "
            "Default: 0 (exact match, identical to the ImageJ macro).  "
            "Increase slightly (e.g. 1.0) for noisy images."
        ),
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help=(
            "HuggingFace SAM2 model ID to load for the preview overlay "
            "(e.g. facebook/sam2-hiera-large).  "
            "Omit to skip the SAM2 preview and collect algorithmic masks only."
        ),
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

    search_subfolders = "**" in args.input_search_pattern
    files = sorted(
        rp.get_files_to_process2(
            args.input_search_pattern,
            search_subfolders=search_subfolders,
        )
    )
    if not files:
        logger.error("No files matched pattern: %s", args.input_search_pattern)
        raise SystemExit(1)

    logger.info("Found %d file(s):", len(files))
    for f in files:
        logger.info("  %s", os.path.basename(f))

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    run_interactive_collection(
        files=files,
        output_folder=output_folder,
        channel=args.channel,
        z_index=args.z_index,
        patch_size=args.patch_size,
        max_patches=args.max_patches,
        margin=args.margin,
        tolerance=args.tolerance,
        model_id=args.model_id,
    )


if __name__ == "__main__":
    main()
