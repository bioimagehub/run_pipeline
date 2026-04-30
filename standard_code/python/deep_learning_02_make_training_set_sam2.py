"""
deep_learning_02_make_training_set_sam2.py
------------------------------------------
SAM2-backed training-set generator for deep-learning centroid detection.

Extends deep_learning_01 by running SAM2 VideoPredictor in a **background thread**
after each napari centroid click.  The predictor propagates the segmentation
across *all* timepoints of the source video, yielding many more training
examples per annotation than the single mid-frame approach in _01.

Output folder structure
-----------------------
<output-folder>/
    manual_labelling/           # session folders with saved annotations
        YYYYMMDD_N/
            coords.npy
            frames.npy
    training_set/
        input/          # sub-patch images  (float32 .tif)
        target/         # Gaussian blob maps (float32 .tif)
        target_sam2/    # SAM2 binary masks  (float32 .tif, values 0.0 or 1.0)

Channel encoding
----------------
When exporting crop frames as JPEG for SAM2, available image channels are mapped
to R/G/B (ch0→R, ch1→G if C≥2, ch2→B if C≥3).  Any unfilled channel copies ch0.
Each channel is normalised independently (percentile 0.1–99.9 → uint8).

TODO
----
- Add a validation step to review SAM2-predicted masks in napari before they
  are written to target_sam2/ (currently saved automatically without review).

MIT License - BIPHUB, University of Oslo
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import queue
import re
import shutil
import tempfile
import threading
from collections import defaultdict
from contextlib import nullcontext

import numpy as np
import napari
import tifffile
import torch
from tqdm import tqdm
from PIL import Image as PILImage
from bioio import BioImage
import bioio_ome_tiff

import bioimage_pipeline_utils as rp
from sam2_utils import load_sam2_predictor, _normalize_to_uint8

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers  (identical to deep_learning_01_make_training_set.py)
# ---------------------------------------------------------------------------

def _get_gaussian(radius: int) -> np.ndarray:
    """Create a normalised 2-D Gaussian kernel of the given pixel radius."""
    x = np.linspace(-radius, radius, radius * 2)
    y = np.linspace(-radius, radius, radius * 2)
    xx, yy = np.meshgrid(x, y)
    d = np.sqrt(xx ** 2 + yy ** 2)
    sigma = radius / 2.0
    gauss = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
    gauss = (gauss - gauss.min()) / (gauss.max() - gauss.min())
    return gauss


def centroids2images(
    point_list: list[list[float]],
    im_num_row: int,
    im_num_col: int,
    g_radius: int = 20,
) -> np.ndarray:
    """Stamp Gaussian blobs at each centroid onto a blank image."""
    circle_mat = _get_gaussian(g_radius)
    temp = np.zeros(
        (im_num_row + g_radius * 2, im_num_col + g_radius * 2),
        dtype=np.float32,
    )
    for pnt in point_list:
        r, c = int(pnt[0]), int(pnt[1])
        r0 = g_radius + r - g_radius
        r1 = g_radius + r + g_radius
        c0 = g_radius + c - g_radius
        c1 = g_radius + c + g_radius
        temp[r0:r1, c0:c1] = np.maximum(temp[r0:r1, c0:c1], circle_mat)
    return temp[g_radius:-g_radius, g_radius:-g_radius]


_SESS_RE = re.compile(r"^(\d{8})_(\d+)$")


def _find_latest_session(root: str) -> str | None:
    """Return the path to the most recent annotation session, or None."""
    if not os.path.isdir(root):
        return None
    sessions: list[tuple[str, int, str]] = []
    for name in os.listdir(root):
        m = _SESS_RE.match(name)
        if m and os.path.isdir(os.path.join(root, name)):
            sessions.append((m.group(1), int(m.group(2)), name))
    if not sessions:
        return None
    sessions.sort(key=lambda x: (x[0], x[1]))
    return os.path.join(root, sessions[-1][2])


def _create_new_session(root: str) -> str:
    """Create and return a new timestamped session folder inside *root*."""
    os.makedirs(root, exist_ok=True)
    today = datetime.datetime.now().strftime("%Y%m%d")
    today_re = re.compile(rf"^{today}_(\d+)$")
    existing: list[int] = []
    for name in os.listdir(root):
        m = today_re.match(name)
        if m and os.path.isdir(os.path.join(root, name)):
            existing.append(int(m.group(1)))
    idx = max(existing, default=0) + 1
    folder = os.path.join(root, f"{today}_{idx}")
    os.makedirs(folder)
    return folder


# ---------------------------------------------------------------------------
# Thread-safe save-index counter
# ---------------------------------------------------------------------------

_save_lock = threading.Lock()
_save_counter = [0]  # mutable int in a list for thread-safe mutation


def _init_save_counter(dirs: list[str]) -> int:
    """Scan existing .tif files across output dirs and return next free index."""
    max_idx = -1
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if fname.endswith(".tif"):
                try:
                    idx = int(os.path.splitext(fname)[0])
                    max_idx = max(max_idx, idx)
                except ValueError:
                    pass
    return max_idx + 1


def _next_save_idx() -> int:
    with _save_lock:
        idx = _save_counter[0]
        _save_counter[0] += 1
    return idx


# ---------------------------------------------------------------------------
# Multi-channel JPEG export for SAM2
# ---------------------------------------------------------------------------

def _save_crop_as_jpeg(
    img_da,
    t: int,
    r0: int, r1: int,
    c0: int, c1: int,
    out_path: str,
    jpeg_quality: int = 95,
) -> None:
    """Export one spatially-cropped frame as a JPEG with multi-channel RGB mapping.

    img_da must be either (T, C, H, W) or (T, H, W).

    Channel mapping:
      ch 0 → R
      ch 1 → G  (if C ≥ 2, else copies ch 0)
      ch 2 → B  (if C ≥ 3, else copies ch 0)

    Each channel is normalised independently via percentile clipping.
    """
    if img_da.ndim == 4:
        C = img_da.shape[1]
        r_ch = _normalize_to_uint8(np.array(img_da[t, 0, r0:r1, c0:c1]))
        g_ch = _normalize_to_uint8(np.array(img_da[t, 1, r0:r1, c0:c1])) if C >= 2 else r_ch
        b_ch = _normalize_to_uint8(np.array(img_da[t, 2, r0:r1, c0:c1])) if C >= 3 else r_ch
    else:  # (T, H, W)
        r_ch = _normalize_to_uint8(np.array(img_da[t, r0:r1, c0:c1]))
        g_ch = b_ch = r_ch

    rgb = np.stack([r_ch, g_ch, b_ch], axis=-1)
    PILImage.fromarray(rgb).save(out_path, quality=jpeg_quality)


# ---------------------------------------------------------------------------
# SAM2 background worker
# ---------------------------------------------------------------------------

def _run_sam2_job(
    job: dict,
    predictor,
    train_input_dir: str,
    train_target_dir: str,
    train_target_sam2_dir: str,
    sub_patch_size: int,
    gaussian_radius: int,
    jpeg_quality: int = 95,
) -> None:
    """Process one SAM2 propagation job and stream-save the results.

    job keys
    --------
    file_path : str   — absolute path to the OME-TIFF timelapse
    r0, r1    : int   — crop row bounds (full-frame pixels)
    c0, c1    : int   — crop column bounds (full-frame pixels)
    y_crop    : float — click Y position within the crop (row, SAM2 uses (x,y))
    x_crop    : float — click X position within the crop (column)
    patch_idx : int   — index into all_patches_arr (for logging only)
    """
    file_path = job["file_path"]
    r0, r1    = job["r0"], job["r1"]
    c0, c1    = job["c0"], job["c1"]
    y_crop    = job["y_crop"]
    x_crop    = job["x_crop"]
    patch_idx = job["patch_idx"]

    crop_h = r1 - r0
    crop_w = c1 - c0
    sps    = sub_patch_size

    logger.info(
        "SAM2 job start: patch_idx=%d  file=%s",
        patch_idx, os.path.basename(file_path),
    )

    # ---- Load timelapse lazily (dask) ---------------------------------------
    bio = BioImage(file_path, reader=bioio_ome_tiff.Reader)
    try:
        img_da = bio.get_image_dask_data("TCYX", Z=0)
    except Exception:
        img_da = bio.get_image_dask_data("TYX", C=0, Z=0)

    T     = img_da.shape[0]
    mid_t = T // 2

    device = predictor._device
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device == "cuda" else nullcontext()
    )

    # ---- Export crop frames as JPEGs ----------------------------------------
    tmp_dir = tempfile.mkdtemp(prefix="sam2_job_")
    try:
        for t in tqdm(range(T), desc=f"Exporting JPEGs patch={patch_idx}", unit="frame", leave=False):
            _save_crop_as_jpeg(
                img_da, t, r0, r1, c0, c1,
                os.path.join(tmp_dir, f"{t:05d}.jpg"),
                jpeg_quality,
            )

        # SAM2 point prompt: (x, y) = (column, row) convention
        pts    = np.array([[x_crop, y_crop]], dtype=np.float32)
        labels = np.array([1],               dtype=np.int32)

        # ---- Forward propagation: mid_t → T-1 --------------------------------
        masks: dict[int, np.ndarray] = {}  # frame_idx -> (crop_h, crop_w) bool

        with torch.inference_mode(), autocast_ctx:
            inf_state = predictor.init_state(
                video_path=tmp_dir,
                offload_video_to_cpu=(device == "cpu"),
            )
            predictor.add_new_points_or_box(
                inf_state,
                frame_idx=mid_t,
                obj_id=1,
                points=pts,
                labels=labels,
            )
            for frame_idx, _obj_ids, video_res_masks in predictor.propagate_in_video(
                inf_state, start_frame_idx=mid_t,
            ):
                masks[frame_idx] = (
                    video_res_masks[0, 0].float() > 0.0
                ).cpu().numpy()
            predictor.reset_state(inf_state)

        # ---- Backward propagation: mid_t → 0 --------------------------------
        with torch.inference_mode(), autocast_ctx:
            inf_state = predictor.init_state(
                video_path=tmp_dir,
                offload_video_to_cpu=(device == "cpu"),
            )
            predictor.add_new_points_or_box(
                inf_state,
                frame_idx=mid_t,
                obj_id=1,
                points=pts,
                labels=labels,
            )
            for frame_idx, _obj_ids, video_res_masks in predictor.propagate_in_video(
                inf_state, start_frame_idx=mid_t, reverse=True,
            ):
                if frame_idx not in masks:  # forward result takes priority at mid_t
                    masks[frame_idx] = (
                        video_res_masks[0, 0].float() > 0.0
                    ).cpu().numpy()
            predictor.reset_state(inf_state)

        # ---- Load raw ch-0 crops for input sub-patches ----------------------
        if img_da.ndim == 4:
            raw_crops = {
                t: np.array(img_da[t, 0, r0:r1, c0:c1])
                for t in tqdm(masks, desc=f"Loading crops patch={patch_idx}", unit="frame", leave=False)
            }
        else:
            raw_crops = {
                t: np.array(img_da[t, r0:r1, c0:c1])
                for t in tqdm(masks, desc=f"Loading crops patch={patch_idx}", unit="frame", leave=False)
            }

        # ---- Stream-save sub-patches for each valid frame -------------------
        os.makedirs(train_input_dir,       exist_ok=True)
        os.makedirs(train_target_dir,      exist_ok=True)
        os.makedirs(train_target_sam2_dir, exist_ok=True)

        n_saved      = 0
        valid_frames = 0

        for t, mask in tqdm(sorted(masks.items()), desc=f"Saving patches patch={patch_idx}", unit="frame", leave=False):
            if mask.sum() == 0:
                continue
            valid_frames += 1

            raw_crop = raw_crops[t]

            # Pad to full crop size if the frame was at the image boundary
            if raw_crop.shape != (crop_h, crop_w):
                tmp = np.zeros((crop_h, crop_w), dtype=raw_crop.dtype)
                tmp[: raw_crop.shape[0], : raw_crop.shape[1]] = raw_crop
                raw_crop = tmp

            mask_full = np.zeros((crop_h, crop_w), dtype=bool)
            mask_full[: mask.shape[0], : mask.shape[1]] = mask

            # Derive Gaussian label from SAM2 mask centroid
            ys, xs = np.where(mask_full)
            cy = float(ys.mean())
            cx = float(xs.mean())
            blob = centroids2images([[cy, cx]], crop_h, crop_w, g_radius=gaussian_radius)

            # Split crop into sub-patches
            for sr in range(0, crop_h, sps):
                for sc in range(0, crop_w, sps):
                    inp_sub  = raw_crop[sr : sr + sps, sc : sc + sps]
                    blob_sub = blob[sr : sr + sps, sc : sc + sps]
                    mask_sub = mask_full[sr : sr + sps, sc : sc + sps].astype(np.float32)

                    # Skip sub-patches smaller than sps (crop edge)
                    if inp_sub.shape != (sps, sps):
                        continue

                    idx   = _next_save_idx()
                    fname = f"{idx}.tif"
                    tifffile.imwrite(
                        os.path.join(train_input_dir,       fname),
                        inp_sub.astype(np.float32),
                    )
                    tifffile.imwrite(
                        os.path.join(train_target_dir,      fname),
                        blob_sub.astype(np.float32),
                    )
                    tifffile.imwrite(
                        os.path.join(train_target_sam2_dir, fname),
                        mask_sub,
                    )
                    n_saved += 1

        logger.info(
            "SAM2 job done: patch_idx=%d  T_valid=%d/%d  saved=%d sub-patches",
            patch_idx, valid_frames, T, n_saved,
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _worker_loop(
    job_queue: queue.Queue,
    predictor,
    train_input_dir: str,
    train_target_dir: str,
    train_target_sam2_dir: str,
    sub_patch_size: int,
    gaussian_radius: int,
    jpeg_quality: int,
) -> None:
    """Background worker: pulls jobs from the queue and runs SAM2 propagation."""
    while True:
        job = job_queue.get()
        if job is None:  # sentinel value — shut down cleanly
            job_queue.task_done()
            break
        try:
            _run_sam2_job(
                job, predictor,
                train_input_dir, train_target_dir, train_target_sam2_dir,
                sub_patch_size, gaussian_radius, jpeg_quality,
            )
        except Exception:
            logger.exception("SAM2 job failed for patch_idx=%s", job.get("patch_idx"))
        finally:
            job_queue.task_done()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Annotate microscopy patches with napari and generate a SAM2-backed "
            "training set with temporal propagation across all video timepoints."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: SAM2-backed training set generation (default settings)
  environment: uv@3.11:deep-learning
  commands:
  - python
  - '%REPO%/standard_code/python/deep_learning_02_make_training_set_sam2.py'
  - --input-search-pattern: '%YAML%/input_drift_corrected/**/*.ome.tif'
  - --output-folder: '%YAML%/deep_learning_output'

- name: SAM2-backed training set generation (custom parameters)
  environment: uv@3.11:deep-learning
  commands:
  - python
  - '%REPO%/standard_code/python/deep_learning_02_make_training_set_sam2.py'
  - --input-search-pattern: '%YAML%/input_drift_corrected/**/*.ome.tif'
  - --output-folder: '%YAML%/deep_learning_output'
  - --patch-size: 512
  - --sub-patch-size: 256
  - --gaussian-radius: 20
  - --sam2-model: facebook/sam2-hiera-large
  - --max-patches: 100
  - --zero-keep-ratio: 0.10
  - --log-level: INFO

- name: Pause for manual inspection of training set
  type: pause
  message: 'Review generated training patches before continuing.'

- name: Stop intentionally
  type: stop
  message: 'Pipeline stopped intentionally.'

- name: Force reprocessing for later segments
  type: force
  message: 'Reprocessing all subsequent steps.'

Note: --no-force, --no-parallel, and --maxcores are not supported by this module.
This script is an interactive napari annotation tool; it manages session resumption
via timestamped session folders in manual_labelling/ rather than per-file skipping.
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
        help=(
            "Root output folder.  Sub-folders 'manual_labelling/' and "
            "'training_set/{input,target,target_sam2}/' are created inside it."
        ),
    )
    parser.add_argument(
        "--patch-size",
        type=int, default=512,
        help="Size of square patches extracted from the mid-timepoint frame (default: 512).",
    )
    parser.add_argument(
        "--sub-patch-size",
        type=int, default=256,
        help="Size of square sub-patches split from each patch for saving (default: 256).",
    )
    parser.add_argument(
        "--max-patches",
        type=int, default=None,
        help="Limit the number of patches shown in napari. Default: all patches.",
    )
    parser.add_argument(
        "--gaussian-radius",
        type=int, default=20,
        help="Radius (px) for Gaussian blobs placed at each centroid (default: 20).",
    )
    parser.add_argument(
        "--zero-keep-ratio",
        type=float, default=0.10,
        help=(
            "Fraction of non-zero patch count to keep as background patches "
            "in the mid-frame Gaussian-blob pass (default: 0.10)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int, default=42,
        help="Random seed for reproducible background-patch downsampling (default: 42).",
    )
    parser.add_argument(
        "--sam2-model",
        default="facebook/sam2-hiera-large",
        help="HuggingFace model ID for SAM2 VideoPredictor (default: facebook/sam2-hiera-large).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int, default=95,
        help="JPEG quality for SAM2 crop frame export (default: 95).",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: WARNING).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    output_folder         = os.path.abspath(args.output_folder)
    labelling_dir         = os.path.join(output_folder, "manual_labelling")
    train_input_dir       = os.path.join(output_folder, "training_set", "input")
    train_target_dir      = os.path.join(output_folder, "training_set", "target")
    train_target_sam2_dir = os.path.join(output_folder, "training_set", "target_sam2")

    ps  = args.patch_size
    sps = args.sub_patch_size

    # ------------------------------------------------------------------- #
    # 1. Discover input files                                              #
    # ------------------------------------------------------------------- #
    search_subfolders = "**" in args.input_search_pattern
    files = sorted(
        rp.get_files_to_process2(
            args.input_search_pattern,
            search_subfolders=search_subfolders,
        )
    )
    if not files:
        raise SystemExit(f"No files matched pattern: {args.input_search_pattern}")

    logger.info("Found %d file(s)", len(files))
    for f in files:
        logger.info("  %s", os.path.basename(f))

    # ------------------------------------------------------------------- #
    # 2. Load mid-timepoint frame from each timelapse for napari display   #
    # ------------------------------------------------------------------- #
    frames: list[np.ndarray] = []
    for fpath in tqdm(files, desc="Loading frames", unit="file"):
        img    = rp.load_tczyx_image(fpath)
        t_mid  = img.dims.T // 2
        block  = img.get_image_data("ZYX", T=t_mid, C=0)
        frame  = np.max(block, axis=0) if block.shape[0] > 1 else block[0]
        frames.append(frame)

    max_h = max(f.shape[0] for f in frames)
    max_w = max(f.shape[1] for f in frames)
    padded_frames: list[np.ndarray] = []
    for f in frames:
        pad = np.zeros((max_h, max_w), dtype=f.dtype)
        pad[: f.shape[0], : f.shape[1]] = f
        padded_frames.append(pad)

    stacked = np.stack(padded_frames)
    logger.info("Stacked shape: %s  (N files, Y, X)", stacked.shape)

    # ------------------------------------------------------------------- #
    # 3. Extract non-overlapping patches                                   #
    # ------------------------------------------------------------------- #
    all_patches: list[np.ndarray] = []
    patch_meta:  list[dict]       = []   # {file_idx, file_path, r0, r1, c0, c1}

    for file_idx, (frame, fpath) in enumerate(zip(stacked, files)):
        h, w = frame.shape
        for r0 in range(0, h, ps):
            for c0 in range(0, w, ps):
                r1 = min(r0 + ps, h)
                c1 = min(c0 + ps, w)
                patch = np.zeros((ps, ps), dtype=frame.dtype)
                patch[: r1 - r0, : c1 - c0] = frame[r0:r1, c0:c1]
                all_patches.append(patch)
                patch_meta.append({
                    "file_idx":  file_idx,
                    "file_path": fpath,
                    "r0": r0, "r1": r1,
                    "c0": c0, "c1": c1,
                })

    all_patches_arr  = np.stack(all_patches)
    n_patches_total  = all_patches_arr.shape[0]
    n_patches_display = (
        min(n_patches_total, args.max_patches)
        if args.max_patches is not None
        else n_patches_total
    )
    logger.info("Total patches: %d  (shape %s)", n_patches_total, all_patches_arr.shape)

    # ------------------------------------------------------------------- #
    # 4. Session management                                                #
    # ------------------------------------------------------------------- #
    os.makedirs(labelling_dir, exist_ok=True)
    latest_session = _find_latest_session(labelling_dir)
    if latest_session and os.path.exists(os.path.join(latest_session, "coords.npy")):
        coords = np.load(os.path.join(latest_session, "coords.npy"))
        logger.info("Loaded %d points from previous session: %s", len(coords), latest_session)
    else:
        coords = np.empty((0, 3), dtype=np.float32)
        logger.info("No previous annotations found — starting fresh.")

    session_dir = _create_new_session(labelling_dir)
    logger.info("New session folder: %s", session_dir)

    # ------------------------------------------------------------------- #
    # 5. Initialise save counter and load SAM2 predictor                  #
    # ------------------------------------------------------------------- #
    _save_counter[0] = _init_save_counter(
        [train_input_dir, train_target_dir, train_target_sam2_dir]
    )
    logger.info("Save counter initialised to %d.", _save_counter[0])

    logger.info("Loading SAM2 VideoPredictor (%s)...", args.sam2_model)
    predictor = load_sam2_predictor(model_id=args.sam2_model)
    logger.info("SAM2 ready  (device: %s).", predictor._device)

    # ------------------------------------------------------------------- #
    # 6. Start background SAM2 worker thread                               #
    # ------------------------------------------------------------------- #
    sam2_queue: queue.Queue = queue.Queue()
    worker = threading.Thread(
        target=_worker_loop,
        args=(
            sam2_queue, predictor,
            train_input_dir, train_target_dir, train_target_sam2_dir,
            sps, args.gaussian_radius, args.jpeg_quality,
        ),
        daemon=True,
        name="sam2-worker",
    )
    worker.start()
    logger.info("SAM2 background worker started.")

    # ------------------------------------------------------------------- #
    # 7. Napari annotation                                                 #
    # ------------------------------------------------------------------- #
    viewer = napari.Viewer()
    viewer.add_image(all_patches_arr, name="patches")

    points_layer = viewer.add_points(
        coords if len(coords) > 0 else None,
        name="selected",
        ndim=3,
        size=10,
        face_color="red",
    )

    # Track which points have already been enqueued to avoid duplicate jobs.
    # Previous-session points are registered immediately so they are not
    # re-queued when the layer is first populated.
    _enqueued: set[tuple] = set()
    for row in coords:
        _enqueued.add(tuple(row.tolist()))

    def _on_data_change(event=None) -> None:  # noqa: ANN001
        pts_now = points_layer.data
        if len(pts_now) == 0:
            return

        # Autosave
        np.save(os.path.join(session_dir, "coords.npy"), pts_now)
        logger.debug("Auto-saved %d points.", len(pts_now))

        # Enqueue SAM2 jobs for any newly-added points
        for row in pts_now:
            key = tuple(row.tolist())
            if key in _enqueued:
                continue
            _enqueued.add(key)

            pidx = int(round(row[0]))
            if pidx < 0 or pidx >= n_patches_display:
                continue

            meta   = patch_meta[pidx]
            y_crop = float(row[1])
            x_crop = float(row[2])

            sam2_queue.put({
                "file_path": meta["file_path"],
                "r0": meta["r0"], "r1": meta["r1"],
                "c0": meta["c0"], "c1": meta["c1"],
                "y_crop":    y_crop,
                "x_crop":    x_crop,
                "patch_idx": pidx,
            })
            logger.info(
                "Queued SAM2 job: patch_idx=%d  y=%.0f  x=%.0f  queue_depth=%d",
                pidx, y_crop, x_crop, sam2_queue.qsize(),
            )

    viewer.dims.events.current_step.connect(_on_data_change)
    points_layer.events.data.connect(_on_data_change)

    logger.info("Napari is open. Click centroids on the 'selected' layer, then close the viewer.")
    napari.run()

    # ------------------------------------------------------------------- #
    # 8. Save final annotation coordinates                                 #
    # ------------------------------------------------------------------- #
    pts = points_layer.data
    autosaved_coords = os.path.join(session_dir, "coords.npy")
    if len(pts) > 0:
        np.save(autosaved_coords, pts)
        logger.info("Saved %d points from layer.", len(pts))
    elif os.path.exists(autosaved_coords):
        pts = np.load(autosaved_coords)
        logger.info(
            "Points layer empty on close; kept autosaved %d point(s).", len(pts)
        )
    else:
        np.save(autosaved_coords, pts)
        logger.warning("No points annotated in this session.")

    np.save(os.path.join(session_dir, "frames.npy"), all_patches_arr)

    # ------------------------------------------------------------------- #
    # 9. Drain the SAM2 queue before continuing                            #
    # ------------------------------------------------------------------- #
    pending = sam2_queue.qsize()
    if pending > 0:
        logger.info("Waiting for %d pending SAM2 job(s) to finish...", pending)
    sam2_queue.join()
    logger.info("SAM2 queue drained.")

    # Send sentinel to stop the worker thread gracefully
    sam2_queue.put(None)
    worker.join(timeout=30)

    # ------------------------------------------------------------------- #
    # 10. Gaussian-blob pass on mid-frame patches  (same as _01)           #
    #     Saves to training_set/input/ and training_set/target/.           #
    #     Provides balanced background patches and a compatible dataset    #
    #     for models that do not use target_sam2/.                         #
    # ------------------------------------------------------------------- #
    patch_points: dict[int, list[list[float]]] = defaultdict(list)
    for row in pts:
        pidx = int(round(row[0]))
        y, x = float(row[1]), float(row[2])
        if 0 <= pidx < n_patches_display:
            patch_points[pidx].append([y, x])

    label_images = np.zeros((n_patches_display, ps, ps), dtype=np.float32)
    for pidx, points in patch_points.items():
        label_images[pidx] = centroids2images(
            points, ps, ps, g_radius=args.gaussian_radius
        )

    nonzero_total = int((label_images.max(axis=(1, 2)) > 0).sum())
    logger.info(
        "Gaussian-blob labels: non-zero %d / %d patches",
        nonzero_total, n_patches_display,
    )

    work_patches = all_patches_arr[:n_patches_display]

    sub_frames: list[np.ndarray] = []
    sub_labels: list[np.ndarray] = []
    for i in tqdm(range(work_patches.shape[0]), desc="Splitting sub-patches", unit="patch"):
        for r in range(0, ps, sps):
            for c in range(0, ps, sps):
                sub_frames.append(work_patches[i, r : r + sps, c : c + sps])
                sub_labels.append(label_images[i, r : r + sps, c : c + sps])

    sub_frames_arr = np.stack(sub_frames)
    sub_labels_arr = np.stack(sub_labels)

    nonzero_mask  = sub_labels_arr.max(axis=(1, 2)) > 0
    nonzero_idx   = np.where(nonzero_mask)[0]
    zero_idx      = np.where(~nonzero_mask)[0]
    n_keep_zero   = max(1, int(round(len(nonzero_idx) * args.zero_keep_ratio)))
    rng           = np.random.default_rng(args.seed)
    kept_zero_idx = rng.choice(
        zero_idx, size=min(n_keep_zero, len(zero_idx)), replace=False
    )
    final_idx = np.sort(np.concatenate([nonzero_idx, kept_zero_idx]))

    logger.info("Non-zero sub-patches : %d", len(nonzero_idx))
    logger.info("Zero sub-patches kept: %d", len(kept_zero_idx))
    logger.info("Final mid-frame pairs: %d", len(final_idx))

    os.makedirs(train_input_dir,  exist_ok=True)
    os.makedirs(train_target_dir, exist_ok=True)

    for arr_idx in tqdm(final_idx, desc="Saving mid-frame pairs", unit="patch"):
        idx   = _next_save_idx()
        fname = f"{idx}.tif"
        tifffile.imwrite(
            os.path.join(train_input_dir,  fname),
            sub_frames_arr[arr_idx].astype(np.float32),
        )
        tifffile.imwrite(
            os.path.join(train_target_dir, fname),
            sub_labels_arr[arr_idx].astype(np.float32),
        )

    logger.info("Saved %d mid-frame pairs.", len(final_idx))
    logger.info("  input       -> %s", train_input_dir)
    logger.info("  target      -> %s", train_target_dir)
    logger.info("  target_sam2 -> %s", train_target_sam2_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
