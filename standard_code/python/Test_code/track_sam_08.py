# %% Imports
import sys
import os
import importlib
import subprocess
import tempfile
import threading
from contextlib import nullcontext

import numpy as np
import matplotlib.pyplot as plt
from bioio import BioImage
from joblib import Parallel, delayed
from scipy.ndimage import label as scipy_label, center_of_mass
from skimage.morphology import flood_fill, reconstruction
from skimage.measure import label as cc_label
import time
import tifffile
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sam2_utils
importlib.reload(sam2_utils)
from sam2_utils import load_sam2_video_predictor, _save_frames_as_jpeg


# %% GPU monitor (nvidia-smi polling in background thread)
class _GpuMonitor:
    """Background thread that polls nvidia-smi every `interval` seconds."""

    def __init__(self, interval: float = 1.0, gpu_index: int = 0) -> None:
        self.interval = interval
        self.gpu_index = gpu_index
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.util_pct:    list[float] = []
        self.mem_used_mb: list[float] = []

    def _poll(self) -> None:
        cmd = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop_event.is_set():
            try:
                out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
                parts = out.split(",")
                if len(parts) == 2:
                    self.util_pct.append(float(parts[0].strip()))
                    self.mem_used_mb.append(float(parts[1].strip()))
            except Exception:
                pass
            self._stop_event.wait(self.interval)

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def report(self, label: str = "") -> None:
        prefix = f"[GPU{self.gpu_index}]{' ' + label if label else ''}"
        if self.util_pct:
            print(
                f"{prefix} GPU util  — mean: {sum(self.util_pct)/len(self.util_pct):.1f}%  "
                f"peak: {max(self.util_pct):.1f}%"
            )
        if self.mem_used_mb:
            print(
                f"{prefix} VRAM used — mean: {sum(self.mem_used_mb)/len(self.mem_used_mb):.0f} MiB  "
                f"peak: {max(self.mem_used_mb):.0f} MiB"
            )
        if not self.util_pct and not self.mem_used_mb:
            print(f"{prefix} No GPU samples recorded (nvidia-smi unavailable?)")

    def reset(self) -> None:
        self.util_pct.clear()
        self.mem_used_mb.clear()


_gpu_mon = _GpuMonitor(interval=0.5)


# %% Functions

def greyscale_fill_holes(image: np.ndarray) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")
    if image.size == 0 or np.all(image == image.flat[0]):
        return image.copy()
    image_max = image.max()
    inverted = image_max - image
    seed = inverted.copy()
    seed[1:-1, 1:-1] = inverted.min()
    from skimage.morphology import reconstruction
    reconstructed = reconstruction(seed, inverted, method='dilation')
    filled = image_max - reconstructed
    return filled


def detect_from_prediction(
    pred_tyx: np.ndarray,
    threshold: float,
    min_area: int,
    max_area: int,
) -> list[list[tuple[int, int, float]]]:
    """Threshold the probability map and find CC centroids per frame.

    Replaces GPU DoG: the prediction map is already a segmentation probability,
    so thresholding directly is both faster and more accurate.

    Returns
    -------
    List of T lists, each containing (y, x, 1.0) tuples — one per CC centroid.
    The 1.0 sigma placeholder keeps the signature compatible with _build_det_frame.
    """
    T = pred_tyx.shape[0]
    blobs_per_frame: list[list[tuple[int, int, float]]] = []
    for t in range(T):
        binary = (pred_tyx[t] > threshold).astype(np.uint8)
        labeled, n = scipy_label(binary)
        if n == 0:
            blobs_per_frame.append([])
            continue
        centroids = center_of_mass(binary, labeled, range(1, n + 1))
        blobs: list[tuple[int, int, float]] = []
        for cy, cx in centroids:
            iy, ix = int(round(cy)), int(round(cx))
            # Filter by CC area using the labeled mask
            region_area = int((labeled == labeled[iy, ix]).sum())
            if min_area <= region_area <= max_area:
                blobs.append((iy, ix, 1.0))
        blobs_per_frame.append(blobs)
    return blobs_per_frame


def _build_det_frame(
    blobs: list[tuple[int, int, float]],
    raw_frame: np.ndarray,
    frame_H: int,
    frame_W: int,
    _tile_size: int,
    _min_hole_area: int,
    _max_hole_area: int,
    _min_circularity: float,
) -> np.ndarray:
    """Build the binary detection mask for one frame from its blob list.

    Flood-fills a small crop of the raw image centred on each CC centroid from
    the prediction map.  This recovers the precise hole boundary from the
    original fluorescence image rather than the (softer) probability map.

    Designed to be called in parallel (thread-safe: no shared mutable state).
    """
    from skimage.measure import perimeter as sk_perimeter
    _det = np.zeros((frame_H, frame_W), dtype=np.uint8)
    for blob_y, blob_x, _sigma in blobs:
        seed_y = int(np.clip(blob_y, 0, frame_H - 1))
        seed_x = int(np.clip(blob_x, 0, frame_W - 1))
        if _det[seed_y, seed_x] != 0:
            continue
        y0 = max(0, blob_y - _tile_size)
        y1 = min(frame_H, blob_y + _tile_size)
        x0 = max(0, blob_x - _tile_size)
        x1 = min(frame_W, blob_x + _tile_size)
        local_y = seed_y - y0
        local_x = seed_x - x0
        tile = raw_frame[y0:y1, x0:x1].astype(np.float32)
        filled_tile = greyscale_fill_holes(tile)
        if not (0 <= local_y < filled_tile.shape[0] and 0 <= local_x < filled_tile.shape[1]):
            continue
        flooded = flood_fill(filled_tile, seed_point=(local_y, local_x),
                             new_value=-1.0, tolerance=0, connectivity=1)
        raw_mask = flooded == -1.0
        area = int(raw_mask.sum())
        if not (_min_hole_area <= area <= _max_hole_area):
            continue
        perim = sk_perimeter(raw_mask.astype(np.uint8))
        if perim > 0 and (4.0 * np.pi * area / perim ** 2) < _min_circularity:
            continue
        write_region = raw_mask & (_det[y0:y1, x0:x1] == 0)
        _det[y0:y1, x0:x1][write_region] = 1
    return _det


def _label_det_frame(det_frame: np.ndarray, next_id: int) -> tuple[np.ndarray, int]:
    """Label connected components of a binary detection mask with consecutive IDs.

    Returns
    -------
    labeled : (H, W) int32 — each CC gets a unique ID starting from next_id.
    next_id : updated counter (next available ID after this frame).
    """
    labeled_local, n = scipy_label(det_frame)
    out = np.zeros_like(det_frame, dtype=np.int32)
    for lbl in range(1, n + 1):
        out[labeled_local == lbl] = next_id
        next_id += 1
    return out, next_id


def _predict_prev_frame(
    img_next: np.ndarray,
    img_curr: np.ndarray,
    mask_next: np.ndarray,
    predictor,
) -> np.ndarray:
    """Use SAM2 to propagate track IDs from frame t+1 into frame t.

    Encodes a 2-frame video [img_curr, img_next] as JPEGs, registers the
    known masks at frame index 1 (img_next), then propagates backward to
    frame index 0 (img_curr).

    Parameters
    ----------
    img_next  : (H, W) image at t+1 — the frame with known masks.
    img_curr  : (H, W) image at t   — the frame to predict.
    mask_next : (H, W) int32 track-ID map at t+1 (0 = background).
    predictor : SAM2VideoPredictor instance.

    Returns
    -------
    (H, W) int32 predicted track-ID map at t.  0 = background.
    """
    track_ids = [int(tid) for tid in np.unique(mask_next) if tid != 0]
    H, W = img_next.shape
    result = np.zeros((H, W), dtype=np.int32)
    if not track_ids:
        return result

    device = predictor._device
    _autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else nullcontext()
    )

    # Two-frame stack: frame 0 = curr (target), frame 1 = next (prompt source)
    two_frames = np.stack([img_curr, img_next])  # (2, H, W)

    with tempfile.TemporaryDirectory() as tmpdir:
        _save_frames_as_jpeg(two_frames, tmpdir)
        with torch.inference_mode(), _autocast:
            inference_state = predictor.init_state(
                video_path=tmpdir,
                offload_video_to_cpu=(device == "cpu"),
            )
            for tid in track_ids:
                predictor.add_new_mask(
                    inference_state,
                    frame_idx=1,
                    obj_id=tid,
                    mask=(mask_next == tid),
                )
            # propagate_in_video with reverse=True yields frames 1 → 0
            for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(
                inference_state, start_frame_idx=1, reverse=True,
            ):
                if frame_idx != 0:
                    continue  # only collect the target frame
                for i, obj_id in enumerate(obj_ids):
                    binary = (video_res_masks[i, 0].float() > 0.0).cpu().numpy()
                    # First-write-wins when masks overlap
                    result[binary & (result == 0)] = int(obj_id)
            predictor.reset_state(inference_state)

    return result


def _merge_detections(
    sam_pred: np.ndarray,
    det_frame: np.ndarray,
    next_id: int,
) -> tuple[np.ndarray, int]:
    """Merge SAM prediction with per-frame flood-fill detections.

    SAM predictions take priority: any detection CC that overlaps at least one
    SAM-predicted pixel is dropped entirely.  Non-overlapping CCs are assigned
    new consecutive track IDs starting from next_id.

    Parameters
    ----------
    sam_pred  : (H, W) int32 SAM prediction (0 = background, >0 = track ID).
    det_frame : (H, W) uint8 binary detection mask (1 = detected hole).
    next_id   : next available track ID.

    Returns
    -------
    merged : (H, W) int32 combined map.
    next_id : updated counter.
    """
    merged = sam_pred.copy()
    labeled_det, n = scipy_label(det_frame)
    for lbl in range(1, n + 1):
        cc_mask = labeled_det == lbl
        if sam_pred[cc_mask].any():
            continue  # SAM already covers this region — discard
        # New detection: assign a fresh track ID
        merged[cc_mask & (merged == 0)] = next_id
        next_id += 1
    return merged, next_id


# %% Input paths
video_path            = r"E:\Oyvind\BIP-hub-scratch\train_macropinosome_model\input_drift_corrected\Input_crest__KiaWee__250225_RPE-mNG-Phafin2_BSD_10ul_001_1_drift.ome.tif"
prediction_video_path = r"E:\Oyvind\BIP-hub-scratch\train_macropinosome_model\deep_learning_output\predictions\Input_crest__KiaWee__250225_RPE-mNG-Phafin2_BSD_10ul_001_1_drift_pred.ome.tif"
output_folder         = r"C:\Users\oyvinode\Desktop\del"

channel = 0
z_slice = 0

# Detection parameters (prediction-map thresholding + flood-fill crop)
pred_threshold = 0.5   # probability threshold to binarise the prediction map
tile_size      = 40    # half-crop size for flood-fill on raw image (pixels)

# Flood-fill / shape filters
max_hole_area    = 10000  # discard regions larger than this (inter-cell)
min_hole_area    = 8      # discard regions smaller than this (noise)
min_circularity  = 0.5    # 4π·area/perimeter² threshold (1.0 = perfect circle)

# For debug / testing
max_T = None   # set to a small integer (e.g. 10) to limit frames during testing
debug = False  # if True shows QC plots at each stage


# %% Load image metadata
img = BioImage(video_path)
n_total = img.dims.T
print(f"Image shape: {img.shape}  dtype: {img.dtype}")
print(f"Total frames: {n_total}")
if max_T is not None:
    n_total = min(n_total, max_T)

prediction_img = BioImage(prediction_video_path)
print(f"Prediction image shape: {prediction_img.shape}  dtype: {prediction_img.dtype}")


# %% Pre-load all frames into RAM as (T, H, W)
t0_total = time.perf_counter()
_gpu_mon.start()

print(f"Loading {n_total} prediction frames …")
pred_frames = np.stack(
    [prediction_img.get_image_data("YX", T=t, C=channel, Z=z_slice)
     for t in range(n_total)]
)  # (T, H, W) float32 probability map
H, W = pred_frames.shape[1], pred_frames.shape[2]
print(f"Loaded predictions: {pred_frames.shape}  dtype: {pred_frames.dtype}")

print(f"Loading {n_total} raw frames …")
raw_frames = np.stack(
    [img.get_image_data("YX", T=t, C=channel, Z=z_slice)
     for t in range(n_total)]
)  # (T, H, W) uint16 fluorescence
print(f"Loaded raw:         {raw_frames.shape}  dtype: {raw_frames.dtype}")
print(f"  Load elapsed: {time.perf_counter() - t0_total:.1f}s")


# %% Stage 2: threshold prediction map → CC centroids
t0_det = time.perf_counter()
print(f"Detecting holes from prediction map (threshold={pred_threshold}) …")

blobs_per_frame = detect_from_prediction(
    pred_frames, pred_threshold, min_hole_area, max_hole_area,
)
n_total_blobs = sum(len(b) for b in blobs_per_frame)
print(f"  Found {n_total_blobs} blob centres across {n_total} frames.")
print(f"  Stage 2 (detection) elapsed: {time.perf_counter() - t0_det:.1f}s")

if debug:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(pred_frames[-1], cmap="hot", vmin=0, vmax=1)
    for y, x, _ in blobs_per_frame[-1]:
        ax.plot(x, y, 'c+', markersize=6)
    ax.set_title(f"Frame {n_total-1} — {len(blobs_per_frame[-1])} detected centroids")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


# %% Stage 3: flood-fill crops on raw image to get precise hole boundaries
t0_stage3 = time.perf_counter()
print(f"Building detection mask from centroids (parallel, {n_total} frames) …")

frame_dets = Parallel(n_jobs=-1, prefer='threads')(
    delayed(_build_det_frame)(
        blobs_per_frame[t_idx], raw_frames[t_idx],
        H, W, tile_size, min_hole_area, max_hole_area, min_circularity,
    )
    for t_idx in range(n_total)
)
det_mask_tyx = np.stack(frame_dets)  # (T, H, W) uint8 binary

n_det_pixels = int(det_mask_tyx.sum())
n_det_frames = int((det_mask_tyx.sum(axis=(1, 2)) > 0).sum())
print(f"  Detection mask: {n_det_pixels} hole pixels across {n_det_frames}/{n_total} frames.")
print(f"  Stage 3 elapsed: {time.perf_counter() - t0_stage3:.1f}s")

if debug:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(raw_frames[-1], cmap="gray")
    axes[0].set_title(f"Frame {n_total-1} raw")
    axes[0].axis("off")
    axes[1].imshow(raw_frames[-1], cmap="gray")
    ov = np.zeros((*det_mask_tyx[-1].shape, 4), dtype=np.float32)
    ov[det_mask_tyx[-1] != 0] = [1.0, 0.0, 0.0, 0.6]
    axes[1].imshow(ov)
    axes[1].set_title(f"Frame {n_total-1} detection overlay")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()


# %% Stage 4: rolling 2-frame backward SAM2 tracking
# -----------------------------------------------------------------------
# Algorithm:
#   Seed frame (T-1):
#     Label CCs of det_mask_tyx[T-1] with consecutive IDs.
#     Run SAM self-refinement: feed img[T-1] as both frames 0 and 1 in a
#     2-frame video, so SAM's mask decoder cleans up the flood-fill shapes.
#
#   For each step t = T-2 … 0:
#     1. sam_pred ← _predict_prev_frame(raw[t+1], raw[t], global_mask[t+1])
#     2. For each CC in det_mask_tyx[t]:
#          if ANY pixel overlaps sam_pred → discard (SAM already covers it)
#          else → assign new track ID
#     3. global_mask[t] = sam_pred + non-overlapping new detections
# -----------------------------------------------------------------------
print("Loading SAM2 video predictor …")
predictor = load_sam2_video_predictor()

t0_sam = time.perf_counter()
_gpu_mon.reset()

global_mask_tyx = np.zeros((n_total, H, W), dtype=np.int32)
next_id = 1

# --- Seed frame (T-1) ---
seed_t = n_total - 1
print(f"Seeding frame {seed_t} …")

# Label CCs in the detection mask to get initial track IDs
seed_labeled, next_id = _label_det_frame(det_mask_tyx[seed_t], next_id)

# SAM self-refinement on the seed frame: feed the same frame twice
# so SAM smooths the raw flood-fill shapes through its mask decoder.
if seed_labeled.any():
    seed_refined = _predict_prev_frame(
        img_next=raw_frames[seed_t],
        img_curr=raw_frames[seed_t],
        mask_next=seed_labeled,
        predictor=predictor,
    )
    # Fall back to flood-fill labels for any track that SAM dropped
    for tid in np.unique(seed_labeled):
        if tid == 0:
            continue
        if not (seed_refined == tid).any():
            seed_refined[seed_labeled == tid] = tid
    global_mask_tyx[seed_t] = seed_refined
else:
    global_mask_tyx[seed_t] = seed_labeled

n_seed_tracks = int((global_mask_tyx[seed_t] > 0).sum() > 0) and int(global_mask_tyx[seed_t].max())
print(f"  Seed frame: {int(global_mask_tyx[seed_t].max())} tracks seeded.")

# --- Rolling backward loop ---
for t in range(n_total - 2, -1, -1):
    mask_next = global_mask_tyx[t + 1]

    # Step 1: SAM propagation from t+1 → t
    sam_pred = _predict_prev_frame(
        img_next=raw_frames[t + 1],
        img_curr=raw_frames[t],
        mask_next=mask_next,
        predictor=predictor,
    )

    # Step 2 + 3: merge with fresh per-frame detections (SAM wins on overlap)
    global_mask_tyx[t], next_id = _merge_detections(
        sam_pred, det_mask_tyx[t], next_id,
    )

    n_tracks_t = int(global_mask_tyx[t].max())
    n_sam_px   = int((sam_pred > 0).sum())
    n_new_px   = int((global_mask_tyx[t] > 0).sum()) - n_sam_px
    print(f"  t={t:3d}: {n_tracks_t} total IDs, {n_sam_px} SAM px, {n_new_px} new-det px")

n_unique_tracks = int(global_mask_tyx.max())
print(f"Tracking complete. Unique tracks: {n_unique_tracks}")
print(f"  Stage 4 (SAM2) elapsed: {time.perf_counter() - t0_sam:.1f}s")
_gpu_mon.stop()
_gpu_mon.report("Stage4-SAM2")


# %% Debug: colour overlay of all tracked frames
if debug:
    from matplotlib import cm
    unique_ids = np.unique(global_mask_tyx)
    unique_ids = unique_ids[unique_ids != 0]
    n_ids = len(unique_ids)
    color_vol = np.zeros_like(global_mask_tyx, dtype=np.int32)
    for i, tid in enumerate(unique_ids, start=1):
        color_vol[global_mask_tyx == tid] = i
    cmap = cm.get_cmap('tab20', max(n_ids + 1, 2))
    n_cols = min(n_total, 4)
    n_rows = (n_total + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
    for t in range(n_total):
        ax = axes[t // n_cols][t % n_cols]
        ax.imshow(raw_frames[t], cmap='gray')
        ov = cmap(color_vol[t] / max(n_ids, 1))
        ov[..., 3] = np.where(global_mask_tyx[t] != 0, 0.5, 0.0)
        ax.imshow(ov)
        ax.set_title(f"t={t}", fontsize=8)
        ax.axis('off')
    for t in range(n_total, n_rows * n_cols):
        axes[t // n_cols][t % n_cols].axis('off')
    plt.suptitle(f"Global mask — {n_unique_tracks} unique tracks", fontsize=10)
    plt.tight_layout()
    plt.show()


# %% Save output
output_path = os.path.join(output_folder, "global_mask_v8.tif")
tifffile.imwrite(output_path, global_mask_tyx.astype(np.uint16))
print(f"Saved global mask to {output_path}")
print(f"Total elapsed: {time.perf_counter() - t0_total:.1f}s")
# %%
