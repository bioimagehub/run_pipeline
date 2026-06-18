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
from skimage.morphology import flood_fill, label, reconstruction
from skimage.measure import label as cc_label
from scipy.ndimage import median_filter
from skimage.feature import peak_local_max
from scipy.ndimage import binary_fill_holes


import time
import tifffile
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sam2_utils
importlib.reload(sam2_utils)
from sam2_utils import load_sam2_video_predictor, predict_prev_frame


# %% GPU monitor (nvidia-smi polling in background thread)

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

def segment_algorithmic(
    crop_2d: np.ndarray,
    edge_enhancement_radius: float = 1.0,
    enforce_object_in_center: bool = True,
    center: tuple[int, int] | None = None,
) -> tuple[np.ndarray, int, int]:
    """Segment a macropinosome

    Args:
        crop_2d:   (H, W) float or integer image array.
        edge_enhancement_radius: Radius used to derive the median-filter kernel size.

    Returns:
        (H, W) float32 binary mask; 1.0 inside the macropinosome.
    """

    tolerance: float = 0.0

    crop_f = crop_2d.astype(np.float32)
    H, W = crop_f.shape

    crop_f = median_filter(crop_f, footprint=np.ones((3, 3), dtype=bool))

    seed = crop_f.copy()
    seed[1:-1, 1:-1] = crop_f.max()
    filled = reconstruction(seed, crop_f, method="erosion").astype(np.float32)

    score = filled - crop_f


    # plt.imshow(crop_f, cmap="gray")
    # plt.imshow(score, cmap="hot", alpha=0.5)
    # plt.title("Score image (bright = likely inside object)")
    # plt.axis("off")
    # plt.show()




    # Find the brightest pixel and use the connected component around it
    # as the seed centre. This makes the algorithm less sensitive to rectangle
    # placement by choosing the most prominent object in the score image.
    flat_max_idx = int(np.argmax(score))
    y_max, x_max = np.unravel_index(flat_max_idx, score.shape)
    max_val = float(score[y_max, x_max])

    if center is not None:
        center_y, center_x = center
    elif max_val <= 0 or enforce_object_in_center:
        # no positive score, or caller asserts the object is already centred
        center_y, center_x = H // 2, W // 2
    else:
        # threshold at a fraction of the peak to get the object region
        thr = max_val * 0.2
        binary = score >= thr
        labeled_score = label(binary.astype(np.uint8), connectivity=1)
        comp_id = int(labeled_score[y_max, x_max])

        if comp_id == 0:
            # brightest pixel not in any component (unlikely) -> pick largest component
            counts = np.bincount(labeled_score.ravel())
            if counts.size <= 1:
                center_y, center_x = y_max, x_max
            else:
                largest = counts[1:].argmax() + 1
                comp_id = int(largest)

        coords = np.argwhere(labeled_score == comp_id)
        if coords.size == 0:
            center_y, center_x = y_max, x_max
        else:
            center_y = int(coords[:, 0].mean())
            center_x = int(coords[:, 1].mean())


    cy = int(np.clip(center_y, 0, H - 1))
    cx = int(np.clip(center_x, 0, W - 1))
    # Use an odd kernel size so the median filter is centered on each pixel.
    median_radius = max(0, int(round(edge_enhancement_radius)))
    kernel_size = 2 * median_radius + 1
    footprint = np.ones((kernel_size, kernel_size), dtype=bool)
    enhanced = median_filter(filled, footprint=footprint)


    center_val = float(enhanced[cy, cx])

    binary = (enhanced >= center_val - tolerance) & (enhanced <= center_val + tolerance)
    binary = binary_fill_holes(binary)

    labeled = label(binary.astype(np.uint8), connectivity=1)
    comp_id = int(labeled[cy, cx])
    if comp_id == 0:
        logger.warning(
            "No component at seed (%d, %d); returning empty mask.", cy, cx
        )
        return np.zeros((H, W), dtype=np.float32), center_y, center_x

    return (labeled == comp_id).astype(np.float32), center_y, center_x


def detect_from_prediction(
    pred_tyx: np.ndarray,
    threshold: float,
    min_area: int,
    max_area: int,
) -> list[list[tuple[int, int]]]:
    """Threshold the probability map and find CC centroids per frame.

    Replaces GPU DoG: the prediction map is already a segmentation probability,
    so thresholding directly is both faster and more accurate.

    Returns
    -------
    List of T lists, each containing (y, x) tuples — one per CC centroid.
    """
    T = pred_tyx.shape[0]
    blobs_per_frame: list[list[tuple[int, int]]] = []
    for t in range(T):
        binary = (pred_tyx[t] > threshold).astype(np.uint8)
        labeled, n = scipy_label(binary)
        if n == 0:
            blobs_per_frame.append([])
            continue
        centroids = center_of_mass(binary, labeled, range(1, n + 1))
        blobs: list[tuple[int, int]] = []
        for cy, cx in centroids:
            iy, ix = int(round(cy)), int(round(cx))
            # Filter by CC area using the labeled mask
            region_area = int((labeled == labeled[iy, ix]).sum())
            if min_area <= region_area <= max_area:
                blobs.append((iy, ix))
        blobs_per_frame.append(blobs)
    return blobs_per_frame


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
max_T = 10   # set to a small integer (e.g. 10) to limit frames during testing
debug = True  # if True shows QC plots at each stage


# %% Load image metadata
img = BioImage(video_path)
n_total = img.dims.T
print(f"Image shape: {img.shape}  dtype: {img.dtype}")
print(f"Total frames: {n_total}")
if max_T is not None:
    n_total = min(n_total, max_T)

prediction_img = BioImage(prediction_video_path)
print(f"Prediction image shape: {prediction_img.shape}  dtype: {prediction_img.dtype}")


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

if debug:
    # Show that the prediction map roughly matches the raw image (for the last frame)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(raw_frames[-1], cmap="gray")
    axes[0].set_title(f"Frame {n_total-1} raw")
    axes[0].axis("off")
    axes[1].imshow(pred_frames[-1], cmap="hot", vmin=0, vmax=1)
    axes[1].set_title(f"Frame {n_total-1} prediction")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()




# %% Stage 2: Find centroids from prediction map (all frames)
print(f"Detecting holes from prediction map (threshold={pred_threshold}) …")

mp_predictions_per_frame: list[np.ndarray] = [
    peak_local_max(pred_frames[t], min_distance=2, threshold_rel=0.1)
    for t in range(n_total)
]
n_total_preds = sum(len(p) for p in mp_predictions_per_frame)
print(f"  Detected {n_total_preds} centroids across {n_total} frames.")

if debug:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(pred_frames[-1], cmap="hot", vmin=0, vmax=1)
    for y, x in mp_predictions_per_frame[-1]:
        ax.plot(x, y, 'c+', markersize=6)
    ax.set_title(f"Frame {n_total-1} — {len(mp_predictions_per_frame[-1])} detected centroids")
    ax.axis("off")
    plt.tight_layout()
    plt.show()

# %% Loop over all frames and segment macropinosomes (fill holes)
mp_mask = np.zeros_like(pred_frames, dtype=np.float32)

for t in range(n_total):
    for y, x in mp_predictions_per_frame[t]:
        x1 = max(0, x - tile_size)
        x2 = min(W, x + tile_size + 1)
        y1 = max(0, y - tile_size)
        y2 = min(H, y + tile_size + 1)
        tile = raw_frames[t, y1:y2, x1:x2]

        mp = segment_algorithmic(crop_2d=tile, edge_enhancement_radius=1.0, enforce_object_in_center=True)[0]

        mp_mask[t, y1:y2, x1:x2] = np.maximum(mp_mask[t, y1:y2, x1:x2], mp)
        # if debug:
        #     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        #     axes[0].imshow(tile, cmap="gray")
        #     axes[0].set_title(f"t={t} tile")
        #     axes[0].axis("off")
        #     axes[1].imshow(mp, cmap="Reds", alpha=0.5)
        #     axes[1].set_title("Segmented")
        #     axes[1].axis("off")
        #     plt.tight_layout()
        #     plt.show()

    mp_mask[t] = label(mp_mask[t]).astype(np.float32) # Will fuse overlapping masks on purpose

if debug:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(pred_frames[-1], cmap="gray")
    axes[0].set_title(f"Frame {n_total-1} prediction")
    axes[0].axis("off")
    axes[1].imshow(raw_frames[-1], cmap="gray")
    _labeled = mp_mask[-1].astype(np.int32)
    _ids = np.unique(_labeled)
    _ids = _ids[_ids != 0]
    _rng = np.random.default_rng(seed=42)
    _colors = _rng.uniform(0.2, 1.0, size=(len(_ids), 3))
    _ov = np.zeros((*_labeled.shape, 4), dtype=np.float32)
    for _i, _uid in enumerate(_ids):
        _m = _labeled == _uid
        _ov[_m, :3] = _colors[_i]
        _ov[_m, 3] = 0.6
    axes[1].imshow(_ov)
    axes[1].set_title(f"Frame {n_total-1} segmented")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()

# %% Stage 4: Rolling 2-frame backward SAM2 tracking
# Algorithm:
#   1. Seed last frame directly from per-frame detection mask.
#   2. For t = n_total-2 down to 0:
#        a. Feed [raw[t], raw[t+1]] as a 2-frame SAM2 video, seeded with
#           global_mask_tyx[t+1], to get sam_pred at frame t.
#        b. Merge: SAM wins on overlap; new detections at t that don't
#           overlap sam_pred get a fresh track ID.
#        c. Store merged result in global_mask_tyx[t].

sam_video_predictor = load_sam2_video_predictor()

# det_mask_tyx = (mp_mask[:n_total] > 0).astype(np.uint8)
# global_mask_tyx = np.zeros((n_total, H, W), dtype=np.int32)
# next_track_id = 1

# # Seed the last frame
# seed_labeled = label(det_mask_tyx[n_total - 1].astype(np.uint8), connectivity=1).astype(np.int32)
# global_mask_tyx[n_total - 1] = seed_labeled
# next_track_id = int(seed_labeled.max()) + 1
# print(f"Seeded frame {n_total-1} with {int(seed_labeled.max())} objects.")



# t = n_total - 2 # down to 0
for t in range(n_total - 2, -1, -1):
    # Step a: propagate mask from t+1 → t via 2-frame SAM2 call

    img_next = raw_frames[t + 1]
    # Todo: Add a channel with output from fill holes (holescore = filled - raw)

    img_curr = raw_frames[t]
    # Todo: Add a channel with output from fill holes (holescore = filled - raw)
    mask_next = mp_mask[t + 1]


    ids = np.unique(mask_next).astype(np.int32)
    # uid = ids[37]
    for uid in ids:
        if uid == 0:
            continue        
        y, x = center_of_mass(mask_next == uid)
        y, x = int(round(y)), int(round(x))
        x1 = max(0, x - tile_size)
        x2 = min(W, x + tile_size + 1)
        y1 = max(0, y - tile_size)
        y2 = min(H, y + tile_size + 1)
        
        filled_next = greyscale_fill_holes(img_next[y1:y2, x1:x2])
        score_next = filled_next - img_next[y1:y2, x1:x2]
        two_channel_next = np.stack([img_next[y1:y2, x1:x2].astype(np.float32), score_next], axis=0)  # (2, H, W)
        mask_next_crop = (mask_next[y1:y2, x1:x2]).astype(np.uint8)
        

        filled_curr = greyscale_fill_holes(img_curr[y1:y2, x1:x2])
        score_curr = filled_curr - img_curr[y1:y2, x1:x2]
        two_channel_curr = np.stack([img_curr[y1:y2, x1:x2].astype(np.float32), score_curr], axis=0)  # (2, H, W)
  


        sam_pred = predict_prev_frame(
            img_next=two_channel_next,
            img_curr=two_channel_curr,
            mask_next=mask_next_crop,
            predictor=sam_video_predictor,
        )

        # capture the current uid from mask_next_crop and sam_pred
        # plot them as contours on top of the raw images for both frames, with different colours for the SAM prediction and the original mask_next_crop
        predicted_x, predicted_y = center_of_mass(sam_pred == uid)
        if np.isnan(predicted_x) or np.isnan(predicted_y):
            # uid not found in sam_pred (SAM lost the object); skip this uid
            continue
        predicted_x, predicted_y = int(round(predicted_x)), int(round(predicted_y)) 

        predicted_macropinosome_mask = segment_algorithmic(crop_2d=two_channel_curr[0], edge_enhancement_radius=1.0, enforce_object_in_center=False, center=(predicted_x, predicted_y))[0]

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(two_channel_curr[0], cmap="gray")
        axes[0].imshow(two_channel_curr[1], cmap="Reds", alpha=0.5)
        axes[0].imshow(mask_next_crop > 0, cmap="Greens", alpha=0.2)
        axes[0].contour(mask_next_crop == uid, colors='lime', linewidths=1.5)
        
        axes[0].set_title(f"t={t+1} seed for ID {int(uid)}")
        axes[0].axis("off") 
        axes[1].imshow(two_channel_next[0], cmap="gray")
        axes[1].imshow(two_channel_next[1], cmap="Reds", alpha=0.5)
        axes[1].imshow(sam_pred > 0, cmap="Blues", alpha=0.2)
        axes[1].contour(sam_pred == uid, colors='lime', linewidths=1.5)
        axes[1].contour(predicted_macropinosome_mask > 0, colors='cyan', linewidths=1.5)    
        axes[1].set_title(f"t={t} candidate region")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()


        
        torch.cuda.empty_cache()


    # sam_pred_labeled = label(sam_pred.astype(np.uint8), connectivity=1).astype(np.int32)
    # loop over all labels
    # run segment_algorithmic on each mask center
    # lbl = 46
    for lbl in np.unique(sam_pred):           # sam_pred already has correct track IDs
        if lbl == 0:
            continue
        mask = (sam_pred == lbl)
        if not mask.any():
            continue
        cy, cx = center_of_mass(mask)
        cy, cx = int(round(cy)), int(round(cx))
        x1 = max(0, cx - tile_size)
        x2 = min(W, cx + tile_size + 1)
        y1 = max(0, cy - tile_size)
        y2 = min(H, cy + tile_size + 1)
        tile = img_curr[y1:y2, x1:x2]
        mp, _, _ = segment_algorithmic(crop_2d=tile, edge_enhancement_radius=1.0, enforce_object_in_center=True)

        plt.imshow(tile, cmap="gray")
        #plt.imshow(mp, cmap="Reds", alpha=0.5)
        plt.title(f"t={t} tile for label {lbl}")
        plt.axis("off")
        plt.show()

        # torch.cuda.empty_cache()







        # Write the track label (lbl) to the refined pixels only — don't zero the tile
        mp_mask[t, y1:y2, x1:x2][mp > 0] = float(lbl)
    


if debug:
    _all_ids = np.unique(np.concatenate([mp_mask[-1].ravel(), mp_mask[-2].ravel()]))
    _all_ids = _all_ids[_all_ids != 0].astype(np.int32)
    _rng = np.random.default_rng(seed=42)
    _color_map = {int(uid): _rng.uniform(0.2, 1.0, size=3) for uid in _all_ids}

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, t_idx in zip(axes, [-1, -2]):
        _labeled = mp_mask[t_idx].astype(np.int32)
        _ids = np.unique(_labeled)
        _ids = _ids[_ids != 0]
        _ov = np.zeros((*_labeled.shape, 4), dtype=np.float32)
        for _uid in _ids:
            _m = _labeled == _uid
            _ov[_m, :3] = _color_map[int(_uid)]
            _ov[_m, 3] = 0.6
        ax.imshow(raw_frames[t_idx], cmap="gray")
        ax.imshow(_ov)
        ax.set_title(f"Frame {n_total + t_idx} segmented")
        ax.axis("off")
    plt.tight_layout()
    plt.show()



del sam_video_predictor
torch.cuda.empty_cache()
    
    






# # %% Stage 3: flood-fill crops on raw image to get precise hole boundaries
# t0_stage3 = time.perf_counter()
# print(f"Building detection mask from centroids (parallel, {n_total} frames) …")

# frame_dets = Parallel(n_jobs=-1, prefer='threads')(
#     delayed(_build_det_frame)(
#         blobs_per_frame[t_idx], raw_frames[t_idx],
#         H, W, tile_size, min_hole_area, max_hole_area, min_circularity,
#     )
#     for t_idx in range(n_total)
# )
# det_mask_tyx = np.stack(frame_dets)  # (T, H, W) uint8 binary

# n_det_pixels = int(det_mask_tyx.sum())
# n_det_frames = int((det_mask_tyx.sum(axis=(1, 2)) > 0).sum())
# print(f"  Detection mask: {n_det_pixels} hole pixels across {n_det_frames}/{n_total} frames.")
# print(f"  Stage 3 elapsed: {time.perf_counter() - t0_stage3:.1f}s")

# if debug:
#     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#     axes[0].imshow(raw_frames[-1], cmap="gray")
#     axes[0].set_title(f"Frame {n_total-1} raw")
#     axes[0].axis("off")
#     axes[1].imshow(raw_frames[-1], cmap="gray")
#     ov = np.zeros((*det_mask_tyx[-1].shape, 4), dtype=np.float32)
#     ov[det_mask_tyx[-1] != 0] = [1.0, 0.0, 0.0, 0.6]
#     axes[1].imshow(ov)
#     axes[1].set_title(f"Frame {n_total-1} detection overlay")
#     axes[1].axis("off")
#     plt.tight_layout()
#     plt.show()


# # %% Stage 4: rolling 2-frame backward SAM2 tracking
# # -----------------------------------------------------------------------
# # Algorithm:
# #   Seed frame (T-1):
# #     Label CCs of det_mask_tyx[T-1] with consecutive IDs.
# #     Run SAM self-refinement: feed img[T-1] as both frames 0 and 1 in a
# #     2-frame video, so SAM's mask decoder cleans up the flood-fill shapes.
# #
# #   For each step t = T-2 … 0:
# #     1. sam_pred ← _predict_prev_frame(raw[t+1], raw[t], global_mask[t+1])
# #     2. For each CC in det_mask_tyx[t]:
# #          if ANY pixel overlaps sam_pred → discard (SAM already covers it)
# #          else → assign new track ID
# #     3. global_mask[t] = sam_pred + non-overlapping new detections
# # -----------------------------------------------------------------------
# print("Loading SAM2 video predictor …")
# predictor = load_sam2_video_predictor()

# t0_sam = time.perf_counter()
# _gpu_mon.reset()

# global_mask_tyx = np.zeros((n_total, H, W), dtype=np.int32)
# next_id = 1

# # --- Seed frame (T-1) ---
# seed_t = n_total - 1
# print(f"Seeding frame {seed_t} …")

# # Label CCs in the detection mask to get initial track IDs
# seed_labeled, next_id = _label_det_frame(det_mask_tyx[seed_t], next_id)

# # SAM self-refinement on the seed frame: feed the same frame twice
# # so SAM smooths the raw flood-fill shapes through its mask decoder.
# if seed_labeled.any():
#     seed_refined = _predict_prev_frame(
#         img_next=raw_frames[seed_t],
#         img_curr=raw_frames[seed_t],
#         mask_next=seed_labeled,
#         predictor=predictor,
#     )
#     # Fall back to flood-fill labels for any track that SAM dropped
#     for tid in np.unique(seed_labeled):
#         if tid == 0:
#             continue
#         if not (seed_refined == tid).any():
#             seed_refined[seed_labeled == tid] = tid
#     global_mask_tyx[seed_t] = seed_refined
# else:
#     global_mask_tyx[seed_t] = seed_labeled

# n_seed_tracks = int((global_mask_tyx[seed_t] > 0).sum() > 0) and int(global_mask_tyx[seed_t].max())
# print(f"  Seed frame: {int(global_mask_tyx[seed_t].max())} tracks seeded.")

# # --- Rolling backward loop ---
# for t in range(n_total - 2, -1, -1):
#     mask_next = global_mask_tyx[t + 1]

#     # Step 1: SAM propagation from t+1 → t
#     sam_pred = _predict_prev_frame(
#         img_next=raw_frames[t + 1],
#         img_curr=raw_frames[t],
#         mask_next=mask_next,
#         predictor=predictor,
#     )

#     # Step 2 + 3: merge with fresh per-frame detections (SAM wins on overlap)
#     global_mask_tyx[t], next_id = _merge_detections(
#         sam_pred, det_mask_tyx[t], next_id,
#     )

#     n_tracks_t = int(global_mask_tyx[t].max())
#     n_sam_px   = int((sam_pred > 0).sum())
#     n_new_px   = int((global_mask_tyx[t] > 0).sum()) - n_sam_px
#     print(f"  t={t:3d}: {n_tracks_t} total IDs, {n_sam_px} SAM px, {n_new_px} new-det px")

# n_unique_tracks = int(global_mask_tyx.max())
# print(f"Tracking complete. Unique tracks: {n_unique_tracks}")
# print(f"  Stage 4 (SAM2) elapsed: {time.perf_counter() - t0_sam:.1f}s")
# _gpu_mon.stop()
# _gpu_mon.report("Stage4-SAM2")


# # %% Debug: colour overlay of all tracked frames
# if debug:
#     from matplotlib import cm
#     unique_ids = np.unique(global_mask_tyx)
#     unique_ids = unique_ids[unique_ids != 0]
#     n_ids = len(unique_ids)
#     color_vol = np.zeros_like(global_mask_tyx, dtype=np.int32)
#     for i, tid in enumerate(unique_ids, start=1):
#         color_vol[global_mask_tyx == tid] = i
#     cmap = cm.get_cmap('tab20', max(n_ids + 1, 2))
#     n_cols = min(n_total, 4)
#     n_rows = (n_total + n_cols - 1) // n_cols
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
#     for t in range(n_total):
#         ax = axes[t // n_cols][t % n_cols]
#         ax.imshow(raw_frames[t], cmap='gray')
#         ov = cmap(color_vol[t] / max(n_ids, 1))
#         ov[..., 3] = np.where(global_mask_tyx[t] != 0, 0.5, 0.0)
#         ax.imshow(ov)
#         ax.set_title(f"t={t}", fontsize=8)
#         ax.axis('off')
#     for t in range(n_total, n_rows * n_cols):
#         axes[t // n_cols][t % n_cols].axis('off')
#     plt.suptitle(f"Global mask — {n_unique_tracks} unique tracks", fontsize=10)
#     plt.tight_layout()
#     plt.show()


# # %% Save output
# output_path = os.path.join(output_folder, "global_mask_v8.tif")
# tifffile.imwrite(output_path, global_mask_tyx.astype(np.uint16))
# print(f"Saved global mask to {output_path}")
# print(f"Total elapsed: {time.perf_counter() - t0_total:.1f}s")
# # %%

# %%
