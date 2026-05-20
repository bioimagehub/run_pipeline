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
from skimage.morphology import dilation, disk, flood_fill, label, reconstruction
from skimage.measure import label as cc_label, regionprops
from scipy.ndimage import median_filter, gaussian_filter
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from scipy.ndimage import binary_fill_holes


import time
import tifffile
import torch

_script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.path.abspath(r"E:\Oyvind\OF_git\run_pipeline\standard_code\python\Test_code")
sys.path.insert(0, os.path.join(_script_dir, ".."))
import sam2_utils
importlib.reload(sam2_utils)
from sam2_utils import load_sam2_video_predictor, predict_prev_frame
from fill_greyscale_holes import greyscale_fill_holes_kernel_2d

# %% GPU monitor (nvidia-smi polling in background thread)

# %% Functions

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


# %% Input paths
video_path            = r"E:\Oyvind\BIP-hub-scratch\train_macropinosome_model\input_drift_corrected\Input_crest__KiaWee__250225_RPE-mNG-Phafin2_BSD_10ul_001_1_drift.ome.tif"
# prediction_video_path = r"E:\Oyvind\BIP-hub-scratch\train_macropinosome_model\deep_learning_output\predictions\Input_crest__KiaWee__250225_RPE-mNG-Phafin2_BSD_10ul_001_1_drift_pred.ome.tif"
output_folder         = r"C:\Users\oyvinode\Desktop\del"

channel = 0
z_slice = 0

# Detection parameters
tile_size         = 40    # half-crop for debug display (pixels)
ring_search_steps = 25    # steps from hole edge outward to search for the ring
min_hole_area     = 8     # discard flood masks smaller than this (noise)
max_hole_area     = 8000  # discard flood masks larger than this (inter-cell)
# Half-maximum (FWHM) criterion applied to the radial profile FROM THE HOLE EDGE:
#   - hole interior median must be < half_max of ring peak
#   - profile must drop below half_max after the peak (cytoplasm is dark)
#   - ring peak must not be at the last step (outer drop must be visible)

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


print(f"Loading {n_total} raw frames …")

start_time = time.time()
raw_frames = img.get_image_data("TYX", C=channel, Z=z_slice)[:n_total]  # (T, H, W) uint16 fluorescence
print(f"Loaded raw:         {raw_frames.shape}  dtype: {raw_frames.dtype}, elapsed: {time.time() - start_time:.1f}s")


# %% find macropinosomes in the last frame

t = n_total - 1

last_frame = raw_frames[t]
last_frame_filled = greyscale_fill_holes_kernel_2d(last_frame, kernel_size=256, kernel_overlap=50)
last_frame_score = gaussian_filter(last_frame_filled - last_frame, sigma=3)

macropinosome_coordinates = peak_local_max(
    last_frame_score,
    min_distance=5,
    threshold_rel=0.4
)

# --- DEBUG: show ALL local maxima around a known structure to diagnose misses ---
# Lower threshold to catch weak peaks; plot side-by-side with accepted candidates.
if debug:
    _all_candidates = peak_local_max(last_frame_score, min_distance=5, threshold_rel=0.05)
    _investigate_y, _investigate_x = 1550, 593   # adjust to the region of interest
    _inv_pad = 80
    _H_dbg, _W_dbg = last_frame.shape
    _iy0 = max(0, _investigate_y - _inv_pad);  _iy1 = min(_H_dbg, _investigate_y + _inv_pad + 1)
    _ix0 = max(0, _investigate_x - _inv_pad);  _ix1 = min(_W_dbg, _investigate_x + _inv_pad + 1)

    _in_region_all = (_all_candidates[:, 0] >= _iy0) & (_all_candidates[:, 0] < _iy1) & \
                     (_all_candidates[:, 1] >= _ix0) & (_all_candidates[:, 1] < _ix1)
    _in_region_acc = (macropinosome_coordinates[:, 0] >= _iy0) & (macropinosome_coordinates[:, 0] < _iy1) & \
                     (macropinosome_coordinates[:, 1] >= _ix0) & (macropinosome_coordinates[:, 1] < _ix1)

    _score_crop = last_frame_score[_iy0:_iy1, _ix0:_ix1]
    _global_max = float(last_frame_score.max())
    print(f"\n--- Candidate investigation around ({_investigate_y},{_investigate_x}) ---")
    print(f"Global score max = {_global_max:.2f}  |  threshold_rel=0.4 cutoff = {_global_max * 0.4:.2f}")
    for _cy, _cx in _all_candidates[_in_region_all]:
        _sv = float(last_frame_score[_cy, _cx])
        _passed = "ACCEPTED" if any(
            (macropinosome_coordinates[:, 0] == _cy) & (macropinosome_coordinates[:, 1] == _cx)
        ) else f"DROPPED by peak_local_max (score={_sv:.2f} < {_global_max * 0.4:.2f})"
        print(f"  ({_cy},{_cx})  score={_sv:.2f}  rel={_sv / _global_max:.3f}  → {_passed}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(last_frame[_iy0:_iy1, _ix0:_ix1], cmap="gray")
    for _cy, _cx in _all_candidates[_in_region_all]:
        _sv = float(last_frame_score[_cy, _cx])
        _color = 'lime' if any(
            (macropinosome_coordinates[:, 0] == _cy) & (macropinosome_coordinates[:, 1] == _cx)
        ) else 'red'
        axes[0].plot(_cx - _ix0, _cy - _iy0, '+', color=_color, markersize=10, markeredgewidth=1.5)
        axes[0].text(_cx - _ix0 + 2, _cy - _iy0 - 2, f"{_sv:.0f}", color=_color, fontsize=6)
    axes[0].set_title("Raw  (lime=passes threshold_rel=0.4, red=dropped)")
    axes[0].axis("off")
    axes[1].imshow(_score_crop, cmap="magma")
    for _cy, _cx in _all_candidates[_in_region_all]:
        _sv = float(last_frame_score[_cy, _cx])
        _color = 'lime' if any(
            (macropinosome_coordinates[:, 0] == _cy) & (macropinosome_coordinates[:, 1] == _cx)
        ) else 'red'
        axes[1].plot(_cx - _ix0, _cy - _iy0, '+', color=_color, markersize=10, markeredgewidth=1.5)
        axes[1].text(_cx - _ix0 + 2, _cy - _iy0 - 2, f"{_sv:.0f}", color=_color, fontsize=6)
    axes[1].set_title(f"Score  (global_max={_global_max:.0f}, cutoff={_global_max * 0.4:.0f})")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()
# --- END DEBUG ---

last_frame_score_f32 = last_frame_score.astype(np.float32)

# Final labeled mask:
#   odd  labels = dark hole  (1, 3, 5, …)
#   even labels = bright ring (2, 4, 6, …)  where ring 2k encircles hole 2k-1
H_img, W_img = last_frame.shape
hole_ring_mask = np.zeros((H_img, W_img), dtype=np.int32)
next_odd_id = 1

macropinosome_coordinates_filtered = []

for y, x in macropinosome_coordinates:
    # ------------------------------------------------------------------ #
    # Step 1: flood-fill the SCORE image at the candidate centroid at
    # half-maximum to get the approximate hole boundary.  This works
    # regardless of hole size because we use the score image (not raw).
    #
    # Step 2: grow outward from that boundary in the RAW image for a fixed
    # ring_search_steps steps.  The ring should always be within ~15 px of
    # the hole edge, no matter how large the hole itself is.
    #
    # Step 3: FWHM filters: hole interior AND cytoplasm must both be < 50 %
    # of the ring peak intensity.
    # ------------------------------------------------------------------ #
    _pad = ring_search_steps + 55   # crop: flood fill can extend far from centroid
    _y0 = max(0, y - _pad);  _y1 = min(H_img, y + _pad + 1)
    _x0 = max(0, x - _pad);  _x1 = min(W_img, x + _pad + 1)
    tile_score = last_frame_score_f32[_y0:_y1, _x0:_x1]
    tile_raw   = last_frame[_y0:_y1, _x0:_x1]
    cy_c, cx_c = y - _y0, x - _x0

    peak_score = float(tile_score[cy_c, cx_c])
    if peak_score <= 0:
        continue

    # Score-image flood fill at half-max of the candidate peak
    filled_s   = flood_fill(tile_score, (cy_c, cx_c), new_value=-1.0, tolerance=peak_score / 2)
    flood_mask = (filled_s == -1.0).astype(np.uint8)

    props = regionprops(flood_mask)
    if not props:
        continue
    area = props[0].area
    if not (min_hole_area <= area <= max_hole_area):
        if debug:
            print(f"Discarded ({y},{x}): flood area={area} outside [{min_hole_area}, {max_hole_area}]")
        continue

    # Radial profile: grow outward from flood_mask boundary in RAW image
    prev = flood_mask.copy()
    ring_medians: list[float] = []
    for _r in range(ring_search_steps):
        curr = dilation(prev, disk(1))
        annulus = (curr == 1) & (prev == 0)
        if not annulus.any():
            break
        ring_medians.append(float(np.median(tile_raw[annulus])))
        prev = curr

    if len(ring_medians) < 3:
        continue

    profile        = np.array(ring_medians, dtype=np.float32)
    ring_peak_step = int(np.argmax(profile))        # 0-based step from hole edge
    ring_peak_val  = float(profile[ring_peak_step])
    half_max       = ring_peak_val / 2.0

    # --- Background-corrected thresholds ---
    # Estimate background from the tail of the profile (last few steps).
    # Use 50% of (peak - background) above background as the ring boundary.
    # This avoids the absolute half_max being below background in high-
    # fluorescence regions, which would make the outer filter impossible to pass.
    _n_bg = max(1, len(profile) - ring_peak_step - 2)
    bg_estimate    = float(np.min(profile[-_n_bg:]))
    corrected_half = bg_estimate + (ring_peak_val - bg_estimate) * 0.5

    # --- FWHM filters ---
    hole_median_val  = float(np.median(tile_raw[flood_mask == 1]))
    # Hole must be darker than the ring (using absolute contrast: hole < peak)
    is_dark_inside   = hole_median_val < ring_peak_val * 0.90
    # Outside must drop to at least the bg-corrected midpoint after the peak
    has_dark_outside = (ring_peak_step < len(profile) - 1 and
                        float(np.min(profile[ring_peak_step + 1:])) < corrected_half)
    is_ring_within_range = ring_peak_step < len(profile) - 2

    if not (is_dark_inside and has_dark_outside and is_ring_within_range):
        if debug:
            reason = []
            if not is_dark_inside:
                reason.append(f"hole_median={hole_median_val:.0f} >= 90% peak={ring_peak_val * 0.90:.0f}")
            if not has_dark_outside:
                reason.append(f"outside never drops below bg_half={corrected_half:.0f}")
            if not is_ring_within_range:
                reason.append(f"ring peak at last step {ring_peak_step+1} (no outer drop visible)")
            print(f"Discarded ({y},{x}): {', '.join(reason)}")
        continue

    # --- Outer boundary: first step after peak below bg-corrected half ---
    outer_half_step = ring_peak_step + 2  # fallback: 2 steps past peak
    for _i in range(ring_peak_step + 1, len(profile)):
        if profile[_i] < corrected_half:
            outer_half_step = _i + 1  # 1-based dilation count from flood_mask edge
            break

    # --- Filter: discard if ring extends too far from hole edge ---
    max_outer_half_step = 10
    if outer_half_step > max_outer_half_step:
        if debug:
            print(f"Discarded ({y},{x}): outer_half_step={outer_half_step} > {max_outer_half_step}")
        continue

    # --- Build masks ---
    # Hole: fill any interior gaps in the score flood region
    hole_mask_crop = binary_fill_holes(flood_mask).astype(np.uint8)
    # Ring: Euclidean distance from the hole boundary gives a circular ring
    # regardless of the hole shape.  Pixels within [1, outer_half_step] px of
    # the hole edge form the annulus; already-claimed pixels are excluded so
    # adjacent macropinosomes never eat into each other's territory.
    _dist_from_hole = distance_transform_edt(hole_mask_crop == 0)
    _existing_crop  = hole_ring_mask[_y0:_y1, _x0:_x1]
    _forbidden      = (_existing_crop != 0)
    ring_mask_crop  = ((_dist_from_hole > 0) &
                       (_dist_from_hole <= outer_half_step) &
                       (~_forbidden)).astype(np.uint8)

    # --- Assign labels; never overwrite an earlier detection ---
    hole_id = next_odd_id
    ring_id  = next_odd_id + 1
    next_odd_id += 2

    _view = hole_ring_mask[_y0:_y1, _x0:_x1]
    _view[(hole_mask_crop == 1) & (_view == 0)] = hole_id
    _view[(ring_mask_crop == 1) & (_view == 0)] = ring_id

    macropinosome_coordinates_filtered.append((y, x))

    if debug:
        _dy0 = max(0, y - tile_size * 2);  _dy1 = min(H_img, y + tile_size * 2 + 1)
        _dx0 = max(0, x - tile_size * 2);  _dx1 = min(W_img, x + tile_size * 2 + 1)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].imshow(last_frame[_dy0:_dy1, _dx0:_dx1], cmap="gray")
        axes[0].contour(hole_ring_mask[_dy0:_dy1, _dx0:_dx1] == hole_id, colors='cyan',   linewidths=1.5)
        axes[0].contour(hole_ring_mask[_dy0:_dy1, _dx0:_dx1] == ring_id, colors='yellow', linewidths=1.5)
        axes[0].plot(x - _dx0, y - _dy0, 'c+', markersize=12)
        axes[0].set_title(f"Raw ({y},{x})  cyan=hole  yellow=ring")
        axes[0].axis("off")
        axes[1].imshow(last_frame_score[_dy0:_dy1, _dx0:_dx1], cmap="magma")
        axes[1].contour(hole_ring_mask[_dy0:_dy1, _dx0:_dx1] == hole_id, colors='cyan',   linewidths=1.5)
        axes[1].contour(hole_ring_mask[_dy0:_dy1, _dx0:_dx1] == ring_id, colors='yellow', linewidths=1.5)
        axes[1].set_title(
            f"Score  area={area}px²  peak_step={ring_peak_step+1}  outer_step={outer_half_step}"
        )
        axes[1].axis("off")
        _rr = range(1, len(profile) + 1)
        axes[2].plot(_rr, profile, marker='o', color='steelblue', markersize=4)
        axes[2].axhline(half_max,        color='white',  linestyle=':',  linewidth=1.2, label=f'half_max={half_max:.0f}')
        axes[2].axhline(bg_estimate,       color='gray',   linestyle=':',  linewidth=1.0, label=f'bg={bg_estimate:.0f}')
        axes[2].axhline(corrected_half,    color='cyan',   linestyle=':',  linewidth=1.2, label=f'bg_half={corrected_half:.0f}')
        axes[2].axvline(ring_peak_step+1, color='red',    linestyle='--', label=f'peak step={ring_peak_step+1}')
        axes[2].axvline(outer_half_step,  color='orange', linestyle='--', label=f'outer step={outer_half_step}')
        axes[2].axhline(ring_peak_val,    color='yellow', linestyle=':',  label=f'peak={ring_peak_val:.0f}')
        axes[2].set_xlabel("Steps from hole edge (px)")
        axes[2].set_ylabel("Annulus median intensity")
        axes[2].set_title("Radial profile from hole edge (FWHM)")
        axes[2].legend(fontsize=7)
        plt.tight_layout()
        plt.show()

print(f"\nAccepted {len(macropinosome_coordinates_filtered)} / {len(macropinosome_coordinates)} candidates.")
print(f"Hole labels (odd):  1, 3, 5, …  up to {next_odd_id - 2 if next_odd_id > 1 else 'none'}")
print(f"Ring labels (even): 2, 4, 6, …  up to {next_odd_id - 1 if next_odd_id > 1 else 'none'}")

# %% QC: zoomed tiles for every accepted macropinosome
if debug and macropinosome_coordinates_filtered:
    n_mp   = len(macropinosome_coordinates_filtered)
    n_cols = min(8, n_mp)
    n_rows = (n_mp + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.5, n_rows * 2.5),
                             squeeze=False)
    for _idx, (fy, fx) in enumerate(macropinosome_coordinates_filtered):
        hole_id_q = 2 * _idx + 1
        ring_id_q = 2 * _idx + 2
        _dy0 = max(0, fy - tile_size);  _dy1 = min(H_img, fy + tile_size + 1)
        _dx0 = max(0, fx - tile_size);  _dx1 = min(W_img, fx + tile_size + 1)
        crop_raw  = last_frame[_dy0:_dy1, _dx0:_dx1]
        crop_mask = hole_ring_mask[_dy0:_dy1, _dx0:_dx1]
        ov = np.zeros((*crop_raw.shape, 4), dtype=np.float32)
        ov[crop_mask == hole_id_q] = [1.0, 0.2, 0.2, 0.6]   # hole  → red
        ov[crop_mask == ring_id_q] = [0.3, 0.7, 1.0, 0.45]  # ring  → blue
        ax = axes[_idx // n_cols][_idx % n_cols]
        ax.imshow(crop_raw, cmap="gray")
        ax.imshow(ov)
        ax.contour(crop_mask == hole_id_q, colors='cyan',   linewidths=0.8)
        ax.contour(crop_mask == ring_id_q, colors='yellow', linewidths=0.8)
        ax.plot(fx - _dx0, fy - _dy0, 'c+', markersize=8)
        ax.set_title(f"#{hole_id_q}  ({fy},{fx})", fontsize=7)
        ax.axis("off")
    # hide unused axes
    for _idx in range(n_mp, n_rows * n_cols):
        axes[_idx // n_cols][_idx % n_cols].axis("off")
    fig.suptitle(
        f"{n_mp} macropinosomes  (red=hole, blue=ring, ±{tile_size}px zoom)",
        fontsize=10
    )
    plt.tight_layout()
    plt.show()




# if debug:
#     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#     axes[0].imshow(last_frame[500:1000, 700:1200], cmap="gray")
#     axes[0].set_title(f"t={t} tile")
#     axes[0].axis("off")
#     axes[1].imshow(last_frame_score[500:1000, 700:1200], cmap="magma")
#     in_crop = (macropinosome_coordinates[:, 0] >= 500) & (macropinosome_coordinates[:, 0] < 1000) & \
#               (macropinosome_coordinates[:, 1] >= 700) & (macropinosome_coordinates[:, 1] < 1200)
#     crop_holes = macropinosome_coordinates[in_crop]
#     axes[1].plot(crop_holes[:, 1] - 700, crop_holes[:, 0] - 500, 'c+', markersize=12, color='cyan')
#     #axes[1].set_title("Segmented")
#     #axes[1].axis("off")
#     plt.tight_layout()
#     plt.show()









# # %% Stage 2: Find centroids from prediction map (all frames)
# print(f"Detecting holes from prediction map (threshold={pred_threshold}) …")

# mp_predictions_per_frame: list[np.ndarray] = [
#     peak_local_max(pred_frames[t], min_distance=2, threshold_rel=0.1)
#     for t in range(n_total)
# ]
# n_total_preds = sum(len(p) for p in mp_predictions_per_frame)
# print(f"  Detected {n_total_preds} centroids across {n_total} frames.")

# print(f"Detecting holes from prediction map (threshold={pred_threshold}) …")

# mp_predictions_per_frame: list[np.ndarray] = [
#     peak_local_max(pred_frames[t], min_distance=2, threshold_rel=0.1)
#     for t in range(n_total)
# ]
# n_total_preds = sum(len(p) for p in mp_predictions_per_frame)
# print(f"  Detected {n_total_preds} centroids across {n_total} frames.")





# if debug:
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.imshow(pred_frames[-1], cmap="hot", vmin=0, vmax=1)
#     for y, x in mp_predictions_per_frame[-1]:
#         ax.plot(x, y, 'c+', markersize=6)
#     ax.set_title(f"Frame {n_total-1} — {len(mp_predictions_per_frame[-1])} detected centroids")
#     ax.axis("off")
#     plt.tight_layout()
#     plt.show()

# # %% Loop over all frames and segment macropinosomes (fill holes)
# mp_mask = np.zeros_like(pred_frames, dtype=np.float32)

# for t in range(n_total):
#     for y, x in mp_predictions_per_frame[t]:
#         x1 = max(0, x - tile_size)
#         x2 = min(W, x + tile_size + 1)
#         y1 = max(0, y - tile_size)
#         y2 = min(H, y + tile_size + 1)
#         tile = raw_frames[t, y1:y2, x1:x2]

#         mp = segment_algorithmic(crop_2d=tile, edge_enhancement_radius=1.0, enforce_object_in_center=True)[0]

#         mp_mask[t, y1:y2, x1:x2] = np.maximum(mp_mask[t, y1:y2, x1:x2], mp)
#         # if debug:
#         #     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#         #     axes[0].imshow(tile, cmap="gray")
#         #     axes[0].set_title(f"t={t} tile")
#         #     axes[0].axis("off")
#         #     axes[1].imshow(mp, cmap="Reds", alpha=0.5)
#         #     axes[1].set_title("Segmented")
#         #     axes[1].axis("off")
#         #     plt.tight_layout()
#         #     plt.show()

#     mp_mask[t] = label(mp_mask[t]).astype(np.float32) # Will fuse overlapping masks on purpose

# if debug:
#     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#     axes[0].imshow(pred_frames[-1], cmap="gray")
#     axes[0].set_title(f"Frame {n_total-1} prediction")
#     axes[0].axis("off")
#     axes[1].imshow(raw_frames[-1], cmap="gray")
#     _labeled = mp_mask[-1].astype(np.int32)
#     _ids = np.unique(_labeled)
#     _ids = _ids[_ids != 0]
#     _rng = np.random.default_rng(seed=42)
#     _colors = _rng.uniform(0.2, 1.0, size=(len(_ids), 3))
#     _ov = np.zeros((*_labeled.shape, 4), dtype=np.float32)
#     for _i, _uid in enumerate(_ids):
#         _m = _labeled == _uid
#         _ov[_m, :3] = _colors[_i]
#         _ov[_m, 3] = 0.6
#     axes[1].imshow(_ov)
#     axes[1].set_title(f"Frame {n_total-1} segmented")
#     axes[1].axis("off")
#     plt.tight_layout()
#     plt.show()

# # %% Stage 4: Rolling 2-frame backward SAM2 tracking
# # Algorithm:
# #   1. Seed last frame directly from per-frame detection mask.
# #   2. For t = n_total-2 down to 0:
# #        a. Feed [raw[t], raw[t+1]] as a 2-frame SAM2 video, seeded with
# #           global_mask_tyx[t+1], to get sam_pred at frame t.
# #        b. Merge: SAM wins on overlap; new detections at t that don't
# #           overlap sam_pred get a fresh track ID.
# #        c. Store merged result in global_mask_tyx[t].

# sam_video_predictor = load_sam2_video_predictor()

# # det_mask_tyx = (mp_mask[:n_total] > 0).astype(np.uint8)
# # global_mask_tyx = np.zeros((n_total, H, W), dtype=np.int32)
# # next_track_id = 1

# # # Seed the last frame
# # seed_labeled = label(det_mask_tyx[n_total - 1].astype(np.uint8), connectivity=1).astype(np.int32)
# # global_mask_tyx[n_total - 1] = seed_labeled
# # next_track_id = int(seed_labeled.max()) + 1
# # print(f"Seeded frame {n_total-1} with {int(seed_labeled.max())} objects.")



# # t = n_total - 2 # down to 0
# for t in range(n_total - 2, -1, -1):
#     # Step a: propagate mask from t+1 → t via 2-frame SAM2 call

#     img_next = raw_frames[t + 1]
#     # Todo: Add a channel with output from fill holes (holescore = filled - raw)

#     img_curr = raw_frames[t]
#     # Todo: Add a channel with output from fill holes (holescore = filled - raw)
#     mask_next = mp_mask[t + 1]


#     ids = np.unique(mask_next).astype(np.int32)
#     # uid = ids[37]
#     for uid in ids:
#         if uid == 0:
#             continue        
#         y, x = center_of_mass(mask_next == uid)
#         y, x = int(round(y)), int(round(x))
#         x1 = max(0, x - tile_size)
#         x2 = min(W, x + tile_size + 1)
#         y1 = max(0, y - tile_size)
#         y2 = min(H, y + tile_size + 1)
        
#         filled_next = greyscale_fill_holes(img_next[y1:y2, x1:x2])
#         score_next = filled_next - img_next[y1:y2, x1:x2]
#         two_channel_next = np.stack([img_next[y1:y2, x1:x2].astype(np.float32), score_next], axis=0)  # (2, H, W)
#         mask_next_crop = (mask_next[y1:y2, x1:x2]).astype(np.uint8)
        

#         filled_curr = greyscale_fill_holes(img_curr[y1:y2, x1:x2])
#         score_curr = filled_curr - img_curr[y1:y2, x1:x2]
#         two_channel_curr = np.stack([img_curr[y1:y2, x1:x2].astype(np.float32), score_curr], axis=0)  # (2, H, W)
  


#         sam_pred = predict_prev_frame(
#             img_next=two_channel_next,
#             img_curr=two_channel_curr,
#             mask_next=mask_next_crop,
#             predictor=sam_video_predictor,
#         )

#         # capture the current uid from mask_next_crop and sam_pred
#         # plot them as contours on top of the raw images for both frames, with different colours for the SAM prediction and the original mask_next_crop
#         predicted_x, predicted_y = center_of_mass(sam_pred == uid)
#         if np.isnan(predicted_x) or np.isnan(predicted_y):
#             # uid not found in sam_pred (SAM lost the object); skip this uid
#             continue
#         predicted_x, predicted_y = int(round(predicted_x)), int(round(predicted_y)) 

#         predicted_macropinosome_mask = segment_algorithmic(crop_2d=two_channel_curr[0], edge_enhancement_radius=1.0, enforce_object_in_center=False, center=(predicted_x, predicted_y))[0]

#         fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#         axes[0].imshow(two_channel_curr[0], cmap="gray")
#         axes[0].imshow(two_channel_curr[1], cmap="Reds", alpha=0.5)
#         axes[0].imshow(mask_next_crop > 0, cmap="Greens", alpha=0.2)
#         axes[0].contour(mask_next_crop == uid, colors='lime', linewidths=1.5)
        
#         axes[0].set_title(f"t={t+1} seed for ID {int(uid)}")
#         axes[0].axis("off") 
#         axes[1].imshow(two_channel_next[0], cmap="gray")
#         axes[1].imshow(two_channel_next[1], cmap="Reds", alpha=0.5)
#         axes[1].imshow(sam_pred > 0, cmap="Blues", alpha=0.2)
#         axes[1].contour(sam_pred == uid, colors='lime', linewidths=1.5)
#         axes[1].contour(predicted_macropinosome_mask > 0, colors='cyan', linewidths=1.5)    
#         axes[1].set_title(f"t={t} candidate region")
#         axes[1].axis("off")
#         plt.tight_layout()
#         plt.show()


        
#         torch.cuda.empty_cache()


#     # sam_pred_labeled = label(sam_pred.astype(np.uint8), connectivity=1).astype(np.int32)
#     # loop over all labels
#     # run segment_algorithmic on each mask center
#     # lbl = 46
#     for lbl in np.unique(sam_pred):           # sam_pred already has correct track IDs
#         if lbl == 0:
#             continue
#         mask = (sam_pred == lbl)
#         if not mask.any():
#             continue
#         cy, cx = center_of_mass(mask)
#         cy, cx = int(round(cy)), int(round(cx))
#         x1 = max(0, cx - tile_size)
#         x2 = min(W, cx + tile_size + 1)
#         y1 = max(0, cy - tile_size)
#         y2 = min(H, cy + tile_size + 1)
#         tile = img_curr[y1:y2, x1:x2]
#         mp, _, _ = segment_algorithmic(crop_2d=tile, edge_enhancement_radius=1.0, enforce_object_in_center=True)

#         plt.imshow(tile, cmap="gray")
#         #plt.imshow(mp, cmap="Reds", alpha=0.5)
#         plt.title(f"t={t} tile for label {lbl}")
#         plt.axis("off")
#         plt.show()

#         # torch.cuda.empty_cache()







#         # Write the track label (lbl) to the refined pixels only — don't zero the tile
#         mp_mask[t, y1:y2, x1:x2][mp > 0] = float(lbl)
    


# if debug:
#     _all_ids = np.unique(np.concatenate([mp_mask[-1].ravel(), mp_mask[-2].ravel()]))
#     _all_ids = _all_ids[_all_ids != 0].astype(np.int32)
#     _rng = np.random.default_rng(seed=42)
#     _color_map = {int(uid): _rng.uniform(0.2, 1.0, size=3) for uid in _all_ids}

#     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#     for ax, t_idx in zip(axes, [-1, -2]):
#         _labeled = mp_mask[t_idx].astype(np.int32)
#         _ids = np.unique(_labeled)
#         _ids = _ids[_ids != 0]
#         _ov = np.zeros((*_labeled.shape, 4), dtype=np.float32)
#         for _uid in _ids:
#             _m = _labeled == _uid
#             _ov[_m, :3] = _color_map[int(_uid)]
#             _ov[_m, 3] = 0.6
#         ax.imshow(raw_frames[t_idx], cmap="gray")
#         ax.imshow(_ov)
#         ax.set_title(f"Frame {n_total + t_idx} segmented")
#         ax.axis("off")
#     plt.tight_layout()
#     plt.show()



# del sam_video_predictor
# torch.cuda.empty_cache()
    
    






# # # %% Stage 3: flood-fill crops on raw image to get precise hole boundaries
# # t0_stage3 = time.perf_counter()
# # print(f"Building detection mask from centroids (parallel, {n_total} frames) …")

# # frame_dets = Parallel(n_jobs=-1, prefer='threads')(
# #     delayed(_build_det_frame)(
# #         blobs_per_frame[t_idx], raw_frames[t_idx],
# #         H, W, tile_size, min_hole_area, max_hole_area, min_circularity,
# #     )
# #     for t_idx in range(n_total)
# # )
# # det_mask_tyx = np.stack(frame_dets)  # (T, H, W) uint8 binary

# # n_det_pixels = int(det_mask_tyx.sum())
# # n_det_frames = int((det_mask_tyx.sum(axis=(1, 2)) > 0).sum())
# # print(f"  Detection mask: {n_det_pixels} hole pixels across {n_det_frames}/{n_total} frames.")
# # print(f"  Stage 3 elapsed: {time.perf_counter() - t0_stage3:.1f}s")

# # if debug:
# #     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# #     axes[0].imshow(raw_frames[-1], cmap="gray")
# #     axes[0].set_title(f"Frame {n_total-1} raw")
# #     axes[0].axis("off")
# #     axes[1].imshow(raw_frames[-1], cmap="gray")
# #     ov = np.zeros((*det_mask_tyx[-1].shape, 4), dtype=np.float32)
# #     ov[det_mask_tyx[-1] != 0] = [1.0, 0.0, 0.0, 0.6]
# #     axes[1].imshow(ov)
# #     axes[1].set_title(f"Frame {n_total-1} detection overlay")
# #     axes[1].axis("off")
# #     plt.tight_layout()
# #     plt.show()


# # # %% Stage 4: rolling 2-frame backward SAM2 tracking
# # # -----------------------------------------------------------------------
# # # Algorithm:
# # #   Seed frame (T-1):
# # #     Label CCs of det_mask_tyx[T-1] with consecutive IDs.
# # #     Run SAM self-refinement: feed img[T-1] as both frames 0 and 1 in a
# # #     2-frame video, so SAM's mask decoder cleans up the flood-fill shapes.
# # #
# # #   For each step t = T-2 … 0:
# # #     1. sam_pred ← _predict_prev_frame(raw[t+1], raw[t], global_mask[t+1])
# # #     2. For each CC in det_mask_tyx[t]:
# # #          if ANY pixel overlaps sam_pred → discard (SAM already covers it)
# # #          else → assign new track ID
# # #     3. global_mask[t] = sam_pred + non-overlapping new detections
# # # -----------------------------------------------------------------------
# # print("Loading SAM2 video predictor …")
# # predictor = load_sam2_video_predictor()

# # t0_sam = time.perf_counter()
# # _gpu_mon.reset()

# # global_mask_tyx = np.zeros((n_total, H, W), dtype=np.int32)
# # next_id = 1

# # # --- Seed frame (T-1) ---
# # seed_t = n_total - 1
# # print(f"Seeding frame {seed_t} …")

# # # Label CCs in the detection mask to get initial track IDs
# # seed_labeled, next_id = _label_det_frame(det_mask_tyx[seed_t], next_id)

# # # SAM self-refinement on the seed frame: feed the same frame twice
# # # so SAM smooths the raw flood-fill shapes through its mask decoder.
# # if seed_labeled.any():
# #     seed_refined = _predict_prev_frame(
# #         img_next=raw_frames[seed_t],
# #         img_curr=raw_frames[seed_t],
# #         mask_next=seed_labeled,
# #         predictor=predictor,
# #     )
# #     # Fall back to flood-fill labels for any track that SAM dropped
# #     for tid in np.unique(seed_labeled):
# #         if tid == 0:
# #             continue
# #         if not (seed_refined == tid).any():
# #             seed_refined[seed_labeled == tid] = tid
# #     global_mask_tyx[seed_t] = seed_refined
# # else:
# #     global_mask_tyx[seed_t] = seed_labeled

# # n_seed_tracks = int((global_mask_tyx[seed_t] > 0).sum() > 0) and int(global_mask_tyx[seed_t].max())
# # print(f"  Seed frame: {int(global_mask_tyx[seed_t].max())} tracks seeded.")

# # # --- Rolling backward loop ---
# # for t in range(n_total - 2, -1, -1):
# #     mask_next = global_mask_tyx[t + 1]

# #     # Step 1: SAM propagation from t+1 → t
# #     sam_pred = _predict_prev_frame(
# #         img_next=raw_frames[t + 1],
# #         img_curr=raw_frames[t],
# #         mask_next=mask_next,
# #         predictor=predictor,
# #     )

# #     # Step 2 + 3: merge with fresh per-frame detections (SAM wins on overlap)
# #     global_mask_tyx[t], next_id = _merge_detections(
# #         sam_pred, det_mask_tyx[t], next_id,
# #     )

# #     n_tracks_t = int(global_mask_tyx[t].max())
# #     n_sam_px   = int((sam_pred > 0).sum())
# #     n_new_px   = int((global_mask_tyx[t] > 0).sum()) - n_sam_px
# #     print(f"  t={t:3d}: {n_tracks_t} total IDs, {n_sam_px} SAM px, {n_new_px} new-det px")

# # n_unique_tracks = int(global_mask_tyx.max())
# # print(f"Tracking complete. Unique tracks: {n_unique_tracks}")
# # print(f"  Stage 4 (SAM2) elapsed: {time.perf_counter() - t0_sam:.1f}s")
# # _gpu_mon.stop()
# # _gpu_mon.report("Stage4-SAM2")


# # # %% Debug: colour overlay of all tracked frames
# # if debug:
# #     from matplotlib import cm
# #     unique_ids = np.unique(global_mask_tyx)
# #     unique_ids = unique_ids[unique_ids != 0]
# #     n_ids = len(unique_ids)
# #     color_vol = np.zeros_like(global_mask_tyx, dtype=np.int32)
# #     for i, tid in enumerate(unique_ids, start=1):
# #         color_vol[global_mask_tyx == tid] = i
# #     cmap = cm.get_cmap('tab20', max(n_ids + 1, 2))
# #     n_cols = min(n_total, 4)
# #     n_rows = (n_total + n_cols - 1) // n_cols
# #     fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
# #     for t in range(n_total):
# #         ax = axes[t // n_cols][t % n_cols]
# #         ax.imshow(raw_frames[t], cmap='gray')
# #         ov = cmap(color_vol[t] / max(n_ids, 1))
# #         ov[..., 3] = np.where(global_mask_tyx[t] != 0, 0.5, 0.0)
# #         ax.imshow(ov)
# #         ax.set_title(f"t={t}", fontsize=8)
# #         ax.axis('off')
# #     for t in range(n_total, n_rows * n_cols):
# #         axes[t // n_cols][t % n_cols].axis('off')
# #     plt.suptitle(f"Global mask — {n_unique_tracks} unique tracks", fontsize=10)
# #     plt.tight_layout()
# #     plt.show()


# # # %% Save output
# # output_path = os.path.join(output_folder, "global_mask_v8.tif")
# # tifffile.imwrite(output_path, global_mask_tyx.astype(np.uint16))
# # print(f"Saved global mask to {output_path}")
# # print(f"Total elapsed: {time.perf_counter() - t0_total:.1f}s")
# # # %%

# # %%

# %%
