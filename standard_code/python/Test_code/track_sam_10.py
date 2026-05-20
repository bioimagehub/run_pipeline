# %% Imports
# Standard library and third-party imports.
# - BioImage: reads multi-dimensional microscopy files (OME-TIFF, ND2, etc.) and
#   exposes them as labelled 5-D arrays (T, C, Z, Y, X).
# - skimage / scipy: morphological operations (dilation, flood_fill, reconstruction),
#   distance transforms, peak detection and region measurements.
# - sam2_utils / fill_greyscale_holes: local project modules — SAM2 video predictor
#   wrapper and the GPU-accelerated hole-fill kernel.
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
from sam2_utils import load_sam2_video_predictor, predict_prev_frame, _run_tracking_on_region
from fill_greyscale_holes import greyscale_fill_holes_kernel_2d

# %% GPU monitor (nvidia-smi polling in background thread)
# Placeholder for an optional background thread that polls nvidia-smi to log
# GPU memory / utilisation while the heavy processing runs.  Currently empty —
# add a threading.Thread here if you want live GPU stats printed to the console.

# %% Functions
# Helper function used later when tracking backwards through frames.
# segment_algorithmic() takes a single 2-D greyscale crop around a candidate
# macropinosome and returns a binary mask of just that object.
#
# How it works:
#   1. Morphological reconstruction (erosion-based) fills the dark interior of
#      the macropinosome by "flooding" from the image border inward, producing
#      a "filled" version where holes are filled with the surrounding intensity.
#   2. score = filled – raw  →  bright where the raw image had a dark hole.
#   3. The seed point (centre of the crop, or a caller-supplied coordinate)
#      seeds a flood-fill on the "filled" image at equal intensity, extracting
#      the connected plateau that corresponds to the object interior.
#   4. Median filtering of the filled image + a final connected-component step
#      clean up noise and return the largest component that touches the seed.

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
# All user-facing configuration lives here — change these variables to point at
# a different dataset without touching any other part of the script.
#
# Key parameters:
#   tile_size         – half-width (px) of the debug zoom crop shown around each
#                       accepted macropinosome.  Does NOT affect detection.
#   ring_search_steps – how many 1-pixel dilation steps to take outward from the
#                       flood-filled hole boundary when building the radial profile.
#                       25 steps = up to 25 px from the hole edge are sampled.
#   min/max_hole_area – flood-fill areas outside this range are rejected as noise
#                       (too small) or whole-cell background (too large).
#   max_T             – set to a small number (e.g. 10) to test on just the first
#                       N frames; set to None to process the entire video.
#   debug             – when True, matplotlib plots are shown at every stage so
#                       you can inspect what the algorithm is doing interactively.
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
# Open the OME-TIFF with BioImage (lazy — nothing is read from disk yet) and
# print the full array shape so it is easy to spot mis-configured channel/Z
# indices.  Then eagerly load all T frames for the chosen channel + Z slice
# into a single NumPy array of shape (T, H, W) in memory.  Loading all frames
# at once is faster than repeated random access later and keeps downstream code
# simple (no lazy-loading bookkeeping).
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
# Detection is done on the LAST frame only.  The idea is that macropinosomes
# are most mature (largest, brightest ring) at the end of the timelapse, so
# this frame gives the strongest signal for initial detection.  Tracking then
# propagates the found objects backwards to earlier frames.
#
# Detection pipeline:
#   1. greyscale_fill_holes_kernel_2d() — morphologically fills dark holes in
#      the image using a sliding 256-px kernel.  The result (last_frame_filled)
#      is a version of the image where macropinosome interiors are filled with
#      the surrounding ring intensity.
#   2. score = filled – raw, smoothed with a Gaussian.  Bright pixels in this
#      "score" image correspond to locations where the raw image was darker than
#      its surroundings — i.e. the dark interior of a macropinosome hole.
#   3. peak_local_max() on the score image finds local brightness peaks that are
#      at least threshold_rel × global_max bright and at least min_distance px
#      apart.  Each peak is a candidate macropinosome centre.
#
# After detection, a DEBUG block (controlled by the debug flag) re-runs
# peak_local_max at a very low threshold and prints/plots all candidate peaks
# in a region of interest so you can see exactly why a specific structure was
# accepted or rejected at this stage.
#
# The main per-candidate validation loop that follows then applies stricter
# FWHM-based filters on the radial fluorescence profile to confirm each peak
# is truly a macropinosome (dark hole + bright surrounding ring + dark cytoplasm
# outside) and builds a labeled mask (hole_ring_mask) where odd labels are
# hole interiors and even labels are the encircling fluorescent rings.
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
mp_outer_half_steps: list[int] = []  # ring width (px from hole edge) for each accepted MP

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
    mp_outer_half_steps.append(outer_half_step)

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
# Summary QC plot shown after the main detection loop completes.
# For each accepted macropinosome a small zoomed crop (±tile_size px) is shown
# with a red overlay on the hole interior and a blue overlay on the ring annulus,
# plus cyan/yellow contour lines so the mask boundaries are easy to judge.
# This lets you quickly scan all detections side-by-side and spot any that look
# wrong before moving on to the tracking stage.
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


# %% Build global tracking mask volume
# Now that we have confirmed macropinosome locations in the last frame we can
# create a 3-D volume that will hold per-frame masks for the entire video.
#
# Shape: (T, H, W) — same time axis as raw_frames, same spatial dimensions.
# Dtype: int32 — matches hole_ring_mask; odd values = hole interiors,
#                even values = ring annuli (0 = background).
#
# For now only the last frame (index t = n_total - 1) is filled in with the
# detections we just computed.  All earlier frames are still zero and will be
# populated by the tracking loop in the next cells.
#
# Label meaning (same convention carried through to all frames):
#   1, 3, 5, …  → hole interior of macropinosome #1, #2, #3, …
#   2, 4, 6, …  → ring annulus  of macropinosome #1, #2, #3, …
tracking_mask = np.zeros((n_total, H_img, W_img), dtype=np.int32)
tracking_mask[t] = hole_ring_mask

print(f"tracking_mask shape: {tracking_mask.shape}  dtype: {tracking_mask.dtype}")
print(f"Frame {t} seeded with {len(macropinosome_coordinates_filtered)} macropinosomes.")


# %% Load SAM2 predictor
# Load the SAM2 video predictor.  Device (CUDA or CPU) is auto-detected
# inside load_sam2_video_predictor().  This only needs to happen once — the same predictor instance
# is reused for all per-macropinosome crops below.
#
# The checkpoint path is relative to the script's parent directory where
# the SAM2 model weights are expected to live.  Adjust if needed.
print("Loading SAM2 predictor …")
predictor = load_sam2_video_predictor()
print("SAM2 predictor ready.")


# %% SAM2 backward tracking
# Now we track each confirmed macropinosome backward through all frames.
#
# Strategy per macropinosome:
#   1. Cut a fixed spatial crop (tile_size pixels each side of the centroid)
#      that stays at the same pixel coordinates in every frame.
#   2. For every frame, build a 3-channel version of that crop:
#        ch 0 = raw fluorescence
#        ch 1 = hole score  (filled – raw, Gaussian-smoothed) — highlights dark
#               holes with bright rings, the same signal used for detection
#        ch 2 = raw fluorescence again
#      This gives SAM extra contrast on the macropinosome even though it was
#      not trained on these tiny vesicles.
#   3. Seed SAM2 at the last frame with the hole masks of ALL macropinosomes
#      present in the crop (not just the target), so SAM can track multiple
#      objects and distinguish them from each other when they come close.
#   4. Run SAM2 backward propagation.
#   5. Identify which SAM2 track ID corresponds to our target macropinosome
#      (the one whose mask covers the seed centroid in the last frame).
#   6. For every earlier frame: extract that track's hole mask, reconstruct
#      the ring via EDT with the same outer_half_step width from detection,
#      and write both back to tracking_mask.

# Context padding for fill algorithm: large enough that the hole-fill kernel
# has enough surrounding tissue to compute an accurate filled image.
_fill_context_pad = max(64, tile_size)

for mp_idx, (mp_y, mp_x) in enumerate(macropinosome_coordinates_filtered):
    hole_id_target = 2 * mp_idx + 1
    ring_id_target = 2 * mp_idx + 2
    outer_r        = mp_outer_half_steps[mp_idx]   # ring width (px) from detection

    print(f"\n--- Tracking MP#{hole_id_target} at ({mp_y},{mp_x}), ring_width={outer_r} px ---")

    # Fixed spatial crop bounds (same for every frame)
    _cy0 = max(0, mp_y - tile_size)
    _cy1 = min(H_img, mp_y + tile_size + 1)
    _cx0 = max(0, mp_x - tile_size)
    _cx1 = min(W_img, mp_x + tile_size + 1)
    cH = _cy1 - _cy0
    cW = _cx1 - _cx0

    # Build 3-channel video: shape (n_total, 3, cH, cW)
    video_3ch = np.zeros((n_total, 3, cH, cW), dtype=np.float32)

    for frame_t in range(n_total):
        raw_crop = raw_frames[frame_t, _cy0:_cy1, _cx0:_cx1].astype(np.float32)

        # Compute score on a larger crop so fill has enough context around the hole.
        # A kernel_size smaller than the full-frame value is fine here because
        # the context area is already limited to _fill_context_pad pixels.
        _fy0 = max(0, _cy0 - _fill_context_pad); _fy1 = min(H_img, _cy1 + _fill_context_pad)
        _fx0 = max(0, _cx0 - _fill_context_pad); _fx1 = min(W_img, _cx1 + _fill_context_pad)
        raw_large    = raw_frames[frame_t, _fy0:_fy1, _fx0:_fx1]
        filled_large = greyscale_fill_holes_kernel_2d(raw_large, kernel_size=128, kernel_overlap=25)
        _ry0 = _cy0 - _fy0; _ry1 = _ry0 + cH
        _rx0 = _cx0 - _fx0; _rx1 = _rx0 + cW
        score_crop = gaussian_filter(
            filled_large[_ry0:_ry1, _rx0:_rx1].astype(np.float32) - raw_crop,
            sigma=3,
        )

        video_3ch[frame_t, 0] = raw_crop
        video_3ch[frame_t, 1] = score_crop
        video_3ch[frame_t, 2] = raw_crop

    # Seed mask: binary hole pixels at last frame for ALL macropinosomes in crop.
    # SAM will see them as separate connected components and assign distinct track IDs.
    seed_frame_crop = hole_ring_mask[_cy0:_cy1, _cx0:_cx1]
    det_mask = np.zeros((n_total, cH, cW), dtype=np.uint8)
    det_mask[t] = ((seed_frame_crop % 2 == 1) & (seed_frame_crop > 0)).astype(np.uint8)

    # Run SAM2 backward tracking on the 3-channel crop
    sam_result = _run_tracking_on_region(
        video_3ch, det_mask, predictor, max_centroid_dist=float(tile_size),
        label=f"MP#{hole_id_target} ",
    )

    # Identify which SAM2 track ID covers our target centroid in the last frame
    _local_y = mp_y - _cy0
    _local_x = mp_x - _cx0
    target_sam_id = int(sam_result[t, _local_y, _local_x])

    if target_sam_id == 0:
        print(f"  Warning: SAM2 produced no mask at seed location — skipping.")
        continue
    print(f"  SAM2 track ID for this macropinosome = {target_sam_id}")

    # Write hole + ring for every earlier frame back to tracking_mask.
    # The last frame is already filled from the detection step; skip it.
    for frame_t in range(n_total):
        if frame_t == t:
            continue

        hole_crop_t = (sam_result[frame_t] == target_sam_id)
        if not hole_crop_t.any():
            continue

        # Fill small gaps inside the SAM mask (same as detection step)
        hole_filled_t = binary_fill_holes(hole_crop_t).astype(np.uint8)

        # Reconstruct ring via EDT: pixels within [1, outer_r] px of hole boundary
        dist_t     = distance_transform_edt(hole_filled_t == 0)
        ring_crop_t = ((dist_t > 0) & (dist_t <= outer_r)).astype(np.uint8)

        # Write to tracking_mask without overwriting earlier detections
        _view = tracking_mask[frame_t, _cy0:_cy1, _cx0:_cx1]
        _view[(hole_filled_t == 1) & (_view == 0)] = hole_id_target
        _view[(ring_crop_t  == 1) & (_view == 0)] = ring_id_target

print(f"\nTracking complete.")
print(f"Frames with at least one mask: {(tracking_mask.any(axis=(1, 2))).sum()} / {n_total}")


# %% QC: tracked macropinosomes over time
# Debug plot showing sample frames throughout the video with tracked holes (red)
# and rings (blue) overlaid on the raw image. Displays up to 5 frames evenly
# spaced across the temporal range.
if debug and len(macropinosome_coordinates_filtered) > 0:
    # Select up to 5 frames evenly spaced across the video
    n_display = min(5, n_total)
    frame_indices = np.linspace(0, n_total - 1, n_display, dtype=int)

    fig, axes = plt.subplots(1, n_display, figsize=(n_display * 4, 4), squeeze=False)

    for plot_idx, frame_t in enumerate(frame_indices):
        ax = axes[0, plot_idx]

        # Show full-frame raw
        ax.imshow(raw_frames[frame_t], cmap="gray")

        # Overlay holes (red) and rings (blue)
        frame_mask = tracking_mask[frame_t]
        ov = np.zeros((*raw_frames[frame_t].shape, 4), dtype=np.float32)

        # Holes: odd labels
        hole_pixels = (frame_mask % 2 == 1) & (frame_mask > 0)
        ov[hole_pixels] = [1.0, 0.2, 0.2, 0.5]  # red

        # Rings: even labels
        ring_pixels = (frame_mask % 2 == 0) & (frame_mask > 0)
        ov[ring_pixels] = [0.3, 0.7, 1.0, 0.4]  # blue

        ax.imshow(ov)

        # Draw contours around each macropinosome's hole and ring
        for hole_id in np.unique(frame_mask):
            if hole_id == 0 or hole_id % 2 == 0:
                continue  # Skip background and ring labels
            ring_id = hole_id + 1
            ax.contour(frame_mask == hole_id, colors="cyan", linewidths=1.0)
            ax.contour(frame_mask == ring_id, colors="yellow", linewidths=0.8)

        n_objects = len(np.unique(frame_mask)) // 2  # pairs of (hole, ring)
        ax.set_title(f"t={frame_t}  ({n_objects} objects)", fontsize=9)
        ax.axis("off")

    fig.suptitle("Tracked macropinosomes (red=hole, blue=ring)", fontsize=11)
    plt.tight_layout()
    plt.show()


# %% QC: zoomed macropinosome tracks over time
# Grid showing each accepted macropinosome across multiple timepoints.
# Rows = different macropinosomes; Columns = evenly spaced timepoints.
# Each cell is a ±tile_size crop centered on the macropinosome with hole (red)
# and ring (blue) overlays.
if debug and len(macropinosome_coordinates_filtered) > 0:
    n_mp = len(macropinosome_coordinates_filtered)
    n_time_samples = min(5, n_total)
    time_samples = np.linspace(0, n_total - 1, n_time_samples, dtype=int)

    fig, axes = plt.subplots(n_mp, n_time_samples, figsize=(n_time_samples * 3, n_mp * 3), squeeze=False)

    for mp_row, (mp_y, mp_x) in enumerate(macropinosome_coordinates_filtered):
        hole_id_mp = 2 * mp_row + 1
        ring_id_mp = 2 * mp_row + 2

        for time_col, frame_t in enumerate(time_samples):
            ax = axes[mp_row, time_col]

            # Crop bounds around this macropinosome
            _cy0 = max(0, mp_y - tile_size)
            _cy1 = min(H_img, mp_y + tile_size + 1)
            _cx0 = max(0, mp_x - tile_size)
            _cx1 = min(W_img, mp_x + tile_size + 1)

            # Extract crop from raw and mask
            raw_crop = raw_frames[frame_t, _cy0:_cy1, _cx0:_cx1]
            mask_crop = tracking_mask[frame_t, _cy0:_cy1, _cx0:_cx1]

            # Show raw image
            ax.imshow(raw_crop, cmap="gray")

            # Overlay holes and rings for this macropinosome only
            ov = np.zeros((*raw_crop.shape, 4), dtype=np.float32)
            hole_pixels = (mask_crop == hole_id_mp)
            ring_pixels = (mask_crop == ring_id_mp)

            ov[hole_pixels] = [1.0, 0.2, 0.2, 0.6]  # red for hole
            ov[ring_pixels] = [0.3, 0.7, 1.0, 0.5]  # blue for ring

            ax.imshow(ov)

            # Contours for clarity
            ax.contour(hole_pixels, colors="cyan", linewidths=1.0)
            ax.contour(ring_pixels, colors="yellow", linewidths=0.8)

            # Mark the centroid
            _local_y = mp_y - _cy0
            _local_x = mp_x - _cx0
            if 0 <= _local_y < raw_crop.shape[0] and 0 <= _local_x < raw_crop.shape[1]:
                ax.plot(_local_x, _local_y, "c+", markersize=10, markeredgewidth=1.5)

            ax.set_title(f"MP#{hole_id_mp} t={frame_t}", fontsize=8)
            ax.axis("off")

    fig.suptitle(f"{n_mp} macropinosomes tracked across {n_time_samples} timepoints", fontsize=11)
    plt.tight_layout()
    plt.show()







# %%
