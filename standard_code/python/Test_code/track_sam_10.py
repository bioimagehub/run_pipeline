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


QC_FAIL_HOLE_ID = 1
QC_FAIL_RING_ID = 2
FIRST_TRACK_HOLE_ID = 3


def _qc_macropinosome_mask(
    frame_raw: np.ndarray,
    hole_mask_crop: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> dict[str, object]:
    """Run macropinosome QC on one candidate hole mask inside one frame crop.

    The returned dict contains:
      passes_qc, reason, hole_mask_crop, ring_mask_crop, outer_half_step
    """
    tile_raw = frame_raw[y0:y1, x0:x1]
    hole_mask_crop = binary_fill_holes(hole_mask_crop).astype(np.uint8)
    area = int(hole_mask_crop.sum())

    if not (min_hole_area <= area <= max_hole_area):
        return {
            "passes_qc": False,
            "reason": f"area={area} outside [{min_hole_area}, {max_hole_area}]",
            "hole_mask_crop": hole_mask_crop,
            "ring_mask_crop": np.zeros_like(hole_mask_crop, dtype=np.uint8),
            "outer_half_step": None,
        }

    prev = hole_mask_crop.copy()
    ring_medians: list[float] = []
    for _r in range(ring_search_steps):
        curr = dilation(prev, disk(1))
        annulus = (curr == 1) & (prev == 0)
        if not annulus.any():
            break
        ring_medians.append(float(np.median(tile_raw[annulus])))
        prev = curr

    if len(ring_medians) < 3:
        return {
            "passes_qc": False,
            "reason": "ring profile too short (<3 annuli)",
            "hole_mask_crop": hole_mask_crop,
            "ring_mask_crop": np.zeros_like(hole_mask_crop, dtype=np.uint8),
            "outer_half_step": None,
        }

    profile = np.array(ring_medians, dtype=np.float32)
    ring_peak_step = int(np.argmax(profile))
    ring_peak_val = float(profile[ring_peak_step])

    _n_bg = max(1, len(profile) - ring_peak_step - 2)
    bg_estimate = float(np.min(profile[-_n_bg:]))
    corrected_half = bg_estimate + (ring_peak_val - bg_estimate) * 0.5

    hole_median_val = float(np.median(tile_raw[hole_mask_crop == 1]))
    is_dark_inside = hole_median_val < ring_peak_val * 0.90
    has_dark_outside = (
        ring_peak_step < len(profile) - 1 and
        float(np.min(profile[ring_peak_step + 1:])) < corrected_half
    )
    is_ring_within_range = ring_peak_step < len(profile) - 2

    outer_half_step = ring_peak_step + 2
    for _i in range(ring_peak_step + 1, len(profile)):
        if profile[_i] < corrected_half:
            outer_half_step = _i + 1
            break

    dist_from_hole = distance_transform_edt(hole_mask_crop == 0)
    ring_mask_crop = ((dist_from_hole > 0) & (dist_from_hole <= outer_half_step)).astype(np.uint8)

    reasons: list[str] = []
    if not is_dark_inside:
        reasons.append(f"hole_median={hole_median_val:.0f} >= 90% peak={ring_peak_val * 0.90:.0f}")
    if not has_dark_outside:
        reasons.append(f"outside never drops below bg_half={corrected_half:.0f}")
    if not is_ring_within_range:
        reasons.append(f"ring peak at last step {ring_peak_step + 1}")
    if outer_half_step > 10:
        reasons.append(f"outer_half_step={outer_half_step} > 10")

    return {
        "passes_qc": len(reasons) == 0,
        "reason": ", ".join(reasons) if reasons else "pass",
        "hole_mask_crop": hole_mask_crop,
        "ring_mask_crop": ring_mask_crop,
        "outer_half_step": int(outer_half_step),
    }


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
#   1/2 reserved for QC-failed hole/ring masks
#   odd  labels >= 3 = dark hole  (3, 5, 7, …)
#   even labels >= 4 = bright ring (4, 6, 8, …)  where ring 2k encircles hole 2k-1
H_img, W_img = last_frame.shape
hole_ring_mask = np.zeros((H_img, W_img), dtype=np.int32)
next_odd_id = FIRST_TRACK_HOLE_ID

macropinosome_coordinates_filtered = []
mp_outer_half_steps: list[int] = []  # ring width (px from hole edge) for each accepted MP
accepted_hole_ids: list[int] = []

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

    hole_mask_crop = binary_fill_holes(flood_mask).astype(np.uint8)
    qc_result = _qc_macropinosome_mask(last_frame, hole_mask_crop, _y0, _y1, _x0, _x1)

    if qc_result["passes_qc"]:
        hole_id = next_odd_id
        ring_id = next_odd_id + 1
        next_odd_id += 2
        macropinosome_coordinates_filtered.append((y, x))
        mp_outer_half_steps.append(int(qc_result["outer_half_step"]))
        accepted_hole_ids.append(hole_id)
    else:
        hole_id = QC_FAIL_HOLE_ID
        ring_id = QC_FAIL_RING_ID
        if debug:
            print(f"QC-failed ({y},{x}): {qc_result['reason']}")

    _existing_crop = hole_ring_mask[_y0:_y1, _x0:_x1]
    _forbidden = (_existing_crop != 0)
    ring_mask_crop = qc_result["ring_mask_crop"] & (~_forbidden)
    _view = hole_ring_mask[_y0:_y1, _x0:_x1]
    _view[(qc_result["hole_mask_crop"] == 1) & (_view == 0)] = hole_id
    _view[(ring_mask_crop == 1) & (_view == 0)] = ring_id

    if debug:
        outer_half_step = qc_result["outer_half_step"] if qc_result["outer_half_step"] is not None else -1
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
            f"Score  area={area}px²  QC={'pass' if qc_result['passes_qc'] else 'fail'}  outer_step={outer_half_step}"
        )
        axes[1].axis("off")
        axes[2].imshow(qc_result["ring_mask_crop"], cmap="Blues")
        axes[2].set_title(f"QC={'pass' if qc_result['passes_qc'] else 'fail'}  outer_step={outer_half_step}")
        axes[2].axis("off")
        plt.tight_layout()
        plt.show()

print(f"\nAccepted {len(macropinosome_coordinates_filtered)} / {len(macropinosome_coordinates)} candidates.")
print(f"QC-fail reserve labels: hole={QC_FAIL_HOLE_ID}, ring={QC_FAIL_RING_ID}")
print(f"Hole labels (odd):  3, 5, 7, …  up to {next_odd_id - 2 if next_odd_id > FIRST_TRACK_HOLE_ID else 'none'}")
print(f"Ring labels (even): 4, 6, 8, …  up to {next_odd_id - 1 if next_odd_id > FIRST_TRACK_HOLE_ID else 'none'}")

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
        hole_id_q = accepted_hole_ids[_idx]
        ring_id_q = hole_id_q + 1
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
# Stage A: Track all macropinosomes seeded in the last frame.
# Stage B: For each earlier frame (T-2 ... 0), detect new macropinosomes.
#          If a detected center is outside all currently tracked masks, seed a
#          new object there and run backward SAM2 tracking from that frame.

def _detect_macropinosomes_in_frame(
    frame_raw: np.ndarray,
    threshold_rel: float = 0.4,
) -> list[dict[str, object]]:
    """Detect macropinosomes in one frame using the same logic as frame T-1.

    Returns a list of dicts with keys:
      y, x, outer_half_step, hole_mask_crop, crop_bounds
    """
    frame_filled = greyscale_fill_holes_kernel_2d(frame_raw, kernel_size=256, kernel_overlap=50)
    frame_score = gaussian_filter(frame_filled - frame_raw, sigma=3)
    coords = peak_local_max(frame_score, min_distance=5, threshold_rel=threshold_rel)

    frame_score_f32 = frame_score.astype(np.float32)
    accepted: list[dict[str, object]] = []

    for y, x in coords:
        _pad = ring_search_steps + 55
        _y0 = max(0, y - _pad); _y1 = min(H_img, y + _pad + 1)
        _x0 = max(0, x - _pad); _x1 = min(W_img, x + _pad + 1)
        tile_score = frame_score_f32[_y0:_y1, _x0:_x1]
        tile_raw = frame_raw[_y0:_y1, _x0:_x1]
        cy_c, cx_c = y - _y0, x - _x0

        peak_score = float(tile_score[cy_c, cx_c])
        if peak_score <= 0:
            continue

        filled_s = flood_fill(tile_score, (cy_c, cx_c), new_value=-1.0, tolerance=peak_score / 2)
        flood_mask = (filled_s == -1.0).astype(np.uint8)

        props = regionprops(flood_mask)
        if not props:
            continue
        area = props[0].area
        if not (min_hole_area <= area <= max_hole_area):
            continue

        hole_mask_crop = binary_fill_holes(flood_mask).astype(np.uint8)
        qc_result = _qc_macropinosome_mask(frame_raw, hole_mask_crop, _y0, _y1, _x0, _x1)
        if not qc_result["passes_qc"]:
            continue
        accepted.append({
            "y": int(y),
            "x": int(x),
            "outer_half_step": int(qc_result["outer_half_step"]),
            "hole_mask_crop": qc_result["hole_mask_crop"],
            "crop_bounds": (_y0, _y1, _x0, _x1),
        })

    return accepted


def _build_video_3ch_until(
    last_frame_idx: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    context_pad: int,
) -> np.ndarray:
    """Build (last_frame_idx+1, 3, H, W) video with channels [raw, score, raw]."""
    cH = y1 - y0
    cW = x1 - x0
    out = np.zeros((last_frame_idx + 1, 3, cH, cW), dtype=np.float32)

    for frame_t in range(last_frame_idx + 1):
        raw_crop = raw_frames[frame_t, y0:y1, x0:x1].astype(np.float32)

        fy0 = max(0, y0 - context_pad); fy1 = min(H_img, y1 + context_pad)
        fx0 = max(0, x0 - context_pad); fx1 = min(W_img, x1 + context_pad)
        raw_large = raw_frames[frame_t, fy0:fy1, fx0:fx1]
        filled_large = greyscale_fill_holes_kernel_2d(raw_large, kernel_size=128, kernel_overlap=25)
        ry0 = y0 - fy0; ry1 = ry0 + cH
        rx0 = x0 - fx0; rx1 = rx0 + cW
        score_crop = gaussian_filter(
            filled_large[ry0:ry1, rx0:rx1].astype(np.float32) - raw_crop,
            sigma=3,
        )

        out[frame_t, 0] = raw_crop
        out[frame_t, 1] = score_crop
        out[frame_t, 2] = raw_crop

    return out


def _track_single_seed_backward(
    seed_t: int,
    seed_y: int,
    seed_x: int,
    hole_id_target: int,
    ring_id_target: int,
    outer_r: int,
    candidate_hole_crop: np.ndarray | None = None,
) -> bool:
    """Track one seed backward from frame seed_t and write into tracking_mask."""
    cy0 = max(0, seed_y - tile_size)
    cy1 = min(H_img, seed_y + tile_size + 1)
    cx0 = max(0, seed_x - tile_size)
    cx1 = min(W_img, seed_x + tile_size + 1)
    cH = cy1 - cy0
    cW = cx1 - cx0

    if candidate_hole_crop is not None and candidate_hole_crop.any():
        qc_seed = _qc_macropinosome_mask(raw_frames[seed_t], candidate_hole_crop, cy0, cy1, cx0, cx1)
        view_seed = tracking_mask[seed_t, cy0:cy1, cx0:cx1]
        hole_seed_id = hole_id_target if qc_seed["passes_qc"] else QC_FAIL_HOLE_ID
        ring_seed_id = ring_id_target if qc_seed["passes_qc"] else QC_FAIL_RING_ID
        view_seed[(qc_seed["hole_mask_crop"] == 1) & (view_seed == 0)] = hole_seed_id
        view_seed[(qc_seed["ring_mask_crop"] == 1) & (view_seed == 0)] = ring_seed_id

    video_3ch = _build_video_3ch_until(seed_t, cy0, cy1, cx0, cx1, context_pad=max(64, tile_size))

    det_mask = np.zeros((seed_t + 1, cH, cW), dtype=np.uint8)
    seed_frame_crop = tracking_mask[seed_t, cy0:cy1, cx0:cx1]
    det_mask[seed_t] = ((seed_frame_crop % 2 == 1) & (seed_frame_crop > 0)).astype(np.uint8)

    sam_result = _run_tracking_on_region(
        video_3ch,
        det_mask,
        predictor,
        max_centroid_dist=float(tile_size),
        label=f"seed@t{seed_t} id{hole_id_target} ",
    )

    local_y = seed_y - cy0
    local_x = seed_x - cx0
    target_sam_id = 0

    if candidate_hole_crop is not None and candidate_hole_crop.any():
        ids = sam_result[seed_t][candidate_hole_crop > 0]
        ids = ids[ids != 0]
        if ids.size > 0:
            vals, counts = np.unique(ids, return_counts=True)
            target_sam_id = int(vals[np.argmax(counts)])

    if target_sam_id == 0 and 0 <= local_y < cH and 0 <= local_x < cW:
        target_sam_id = int(sam_result[seed_t, local_y, local_x])

    if target_sam_id == 0:
        if debug:
            print(f"  Warning: SAM2 produced no mask at seed ({seed_y},{seed_x}) in t={seed_t}.")
        return False

    for frame_t in range(seed_t):
        hole_crop_t = (sam_result[frame_t] == target_sam_id)
        if not hole_crop_t.any():
            continue

        qc_result = _qc_macropinosome_mask(raw_frames[frame_t], hole_crop_t.astype(np.uint8), cy0, cy1, cx0, cx1)
        write_hole_id = hole_id_target if qc_result["passes_qc"] else QC_FAIL_HOLE_ID
        write_ring_id = ring_id_target if qc_result["passes_qc"] else QC_FAIL_RING_ID

        view = tracking_mask[frame_t, cy0:cy1, cx0:cx1]
        view[(qc_result["hole_mask_crop"] == 1) & (view == 0)] = write_hole_id
        view[(qc_result["ring_mask_crop"] == 1) & (view == 0)] = write_ring_id

    return True


print("\nStage A: Tracking seeds from the last frame ...")
for mp_idx, (mp_y, mp_x) in enumerate(macropinosome_coordinates_filtered):
    hole_id_target = accepted_hole_ids[mp_idx]
    ring_id_target = hole_id_target + 1
    outer_r = mp_outer_half_steps[mp_idx]
    print(f"  seed MP#{hole_id_target} at t={t}, center=({mp_y},{mp_x}), ring_width={outer_r}")
    _track_single_seed_backward(
        seed_t=t,
        seed_y=mp_y,
        seed_x=mp_x,
        hole_id_target=hole_id_target,
        ring_id_target=ring_id_target,
        outer_r=outer_r,
        candidate_hole_crop=None,
    )


print("\nStage B: Backward discovery of new macropinosomes ...")
new_tracks_added = 0
for seed_t in range(t - 1, -1, -1):
    candidates = _detect_macropinosomes_in_frame(raw_frames[seed_t], threshold_rel=0.4)
    if debug:
        print(f"t={seed_t}: detected {len(candidates)} candidate(s) before overlap filtering")

    accepted_new_in_frame = 0
    for cand in candidates:
        seed_y = int(cand["y"])
        seed_x = int(cand["x"])

        # Reject candidates whose center is already covered by any tracked mask.
        if tracking_mask[seed_t, seed_y, seed_x] != 0:
            continue

        hole_id_target = next_odd_id
        ring_id_target = next_odd_id + 1
        next_odd_id += 2
        outer_r = int(cand["outer_half_step"])

        # Build candidate hole mask in this seed tile coordinates.
        cy0 = max(0, seed_y - tile_size)
        cy1 = min(H_img, seed_y + tile_size + 1)
        cx0 = max(0, seed_x - tile_size)
        cx1 = min(W_img, seed_x + tile_size + 1)
        cH = cy1 - cy0
        cW = cx1 - cx0
        candidate_hole_crop = np.zeros((cH, cW), dtype=np.uint8)

        y0_det, y1_det, x0_det, x1_det = cand["crop_bounds"]
        hole_mask_det = cand["hole_mask_crop"]
        iy0 = max(cy0, y0_det); iy1 = min(cy1, y1_det)
        ix0 = max(cx0, x0_det); ix1 = min(cx1, x1_det)
        if iy1 <= iy0 or ix1 <= ix0:
            continue

        src_y0, src_y1 = iy0 - y0_det, iy1 - y0_det
        src_x0, src_x1 = ix0 - x0_det, ix1 - x0_det
        dst_y0, dst_y1 = iy0 - cy0, iy1 - cy0
        dst_x0, dst_x1 = ix0 - cx0, ix1 - cx0
        candidate_hole_crop[dst_y0:dst_y1, dst_x0:dst_x1] = hole_mask_det[src_y0:src_y1, src_x0:src_x1]

        if not candidate_hole_crop.any():
            continue

        ok = _track_single_seed_backward(
            seed_t=seed_t,
            seed_y=seed_y,
            seed_x=seed_x,
            hole_id_target=hole_id_target,
            ring_id_target=ring_id_target,
            outer_r=outer_r,
            candidate_hole_crop=candidate_hole_crop,
        )

        if ok:
            accepted_new_in_frame += 1
            new_tracks_added += 1
            if debug:
                print(
                    f"  Added new track hole_id={hole_id_target} at t={seed_t}, "
                    f"center=({seed_y},{seed_x}), ring_width={outer_r}"
                )

    if debug and accepted_new_in_frame > 0:
        print(f"t={seed_t}: accepted {accepted_new_in_frame} new track(s)")

print(f"\nTracking complete.")
print(f"Frames with at least one mask: {(tracking_mask.any(axis=(1, 2))).sum()} / {n_total}")
print(f"New tracks discovered after t={t}: {new_tracks_added}")


# %% QC: tracked macropinosomes over time
# Debug plot showing sample frames throughout the video with tracked holes (red)
# and rings (blue) overlaid on the raw image. Displays up to 5 frames evenly
# spaced across the temporal range.
if debug and len(macropinosome_coordinates_filtered) > 0:
    # Select up to 10 frames evenly spaced across the video
    n_display = min(10, n_total)
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
# Grid showing tracked macropinosomes plus explicit failed-QC examples.
# Rows = different tracks / failed-QC components; Columns = evenly spaced
# timepoints. Each cell is a ±tile_size crop with hole/ring overlays.
if debug and (tracking_mask > 0).any():
    n_time_samples = min(10, n_total)
    time_samples = np.linspace(0, n_total - 1, n_time_samples, dtype=int)

    row_specs: list[dict[str, object]] = []

    # Normal tracked rows: all odd IDs except the reserved QC-fail bucket.
    hole_ids_all = sorted(
        int(_id) for _id in np.unique(tracking_mask)
        if _id >= FIRST_TRACK_HOLE_ID and _id % 2 == 1
    )
    for hole_id_mp in hole_ids_all:
        frames_present = np.where((tracking_mask == hole_id_mp).any(axis=(1, 2)))[0]
        if frames_present.size == 0:
            continue
        ref_t = int(frames_present[-1])
        ys, xs = np.where(tracking_mask[ref_t] == hole_id_mp)
        if ys.size == 0:
            continue
        cy = int(np.round(float(ys.mean())))
        cx = int(np.round(float(xs.mean())))
        row_specs.append({
            "label": f"MP#{hole_id_mp}",
            "hole_id": hole_id_mp,
            "ring_id": hole_id_mp + 1,
            "center": (cy, cx),
            "is_fail": False,
        })

    # Failed QC rows: one row per connected failed component in sampled frames.
    for frame_t in time_samples:
        fail_hole = (tracking_mask[frame_t] == QC_FAIL_HOLE_ID).astype(np.uint8)
        fail_ring = (tracking_mask[frame_t] == QC_FAIL_RING_ID).astype(np.uint8)
        labeled_fail = cc_label(fail_hole, connectivity=1)
        fail_ids = [int(_id) for _id in np.unique(labeled_fail) if _id != 0]
        for fail_comp_id in fail_ids:
            ys, xs = np.where(labeled_fail == fail_comp_id)
            if ys.size == 0:
                continue
            cy = int(np.round(float(ys.mean())))
            cx = int(np.round(float(xs.mean())))
            row_specs.append({
                "label": f"FAIL@t={frame_t}#{fail_comp_id}",
                "hole_id": QC_FAIL_HOLE_ID,
                "ring_id": QC_FAIL_RING_ID,
                "center": (cy, cx),
                "is_fail": True,
                "ref_t": int(frame_t),
            })

    n_rows_total = len(row_specs)
    if n_rows_total == 0:
        plt.figure(figsize=(6, 2))
        plt.title("No tracked or failed-QC masks found for zoom QC")
        plt.axis("off")
        plt.show()
    else:
        fig, axes = plt.subplots(
            n_rows_total,
            n_time_samples,
            figsize=(n_time_samples * 3, n_rows_total * 3),
            squeeze=False,
        )

        for mp_row, row_spec in enumerate(row_specs):
            hole_id_mp = int(row_spec["hole_id"])
            ring_id_mp = int(row_spec["ring_id"])
            mp_y, mp_x = row_spec["center"]
            is_fail = bool(row_spec["is_fail"])
            ref_t = int(row_spec.get("ref_t", -1))

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
                if is_fail and frame_t == ref_t:
                    fail_hole_crop = (mask_crop == QC_FAIL_HOLE_ID).astype(np.uint8)
                    fail_labeled_crop = cc_label(fail_hole_crop, connectivity=1)
                    local_y = mp_y - _cy0
                    local_x = mp_x - _cx0
                    fail_comp_id = 0
                    if 0 <= local_y < fail_labeled_crop.shape[0] and 0 <= local_x < fail_labeled_crop.shape[1]:
                        fail_comp_id = int(fail_labeled_crop[local_y, local_x])
                    hole_pixels = fail_labeled_crop == fail_comp_id if fail_comp_id != 0 else np.zeros_like(fail_hole_crop, dtype=bool)
                    ring_pixels = (mask_crop == QC_FAIL_RING_ID) & dilation(hole_pixels.astype(np.uint8), disk(2)).astype(bool)
                else:
                    hole_pixels = (mask_crop == hole_id_mp)
                    ring_pixels = (mask_crop == ring_id_mp)

                if is_fail:
                    ov[hole_pixels] = [1.0, 0.0, 0.8, 0.65]  # magenta for QC-failed hole
                    ov[ring_pixels] = [1.0, 0.6, 0.0, 0.50]  # orange for QC-failed ring
                else:
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

                ax.set_title(f"{row_spec['label']} t={frame_t}", fontsize=8)
                ax.axis("off")

        n_fail_rows = sum(1 for row in row_specs if bool(row["is_fail"]))
        fig.suptitle(
            f"{len(hole_ids_all)} tracked macropinosomes + {n_fail_rows} failed-QC rows across {n_time_samples} timepoints",
            fontsize=11,
        )
        plt.tight_layout()
        plt.show()







# %%
