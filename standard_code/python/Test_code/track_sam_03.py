# %% Imports
import sys
import os
import importlib

import numpy as np
import matplotlib.pyplot as plt
from bioio import BioImage
from joblib import Parallel, delayed
from scipy.ndimage import label as scipy_label
from skimage.morphology import flood_fill, reconstruction
from skimage.feature import blob_dog

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sam2_utils
importlib.reload(sam2_utils)
from sam2_utils import load_sam2_video_predictor, run_tracking


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
    reconstructed = reconstruction(seed, inverted, method='dilation')
    filled = image_max - reconstructed
    return filled


def detect_blobs_in_frame(
    frame: np.ndarray,
    dog_min_sigma: float,
    dog_max_sigma: float,
    dog_threshold_rel: float,
    max_hole_area: int,
) -> np.ndarray:
    """Fill holes + DoG + flood_fill for a single frame.

    Returns binary (H, W) uint8: 1 where a valid hole blob was detected.
    Regions larger than max_hole_area are discarded as inter-cell fills.
    """
    frame_f32 = frame.astype(np.float32)
    filled = greyscale_fill_holes(frame_f32)
    holes = filled - frame_f32

    if holes.max() == 0.0:
        return np.zeros(frame.shape, dtype=np.uint8)

    blobs = blob_dog(
        holes,
        min_sigma=dog_min_sigma,
        max_sigma=dog_max_sigma,
        threshold_rel=dog_threshold_rel,
    )

    det = np.zeros(frame.shape, dtype=np.uint8)
    H, W = frame.shape
    for y, x, _ in blobs:
        cy, cx = int(round(y)), int(round(x))
        if not (0 <= cy < H and 0 <= cx < W):
            continue
        # Use -1 as sentinel — impossible for non-negative intensity images
        flooded = flood_fill(filled, seed_point=(cy, cx),
                             new_value=-1.0, tolerance=0, connectivity=1)
        region = flooded == -1.0
        if region.sum() > max_hole_area:
            continue  # discard large inter-cell fills
        det[region] = 1

    return det


# %% Input paths
video_path        = r"E:\Oyvind\BIP-hub-scratch\train_macropinosome_model\input_drift_corrected\Input_crest__KiaWee__250225_RPE-mNG-Phafin2_BSD_10ul_001_1_drift.ome.tif"
output_folder     = r"C:\Users\oyvinode\Desktop\del"

size              = 2000   # spatial crop (YX) — reduce if RAM is tight
channel           = 0
z_slice           = 0

# DoG + flood_fill detection parameters
dog_min_sigma     = 2      # smallest expected hole radius in pixels
dog_max_sigma     = 30     # largest expected hole radius in pixels
dog_threshold_rel = 0.1    # DoG response threshold (lower = more sensitive)
max_hole_area     = 10000  # discard flood-fill regions larger than this (inter-cell)

# SAM2 tracking parameter
max_centroid_dist = 50.0   # max px a vesicle can move frame-to-frame for ID linking


# %% Load image metadata
img = BioImage(video_path)
n_total = img.dims.T
print(f"Image shape: {img.shape}  dtype: {img.dtype}")
print(f"Total frames: {n_total}")


# %% Pre-load all frames into RAM as (T, H, W)
print(f"Loading {n_total} frames …")
all_frames = np.stack(
    [img.get_image_data("YX", T=t, C=channel, Z=z_slice)[:size, :size]
     for t in range(n_total)]
)  # (T, H, W)
H, W = all_frames.shape[1], all_frames.shape[2]
print(f"Loaded: {all_frames.shape}  dtype: {all_frames.dtype}")


# %% Per-frame detection: fill-holes + DoG + flood_fill  (parallel, threaded)
print(f"Detecting blobs in {n_total} frames (parallel threads) …")
det_list = Parallel(n_jobs=-1, prefer="threads", verbose=5)(
    delayed(detect_blobs_in_frame)(
        all_frames[t], dog_min_sigma, dog_max_sigma, dog_threshold_rel, max_hole_area
    )
    for t in range(n_total)
)
det_mask_tyx = np.stack(det_list).astype(np.int32)  # (T, H, W)  values: 0 or 1

n_frames_det = int((det_mask_tyx.sum(axis=(1, 2)) > 0).sum())
print(f"Detections: {n_frames_det}/{n_total} frames contain at least one blob.")


# %% QC: detection overlay on evenly-spaced frames
n_show  = min(6, n_total)
step    = max(1, n_total // n_show)
show_ts = list(range(0, n_total, step))[:n_show]

fig, axes = plt.subplots(1, len(show_ts), figsize=(4 * len(show_ts), 4))
if len(show_ts) == 1:
    axes = [axes]
for ax, t in zip(axes, show_ts):
    ax.imshow(all_frames[t], cmap="gray")
    overlay = np.zeros((*all_frames[t].shape, 4), dtype=np.float32)
    overlay[det_mask_tyx[t] > 0] = [1.0, 0.2, 0.2, 0.6]
    ax.imshow(overlay)
    n_cc = len(np.unique(scipy_label(det_mask_tyx[t])[0])) - 1
    ax.set_title(f"T={t}\n{n_cc} blobs", fontsize=8)
    ax.axis("off")
plt.suptitle("Per-frame DoG + flood_fill detections", fontsize=9)
plt.tight_layout()
plt.show()


# %% Load SAM2 video predictor  (GPU if CUDA available, CPU otherwise)
predictor = load_sam2_video_predictor()


# %% Run SAM2 tracking
print("Running SAM2 tracking …")
tracked_masks = run_tracking(
    img_tyx=all_frames.astype(np.float32),
    det_mask_tyx=det_mask_tyx,
    predictor=predictor,
    max_centroid_dist=max_centroid_dist,
)
n_tracks = int(tracked_masks.max())
print(f"Tracking complete.  Output shape: {tracked_masks.shape}  Track IDs: {n_tracks}")


# %% QC: grid of SAM2-tracked frames
fig, axes = plt.subplots(1, len(show_ts), figsize=(4 * len(show_ts), 4))
if len(show_ts) == 1:
    axes = [axes]
for ax, t in zip(axes, show_ts):
    ax.imshow(all_frames[t], cmap="gray")
    tm = tracked_masks[t]
    if tm.max() > 0:
        overlay = np.zeros((*tm.shape, 4), dtype=np.float32)
        for tid in np.unique(tm):
            if tid == 0:
                continue
            c = plt.cm.tab20(int(tid) % 20 / 20.0)
            overlay[tm == tid] = [c[0], c[1], c[2], 0.55]
        ax.imshow(overlay)
    ax.set_title(f"T={t}\n{len(np.unique(tm)) - 1} tracks", fontsize=8)
    ax.axis("off")
plt.suptitle("SAM2 tracked masks", fontsize=9)
plt.tight_layout()
plt.show()


# %% Save 2-channel TIFF: C=0 raw image, C=1 tracked mask  (T, C, H, W) float32
import tifffile

os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "tracked_image_and_mask.tif")

combined = np.stack(
    [all_frames.astype(np.float32), tracked_masks.astype(np.float32)],
    axis=1,
)  # (T, 2, H, W)

tifffile.imwrite(
    output_path,
    combined,
    imagej=True,
    metadata={"axes": "TCYX"},
)
print(f"Saved to: {output_path}  shape={combined.shape}")

# %%