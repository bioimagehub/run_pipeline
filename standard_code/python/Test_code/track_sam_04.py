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
import torch
import torch.nn.functional as F

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


def _gauss_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    """Return a (1, 1, k, k) normalised Gaussian kernel for F.conv2d."""
    k = int(6 * sigma + 1) | 1  # ensure odd
    ax = torch.arange(k, dtype=torch.float32) - k // 2
    g = torch.exp(-ax ** 2 / (2.0 * sigma ** 2))
    g2d = torch.outer(g, g)
    g2d = g2d / g2d.sum()
    return g2d[None, None].to(device)


def dog_detect_gpu(
    holes_tyx: np.ndarray,
    dog_min_sigma: float,
    dog_max_sigma: float,
    dog_threshold_rel: float,
    device: torch.device,
) -> list[list[tuple[int, int]]]:
    """Batched GPU DoG + NMS on a (T, H, W) holes array.

    Returns a list of T lists, each containing (y, x) blob-centre tuples.
    """
    t_in = torch.from_numpy(holes_tyx[:, None].astype(np.float32)).to(device)  # (T,1,H,W)

    k1 = _gauss_kernel(dog_min_sigma, device)
    k2 = _gauss_kernel(dog_max_sigma, device)
    p1, p2 = k1.shape[-1] // 2, k2.shape[-1] // 2

    dog = F.conv2d(t_in, k1, padding=p1) - F.conv2d(t_in, k2, padding=p2)  # (T,1,H,W)

    # Per-frame relative threshold
    frame_max = dog.amax(dim=(1, 2, 3), keepdim=True)
    above_thr = dog >= (frame_max * dog_threshold_rel)

    # Non-maximum suppression: keep only strict local maxima
    nms_k = max(3, int(dog_min_sigma * 2 + 1) | 1)
    local_max = F.max_pool2d(dog, kernel_size=nms_k, stride=1, padding=nms_k // 2)
    is_peak = (dog == local_max) & above_thr  # (T,1,H,W)

    is_peak_np = is_peak.squeeze(1).cpu().numpy()  # (T,H,W) bool
    blobs_per_frame: list[list[tuple[int, int]]] = []
    for ti in range(holes_tyx.shape[0]):
        ys, xs = np.where(is_peak_np[ti])
        blobs_per_frame.append(list(zip(ys.tolist(), xs.tolist())))
    return blobs_per_frame


def flood_fill_frame(
    filled_frame: np.ndarray,
    blob_centers: list[tuple[int, int]],
    max_hole_area: int,
) -> np.ndarray:
    """Flood-fill from each detected centre to recover the exact hole mask."""
    H, W = filled_frame.shape
    det = np.zeros((H, W), dtype=np.uint8)
    for cy, cx in blob_centers:
        if not (0 <= cy < H and 0 <= cx < W):
            continue
        flooded = flood_fill(filled_frame, seed_point=(cy, cx),
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

# SAM2 tracking parameters
max_centroid_dist = 50.0   # max px a vesicle can move frame-to-frame for ID linking
tile_size         = 512    # SAM2 runs on tiles of this size (reduces GPU VRAM vs full frame)
tile_overlap      = 64     # extra context pixels on each side — discarded at stitch time


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


# %% Stage 1: fill-holes for all frames  (CPU parallel threads)
print(f"Computing fill-holes for {n_total} frames (parallel threads) …")
filled_list = Parallel(n_jobs=-1, prefer="threads", verbose=5)(
    delayed(greyscale_fill_holes)(all_frames[t].astype(np.float32))
    for t in range(n_total)
)
filled_tyx = np.stack(filled_list)                        # (T, H, W) float32
holes_tyx  = filled_tyx - all_frames.astype(np.float32)  # (T, H, W) float32 — only hole pixels > 0
print(f"Fill-holes complete.")


# %% Stage 2: batched GPU DoG + NMS
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running batched DoG on {_device} …")
blobs_per_frame = dog_detect_gpu(
    holes_tyx, dog_min_sigma, dog_max_sigma, dog_threshold_rel, _device
)
n_total_blobs = sum(len(b) for b in blobs_per_frame)
print(f"  Found {n_total_blobs} blob centres across {n_total} frames.")


# %% Stage 3: flood-fill per frame to recover exact hole masks  (CPU parallel threads)
print("Flood-filling detected blobs (parallel threads) …")
det_list = Parallel(n_jobs=-1, prefer="threads", verbose=5)(
    delayed(flood_fill_frame)(filled_tyx[t], blobs_per_frame[t], max_hole_area)
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


# %% Run SAM2 tracking  (tiled — keeps GPU VRAM within dedicated 24 GB)
#
# Each tile is stride×stride, extended by tile_overlap px on all sides for
# context.  Only the center (non-overlapping) region is written back.
# Track IDs are offset per tile so they are globally unique.

stride    = tile_size
tile_rows = list(range(0, H, stride))
tile_cols = list(range(0, W, stride))
n_tiles   = len(tile_rows) * len(tile_cols)
print(f"Running SAM2 tracking (tiled: {tile_size}px + {tile_overlap}px overlap) …")
print(f"  Grid: {len(tile_rows)} rows × {len(tile_cols)} cols = {n_tiles} tiles")

tracked_masks = np.zeros((n_total, H, W), dtype=np.int32)
id_offset     = 0  # running max track ID across all processed tiles

for ti, r_start in enumerate(tile_rows):
    for tj, c_start in enumerate(tile_cols):
        tile_num = ti * len(tile_cols) + tj + 1

        # Extended tile with overlap context on all sides
        r0 = max(0, r_start - tile_overlap)
        r1 = min(H, r_start + stride + tile_overlap)
        c0 = max(0, c_start - tile_overlap)
        c1 = min(W, c_start + stride + tile_overlap)

        tile_frames = all_frames[:, r0:r1, c0:c1]    # (T, th, tw)
        tile_det    = det_mask_tyx[:, r0:r1, c0:c1]  # (T, th, tw)

        if tile_det.sum() == 0:
            print(f"  [{tile_num}/{n_tiles}] tile({ti},{tj}) — no detections, skipped.")
            continue

        n_det_frames = int((tile_det.sum(axis=(1, 2)) > 0).sum())
        print(f"  [{tile_num}/{n_tiles}] tile({ti},{tj})  "
              f"extent=({r0}:{r1}, {c0}:{c1})  "
              f"detections in {n_det_frames} frames …")

        tile_tracked = run_tracking(
            img_tyx=tile_frames.astype(np.float32),
            det_mask_tyx=tile_det,
            predictor=predictor,
            max_centroid_dist=max_centroid_dist,
        )  # (T, th, tw) int32, values 0..K

        # Offset IDs so they are globally unique across tiles
        if tile_tracked.max() > 0:
            tile_tracked[tile_tracked > 0] += id_offset
            id_offset = int(tile_tracked.max())

        # Write back the center (non-overlap) stripe only
        src_r0 = r_start - r0                         # offset inside extended tile
        src_r1 = src_r0 + min(stride, H - r_start)
        src_c0 = c_start - c0
        src_c1 = src_c0 + min(stride, W - c_start)
        dst_r1 = min(r_start + stride, H)
        dst_c1 = min(c_start + stride, W)

        tracked_masks[:, r_start:dst_r1, c_start:dst_c1] = \
            tile_tracked[:, src_r0:src_r1, src_c0:src_c1]

n_tracks = int(tracked_masks.max())
print(f"Tiled tracking complete.  Output shape: {tracked_masks.shape}  Total track IDs: {n_tracks}")


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