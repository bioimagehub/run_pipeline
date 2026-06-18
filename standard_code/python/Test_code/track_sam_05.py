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
from sam2_utils import load_sam2_video_predictor, run_tracking, find_similar_object


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


def _aspect_ratio(dog_frame: np.ndarray, cy: int, cx: int, sigma: float) -> float:
    """Intensity-weighted eigenvalue aspect ratio (λ_max / λ_min) of a DoG patch.

    Returns np.inf for degenerate patches (zero response or point source).
    Circular blobs → ratio ≈ 1.0; elongated structures → ratio >> 1.
    """
    r = max(3, int(sigma * 2.0))
    y0, y1 = max(0, cy - r), min(dog_frame.shape[0], cy + r + 1)
    x0, x1 = max(0, cx - r), min(dog_frame.shape[1], cx + r + 1)
    patch = dog_frame[y0:y1, x0:x1].astype(np.float64)
    patch = np.clip(patch, 0, None)  # only positive DoG response
    total = patch.sum()
    if total == 0:
        return np.inf
    gy, gx = np.mgrid[y0:y1, x0:x1]
    my = (gy * patch).sum() / total
    mx = (gx * patch).sum() / total
    cov_yy = ((gy - my) ** 2 * patch).sum() / total
    cov_xx = ((gx - mx) ** 2 * patch).sum() / total
    cov_yx = ((gy - my) * (gx - mx) * patch).sum() / total
    eigvals = np.linalg.eigvalsh([[cov_yy, cov_yx], [cov_yx, cov_xx]])
    l_min, l_max = eigvals[0], eigvals[1]
    if l_min <= 0:
        return np.inf
    return float(l_max / l_min)


def dog_detect_gpu(
    holes_tyx: np.ndarray,
    dog_min_sigma: float,
    dog_max_sigma: float,
    dog_threshold_rel: float,
    device: torch.device,
    n_scales: int = 8,
    max_aspect_ratio: float = 3.0,
) -> list[list[tuple[int, int, float]]]:
    """Multi-scale batched GPU DoG + NMS on a (T, H, W) holes array.

    Returns a list of T lists, each containing (y, x, sigma) blob-centre tuples.
    sigma is the representative scale at which the blob had maximum DoG response.
    Blobs whose intensity-weighted aspect ratio exceeds max_aspect_ratio are
    rejected as elongated structures (cell edges, filopodia, etc.).
    """
    t_in = torch.from_numpy(holes_tyx[:, None].astype(np.float32)).to(device)  # (T,1,H,W)

    # Log-spaced sigmas; each adjacent pair forms one DoG scale
    sigmas = np.geomspace(dog_min_sigma, dog_max_sigma, n_scales + 1)
    scale_sigmas = np.array([(sigmas[i] + sigmas[i + 1]) / 2 for i in range(n_scales)])

    # Compute DoG response for each scale and stack: (T, S, H, W)
    dog_scales = []
    for i in range(n_scales):
        k1 = _gauss_kernel(float(sigmas[i]), device)
        k2 = _gauss_kernel(float(sigmas[i + 1]), device)
        p1, p2 = k1.shape[-1] // 2, k2.shape[-1] // 2
        dog = F.conv2d(t_in, k1, padding=p1) - F.conv2d(t_in, k2, padding=p2)  # (T,1,H,W)
        dog_scales.append(dog.squeeze(1))  # (T,H,W)

    dog_vol = torch.stack(dog_scales, dim=1)  # (T, S, H, W)

    # Max across scales → combined response map and best-scale index
    dog_max, best_scale_idx = dog_vol.max(dim=1)  # both (T, H, W)

    # Per-frame relative threshold
    frame_max = dog_max.amax(dim=(1, 2), keepdim=True)
    above_thr = dog_max >= (frame_max * dog_threshold_rel)

    # Non-maximum suppression: keep only strict local maxima
    nms_k = max(3, int(dog_min_sigma * 2 + 1) | 1)
    local_max = F.max_pool2d(dog_max[:, None], kernel_size=nms_k, stride=1, padding=nms_k // 2).squeeze(1)
    is_peak = (dog_max == local_max) & above_thr  # (T, H, W)

    is_peak_np = is_peak.cpu().numpy()           # (T, H, W) bool
    best_scale_np = best_scale_idx.cpu().numpy()  # (T, H, W) int
    dog_max_np = dog_max.cpu().numpy()            # (T, H, W) float32

    blobs_per_frame: list[list[tuple[int, int, float]]] = []
    for ti in range(holes_tyx.shape[0]):
        ys, xs = np.where(is_peak_np[ti])
        blob_sigmas = scale_sigmas[best_scale_np[ti][ys, xs]]
        frame = dog_max_np[ti]
        blobs: list[tuple[int, int, float]] = []
        for y, x, sigma in zip(ys.tolist(), xs.tolist(), blob_sigmas.tolist()):
            if _aspect_ratio(frame, y, x, sigma) <= max_aspect_ratio:
                blobs.append((y, x, sigma))
        blobs_per_frame.append(blobs)
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

def extract_holes_connected(
    holes_frame: np.ndarray,
    max_hole_area: int,
):
    """
    Extract connected regions of equal value in holes_frame.
    Returns a binary mask of valid holes.
    """
    det = np.zeros_like(holes_frame, dtype=np.uint8)

    # Only consider positive (filled) pixels
    values = np.unique(holes_frame)
    values = values[values > 0]

    for v in values:
        mask = (holes_frame == v)

        labeled, num = scipy_label(mask)

        for i in range(1, num + 1):
            region = (labeled == i)
            area = region.sum()

            if area > max_hole_area:
                continue

            det[region] = 1

    return det

from scipy.ndimage import label
import numpy as np

from scipy.ndimage import label
import numpy as np

def extract_holes_connected_optimized(
    holes_frame: np.ndarray,
    max_hole_area: int,
    min_hole_area: int = 0,
) -> np.ndarray:
    """
    Extract connected regions of equal value in holes_frame, 
    filtering based on min and max area using pre-filtering.
    """
    det = np.zeros_like(holes_frame, dtype=np.uint8)
    
    # 1. Use unique and return counts
    values, counts = np.unique(holes_frame, return_counts=True)

    # 2. --- OPTIMIZATION STEP ---
    # Filter both the values and their corresponding counts simultaneously.
    # We keep only values whose total count meets the minimum requirement.
    # (Handle the case where min_hole_area = 0 separately if needed, but 
    # assuming min_hole_area >= 1 for this check.)
    
    if min_hole_area > 0:
        # Create a boolean mask for the required counts
        count_filter = counts >= min_hole_area
        # Apply the filter to both arrays
        values = values[count_filter]
        counts = counts[count_filter]
    
    # We still need to handle the case where a value is zero, 
    # but the logic above assumes we are dealing with the values *present*.
    # We should explicitly ensure we are only looking at positive background values.
    values = values[values > 0]


    # 3. Core Processing Loop
    for v in values:
        # 1. Create the mask for the current value
        mask = (holes_frame == v)
        
        # 2. Label the connected components in this mask
        labeled, num = label(mask)

        # 3. Loop through labels and process
        for i in range(1, num + 1):
            # Check the size of the region i in the labeled array
            region_area = np.count_nonzero(labeled == i)

            # Only check the maximum size here. The minimum size is guaranteed 
            # by the pre-filtering step (Step 2).
            if region_area > max_hole_area:
                continue

            # Mark the entire valid region
            det[labeled == i] = 1

    return det




def _detect_new_objects_at_frame(
    t: int,
    blobs: list[tuple[int, int, float]],
    all_frames: np.ndarray,
    global_mask_tyx: np.ndarray,
    next_track_id: int,
    tile_size: int,
    max_hole_area: int,
    min_hole_area: int,
    max_aspect_ratio: float,
    min_fill_brightness_ratio: float = 1.5,
    min_ring_contrast_ratio: float = 1.2,
    ring_width: int = 3,
    min_circularity: float = 0.5,
    min_hole_depth: float = 0.0,
) -> tuple[dict[int, dict], int]:
    """Flood-fill each DoG blob at frame t and register any new track IDs.

    Skips blobs whose centre pixel already has a non-zero value in
    global_mask_tyx[t] (already tracked by a prior SAM propagation).

    Extra filters applied after flood-fill:
    - fill_brightness_ratio : filled[component].mean() >= ratio * tile.mean()
        Ensures the hole sits inside a genuinely bright structure.
    - ring_contrast_ratio   : tile[ring].mean() / tile[component].mean() >= ratio
        Checks bright membrane ring around a dark interior (macropinosome-specific).
        ring = dilation(component, ring_width) minus component itself.
    - circularity           : 4*pi*area / perimeter^2 >= threshold  (1.0 = circle)
        Rejects crescents/L-shapes that survive the aspect-ratio test.
    - hole_depth            : mean((filled - tile)[component]) >= min_hole_depth
        Ensures the region is actually darker than its filled surroundings.
        Set to 0 to disable.

    Returns
    -------
    new_tracks : {track_id: {'y': cy_global, 'x': cx_global}}
    next_track_id : updated counter
    """
    H, W = all_frames.shape[1], all_frames.shape[2]
    new_tracks: dict[int, dict] = {}

    for blob_y, blob_x, sigma in blobs:
        # Skip if this location is already claimed by an existing track
        if global_mask_tyx[t, blob_y, blob_x] != 0:
            continue

        # Tile bounds
        y0 = max(0, blob_y - tile_size)
        y1 = min(H, blob_y + tile_size)
        x0 = max(0, blob_x - tile_size)
        x1 = min(W, blob_x + tile_size)

        seed_y = blob_y - y0  # seed in tile-local coordinates
        seed_x = blob_x - x0

        tile = all_frames[t, y0:y1, x0:x1].astype(np.float32)
        filled = greyscale_fill_holes(tile)

        # Guard: seed must be within the tile
        if not (0 <= seed_y < filled.shape[0] and 0 <= seed_x < filled.shape[1]):
            continue

        flooded = flood_fill(filled, seed_point=(seed_y, seed_x),
                             new_value=-1.0, tolerance=0, connectivity=1)
        raw_mask = flooded == -1.0

        if raw_mask.sum() > max_hole_area or raw_mask.sum() < min_hole_area:
            continue

        # Keep only the connected component that contains the seed
        labeled_raw, _ = scipy_label(raw_mask)
        seed_label = labeled_raw[seed_y, seed_x]
        if seed_label == 0:
            continue
        component = (labeled_raw == seed_label)

        # Aspect-ratio filter on the seed component
        ys_c, xs_c = np.where(component)
        cy_loc = int(ys_c.mean())
        cx_loc = int(xs_c.mean())
        if _aspect_ratio(tile, cy_loc, cx_loc, sigma) > max_aspect_ratio:
            continue

        # --- Filter 1: fill brightness ratio -----------------------------------
        # The filled value at the hole must be >= ratio * mean tile intensity.
        # Rejects holes in genuinely dark background regions.
        tile_mean = tile.mean()
        if tile_mean > 0 and filled[component].mean() < min_fill_brightness_ratio * tile_mean:
            continue

        # --- Filter 2: hole depth ----------------------------------------------
        # The hole must actually be darker than its filled surroundings.
        hole_depth = (filled.astype(np.float64) - tile.astype(np.float64))[component].mean()
        if hole_depth < min_hole_depth:
            continue

        # --- Filter 3: ring contrast -------------------------------------------
        # The membrane ring around the hole should be brighter than the interior.
        from scipy.ndimage import binary_dilation
        ring = binary_dilation(component, iterations=ring_width) & ~component
        if ring.any() and component.any():
            ring_mean = tile[ring].mean()
            interior_mean = tile[component].mean()
            if interior_mean > 0 and ring_mean / interior_mean < min_ring_contrast_ratio:
                continue

        # --- Filter 4: circularity ---------------------------------------------
        # 4*pi*area / perimeter^2.  Computed from the border pixel count as perimeter.
        from skimage.measure import perimeter as sk_perimeter
        area = component.sum()
        perim = sk_perimeter(component.astype(np.uint8))
        if perim > 0 and (4.0 * np.pi * area / perim ** 2) < min_circularity:
            continue

        # Convert centroid to global coordinates
        cy_global = cy_loc + y0
        cx_global = cx_loc + x0

        # Write to global mask (no-overwrite: first writer wins)
        write_region = component & (global_mask_tyx[t, y0:y1, x0:x1] == 0)
        global_mask_tyx[t, y0:y1, x0:x1] = np.where(
            write_region, next_track_id, global_mask_tyx[t, y0:y1, x0:x1]
        )

        new_tracks[next_track_id] = {'y': cy_global, 'x': cx_global}
        next_track_id += 1

    return new_tracks, next_track_id


def _propagate_one_step(
    t: int,
    active_tracks: dict[int, dict],
    all_frames: np.ndarray,
    global_mask_tyx: np.ndarray,
    predictor,
    tile_size: int,
) -> None:
    """Propagate every active track from frame t to frame t-1 using SAM2.

    Uses find_similar_object with the video path: passes tile_t as the
    reference frame and tile_prev as the single-frame target.  SAM sees
    frame t as frame-0 (prompt) and frame t-1 as frame-1 (propagation target).

    Neighbour tracks within the same tile are included as context labels so
    SAM can delineate boundaries between touching objects.

    Mutates
    -------
    global_mask_tyx[t-1] : written with predicted track IDs (no-overwrite).
    active_tracks         : centroids updated to t-1 position; lost tracks removed.
    """
    H, W = all_frames.shape[1], all_frames.shape[2]
    lost: list[int] = []

    for track_id, state in list(active_tracks.items()):
        cy, cx = state['y'], state['x']

        # Tile centred on the last known position (in frame t)
        y0 = max(0, cy - tile_size)
        y1 = min(H, cy + tile_size)
        x0 = max(0, cx - tile_size)
        x1 = min(W, cx + tile_size)

        tile_t    = all_frames[t,     y0:y1, x0:x1]
        tile_prev = all_frames[t - 1, y0:y1, x0:x1]

        # Build source_mask from global_mask_tyx[t] within the tile
        # Primary track → label 1; neighbours → labels 2, 3, ...
        region_ids = global_mask_tyx[t, y0:y1, x0:x1]
        source_mask = np.zeros_like(region_ids, dtype=np.int32)

        if np.any(region_ids == track_id):
            source_mask[region_ids == track_id] = 1
        else:
            # Track not visible in this tile — mark as lost
            lost.append(track_id)
            continue

        next_label = 2
        for other_id in np.unique(region_ids):
            if other_id == 0 or other_id == track_id:
                continue
            source_mask[region_ids == other_id] = next_label
            next_label += 1

        # find_similar_object: tile_prev as (1, H, W) triggers the video path.
        # Returns (1, H_tile, W_tile) int32; label 1 is our primary track.
        pred = find_similar_object(
            source_image=tile_t,
            source_mask=source_mask,
            target=tile_prev[np.newaxis],
            predictor=predictor,
        )  # (1, th, tw) int32

        primary_pred = pred[0] == 1  # (th, tw) bool

        if primary_pred.sum() == 0:
            lost.append(track_id)
            continue

        # Write to global_mask_tyx[t-1] (no-overwrite: first writer wins)
        write_region = primary_pred & (global_mask_tyx[t - 1, y0:y1, x0:x1] == 0)
        global_mask_tyx[t - 1, y0:y1, x0:x1] = np.where(
            write_region, track_id, global_mask_tyx[t - 1, y0:y1, x0:x1]
        )

        # Update centroid for next step (t-1 → t-2)
        ys_p, xs_p = np.where(primary_pred)
        active_tracks[track_id] = {
            'y': int(ys_p.mean()) + y0,
            'x': int(xs_p.mean()) + x0,
        }

    for tid in lost:
        del active_tracks[tid]


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


# General filtering
max_aspect_ratio          = 2.0    # reject blobs with eigenvalue aspect ratio above this
max_hole_area             = 10000  # discard flood-fill regions larger than this (inter-cell)
min_hole_area             = 8      # discard flood-fill regions smaller than this (noise)
min_fill_brightness_ratio = 0.0    # disabled — fill_ratio median ~0.94 inside bright cells; use hole_depth instead
min_ring_contrast_ratio   = 1.2    # membrane ring mean / hole interior mean must exceed this
ring_width                = 3      # thickness in px of the border ring used for ring contrast
min_circularity           = 0.5    # 4π·area/perimeter² threshold (1.0 = perfect circle)
min_hole_depth            = 5.0    # mean (filled − original) at hole pixels; ensures genuine dark holes

# SAM2 tracking parameters
max_centroid_dist = 50.0   # max px a vesicle can move frame-to-frame for ID linking
tile_size         = 40  # SAM2 runs on tiles of this size (reduces GPU VRAM vs full frame)


# For debug
max_T =   5 # set to None or a small integer to limit frames processed during dev/testing
debug = True  # if True, shows QC plots and prints more info during processing


# %% Load image metadata
img = BioImage(video_path)
n_total = img.dims.T
if max_T is not None:
    n_total = min(n_total, max_T)

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


# %% Stage 2: batched GPU DoG + NMS to find blob centres
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running batched DoG on {_device} …")

blobs_per_frame = dog_detect_gpu(
    holes_tyx, dog_min_sigma, dog_max_sigma, dog_threshold_rel, _device,
    max_aspect_ratio=max_aspect_ratio,
)
n_total_blobs = sum(len(b) for b in blobs_per_frame)
print(f"  Found {n_total_blobs} blob centres across {n_total} frames.")


if debug:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(holes_tyx[-1], cmap="gray")
    for y, x, sigma in blobs_per_frame[-1]:
        circle = plt.Circle((x, y), radius=sigma * 2, color="red", fill=False, linewidth=1)
        ax.add_patch(circle)
    ax.set_title(f"Frame {n_total-1} — {len(blobs_per_frame[-1])} blobs (radius=2σ, aspect ratio ≤ {max_aspect_ratio})")
    ax.axis("off")
    plt.tight_layout()
    plt.show()  


# %% Stage 3 + 4: Backwards tracking — detect holes, then propagate each track
#   one frame at a time from the last frame to the first.
#
#   Loop structure (t = last → 0):
#     A. Detect new blobs at frame t via flood-fill, assign track IDs.
#     B. If t > 0: propagate all active tracks t → t-1 via SAM2 tile calls.
#        Each call uses a 2-frame tile (frame t = prompt, frame t-1 = target).
#        Tile centre follows the track's last known centroid; updates after SAM.
#     C. At t == 0: detection only, no propagation needed.

from tqdm import tqdm

print("Loading SAM2 video predictor …")
predictor = load_sam2_video_predictor()

# Global output: (T, H, W) int32 — pixel value = track ID, 0 = background
global_mask_tyx = np.zeros((n_total, H, W), dtype=np.int32)

# active_tracks: {track_id: {'y': cy_global, 'x': cx_global}}
active_tracks: dict[int, dict] = {}
next_track_id = 1

for t in tqdm(range(n_total - 1, -1, -1), desc="backwards tracking", unit="frame"):

    # --- Step A: detect new holes at this frame ---------------------------------
    new_tracks, next_track_id = _detect_new_objects_at_frame(
        t=t,
        blobs=blobs_per_frame[t],
        all_frames=all_frames,
        global_mask_tyx=global_mask_tyx,
        next_track_id=next_track_id,
        tile_size=tile_size,
        max_hole_area=max_hole_area,
        min_hole_area=min_hole_area,
        max_aspect_ratio=max_aspect_ratio,
        min_fill_brightness_ratio=min_fill_brightness_ratio,
        min_ring_contrast_ratio=min_ring_contrast_ratio,
        ring_width=ring_width,
        min_circularity=min_circularity,
        min_hole_depth=min_hole_depth,
    )
    active_tracks.update(new_tracks)
    print(f"  t={t}: +{len(new_tracks)} new tracks, {len(active_tracks)} active")

    # --- Step B: propagate all active tracks t → t-1 ----------------------------
    if t == 0:
        break  # no preceding frame to propagate to

    _propagate_one_step(
        t=t,
        active_tracks=active_tracks,
        all_frames=all_frames,
        global_mask_tyx=global_mask_tyx,
        predictor=predictor,
        tile_size=tile_size,
    )
    print(f"  t={t}: {len(active_tracks)} tracks still active after propagation")

print(f"Tracking complete. Total unique tracks: {next_track_id - 1}")


# %% Debug: overlay global_mask_tyx on all_frames for all tracked frames
if debug:
    from matplotlib.colors import BoundaryNorm
    from matplotlib import cm

    unique_ids = np.unique(global_mask_tyx)
    unique_ids = unique_ids[unique_ids != 0]
    n_ids = len(unique_ids)
    id_to_color_idx = {tid: i + 1 for i, tid in enumerate(unique_ids)}

    # Build a colour index volume: 0 = background, 1..n = each track
    color_vol = np.zeros_like(global_mask_tyx, dtype=np.int32)
    for tid, ci in id_to_color_idx.items():
        color_vol[global_mask_tyx == tid] = ci

    cmap = cm.get_cmap('tab20', max(n_ids + 1, 2))

    n_cols = min(n_total, 5)
    n_rows = (n_total + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows),
                             squeeze=False)
    for t in range(n_total):
        ax = axes[t // n_cols][t % n_cols]
        ax.imshow(all_frames[t], cmap='gray')
        overlay = cmap(color_vol[t] / max(n_ids, 1))
        overlay[..., 3] = np.where(global_mask_tyx[t] != 0, 0.5, 0.0)
        ax.imshow(overlay)
        ax.set_title(f"t={t}", fontsize=8)
        ax.axis('off')
    for t in range(n_total, n_rows * n_cols):
        axes[t // n_cols][t % n_cols].axis('off')
    plt.suptitle(f"Global mask — {next_track_id - 1} tracks", fontsize=10)
    plt.tight_layout()
    plt.show()



# %%
