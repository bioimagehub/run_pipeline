# %% Imports
import sys
import os
import importlib
import subprocess
import threading

import numpy as np
import matplotlib.pyplot as plt
from bioio import BioImage
from joblib import Parallel, delayed
from scipy.ndimage import label as scipy_label
from skimage.morphology import flood_fill, reconstruction
import time
import tifffile
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sam2_utils
importlib.reload(sam2_utils)
from sam2_utils import load_sam2_video_predictor, run_tracking, find_similar_object


# %% GPU monitor (nvidia-smi polling in background thread)
class _GpuMonitor:
    """Background thread that polls nvidia-smi every `interval` seconds.

    Usage::
        mon = _GpuMonitor(interval=1.0)
        mon.start()
        ...
        mon.stop()
        mon.report()
    """

    def __init__(self, interval: float = 1.0, gpu_index: int = 0) -> None:
        self.interval = interval
        self.gpu_index = gpu_index
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.util_pct:    list[float] = []
        self.mem_used_mb: list[float] = []

    def _poll(self) -> None:
        query = "utilization.gpu,memory.used"
        cmd = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            f"--query-gpu={query}",
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
        """Clear accumulated samples (call between stages to get per-stage stats)."""
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
    reconstructed = reconstruction(seed, inverted, method='dilation')
    filled = image_max - reconstructed
    return filled


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
    batch_size: int = 16,
) -> list[list[tuple[int, int, float]]]:
    """Multi-scale batched GPU DoG + NMS on a (T, H, W) holes array.

    Returns a list of T lists, each containing (y, x, sigma) blob-centre tuples.
    sigma is the representative scale at which the blob had maximum DoG response.
    Blobs whose intensity-weighted aspect ratio exceeds max_aspect_ratio are
    rejected as elongated structures (cell edges, filopodia, etc.).

    Memory strategy
    ---------------
    Frames are processed in temporal batches of ``batch_size`` to bound GPU
    memory use.  Within each batch a running max replaces the full
    (B, S, H, W) scale stack — only two (B, H, W) tensors (running max +
    current scale) are live at once, cutting peak VRAM by factor S.
    """
    T = holes_tyx.shape[0]
    sigmas = np.geomspace(dog_min_sigma, dog_max_sigma, n_scales + 1)
    scale_sigmas = np.array([(sigmas[i] + sigmas[i + 1]) / 2 for i in range(n_scales)])
    nms_k = max(3, int(dog_min_sigma * 2 + 1) | 1)

    blobs_per_frame: list[list[tuple[int, int, float]]] = []

    # Precompute all Gaussian kernels once — they are identical across every batch.
    gauss_kernels = [_gauss_kernel(float(s), device) for s in sigmas]

    for batch_start in range(0, T, batch_size):
        batch_end = min(batch_start + batch_size, T)
        t_in = torch.from_numpy(
            holes_tyx[batch_start:batch_end, None].astype(np.float32)
        ).to(device)  # (B, 1, H, W)

        # Running max across scales — never materialise (B, S, H, W).
        # Peak VRAM: t_in + dog_max + best_scale_idx + one dog scale ≈ 4×(B,H,W).
        dog_max: torch.Tensor | None = None
        best_scale_idx: torch.Tensor | None = None
        for i in range(n_scales):
            k1 = gauss_kernels[i]
            k2 = gauss_kernels[i + 1]
            p1, p2 = k1.shape[-1] // 2, k2.shape[-1] // 2
            dog = (F.conv2d(t_in, k1, padding=p1) - F.conv2d(t_in, k2, padding=p2)).squeeze(1)  # (B,H,W)
            if dog_max is None:
                dog_max = dog
                best_scale_idx = torch.zeros(dog.shape, dtype=torch.long, device=device)
            else:
                better = dog > dog_max
                dog_max = torch.where(better, dog, dog_max)
                best_scale_idx = torch.where(
                    better,
                    torch.full_like(best_scale_idx, i),
                    best_scale_idx,
                )
        del t_in

        assert dog_max is not None and best_scale_idx is not None

        # Per-frame relative threshold + NMS
        frame_max = dog_max.amax(dim=(1, 2), keepdim=True)
        above_thr = dog_max >= (frame_max * dog_threshold_rel)
        local_max = F.max_pool2d(
            dog_max[:, None], kernel_size=nms_k, stride=1, padding=nms_k // 2
        ).squeeze(1)
        is_peak = (dog_max == local_max) & above_thr  # (B, H, W)

        is_peak_np     = is_peak.cpu().numpy()
        best_scale_np  = best_scale_idx.cpu().numpy()
        dog_max_np     = dog_max.cpu().numpy()

        del dog_max, best_scale_idx, is_peak, local_max, above_thr, frame_max
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for bi in range(batch_end - batch_start):
            ys, xs = np.where(is_peak_np[bi])
            blob_sigmas = scale_sigmas[best_scale_np[bi][ys, xs]]
            frame = dog_max_np[bi]
            blobs: list[tuple[int, int, float]] = []
            for y, x, sigma in zip(ys.tolist(), xs.tolist(), blob_sigmas.tolist()):
                if _aspect_ratio(frame, y, x, sigma) <= max_aspect_ratio:
                    blobs.append((y, x, sigma))

            # Cross-scale NMS: keep smallest sigma first, suppress larger blobs
            # whose centre falls within 2×sigma of an already-accepted smaller blob.
            blobs.sort(key=lambda b: b[2])
            kept: list[tuple[int, int, float]] = []
            for by, bx, bs in blobs:
                if not any(
                    ((by - ky) ** 2 + (bx - kx) ** 2) ** 0.5 < ks * 2
                    for ky, kx, ks in kept
                ):
                    kept.append((by, bx, bs))
            blobs_per_frame.append(kept)

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
prediction_video_path = r"E:\Oyvind\BIP-hub-scratch\train_macropinosome_model\deep_learning_output\predictions\Input_crest__KiaWee__250225_RPE-mNG-Phafin2_BSD_10ul_001_1_drift_pred.ome.tif"
output_folder     = r"C:\Users\oyvinode\Desktop\del"

channel           = 0
z_slice           = 0

# DoG + flood_fill detection parameters
dog_min_sigma     = 2      # smallest expected hole radius in pixels
dog_max_sigma     = 30     # largest expected hole radius in pixels
dog_threshold_rel = 0.1    # DoG response threshold (lower = more sensitive)
dog_batch_size    = 10     # frames per GPU batch for DoG (reduce if OOM)


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
tile_size         = 40     # crop size for flood-fill (Stage 3)
sam_tile_hw       = None   # SAM2 spatial tile size in px; set None to disable tiling (full frame)
sam_tile_overlap  = 64     # overlap between adjacent SAM2 tiles in px
temporal_window   = 10     # frames per SAM2 call; None = process all frames in one session (high RAM)


# For debug
max_T =   None # set to None or a small integer to limit frames processed during dev/testing
debug = False  # if True, shows QC plots and prints more info during processing


# %% Load image metadata
img = BioImage(video_path)
n_total = img.dims.T
print(f"Image shape: {img.shape}  dtype: {img.dtype}")
print(f"Total frames: {n_total}")
n_total = img.dims.T
if max_T is not None:
    n_total = min(n_total, max_T)


# Used as seeds for tracking segmentation
prediction_img = BioImage(prediction_video_path)
print(f"Prediction image shape: {prediction_img.shape}  dtype: {prediction_img.dtype}")

# %% Pre-load all frames into RAM as (T, H, W)
t0_total = time.perf_counter()
_gpu_mon.start()
print(f"Loading {n_total} prediction frames …")
all_frames = np.stack(
    [prediction_img.get_image_data("YX", T=t, C=channel, Z=z_slice)
     for t in range(n_total)]
)  # (T, H, W) — full frame, no crop
H, W = all_frames.shape[1], all_frames.shape[2]
print(f"Loaded predictions: {all_frames.shape}  dtype: {all_frames.dtype}")

print(f"Loading {n_total} raw frames …")
raw_frames = np.stack(
    [img.get_image_data("YX", T=t, C=channel, Z=z_slice)
     for t in range(n_total)]
)  # (T, H, W) — full frame, no crop
print(f"Loaded raw:         {raw_frames.shape}  dtype: {raw_frames.dtype}")
print(f"  Load elapsed: {time.perf_counter() - t0_total:.1f}s")


# %% Stage 2: batched GPU DoG + NMS to find blob centres
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running batched DoG on {_device} …")
t0_dog = time.perf_counter()
_gpu_mon.reset()

blobs_per_frame = dog_detect_gpu(
    all_frames, dog_min_sigma, dog_max_sigma, dog_threshold_rel, _device,
    max_aspect_ratio=max_aspect_ratio,
    batch_size=dog_batch_size,
)
n_total_blobs = sum(len(b) for b in blobs_per_frame)
print(f"  Found {n_total_blobs} blob centres across {n_total} frames.")
print(f"  Stage 2 (DoG) elapsed: {time.perf_counter() - t0_dog:.1f}s")
_gpu_mon.report("Stage2-DoG")


if debug:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(all_frames[-1], cmap="gray")
    for y, x, sigma in blobs_per_frame[-1]:
        circle = plt.Circle((x, y), radius=sigma * 2, color="red", fill=False, linewidth=1)
        ax.add_patch(circle)
    ax.set_title(f"Frame {n_total-1} — {len(blobs_per_frame[-1])} blobs (radius=2σ, aspect ratio ≤ {max_aspect_ratio})")
    ax.axis("off")
    plt.tight_layout()
    plt.show()  


# %% Stage 3: Build binary detection mask from blob centres via flood-fill (parallel)
t0_stage3 = time.perf_counter()
print(f"Building detection mask from blob centres (parallel, {n_total} frames) …")

frame_dets = Parallel(n_jobs=-1, prefer='threads')(
    delayed(_build_det_frame)(
        blobs_per_frame[t_idx], raw_frames[t_idx],
        H, W, tile_size, min_hole_area, max_hole_area, min_circularity,
    )
    for t_idx in range(n_total)
)
det_mask_tyx = np.stack(frame_dets)

n_det_pixels = int(det_mask_tyx.sum())
n_det_frames = int((det_mask_tyx.sum(axis=(1, 2)) > 0).sum())
print(f"  Detection mask: {n_det_pixels} hole pixels across {n_det_frames}/{n_total} frames.")
print(f"  Stage 3 elapsed: {time.perf_counter() - t0_stage3:.1f}s")

if debug:
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(raw_frames[-1], cmap="gray")
    plt.title(f"Frame {n_total-1} raw")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(raw_frames[-1], cmap="gray")
    red_overlay = np.zeros((*det_mask_tyx[-1].shape, 4), dtype=np.float32)
    red_overlay[det_mask_tyx[-1] != 0] = [1.0, 0.0, 0.0, 0.6]
    plt.imshow(red_overlay)
    plt.title(f"Frame {n_total-1} detection mask overlay")  
    plt.axis("off")
    plt.tight_layout()
    plt.show()



# %% Stage 4: SAM2 backward tracking via sam2_utils.run_tracking
print("Loading SAM2 video predictor …")
predictor = load_sam2_video_predictor()

t0_sam = time.perf_counter()
_gpu_mon.reset()
global_mask_tyx = run_tracking(
    img_tyx=all_frames,
    det_mask_tyx=det_mask_tyx,
    predictor=predictor,
    max_centroid_dist=max_centroid_dist,
    tile_hw=sam_tile_hw,
    tile_overlap=sam_tile_overlap,
    temporal_window=temporal_window,
)
n_unique_tracks = int(global_mask_tyx.max())
print(f"Tracking complete. Unique tracks: {n_unique_tracks}")
print(f"  Stage 4 (SAM2) elapsed: {time.perf_counter() - t0_sam:.1f}s")
_gpu_mon.stop()
_gpu_mon.report("Stage4-SAM2")


# %% Debug: overlay global_mask_tyx on all_frames for all tracked frames
if debug:
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

    n_cols = min(n_total, 2)
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
    plt.suptitle(f"Global mask — {n_unique_tracks} tracks", fontsize=10)
    plt.tight_layout()
    plt.show()



# %% save as tif
output_path = os.path.join(output_folder, "global_mask_v7.tif")
import tifffile
tifffile.imwrite(output_path, global_mask_tyx.astype(np.uint16))
print(f"Saved global mask to {output_path}")
print(f"Total elapsed: {time.perf_counter() - t0_total:.1f}s")
# %%
