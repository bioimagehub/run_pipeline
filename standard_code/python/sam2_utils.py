"""SAM2 utility functions for multi-object tracking.

Provides helpers for frame preparation, track-ID assignment, SAM2 prompt
registration / propagation, and predictor loading.
"""

import os
import tempfile
import warnings

warnings.filterwarnings("ignore", message="cannot import name '_C' from 'sam2'")

import numpy as np
from PIL import Image
from skimage.measure import label as cc_label, regionprops


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# def _normalize_to_uint8(frame: np.ndarray) -> np.ndarray:
#     frame = frame.astype(np.float32)
#     lo, hi = frame.min(), frame.max()
#     if hi == lo:
#         return np.zeros_like(frame, dtype=np.uint8)
#     return ((frame - lo) / (hi - lo) * 255).astype(np.uint8)
def _normalize_to_uint8(frame: np.ndarray) -> np.ndarray:
    frame = frame.astype(np.float32)
    lo = float(np.percentile(frame, 0.1))
    hi = float(np.percentile(frame, 99.9))
    if hi == lo:
        return np.zeros_like(frame, dtype=np.uint8)
    return np.clip((frame - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def _save_frames_as_jpeg(frames: np.ndarray, folder: str) -> None:
    os.makedirs(folder, exist_ok=True)
    for i, frame in enumerate(frames):
        if frame.ndim == 2:
            # (H, W) grayscale → replicate to RGB
            rgb = np.stack([_normalize_to_uint8(frame)] * 3, axis=-1)
        elif frame.ndim == 3:
            # (C, H, W) multi-channel
            n_ch = frame.shape[0]
            if n_ch > 3:
                print(
                    f"Warning: frame has {n_ch} channels, using only the first 3 for RGB."
                )
                frame = frame[:3]
                n_ch = 3
            channels = [_normalize_to_uint8(frame[c]) for c in range(n_ch)]
            # Pad missing channels with zeros (e.g. 2-channel → R, G, black)
            while len(channels) < 3:
                channels.append(np.zeros_like(channels[0]))
            rgb = np.stack(channels, axis=-1)
        else:
            raise ValueError(
                f"_save_frames_as_jpeg: unexpected frame shape {frame.shape}, expected (H, W) or (C, H, W)"
            )
        Image.fromarray(rgb).save(os.path.join(folder, f"{i:05d}.jpg"), quality=95)


# ---------------------------------------------------------------------------
# Step 1: Assign consistent track IDs — iterate BACKWARDS
# ---------------------------------------------------------------------------

def assign_track_ids(
    det_mask_tyx: np.ndarray,
    max_centroid_dist: float = 50.0,
) -> tuple[list[dict[int, int]], int]:
    """Assign consistent IDs by matching detections from last frame to first.

    Iterating in reverse means IDs are seeded by mature, easy-to-detect
    objects and propagated backward toward newly-formed ones.

    Returns
    -------
    frame_assignments : list length T, index = frame; value = {label -> track_id}
    n_tracks : total unique IDs assigned (IDs run 1 ... n_tracks)
    """
    T = det_mask_tyx.shape[0]
    frame_assignments: list[dict[int, int]] = [{} for _ in range(T)]
    # track_id -> centroid of most recently seen detection (going backwards)
    track_last_centroid: dict[int, tuple[float, float]] = {}
    next_id = 1

    for t in range(T - 1, -1, -1):          # REVERSED: last frame first
        hole = (det_mask_tyx[t] == 1).astype(np.uint8)
        labeled = cc_label(hole, connectivity=1)
        regions = regionprops(labeled)

        if not regions:
            continue

        assignments: dict[int, int] = {}
        used_ids: set[int] = set()

        for region in regions:
            cy, cx = region.centroid
            best_dist = max_centroid_dist
            best_id: int | None = None

            for tid, (py, px) in track_last_centroid.items():
                if tid in used_ids:
                    continue
                dist = ((cy - py) ** 2 + (cx - px) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid

            if best_id is None:
                best_id = next_id
                next_id += 1

            assignments[region.label] = best_id
            track_last_centroid[best_id] = (cy, cx)
            used_ids.add(best_id)

        frame_assignments[t] = assignments

    return frame_assignments, next_id - 1


# ---------------------------------------------------------------------------
# Step 2 + 3: Register SAM2 prompts and propagate BACKWARDS
# ---------------------------------------------------------------------------

def _run_tracking_on_region(
    img_tyx: np.ndarray,
    det_mask_tyx: np.ndarray,
    predictor,
    max_centroid_dist: float,
    label: str = "",
    temporal_window: int | None = None,
) -> np.ndarray:
    """Run the full SAM2 backward-tracking pipeline on one (T, H, W) region.

    Track IDs in the returned array start from 1.  The caller is responsible
    for shifting them to globally unique values when stitching tiles.

    Parameters
    ----------
    img_tyx          : (T, H, W) image region (used for JPEG encoding).
    det_mask_tyx     : (T, H, W) uint8 binary detection mask for this region.
    predictor        : SAM2VideoPredictor.
    max_centroid_dist: centroid linking distance (px) for assign_track_ids.
    label            : short string used in progress messages (e.g. tile coords).
    temporal_window  : if set, process this many frames per SAM2 call, carrying
                       the boundary mask forward to ensure track continuity.
                       Reduces peak GPU/RAM usage for long videos.  ``None``
                       processes all T frames in a single SAM2 session.

    Returns
    -------
    (T, H, W) int32 label map; 0 = background, 1..N = track IDs.
    """
    import torch
    from contextlib import nullcontext
    from tqdm import tqdm

    if img_tyx.ndim == 4:
        T, _nc, H, W = img_tyx.shape  # (T, C, H, W) multi-channel input
    else:
        T, H, W = img_tyx.shape       # (T, H, W) grayscale input
    device = predictor._device

    frame_assignments, n_tracks = assign_track_ids(det_mask_tyx, max_centroid_dist)
    print(f"    {label}{n_tracks} tracks across {T} frames.")

    # ------------------------------------------------------------------
    # Windowed temporal path: process `temporal_window` frames at a time,
    # carrying the boundary mask from one window into the next so that
    # tracks propagate continuously even across window boundaries.
    # Processes windows from LAST to FIRST (matching backward propagation).
    # ------------------------------------------------------------------
    if temporal_window is not None and T > temporal_window:
        result = np.zeros((T, H, W), dtype=np.int32)
        carry_mask: np.ndarray | None = None  # (H, W) int32 from the boundary of the next window
        t_end = T
        while t_end > 0:
            t_start = max(0, t_end - temporal_window)
            has_carry = (carry_mask is not None)
            # Extend by 1 carry frame so SAM2 propagates continuity backward
            # from the already-processed next window into this one.
            t_end_ext = min(t_end + 1, T) if has_carry else t_end
            wsize_total = t_end_ext - t_start
            wsize_core  = t_end - t_start

            # Pre-count prompts to skip JPEG encode + SAM2 init when empty.
            n_det_prompts = sum(
                len(frame_assignments[t_start + local_t])
                for local_t in range(wsize_core)
            )
            total_prompts_preview = n_det_prompts + (len(np.unique(carry_mask)) - 1 if has_carry and carry_mask is not None else 0)
            if total_prompts_preview == 0:
                # Nothing to track in this window — skip entirely.
                t_end = t_start
                continue

            print(
                f"    {label}window [{t_start},{t_end}) — "
                f"{wsize_total} frames ({wsize_core} core + {wsize_total - wsize_core} carry)"
            )
            win_img = img_tyx[t_start:t_end_ext]
            with tempfile.TemporaryDirectory() as tmpdir:
                _save_frames_as_jpeg(win_img, tmpdir)
                _autocast = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if device == "cuda"
                    else nullcontext()
                )
                with torch.inference_mode(), _autocast:
                    inference_state = predictor.init_state(
                        video_path=tmpdir,
                        offload_video_to_cpu=(device == "cpu"),
                    )
                    total_prompts = 0
                    # Detection prompts for core frames
                    for local_t in range(wsize_core):
                        global_t = t_start + local_t
                        assignments = frame_assignments[global_t]
                        if not assignments:
                            continue
                        hole = (det_mask_tyx[global_t] == 1).astype(np.uint8)
                        labeled_win = cc_label(hole, connectivity=1)
                        for lbl, track_id in assignments.items():
                            binary_mask = (labeled_win == lbl)
                            predictor.add_new_mask(
                                inference_state, frame_idx=local_t,
                                obj_id=track_id, mask=binary_mask,
                            )
                            total_prompts += 1
                    # Carry mask at the boundary frame (local index = wsize_core)
                    if has_carry:
                        for tid in np.unique(carry_mask):
                            if tid == 0:
                                continue
                            predictor.add_new_mask(
                                inference_state, frame_idx=wsize_core,
                                obj_id=int(tid), mask=(carry_mask == tid),
                            )
                            total_prompts += 1
                    print(f"    {label}window [{t_start},{t_end}): {total_prompts} prompts registered.")
                    with tqdm(total=wsize_total, desc=f"  propagate{' ' + label if label else ''} [{t_start},{t_end})", unit="frame", leave=False) as pbar:
                        if total_prompts > 0:
                            for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(
                                inference_state, start_frame_idx=wsize_total - 1, reverse=True,
                            ):
                                pbar.update(1)
                                if frame_idx >= wsize_core:
                                    continue  # skip carry frame
                                global_idx = t_start + frame_idx
                                for i, obj_id in enumerate(obj_ids):
                                    binary = (video_res_masks[i, 0].float() > 0.0).cpu().numpy()
                                    result[global_idx][binary] = int(obj_id)
                    predictor.reset_state(inference_state)
            # Carry for the next (earlier) window = earliest frame of this window
            new_carry = result[t_start]
            carry_mask = new_carry.copy() if new_carry.any() else None
            t_end = t_start
        return result

    # ------------------------------------------------------------------
    # Non-windowed (original) path — all T frames in a single SAM2 session.
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        _save_frames_as_jpeg(img_tyx, tmpdir)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            inference_state = predictor.init_state(
                video_path=tmpdir,
                offload_video_to_cpu=(device == "cpu"),
            )

            total_prompts = 0
            for t, assignments in enumerate(frame_assignments):
                if not assignments:
                    continue
                hole = (det_mask_tyx[t] == 1).astype(np.uint8)
                labeled = cc_label(hole, connectivity=1)
                for lbl, track_id in assignments.items():
                    binary_mask = (labeled == lbl)
                    predictor.add_new_mask(
                        inference_state,
                        frame_idx=t,
                        obj_id=track_id,
                        mask=binary_mask,
                    )
                    total_prompts += 1
            print(f"    {label}{total_prompts} prompts registered.")

            result = np.zeros((T, H, W), dtype=np.int32)
            with tqdm(total=T, desc=f"  propagate{' ' + label if label else ''}", unit="frame", leave=False) as pbar:
                for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=T - 1,
                    reverse=True,
                ):
                    for i, obj_id in enumerate(obj_ids):
                        binary = (video_res_masks[i, 0].float() > 0.0).cpu().numpy()
                        result[frame_idx][binary] = int(obj_id)
                    pbar.update(1)

            predictor.reset_state(inference_state)

    return result


def run_tracking(
    img_tyx: np.ndarray,
    det_mask_tyx: np.ndarray,
    predictor,
    max_centroid_dist: float = 50.0,
    tile_hw: int | None = None,
    tile_overlap: int = 64,
    temporal_window: int | None = None,
) -> np.ndarray:
    """Backward-pass SAM2 tracking fusing detection masks with optional tiling.

    When the full frame exceeds GPU VRAM, set ``tile_hw`` to process the frame
    in overlapping spatial tiles.  Each tile is tracked independently and the
    results are stitched with globally unique track IDs.  Tracks that straddle
    a tile boundary are handled by writing each tile's contribution to the
    non-overlapping central strip; the overlap region prevents edge artefacts.

    Parameters
    ----------
    img_tyx           : (T, H, W) image frames (used as SAM2 video input).
    det_mask_tyx      : (T, H, W) uint8 binary detection mask (1 = hole).
    predictor         : SAM2VideoPredictor instance.
    max_centroid_dist : max px between frames to link detections as one track.
    tile_hw           : spatial tile side length in pixels.  ``None`` = full
                        frame (no tiling).  Typical value: 512 or 768.
    tile_overlap      : overlap between adjacent tiles in pixels (default 64).
                        Each tile's contribution is written only in its central
                        strip; the overlap width on each edge is ``tile_overlap//2``.

    Returns
    -------
    (T, H, W) int32 array.  Pixel value = track ID (0 = background).
    """
    T, H, W = img_tyx.shape

    if tile_hw is None or (tile_hw >= H and tile_hw >= W):
        print("Running SAM2 tracking on full frame...")
        return _run_tracking_on_region(
            img_tyx, det_mask_tyx, predictor, max_centroid_dist,
            temporal_window=temporal_window,
        )

    # ------------------------------------------------------------------
    # Spatial tiling
    # ------------------------------------------------------------------
    stride = max(1, tile_hw - tile_overlap)
    half_ov = tile_overlap // 2

    # Tile origins: ensure the last tile always reaches the frame edge
    def _origins(size: int) -> list[int]:
        origins = list(range(0, size, stride))
        if origins[-1] + tile_hw < size:
            origins.append(size - tile_hw)
        return origins

    ys = _origins(H)
    xs = _origins(W)
    total_tiles = len(ys) * len(xs)
    print(
        f"Tiled SAM2 tracking: {len(ys)}×{len(xs)} = {total_tiles} tiles "
        f"(tile_hw={tile_hw}, overlap={tile_overlap}, stride={stride})"
    )

    result = np.zeros((T, H, W), dtype=np.int32)
    global_id_offset = 0

    for ti, ty in enumerate(ys):
        for tj, tx in enumerate(xs):
            y0 = ty
            y1 = min(ty + tile_hw, H)
            x0 = tx
            x1 = min(tx + tile_hw, W)

            th, tw = y1 - y0, x1 - x0
            tile_det = det_mask_tyx[:, y0:y1, x0:x1]
            n_det = int(tile_det.sum())
            print(f"  Tile ({ti},{tj}) [{y0}:{y1}, {x0}:{x1}] ({th}×{tw}): {n_det} detection pixels")
            if n_det == 0:
                continue

            tile_img = img_tyx[:, y0:y1, x0:x1]

            # Zero-pad edge tiles to tile_hw × tile_hw so SAM2 sees
            # consistent frame dimensions across all tiles.
            if th < tile_hw or tw < tile_hw:
                pad_img = np.zeros((T, tile_hw, tile_hw), dtype=tile_img.dtype)
                pad_det = np.zeros((T, tile_hw, tile_hw), dtype=tile_det.dtype)
                pad_img[:, :th, :tw] = tile_img
                pad_det[:, :th, :tw] = tile_det
                tile_img, tile_det = pad_img, pad_det

            tile_result = _run_tracking_on_region(
                tile_img, tile_det, predictor, max_centroid_dist,
                label=f"tile({ti},{tj}) ",
                temporal_window=temporal_window,
            )
            # Crop result back to the actual (unpadded) tile size
            tile_result = tile_result[:, :th, :tw]

            max_tile_id = int(tile_result.max())
            if max_tile_id == 0:
                continue

            # Shift IDs to globally unique range
            tile_result[tile_result != 0] += global_id_offset
            global_id_offset += max_tile_id

            # "Own" region: strip half_ov from edges that abut a neighbour tile
            # (but not from edges that are at the frame boundary).
            ly0 = half_ov if ty > 0 else 0
            ly1 = (y1 - y0) - (half_ov if y1 < H else 0)
            lx0 = half_ov if tx > 0 else 0
            lx1 = (x1 - x0) - (half_ov if x1 < W else 0)

            dst_y0, dst_y1 = y0 + ly0, y0 + ly1
            dst_x0, dst_x1 = x0 + lx0, x0 + lx1

            src = tile_result[:, ly0:ly1, lx0:lx1]
            dst = result[:, dst_y0:dst_y1, dst_x0:dst_x1]
            # First-write-wins: don't overwrite IDs already placed by an
            # earlier tile (can happen in the overlap zone).
            result[:, dst_y0:dst_y1, dst_x0:dst_x1] = np.where(dst != 0, dst, src)

    return result


def predict_prev_frame(
    img_next: np.ndarray,
    img_curr: np.ndarray,
    mask_next: np.ndarray,
    predictor,
) -> np.ndarray:
    """Propagate a labeled mask one step backward using a 2-frame SAM2 call.

    Builds a mini 2-frame video ``[img_curr, img_next]``, seeds all object IDs
    found in ``mask_next`` at frame index 1, and returns the predicted labeled
    mask at frame index 0 (``img_curr``).

    Parameters
    ----------
    img_next  : (H, W) or (C, H, W) raw image at time t+1.
    img_curr  : (H, W) or (C, H, W) raw image at time t.
    mask_next : (H, W) int32 labeled mask at t+1 (0 = background, 1..N = IDs).
    predictor : SAM2VideoPredictor instance.

    Returns
    -------
    (H, W) int32 predicted mask at time t.
    """
    import torch
    from contextlib import nullcontext

    H, W = img_curr.shape[-2], img_curr.shape[-1]
    track_ids = [int(tid) for tid in np.unique(mask_next) if tid != 0]
    if not track_ids:
        return np.zeros((H, W), dtype=np.int32)

    two_frames = np.stack([img_curr, img_next])  # (2, H, W) or (2, C, H, W)
    result = np.zeros((H, W), dtype=np.int32)

    device = predictor._device
    _autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else nullcontext()
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        _save_frames_as_jpeg(two_frames, tmpdir)
        with torch.inference_mode(), _autocast:
            inference_state = predictor.init_state(
                video_path=tmpdir,
                offload_video_to_cpu=(device == "cpu"),
            )
            for tid in track_ids:
                binary_mask = (mask_next == tid)
                predictor.add_new_mask(
                    inference_state, frame_idx=1, obj_id=tid, mask=binary_mask,
                )
            for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(
                inference_state, start_frame_idx=1, reverse=True,
            ):
                if frame_idx == 0:
                    for i, obj_id in enumerate(obj_ids):
                        binary = (video_res_masks[i, 0].float() > 0.0).cpu().numpy()
                        result[binary] = int(obj_id)
            predictor.reset_state(inference_state)

    return result


def track_sam2_video(
    img_tyx: np.ndarray,
    det_mask_tyx: np.ndarray,
    predictor,
    max_centroid_dist: float = 50.0,
    tile_hw: int | None = None,
    tile_overlap: int = 64,
    temporal_window: int | None = None,
) -> np.ndarray:
    """Track objects across all frames using SAM2 backward propagation.

    Convenience wrapper around :func:`run_tracking`.  Seeds every frame with
    its per-frame detection mask, links detections backward in time using
    centroid matching, and lets SAM2 propagate masks cleanly.

    Parameters
    ----------
    img_tyx          : (T, H, W) raw image frames (used as SAM2 video input).
    det_mask_tyx     : (T, H, W) uint8 binary detection mask (1 = object).
    predictor        : SAM2VideoPredictor instance from ``load_sam2_video_predictor``.
    max_centroid_dist: max px between frames to link detections as one track.
    tile_hw          : spatial tile size; ``None`` = full frame.
    tile_overlap     : overlap between adjacent tiles in pixels.
    temporal_window  : frames per SAM2 call; ``None`` = all frames at once.

    Returns
    -------
    (T, H, W) int32 array.  Pixel value = track ID (0 = background).
    """
    return run_tracking(
        img_tyx=img_tyx,
        det_mask_tyx=det_mask_tyx,
        predictor=predictor,
        max_centroid_dist=max_centroid_dist,
        tile_hw=tile_hw,
        tile_overlap=tile_overlap,
        temporal_window=temporal_window,
    )


# ---------------------------------------------------------------------------
# Predictor loader
# ---------------------------------------------------------------------------

def load_sam2_video_predictor(model_id: str = "facebook/sam2-hiera-large"):
    import torch
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"SAM2 device: {device}")
    predictor = SAM2VideoPredictor.from_pretrained(model_id, device=device)
    if device == "cpu":
        predictor = predictor.float()
    predictor._device = device
    return predictor


def load_sam2_image_predictor(model_id: str = "facebook/sam2-hiera-large"):
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"SAM2 device: {device}")
    predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)
    if device == "cpu":
        predictor = predictor.float()
    predictor._device = device
    return predictor


# ---------------------------------------------------------------------------
# Interactive single-frame point labeling
# ---------------------------------------------------------------------------

def interactive_point_label(
    frame: np.ndarray,
    auto_coords: list[tuple[float, float]],
    image_predictor,
) -> list[np.ndarray]:
    """Interactively refine SAM2 predictions for auto-detected objects on one frame.

    For each coordinate in ``auto_coords`` the function opens a matplotlib
    window showing the frame and the current SAM2 mask prediction.  The user
    can click to add extra prompts; the prediction refreshes after every click.

    Controls
    --------
    Left-click   : add positive point (include region)
    Right-click  : add negative point (exclude region)
    R            : reset to the original auto-detected point only
    Enter / Space: accept current mask and move to next object
    S            : skip this object (mask not recorded)
    Q            : quit — stop iterating, return masks accepted so far

    Parameters
    ----------
    frame : (H, W) grayscale image array.
    auto_coords : list of ``(x, y)`` pixel coordinates (col, row) for each
        auto-detected object to iterate over.
    image_predictor : ``SAM2ImagePredictor`` instance (from
        ``load_sam2_image_predictor``).

    Returns
    -------
    List of accepted ``(H, W)`` bool masks, one per accepted object.
    """
    import torch
    import matplotlib.pyplot as plt

    rgb = np.stack([_normalize_to_uint8(frame)] * 3, axis=-1)
    image_predictor.set_image(rgb)

    accepted_masks: list[np.ndarray] = []
    n_objects = len(auto_coords)

    for obj_idx, (init_x, init_y) in enumerate(auto_coords):
        state: dict = {
            "coords": [[init_x, init_y]],
            "labels": [1],   # 1 = positive, 0 = negative
            "quit": False,
            "skip": False,
        }

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.canvas.manager.set_window_title(
            f"Object {obj_idx + 1}/{n_objects} — "
            "LClick=positive  RClick=negative  R=reset  Enter=accept  S=skip  Q=quit"
        )

        current_mask: list[np.ndarray | None] = [None]

        def _refresh():
            coords = np.array(state["coords"], dtype=np.float32)
            labels = np.array(state["labels"], dtype=np.int32)
            with torch.inference_mode():
                masks, scores, _ = image_predictor.predict(
                    point_coords=coords,
                    point_labels=labels,
                    multimask_output=False,
                )
            mask = masks[0].astype(bool)  # (H, W) bool
            current_mask[0] = mask

            ax.clear()
            ax.imshow(rgb)

            # Green mask overlay
            overlay = np.zeros((*rgb.shape[:2], 4), dtype=np.float32)
            overlay[mask] = [0.0, 1.0, 0.0, 0.4]
            ax.imshow(overlay)

            # Draw prompt points
            for (cx, cy), lbl in zip(state["coords"], state["labels"]):
                color = "lime" if lbl == 1 else "red"
                marker = "P" if lbl == 1 else "X"
                ax.plot(cx, cy, marker=marker, color=color,
                        markersize=10, markeredgewidth=1.5,
                        markeredgecolor="black", linestyle="none")

            ax.set_title(
                f"Object {obj_idx + 1}/{n_objects}   score={scores[0]:.3f}\n"
                "LClick=positive  RClick=negative  R=reset  Enter=accept  S=skip  Q=quit"
            )
            ax.axis("off")
            fig.canvas.draw_idle()

        def on_click(event):
            if event.inaxes is not ax or event.xdata is None:
                return
            x, y = event.xdata, event.ydata
            if event.button == 1:       # left → positive
                state["coords"].append([x, y])
                state["labels"].append(1)
            elif event.button == 3:     # right → negative
                state["coords"].append([x, y])
                state["labels"].append(0)
            else:
                return
            _refresh()

        def on_key(event):
            if event.key in ("enter", " "):
                plt.close(fig)
            elif event.key in ("s", "S"):
                state["skip"] = True
                plt.close(fig)
            elif event.key in ("q", "Q"):
                state["skip"] = True
                state["quit"] = True
                plt.close(fig)
            elif event.key in ("r", "R"):
                state["coords"] = [[init_x, init_y]]
                state["labels"] = [1]
                _refresh()

        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)

        _refresh()
        plt.tight_layout()
        plt.show()   # blocks until window is closed

        if not state["skip"] and current_mask[0] is not None:
            accepted_masks.append(current_mask[0])

        if state["quit"]:
            break

    return accepted_masks



# ---------------------------------------------------------------------------
# Find / track object(s) from a reference frame in an image or video
# ---------------------------------------------------------------------------


def find_similar_object(
    source_image: np.ndarray,
    source_mask: np.ndarray,
    target: np.ndarray,
    predictor,
) -> np.ndarray:
    """Locate or track object(s) described by a mask in a reference frame.

    Parameters
    ----------
    source_image : (H, W) greyscale reference frame containing the object(s).
    source_mask  : (H, W) mask.  A bool array or a single-value integer array
                   describes one object.  An integer-labelled array (multiple
                   non-zero unique values) describes several objects; each is
                   tracked independently.
    target       : (H, W) single frame  **or**  (T, H, W) timeseries.
                   For a single frame pass an ``SAM2ImagePredictor`` as
                   ``predictor``.  For a timeseries pass a
                   ``SAM2VideoPredictor``; ``source_image`` is prepended as
                   frame 0 and the objects are propagated forward through all
                   T target frames.
    predictor    : ``SAM2ImagePredictor`` for a 2-D target,
                   ``SAM2VideoPredictor`` for a 3-D (T, H, W) target.

    Returns
    -------
    np.ndarray
        * 2-D target, one object   → (H, W) bool mask.
        * 2-D target, many objects → (H, W) int32 label map (0 = background).
        * 3-D target               → (T, H, W) int32 label map (0 = background).
    """
    import torch
    from PIL import Image as _PILImage
    from skimage.measure import regionprops, label as sk_label

    # Normalise mask to integer labels.
    labeled_mask = source_mask.astype(np.int32)
    object_ids = [int(v) for v in np.unique(labeled_mask) if v != 0]

    if not object_ids:
        raise ValueError("source_mask contains no labelled objects.")

    # ==================================================================
    # VIDEO PATH  (T, H, W) target → SAM2VideoPredictor
    # ==================================================================
    if target.ndim == 3:
        T, H, W = target.shape

        # Prepend source_image as frame 0; SAM2 sees it as the prompt frame.
        all_frames = np.concatenate([source_image[np.newaxis], target], axis=0)

        result = np.zeros((T, H, W), dtype=np.int32)
        device = predictor._device

        from contextlib import nullcontext
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device == "cuda"
            else nullcontext()
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            _save_frames_as_jpeg(all_frames, tmpdir)

            with torch.inference_mode(), autocast_ctx:
                inference_state = predictor.init_state(
                    video_path=tmpdir,
                    offload_video_to_cpu=(device == "cpu"),
                )

                # Register each object mask at frame 0 (source_image).
                for obj_id in object_ids:
                    binary = (labeled_mask == obj_id)
                    predictor.add_new_mask(
                        inference_state,
                        frame_idx=0,
                        obj_id=obj_id,
                        mask=binary,
                    )

                for frame_idx, obj_ids_out, masks_out in predictor.propagate_in_video(
                    inference_state
                ):
                    if frame_idx == 0:
                        continue  # skip reference frame; only return target frames
                    target_t = frame_idx - 1
                    for i, oid in enumerate(obj_ids_out):
                        binary = (masks_out[i, 0].float() > 0.0).cpu().numpy()
                        result[target_t][binary] = int(oid)

                predictor.reset_state(inference_state)

        return result

    # ==================================================================
    # SINGLE-FRAME PATH  (H, W) target → SAM2ImagePredictor
    # ==================================================================
    H, W = target.shape
    rgb_target = np.stack([_normalize_to_uint8(target)] * 3, axis=-1)
    predictor.set_image(rgb_target)

    def _predict_one(obj_mask: np.ndarray) -> np.ndarray:
        """Predict a single object mask using mask + centroid as prompts."""
        lbl = sk_label(obj_mask.astype(np.uint8))
        props = regionprops(lbl)
        if props:
            cy, cx = props[0].centroid
        else:
            ys, xs = np.where(obj_mask)
            cy, cx = float(ys.mean()), float(xs.mean())

        point_coords = np.array([[cx, cy]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)

        # SAM2 expects a (1, 256, 256) low-res logit mask.
        mask_f = obj_mask.astype(np.float32) * 20.0  # high-confidence logit
        pil_m = _PILImage.fromarray(mask_f).resize((256, 256), _PILImage.NEAREST)
        mask_input = np.array(pil_m)[np.newaxis]  # (1, 256, 256)

        with torch.inference_mode():
            masks, _, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=mask_input,
                multimask_output=False,
            )
        return masks[0].astype(bool)

    if len(object_ids) == 1:
        return _predict_one(labeled_mask == object_ids[0])

    # Multiple objects → integer label map.
    result_2d = np.zeros((H, W), dtype=np.int32)
    for obj_id in object_ids:
        predicted = _predict_one(labeled_mask == obj_id)
        result_2d[predicted] = obj_id
    return result_2d


# ---------------------------------------------------------------------------
# Macropinosome seed-based backwards tracking
# ---------------------------------------------------------------------------

def _mp_greyscale_fill_holes(image: np.ndarray) -> np.ndarray:
    """Morphological hole-fill via reconstruction (inverted dilation)."""
    from skimage.morphology import reconstruction as _sk_reconstruction
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")
    if image.size == 0 or np.all(image == image.flat[0]):
        return image.copy()
    image_max = float(image.max())
    inverted = image_max - image.astype(np.float32)
    seed = inverted.copy()
    seed[1:-1, 1:-1] = float(inverted.min())
    reconstructed = _sk_reconstruction(seed, inverted, method="dilation")
    return image_max - reconstructed


def _mp_aspect_ratio(frame: np.ndarray, cy: int, cx: int, sigma: float) -> float:
    """Intensity-weighted eigenvalue aspect ratio of a frame patch.

    Returns ``np.inf`` for degenerate patches. Circular ≈ 1.0, elongated >> 1.
    """
    r = max(3, int(sigma * 2.0))
    y0, y1 = max(0, cy - r), min(frame.shape[0], cy + r + 1)
    x0, x1 = max(0, cx - r), min(frame.shape[1], cx + r + 1)
    patch = frame[y0:y1, x0:x1].astype(np.float64)
    patch = np.clip(patch, 0, None)
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


def _mp_detect_seeds_at_frame(
    t: int,
    seeds_yx: list[tuple[int, int]],
    all_frames: np.ndarray,
    global_mask_tyx: np.ndarray,
    next_track_id: int,
    tile_size: int,
    max_hole_area: int,
    min_hole_area: int,
    max_aspect_ratio: float,
    min_fill_brightness_ratio: float,
    min_ring_contrast_ratio: float,
    ring_width: int,
    min_circularity: float,
    min_hole_depth: float,
) -> tuple[dict[int, dict], int]:
    """Flood-fill each seed point at frame t and register new track IDs.

    Sigma for the aspect-ratio filter is derived from the detected hole area
    as ``sqrt(area / pi)``.  All other filters mirror those in the DoG-based
    detector in track_sam_05.py.

    Returns
    -------
    new_tracks : {track_id: {'y': cy_global, 'x': cx_global, 't': t}}
    next_track_id : updated counter
    """
    from scipy.ndimage import label as _scipy_label, binary_dilation as _bin_dil
    from skimage.morphology import flood_fill as _flood_fill
    from skimage.measure import perimeter as _sk_perimeter

    H, W = all_frames.shape[1], all_frames.shape[2]
    new_tracks: dict[int, dict] = {}

    for blob_y, blob_x in seeds_yx:
        blob_y, blob_x = int(blob_y), int(blob_x)

        # Skip if this location is already claimed by an existing track
        if global_mask_tyx[t, blob_y, blob_x] != 0:
            continue

        # Tile bounds
        y0 = max(0, blob_y - tile_size)
        y1 = min(H, blob_y + tile_size)
        x0 = max(0, blob_x - tile_size)
        x1 = min(W, blob_x + tile_size)

        seed_y = blob_y - y0
        seed_x = blob_x - x0

        tile = all_frames[t, y0:y1, x0:x1].astype(np.float32)
        filled = _mp_greyscale_fill_holes(tile)

        if not (0 <= seed_y < filled.shape[0] and 0 <= seed_x < filled.shape[1]):
            continue

        flooded = _flood_fill(filled, seed_point=(seed_y, seed_x),
                              new_value=-1.0, tolerance=0, connectivity=1)
        raw_mask = flooded == -1.0

        if raw_mask.sum() > max_hole_area or raw_mask.sum() < min_hole_area:
            continue

        # Keep only the connected component that contains the seed
        labeled_raw, _ = _scipy_label(raw_mask)
        seed_label = labeled_raw[seed_y, seed_x]
        if seed_label == 0:
            continue
        component = (labeled_raw == seed_label)

        # Derive sigma from area for the aspect-ratio check
        area = int(component.sum())
        sigma = float(np.sqrt(area / np.pi))
        ys_c, xs_c = np.where(component)
        cy_loc = int(ys_c.mean())
        cx_loc = int(xs_c.mean())
        if _mp_aspect_ratio(tile, cy_loc, cx_loc, sigma) > max_aspect_ratio:
            continue

        # Filter 1: fill brightness ratio
        tile_mean = float(tile.mean())
        if tile_mean > 0 and filled[component].mean() < min_fill_brightness_ratio * tile_mean:
            continue

        # Filter 2: hole depth — mean(filled − original) over hole pixels
        hole_depth = float(
            (filled.astype(np.float64) - tile.astype(np.float64))[component].mean()
        )
        if hole_depth < min_hole_depth:
            continue

        # Filter 3: ring contrast — bright membrane ring around the dark interior
        ring = _bin_dil(component, iterations=ring_width) & ~component
        if ring.any() and component.any():
            ring_mean = float(tile[ring].mean())
            interior_mean = float(tile[component].mean())
            if interior_mean > 0 and ring_mean / interior_mean < min_ring_contrast_ratio:
                continue

        # Filter 4: circularity — 4π·area / perimeter²
        perim = float(_sk_perimeter(component.astype(np.uint8)))
        if perim > 0 and (4.0 * np.pi * area / perim ** 2) < min_circularity:
            continue

        cy_global = cy_loc + y0
        cx_global = cx_loc + x0

        # Write to global mask (no-overwrite: first writer wins)
        write_region = component & (global_mask_tyx[t, y0:y1, x0:x1] == 0)
        global_mask_tyx[t, y0:y1, x0:x1] = np.where(
            write_region, next_track_id, global_mask_tyx[t, y0:y1, x0:x1]
        )
        new_tracks[next_track_id] = {'y': cy_global, 'x': cx_global, 't': t}
        next_track_id += 1

    return new_tracks, next_track_id


def _mp_propagate_one_step(
    t: int,
    active_tracks: dict[int, dict],
    all_frames: np.ndarray,
    global_mask_tyx: np.ndarray,
    predictor,
    tile_size: int,
) -> None:
    """Propagate every active track from frame t to frame t-1 using SAM2.

    Frame t is the SAM prompt (frame 0 of a 2-frame mini-video); frame t-1 is
    the propagation target (frame 1).  Neighbouring tracks inside the tile are
    included as context labels so SAM can delineate touching objects.

    Mutates
    -------
    global_mask_tyx[t-1] : written with predicted track IDs (no-overwrite).
    active_tracks         : centroids updated to t-1 position; lost tracks removed.
    """
    H, W = all_frames.shape[1], all_frames.shape[2]
    lost: list[int] = []

    for track_id, state in list(active_tracks.items()):
        cy, cx = state['y'], state['x']

        y0 = max(0, cy - tile_size)
        y1 = min(H, cy + tile_size)
        x0 = max(0, cx - tile_size)
        x1 = min(W, cx + tile_size)

        tile_t    = all_frames[t,     y0:y1, x0:x1]
        tile_prev = all_frames[t - 1, y0:y1, x0:x1]

        region_ids = global_mask_tyx[t, y0:y1, x0:x1]
        source_mask = np.zeros_like(region_ids, dtype=np.int32)

        if not np.any(region_ids == track_id):
            lost.append(track_id)
            continue

        source_mask[region_ids == track_id] = 1
        next_label = 2
        for other_id in np.unique(region_ids):
            if other_id == 0 or other_id == track_id:
                continue
            source_mask[region_ids == other_id] = next_label
            next_label += 1

        # tile_prev as (1, H, W) triggers the video path in find_similar_object
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

        write_region = primary_pred & (global_mask_tyx[t - 1, y0:y1, x0:x1] == 0)
        global_mask_tyx[t - 1, y0:y1, x0:x1] = np.where(
            write_region, track_id, global_mask_tyx[t - 1, y0:y1, x0:x1]
        )

        ys_p, xs_p = np.where(primary_pred)
        active_tracks[track_id] = {
            'y': int(ys_p.mean()) + y0,
            'x': int(xs_p.mean()) + x0,
        }

    for tid in lost:
        del active_tracks[tid]


def track_macropinosome_from_seed(
    image: "np.ndarray",
    seeds: list[tuple[int, int, int]],
    predictor,
    *,
    channel: int = 0,
    z_slice: int = 0,
    crop_size: "int | None" = None,
    tile_size: int = 40,
    max_hole_area: int = 10000,
    min_hole_area: int = 8,
    max_aspect_ratio: float = 2.0,
    min_fill_brightness_ratio: float = 0.0,
    min_ring_contrast_ratio: float = 1.2,
    ring_width: int = 3,
    min_circularity: float = 0.5,
    min_hole_depth: float = 5.0,
    verbose: bool = True,
) -> np.ndarray:
    """Track macropinosomes backwards in time from seed coordinates.

    Combines morphological hole-fill detection with SAM2 tile-based
    frame-to-frame propagation.  The algorithm loops backwards from the latest
    seeded frame to frame 0:

    1. At each frame *t*, every seed coordinate assigned to that frame is
       flood-filled to recover the precise hole mask.  Each hole that passes
       all quality filters becomes a new track.
    2. All active tracks are propagated one step backwards (frame *t* → *t*-1)
       using SAM2 on small tiles centred on the last known centroid.
    3. At the next frame (*t*-1), any seed coordinates there are initialised
       and added as new tracks before propagation continues.

    Parameters
    ----------
    image : np.ndarray (T, Y, X) *or* BioImage
        Source video.  If a ``BioImage`` is provided, frames are loaded using
        the ``channel``, ``z_slice``, and ``crop_size`` arguments.
    seeds : list of (t, y, x) tuples
        One tuple per seed coordinate.  Multiple seeds at the same frame are
        evaluated independently; any that pass the hole-fill filters become
        separate tracks.  Seeds outside the valid frame range are ignored with
        a warning.
    predictor : SAM2VideoPredictor
        Loaded via :func:`load_sam2_video_predictor`.
    channel : int
        Channel index — used only when *image* is a ``BioImage``.
    z_slice : int
        Z-slice index — used only when *image* is a ``BioImage``.
    crop_size : int or None
        Spatial crop applied as ``frame[:crop_size, :crop_size]`` before
        tracking.  ``None`` keeps the full frame.  Used only for ``BioImage``.
    tile_size : int
        Half-width (px) of the tile centred on each track centroid that is
        passed to SAM2.  The full tile is ``2*tile_size × 2*tile_size`` px.
        Increase for large or fast-moving objects.
    max_hole_area : int
        Maximum flood-fill region area (px²) accepted as a valid hole.
    min_hole_area : int
        Minimum flood-fill region area (px²) accepted as a valid hole.
    max_aspect_ratio : float
        Intensity-weighted eigenvalue ratio threshold.  Objects above this
        are rejected as elongated (cell edges, filopodia, etc.).
    min_fill_brightness_ratio : float
        ``filled[hole].mean() / tile.mean()`` must exceed this.  Set to
        ``0.0`` to disable (recommended when holes sit inside bright cells).
    min_ring_contrast_ratio : float
        ``tile[ring].mean() / tile[hole].mean()`` threshold for the bright
        membrane ring surrounding the hole.
    ring_width : int
        Dilation iterations used to define the membrane ring.
    min_circularity : float
        ``4π·area / perimeter²`` threshold (1.0 = perfect circle).
    min_hole_depth : float
        ``mean(filled − original)`` over the hole region.  Ensures the hole
        is genuinely darker than its filled surroundings.  Set to ``0.0`` to
        disable.
    verbose : bool
        Print frame-by-frame progress.

    Returns
    -------
    global_mask_tyx : np.ndarray, shape (T, H, W), dtype int32
        Track-ID map.  Pixel value = track ID (1-based); 0 = background.  T
        and spatial dimensions match the input array (or loaded BioImage).
    """
    from collections import defaultdict

    # ------------------------------------------------------------------
    # 1. Load frames into a (T, H, W) float32 array
    # ------------------------------------------------------------------
    _is_bioimage = hasattr(image, "get_image_data") and hasattr(image, "dims")

    if _is_bioimage:
        n_total = image.dims.T
        frames_list = [
            image.get_image_data("YX", T=t, C=channel, Z=z_slice)
            for t in range(n_total)
        ]
        if crop_size is not None:
            frames_list = [f[:crop_size, :crop_size] for f in frames_list]
        all_frames = np.stack(frames_list).astype(np.float32)
    else:
        all_frames = np.asarray(image, dtype=np.float32)
        if all_frames.ndim != 3:
            raise ValueError(
                f"Expected a (T, Y, X) array, got shape {all_frames.shape}"
            )

    T, H, W = all_frames.shape

    # ------------------------------------------------------------------
    # 2. Group seeds by frame
    # ------------------------------------------------------------------
    seeds_per_frame: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for st, sy, sx in seeds:
        st, sy, sx = int(st), int(sy), int(sx)
        if 0 <= st < T:
            seeds_per_frame[st].append((sy, sx))
        elif verbose:
            print(f"  Warning: seed (t={st}, y={sy}, x={sx}) outside T=[0,{T-1}], skipped.")

    if not seeds_per_frame:
        raise ValueError("No valid seeds found within the image time range.")

    t_start = max(seeds_per_frame.keys())

    # ------------------------------------------------------------------
    # 3. Backwards tracking loop
    # ------------------------------------------------------------------
    global_mask_tyx = np.zeros((T, H, W), dtype=np.int32)
    active_tracks: dict[int, dict] = {}
    next_track_id = 1

    try:
        from tqdm import tqdm as _tqdm
        frame_iter = _tqdm(
            range(t_start, -1, -1), desc="backwards tracking", unit="frame"
        ) if verbose else range(t_start, -1, -1)
    except ImportError:
        frame_iter = range(t_start, -1, -1)

    for t in frame_iter:
        # Step A: flood-fill seeds at this frame → new tracks
        new_tracks, next_track_id = _mp_detect_seeds_at_frame(
            t=t,
            seeds_yx=seeds_per_frame.get(t, []),
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

        if verbose:
            print(f"  t={t}: +{len(new_tracks)} new tracks, {len(active_tracks)} active")

        # Step B: propagate all active tracks to frame t-1
        if t == 0 or not active_tracks:
            break

        _mp_propagate_one_step(
            t=t,
            active_tracks=active_tracks,
            all_frames=all_frames,
            global_mask_tyx=global_mask_tyx,
            predictor=predictor,
            tile_size=tile_size,
        )

    if verbose:
        print(f"Tracking complete. Total unique tracks: {next_track_id - 1}")

    return global_mask_tyx

