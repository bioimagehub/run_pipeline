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
from skimage.measure import label, regionprops


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
        rgb = np.stack([_normalize_to_uint8(frame)] * 3, axis=-1)
        Image.fromarray(rgb).save(os.path.join(folder, f"{i:05d}.png")) # lossless


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
        labeled = label(hole, connectivity=1)
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

def run_tracking(
    img_tyx: np.ndarray,
    det_mask_tyx: np.ndarray,
    predictor,
    max_centroid_dist: float = 50.0,
) -> np.ndarray:
    """Full backward-pass tracking pipeline fusing detection masks with SAM2.

    Returns
    -------
    (T, H, W) int32 array. Each pixel's value is the track ID (0 = background).
    """
    import torch

    T, H, W = img_tyx.shape
    device = predictor._device

    print("Assigning track IDs (backwards)...")
    frame_assignments, n_tracks = assign_track_ids(det_mask_tyx, max_centroid_dist)
    print(f"  {n_tracks} unique tracks found across {T} frames.")

    with tempfile.TemporaryDirectory() as tmpdir:
        print("Saving frames as JPEG...")
        _save_frames_as_jpeg(img_tyx, tmpdir)

        # On CUDA, SAM2 expects everything in BFloat16. Wrap all inference
        # in autocast so activations and weights stay consistently BFloat16.
        # On CPU, use nullcontext (float32 throughout).
        from contextlib import nullcontext
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

            print("Registering detection prompts...")
            total_prompts = 0
            from tqdm import tqdm
            for t, assignments in tqdm(enumerate(frame_assignments), total=T, desc="registering prompts", unit="frame"):
                if not assignments:
                    continue
                hole = (det_mask_tyx[t] == 1).astype(np.uint8)
                labeled = label(hole, connectivity=1)
                for label, track_id in assignments.items():
                    binary_mask = (labeled == label)
                    predictor.add_new_mask(
                        inference_state,
                        frame_idx=t,
                        obj_id=track_id,
                        mask=binary_mask,
                    )
                    total_prompts += 1
            print(f"  {total_prompts} prompts registered.")

            print("Propagating backwards with SAM2...")
            result = np.zeros((T, H, W), dtype=np.int32)
            with tqdm(total=T, desc="propagate in video", unit="frame") as pbar:
                for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(
                    inference_state, reverse=True     # REVERSED: strong -> weak
                ):
                    for i, obj_id in enumerate(obj_ids):
                        binary = (video_res_masks[i, 0].float() > 0.0).cpu().numpy()
                        result[frame_idx][binary] = int(obj_id)
                    pbar.update(1)

            predictor.reset_state(inference_state)

    return result


# ---------------------------------------------------------------------------
# Predictor loader
# ---------------------------------------------------------------------------

def load_sam2_predictor(model_id: str = "facebook/sam2-hiera-large"):
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

