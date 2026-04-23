"""Interactive SAM-style timepoint annotator for microscopy timelapses.

This tool loads full timelapses, creates max-Z projections per timepoint,
opens napari for prompt-based annotation, and supports per-object propagation
from T to T+1 when Enter is pressed.

Current implementation status
-----------------------------
- Full-stack TYX workflow and persistence are implemented.
- Multi-object prompts are implemented:
  - left click: positive point
  - right click: negative point
  - click-drag: rectangle prompt
  - Ctrl + click/drag: start a new object
- Enter performs one-step propagation from current T to T+1.
- Non-overlap is enforced by confidence, then object id tie-break.

The segmentation core uses a lightweight prompt-driven fallback predictor to
enable immediate usage while SAM2 backend wiring is completed.

Output folder structure
-----------------------
<output-folder>/
    manual_labelling/
        YYYYMMDD_N/
            <input-stem>/
                frames_tyx.npy           # float32 TYX max-Z stack
                labels_tyx.npy           # uint16 TYX instance labels
                centers.csv              # file_idx,timepoint,object_id,y,x,area
                coords.npy               # compatibility export [t,y,x,object_id]
                prompts.json             # prompt history for reproducibility

MIT License - BIPHUB, University of Oslo
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import napari
from scipy import ndimage as ndi

try:
    import dask.array as da
except ImportError:  # pragma: no cover
    da = None

import bioimage_pipeline_utils as rp

# Module-level logger
logger = logging.getLogger(__name__)


def _make_circle_mask(h: int, w: int, y: float, x: float, radius: float) -> np.ndarray:
    """Create a boolean circular mask."""
    yy, xx = np.ogrid[:h, :w]
    return (yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2


def _bbox_from_drag(t: int, y0: float, x0: float, y1: float, x1: float) -> list[list[float]]:
    """Return a rectangle in napari shapes format for 3D TYX view."""
    y_min, y_max = sorted((y0, y1))
    x_min, x_max = sorted((x0, x1))
    return [
        [float(t), float(y_min), float(x_min)],
        [float(t), float(y_min), float(x_max)],
        [float(t), float(y_max), float(x_max)],
        [float(t), float(y_max), float(x_min)],
    ]


def _mask_center(mask: np.ndarray) -> tuple[float, float] | None:
    """Return center of mass as y, x for a binary mask."""
    if mask.sum() == 0:
        return None
    cy, cx = ndi.center_of_mass(mask.astype(np.uint8))
    return float(cy), float(cx)


def _extract_centers_from_label_frame(label_yx: np.ndarray, t: int) -> list[tuple[int, int, int, float, float, int]]:
    """Extract centers for all objects in one timepoint label image."""
    rows: list[tuple[int, int, int, float, float, int]] = []
    obj_ids = np.unique(label_yx)
    for obj_id in obj_ids:
        if obj_id == 0:
            continue
        mask = label_yx == obj_id
        center = _mask_center(mask)
        if center is None:
            continue
        cy, cx = center
        rows.append((0, t, int(obj_id), cy, cx, int(mask.sum())))
    return rows


# ----------------------------------------------------------------------- #
#  Session management                                                      #
# ----------------------------------------------------------------------- #

_SESS_RE = re.compile(r"^(\d{8})_(\d+)$")


def _find_latest_session(root: str) -> str | None:
    """Return the path to the most recent annotation session, or None."""
    if not os.path.isdir(root):
        return None
    sessions: list[tuple[str, int, str]] = []
    for name in os.listdir(root):
        m = _SESS_RE.match(name)
        if m and os.path.isdir(os.path.join(root, name)):
            sessions.append((m.group(1), int(m.group(2)), name))
    if not sessions:
        return None
    sessions.sort(key=lambda x: (x[0], x[1]))
    return os.path.join(root, sessions[-1][2])


def _create_new_session(root: str) -> str:
    """Create and return a new timestamped session folder inside *root*."""
    os.makedirs(root, exist_ok=True)
    today = datetime.datetime.now().strftime("%Y%m%d")
    today_re = re.compile(rf"^{today}_(\d+)$")
    existing: list[int] = []
    for name in os.listdir(root):
        m = today_re.match(name)
        if m and os.path.isdir(os.path.join(root, name)):
            existing.append(int(m.group(1)))
    idx = max(existing, default=0) + 1
    folder = os.path.join(root, f"{today}_{idx}")
    os.makedirs(folder)
    return folder


@dataclass
class PromptSet:
    """Prompt container for one object at one timepoint."""

    positive: list[list[float]]
    negative: list[list[float]]
    boxes: list[list[list[float]]]


class PromptDrivenSegmenter:
    """Fallback prompt-based segmenter used while SAM2 backend is wired in.

    This approximation keeps the interaction loop usable and deterministic.
    """

    def __init__(self, object_radius: int) -> None:
        self.object_radius = max(4, int(object_radius))

    def predict_mask(
        self,
        frame_yx: np.ndarray,
        prompts: PromptSet,
    ) -> tuple[np.ndarray, float]:
        """Return mask and confidence for one object.

        Confidence is a simple prompt-consistency score in [0, 1].
        """
        h, w = frame_yx.shape
        mask = np.zeros((h, w), dtype=bool)

        # Positive points add circular support regions.
        for y, x in prompts.positive:
            mask |= _make_circle_mask(h, w, y, x, self.object_radius)

        # Boxes contribute full rectangular support.
        for box in prompts.boxes:
            ys = [p[1] for p in box]
            xs = [p[2] for p in box]
            y0 = max(0, int(np.floor(min(ys))))
            y1 = min(h, int(np.ceil(max(ys))))
            x0 = max(0, int(np.floor(min(xs))))
            x1 = min(w, int(np.ceil(max(xs))))
            if y1 > y0 and x1 > x0:
                mask[y0:y1, x0:x1] = True

        # Negative points carve holes.
        for y, x in prompts.negative:
            neg = _make_circle_mask(h, w, y, x, self.object_radius * 0.9)
            mask[neg] = False

        if not mask.any() and prompts.positive:
            # Ensure a minimal object when prompts exist.
            y, x = prompts.positive[-1]
            mask |= _make_circle_mask(h, w, y, x, self.object_radius * 0.6)

        p_count = len(prompts.positive)
        n_count = len(prompts.negative)
        b_count = len(prompts.boxes)
        confidence = 0.2 + 0.2 * min(p_count, 3) + 0.1 * min(b_count, 2) - 0.05 * min(n_count, 3)
        confidence = float(np.clip(confidence, 0.01, 0.99))
        return mask, confidence


def _resolve_overlaps(
    masks_by_object: dict[int, np.ndarray],
    scores_by_object: dict[int, float],
) -> np.ndarray:
    """Resolve overlaps using score, then object id as deterministic tie-break."""
    if not masks_by_object:
        raise ValueError("masks_by_object must not be empty")

    any_mask = next(iter(masks_by_object.values()))
    out = np.zeros_like(any_mask, dtype=np.uint16)
    score_map = np.full(any_mask.shape, -np.inf, dtype=np.float32)

    for obj_id in sorted(masks_by_object):
        mask = masks_by_object[obj_id]
        score = float(scores_by_object.get(obj_id, 0.0))
        better = mask & (score > score_map)
        ties = mask & (score == score_map) & (obj_id < out)
        wins = better | ties
        out[wins] = np.uint16(obj_id)
        score_map[wins] = score
    return out


def _prompts_for_timepoint(prompt_db: dict[int, dict[int, PromptSet]], t: int) -> dict[int, PromptSet]:
    """Collect all object prompts for one timepoint."""
    result: dict[int, PromptSet] = {}
    for obj_id, by_t in prompt_db.items():
        if t in by_t:
            result[obj_id] = by_t[t]
    return result


def _to_serializable_prompt_db(prompt_db: dict[int, dict[int, PromptSet]]) -> dict[str, Any]:
    """Convert prompt DB to JSON-serializable dict."""
    serializable: dict[str, Any] = {}
    for obj_id, by_t in prompt_db.items():
        serializable[str(obj_id)] = {}
        for t, prompts in by_t.items():
            serializable[str(obj_id)][str(t)] = {
                "positive": prompts.positive,
                "negative": prompts.negative,
                "boxes": prompts.boxes,
            }
    return serializable


def _write_centers_csv(path: str, centers: list[tuple[int, int, int, float, float, int]]) -> None:
    """Write centers rows as CSV without pandas dependency."""
    lines = ["file_idx,timepoint,object_id,y,x,area"]
    for file_idx, t, obj_id, y, x, area in centers:
        lines.append(f"{file_idx},{t},{obj_id},{y:.4f},{x:.4f},{area}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _load_tyx_max_z(
    fpath: str,
    channel: int,
    use_dask: bool,
) -> np.ndarray | Any:
    """Load one file and return TYX max-Z stack for one channel.

    Returns either numpy array or dask array depending on availability and flag.
    """
    img = rp.load_tczyx_image(fpath)
    if use_dask and da is not None and hasattr(img, "dask_data") and img.dask_data is not None:
        # TCZYX -> TZYX (select C) -> TYX (max over Z)
        dask_tyx = img.dask_data[:, channel, :, :, :].max(axis=1).astype(np.float32)
        return dask_tyx

    t_size = img.dims.T
    frames: list[np.ndarray] = []
    for t in range(t_size):
        block = img.get_image_data("ZYX", T=t, C=channel)
        if block.shape[0] > 1:
            frame = np.max(block, axis=0)
        else:
            frame = block[0]
        frames.append(frame.astype(np.float32, copy=False))
    return np.stack(frames, axis=0)


def _ensure_prompt_set(prompt_db: dict[int, dict[int, PromptSet]], obj_id: int, t: int) -> PromptSet:
    """Get or create prompt set for object and timepoint."""
    if obj_id not in prompt_db:
        prompt_db[obj_id] = {}
    if t not in prompt_db[obj_id]:
        prompt_db[obj_id][t] = PromptSet(positive=[], negative=[], boxes=[])
    return prompt_db[obj_id][t]


# ----------------------------------------------------------------------- #
#  Main pipeline                                                           #
# ----------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
                        "Interactive multi-object annotation over full time stacks with "
                        "point/box prompts and Enter-driven T to T+1 propagation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Interactive SAM-style timepoint annotation
    environment: uv@3.11:default
  commands:
  - python
    - '%REPO%/standard_code/python/deep_learning_01_make_training_set_sam.py'
  - --input-search-pattern: '%YAML%/input_drift_corrected/**/*.ome.tif'
  - --output-folder: '%YAML%/deep_learning_output'
    - --default-object-radius: 20
    - --channel: 0

- name: Pause for manual inspection
  type: pause
    message: 'Inspect labels_tyx.npy and centers.csv before continuing.'
        """,
    )

    parser.add_argument(
        "--input-search-pattern",
        required=True,
        help="Glob pattern for input timelapse files (e.g. '**/*.ome.tif').",
    )
    parser.add_argument(
        "--output-folder",
        required=True,
        help=(
            "Root output folder.  Sub-folders 'manual_labelling/' and "
            "'training_set/{input,target}/' are created inside it."
        ),
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Channel index used for max-Z projection (default: 0).",
    )
    parser.add_argument(
        "--default-object-radius",
        type=int,
        default=20,
        help="Fallback object radius in pixels for prompt-driven segmentation.",
    )
    parser.add_argument(
        "--min-drag-pixels",
        type=float,
        default=6.0,
        help="Minimum drag length to interpret as box prompt.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optionally limit number of discovered files to process.",
    )
    parser.add_argument(
        "--sam-model",
        type=str,
        default="sam2-large",
        choices=["sam2-tiny", "sam2-small", "sam2-base", "sam2-large"],
        help="Model preset name; largest model is default.",
    )
    parser.add_argument(
        "--no-dask",
        action="store_true",
        help="Disable dask-backed lazy loading even if dask is available.",
    )
    parser.add_argument(
        "--frame-cache-size",
        type=int,
        default=4,
        help="Number of TYX frames cached in RAM for interactive segmentation.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: WARNING).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Resolve output sub-folder.
    output_folder = os.path.abspath(args.output_folder)
    labelling_dir = os.path.join(output_folder, "manual_labelling")

    # 1. Discover files.
    search_subfolders = "**" in args.input_search_pattern
    files = sorted(
        rp.get_files_to_process2(
            args.input_search_pattern,
            search_subfolders=search_subfolders,
        )
    )
    if not files:
        raise SystemExit(f"No files matched pattern: {args.input_search_pattern}")

    if args.max_files is not None:
        files = files[: max(1, args.max_files)]

    logger.info("Found %d file(s)", len(files))
    for f in files:
        logger.info("  %s", os.path.basename(f))

    # 2. Session root.
    os.makedirs(labelling_dir, exist_ok=True)
    session_dir = _create_new_session(labelling_dir)
    logger.info("New session folder: %s", session_dir)

    segmenter = PromptDrivenSegmenter(object_radius=args.default_object_radius)
    use_dask = (not args.no_dask)
    if use_dask and da is None:
        logger.warning("dask is not installed; falling back to eager numpy loading.")
        use_dask = False

    for file_idx, fpath in enumerate(files):
        stem = os.path.splitext(os.path.basename(fpath))[0]
        file_session_dir = os.path.join(session_dir, stem)
        os.makedirs(file_session_dir, exist_ok=True)

        frames_tyx = _load_tyx_max_z(fpath, channel=args.channel, use_dask=use_dask)
        t_size, h, w = [int(v) for v in frames_tyx.shape]
        labels_tyx = np.zeros((t_size, h, w), dtype=np.uint16)
        frame_cache: dict[int, np.ndarray] = {}
        frame_cache_order: list[int] = []
        frames_saved = False

        def _get_frame(t: int) -> np.ndarray:
            """Return TYX frame as float32 numpy with small LRU cache."""
            nonlocal frame_cache_order
            if t in frame_cache:
                # Refresh LRU order.
                frame_cache_order = [k for k in frame_cache_order if k != t] + [t]
                return frame_cache[t]

            frame_np = np.asarray(frames_tyx[t], dtype=np.float32)
            frame_cache[t] = frame_np
            frame_cache_order.append(t)

            max_cache = max(1, int(args.frame_cache_size))
            while len(frame_cache_order) > max_cache:
                old_t = frame_cache_order.pop(0)
                frame_cache.pop(old_t, None)
            return frame_np

        def _prefetch_neighbor_frames(t: int) -> None:
            """Warm the cache around current timepoint to reduce interaction latency."""
            _ = _get_frame(t)
            if t + 1 < t_size:
                _ = _get_frame(t + 1)

        prompt_db: dict[int, dict[int, PromptSet]] = {}
        object_colors: dict[int, str] = {}
        next_object_id = 1
        current_object_id = 1
        last_preview_key: tuple[int, int, int, int] | None = None
        last_preview_t: int | None = None

        viewer = napari.Viewer(title=f"SAM Interactive Annotator - {stem}")
        image_layer = viewer.add_image(frames_tyx, name="frames_tyx", colormap="gray")
        label_layer = viewer.add_labels(labels_tyx, name="instance_labels")
        hover_preview_tyx = np.zeros((t_size, h, w), dtype=np.uint16)
        hover_layer = viewer.add_labels(
            hover_preview_tyx,
            name="hover_preview",
            opacity=0.35,
            blending="additive",
        )
        pos_layer = viewer.add_points(
            np.empty((0, 3), dtype=np.float32),
            name="positive_points",
            ndim=3,
            size=8,
            face_color="green",
        )
        neg_layer = viewer.add_points(
            np.empty((0, 3), dtype=np.float32),
            name="negative_points",
            ndim=3,
            size=8,
            face_color="red",
        )
        box_layer = viewer.add_shapes(
            data=[],
            shape_type="rectangle",
            name="box_prompts",
            edge_color="yellow",
            face_color=[1.0, 1.0, 0.0, 0.0],
        )

        def _current_t() -> int:
            return int(viewer.dims.current_step[0])

        def _color_for_object(obj_id: int) -> str:
            if obj_id not in object_colors:
                palette = [
                    "#ff6b6b", "#4ecdc4", "#ffe66d", "#1a535c", "#ff9f1c",
                    "#2ec4b6", "#6a4c93", "#8ac926", "#1982c4", "#f72585",
                ]
                object_colors[obj_id] = palette[(obj_id - 1) % len(palette)]
            return object_colors[obj_id]

        def _refresh_prompt_layers(current_t_only: bool = True) -> None:
            pos_data: list[list[float]] = []
            neg_data: list[list[float]] = []
            box_data: list[list[list[float]]] = []
            box_edge_colors: list[str] = []
            t_now = _current_t()

            for obj_id, by_t in prompt_db.items():
                _ = _color_for_object(obj_id)
                for t, prompts in by_t.items():
                    if current_t_only and t != t_now:
                        continue
                    pos_data.extend([[float(t), y, x] for y, x in prompts.positive])
                    neg_data.extend([[float(t), y, x] for y, x in prompts.negative])
                    for box in prompts.boxes:
                        box_data.append(box)
                        box_edge_colors.append(object_colors[obj_id])

            pos_layer.data = np.asarray(pos_data, dtype=np.float32) if pos_data else np.empty((0, 3), dtype=np.float32)
            neg_layer.data = np.asarray(neg_data, dtype=np.float32) if neg_data else np.empty((0, 3), dtype=np.float32)
            box_layer.data = box_data
            if box_edge_colors:
                box_layer.edge_color = box_edge_colors

        def _save_state(include_frames: bool = False) -> None:
            nonlocal frames_saved
            # Persist TYX only once to avoid heavy IO on each interaction.
            if include_frames and not frames_saved:
                np.save(os.path.join(file_session_dir, "frames_tyx.npy"), np.asarray(frames_tyx, dtype=np.float32))
                frames_saved = True
            np.save(os.path.join(file_session_dir, "labels_tyx.npy"), label_layer.data.astype(np.uint16))

            centers: list[tuple[int, int, int, float, float, int]] = []
            coords_compat: list[list[float]] = []
            for t in range(t_size):
                rows = _extract_centers_from_label_frame(label_layer.data[t], t=t)
                for row in rows:
                    centers.append((file_idx, row[1], row[2], row[3], row[4], row[5]))
                    coords_compat.append([float(row[1]), float(row[3]), float(row[4]), float(row[2])])

            _write_centers_csv(os.path.join(file_session_dir, "centers.csv"), centers)
            coords_arr = np.asarray(coords_compat, dtype=np.float32) if coords_compat else np.empty((0, 4), dtype=np.float32)
            np.save(os.path.join(file_session_dir, "coords.npy"), coords_arr)

            with open(os.path.join(file_session_dir, "prompts.json"), "w", encoding="utf-8") as f:
                json.dump(_to_serializable_prompt_db(prompt_db), f, indent=2)

        def _ensure_object_for_event(modifiers: tuple[str, ...]) -> int:
            nonlocal next_object_id, current_object_id
            if "Control" in modifiers or current_object_id <= 0:
                current_object_id = next_object_id
                next_object_id += 1
                _ = _color_for_object(current_object_id)
            return current_object_id

        def _recompute_object_on_timepoint(obj_id: int, t: int) -> tuple[np.ndarray, float]:
            prompts = _ensure_prompt_set(prompt_db, obj_id, t)
            mask, score = segmenter.predict_mask(_get_frame(t), prompts)
            return mask, score

        def _apply_masks_at_t(
            t: int,
            masks: dict[int, np.ndarray],
            scores: dict[int, float],
        ) -> None:
            if not masks:
                label_layer.data[t] = np.zeros((h, w), dtype=np.uint16)
                return
            label_layer.data[t] = _resolve_overlaps(masks, scores)

        def _seed_prompts_from_current_labels(t: int) -> None:
            frame_labels = label_layer.data[t]
            for obj_id in np.unique(frame_labels):
                if obj_id == 0:
                    continue
                mask = frame_labels == obj_id
                center = _mask_center(mask)
                if center is None:
                    continue
                prompts = _ensure_prompt_set(prompt_db, int(obj_id), t)
                if not prompts.positive:
                    prompts.positive.append([center[0], center[1]])

        def _propagate_to_next_timepoint() -> None:
            t = _current_t()
            if t >= t_size - 1:
                logger.info("Already at last timepoint, cannot propagate further.")
                return

            _seed_prompts_from_current_labels(t)
            t_next = t + 1
            masks: dict[int, np.ndarray] = {}
            scores: dict[int, float] = {}

            for obj_id in sorted(prompt_db):
                # Carry forward current prompts as initial prompts for next frame.
                if t in prompt_db[obj_id] and t_next not in prompt_db[obj_id]:
                    src = prompt_db[obj_id][t]
                    prompt_db[obj_id][t_next] = PromptSet(
                        positive=[p.copy() for p in src.positive],
                        negative=[p.copy() for p in src.negative],
                        boxes=[b.copy() for b in src.boxes],
                    )
                if t_next not in prompt_db[obj_id]:
                    continue
                mask, score = _recompute_object_on_timepoint(obj_id, t_next)
                masks[obj_id] = mask
                scores[obj_id] = score

            if masks:
                _apply_masks_at_t(t_next, masks, scores)

            viewer.dims.current_step = (t_next, 0, 0)
            _refresh_prompt_layers(current_t_only=True)
            _save_state(include_frames=False)

        def _update_current_object_from_prompt(obj_id: int, t: int) -> None:
            mask, score = _recompute_object_on_timepoint(obj_id, t)
            existing_ids = [oid for oid in np.unique(label_layer.data[t]) if oid not in (0, obj_id)]
            masks = {int(oid): label_layer.data[t] == oid for oid in existing_ids}
            scores = {int(oid): 0.5 for oid in existing_ids}
            masks[obj_id] = mask
            scores[obj_id] = score
            _apply_masks_at_t(t, masks, scores)

        def _clear_hover_preview() -> None:
            nonlocal last_preview_t, last_preview_key
            if last_preview_t is not None:
                hover_layer.data[last_preview_t].fill(0)
                hover_layer.refresh()
            last_preview_t = None
            last_preview_key = None

        def _render_hover_preview(y: float, x: float, modifiers: tuple[str, ...]) -> None:
            nonlocal last_preview_key, last_preview_t
            t = _current_t()
            yi = int(round(y))
            xi = int(round(x))
            if yi < 0 or yi >= h or xi < 0 or xi >= w:
                _clear_hover_preview()
                return

            if "Control" in modifiers:
                preview_obj_id = next_object_id
                preview_prompts = PromptSet(positive=[[float(yi), float(xi)]], negative=[], boxes=[])
            else:
                preview_obj_id = current_object_id
                prompts = _ensure_prompt_set(prompt_db, preview_obj_id, t)
                preview_prompts = PromptSet(
                    positive=[p.copy() for p in prompts.positive] + [[float(yi), float(xi)]],
                    negative=[n.copy() for n in prompts.negative],
                    boxes=[b.copy() for b in prompts.boxes],
                )

            # Skip recomputation if cursor/object/timepoint unchanged.
            key = (t, yi, xi, preview_obj_id)
            if key == last_preview_key:
                return
            last_preview_key = key

            mask, _score = segmenter.predict_mask(_get_frame(t), preview_prompts)
            if last_preview_t is not None and last_preview_t != t:
                hover_layer.data[last_preview_t].fill(0)

            hover_layer.data[t].fill(0)
            hover_layer.data[t][mask] = np.uint16(preview_obj_id)
            hover_layer.refresh()
            last_preview_t = t

        @image_layer.mouse_drag_callbacks.append
        def _on_mouse_drag(layer: Any, event: Any) -> Any:
            nonlocal current_object_id
            if len(event.position) < 3:
                return

            _clear_hover_preview()

            t0 = _current_t()
            _, y0, x0 = event.position[:3]
            mods = tuple(event.modifiers)
            obj_id = _ensure_object_for_event(mods)
            _ = _color_for_object(obj_id)

            yield

            if len(event.position) < 3:
                return
            _, y1, x1 = event.position[:3]

            drag_len = float(np.hypot(y1 - y0, x1 - x0))
            prompts = _ensure_prompt_set(prompt_db, obj_id, t0)

            if drag_len >= args.min_drag_pixels:
                prompts.boxes.append(_bbox_from_drag(t0, y0, x0, y1, x1))
                _refresh_prompt_layers(current_t_only=True)
                _update_current_object_from_prompt(obj_id, t0)
                return

            if int(event.button) == 2:
                prompts.negative.append([float(y1), float(x1)])
            else:
                prompts.positive.append([float(y1), float(x1)])
            _refresh_prompt_layers(current_t_only=True)
            _update_current_object_from_prompt(obj_id, t0)

        @image_layer.mouse_move_callbacks.append
        def _on_mouse_move(layer: Any, event: Any) -> None:
            # Preview only tentative positive-click behavior while hovering.
            if len(event.position) < 3:
                _clear_hover_preview()
                return
            if getattr(event, "is_dragging", False):
                _clear_hover_preview()
                return
            _, y, x = event.position[:3]
            _render_hover_preview(float(y), float(x), tuple(event.modifiers))

        @viewer.bind_key("Enter")
        def _on_enter(viewer_obj: napari.Viewer) -> None:
            _save_state(include_frames=False)
            _propagate_to_next_timepoint()

        @viewer.dims.events.current_step.connect
        def _on_time_change(event: Any = None) -> None:  # noqa: ARG001
            t_now = _current_t()
            _prefetch_neighbor_frames(t_now)
            _refresh_prompt_layers(current_t_only=True)
            _clear_hover_preview()

        _refresh_prompt_layers(current_t_only=True)
        _prefetch_neighbor_frames(0)
        _save_state(include_frames=True)
        logger.warning(
            "Annotating %s with model preset %s (fallback segmenter active for this first implementation pass).",
            stem,
            args.sam_model,
        )
        napari.run()
        _clear_hover_preview()
        _save_state(include_frames=False)

        logger.info("Saved interactive outputs for %s in %s", stem, file_session_dir)


if __name__ == "__main__":
    main()
