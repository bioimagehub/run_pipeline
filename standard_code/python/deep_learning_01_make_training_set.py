"""
Interactive training-set generator for deep-learning centroid detection.

Loads microscopy timelapses, extracts patches, opens napari for manual point
annotation, then generates Gaussian-blob label images and saves balanced
train/target sub-patch pairs.

Output folder structure
-----------------------
<output-folder>/
    manual_labelling/           # session folders with saved annotations
        YYYYMMDD_N/
            coords.npy
            frames.npy
            labels.npy
    training_set/
        input/                  # sub-patch images  (float32 .tif)
        target/                 # Gaussian blob maps (float32 .tif)

MIT License - BIPHUB, University of Oslo
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import napari
import tifffile

import bioimage_pipeline_utils as rp

# Module-level logger
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------- #
#  Utility helpers                                                         #
# ----------------------------------------------------------------------- #

def _get_gaussian(radius: int) -> np.ndarray:
    """Create a normalised 2-D Gaussian kernel of the given pixel radius."""
    x = np.linspace(-radius, radius, radius * 2)
    y = np.linspace(-radius, radius, radius * 2)
    xx, yy = np.meshgrid(x, y)
    d = np.sqrt(xx ** 2 + yy ** 2)
    sigma = radius / 2.0
    gauss = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
    gauss = (gauss - gauss.min()) / (gauss.max() - gauss.min())
    return gauss


def centroids2images(
    point_list: list[list[float]],
    im_num_row: int,
    im_num_col: int,
    g_radius: int = 20,
) -> np.ndarray:
    """Stamp Gaussian blobs at each centroid onto a blank image.

    Reproduced from centroid-unet / DataUtils.py.
    """
    circle_mat = _get_gaussian(g_radius)
    temp = np.zeros(
        (im_num_row + g_radius * 2, im_num_col + g_radius * 2),
        dtype=np.float32,
    )
    for pnt in point_list:
        r, c = int(pnt[0]), int(pnt[1])
        r0 = g_radius + r - g_radius
        r1 = g_radius + r + g_radius
        c0 = g_radius + c - g_radius
        c1 = g_radius + c + g_radius
        temp[r0:r1, c0:c1] = np.maximum(temp[r0:r1, c0:c1], circle_mat)
    return temp[g_radius:-g_radius, g_radius:-g_radius]


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


# ----------------------------------------------------------------------- #
#  Main pipeline                                                           #
# ----------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Annotate microscopy patches with napari and generate a "
            "training set of sub-patch input/target pairs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Annotate patches and generate training set
  environment: uv@3.11:deep-learning
  commands:
  - python
  - '%REPO%/standard_code/python/deep_learning_01_make_training_set.py'
  - --input-search-pattern: '%YAML%/input_drift_corrected/**/*.ome.tif'
  - --output-folder: '%YAML%/deep_learning_output'

- name: Generate training set (custom patch sizes)
  environment: uv@3.11:deep-learning
  commands:
  - python
  - '%REPO%/standard_code/python/deep_learning_01_make_training_set.py'
  - --input-search-pattern: '%YAML%/input_drift_corrected/**/*.ome.tif'
  - --output-folder: '%YAML%/deep_learning_output'
  - --patch-size: 256
  - --sub-patch-size: 128
  - --gaussian-radius: 15
  - --zero-keep-ratio: 0.2

- name: Pause for manual inspection
  type: pause
  message: 'Inspect training set outputs before continuing.'
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
        "--patch-size",
        type=int,
        default=512,
        help="Size of square patches extracted from each frame.",
    )
    parser.add_argument(
        "--sub-patch-size",
        type=int,
        default=256,
        help="Size of square sub-patches split from each patch.",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help=(
            "Limit the number of patches sent to napari / label generation. "
            "Default: use all patches."
        ),
    )
    parser.add_argument(
        "--gaussian-radius",
        type=int,
        default=20,
        help="Radius for Gaussian blobs placed at each clicked centroid.",
    )
    parser.add_argument(
        "--zero-keep-ratio",
        type=float,
        default=0.10,
        help="Fraction of non-zero patch count to keep as empty patches.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible zero-patch downsampling.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # ------------------------------------------------------------------- #
    # Resolve output sub-folders                                           #
    # ------------------------------------------------------------------- #
    output_folder = os.path.abspath(args.output_folder)
    labelling_dir = os.path.join(output_folder, "manual_labelling")
    train_input_dir = os.path.join(output_folder, "training_set", "input")
    train_target_dir = os.path.join(output_folder, "training_set", "target")

    ps = args.patch_size
    sps = args.sub_patch_size

    # ------------------------------------------------------------------- #
    # 1. Discover files                                                    #
    # ------------------------------------------------------------------- #
    search_subfolders = "**" in args.input_search_pattern
    files = sorted(
        rp.get_files_to_process2(
            args.input_search_pattern,
            search_subfolders=search_subfolders,
        )
    )
    if not files:
        raise SystemExit(
            f"No files matched pattern: {args.input_search_pattern}"
        )

    logger.info("Found %d file(s)", len(files))
    for f in files:
        logger.info("  %s", os.path.basename(f))

    # ------------------------------------------------------------------- #
    # 2. Load mid-timepoint from each TYX timelapse                        #
    # ------------------------------------------------------------------- #
    frames: list[np.ndarray] = []
    for fpath in files:
        img = rp.load_tczyx_image(fpath)
        # Pick the middle timepoint, first channel, max-project Z
        t_mid = img.dims.T // 2
        # Get CZYX at mid-T, then take channel 0 and max-project Z
        block = img.get_image_data("ZYX", T=t_mid, C=0)
        if block.shape[0] > 1:
            frame = np.max(block, axis=0)  # max-Z-project -> YX
        else:
            frame = block[0]  # single Z slice -> YX
        frames.append(frame)

    # Pad all to the largest common shape
    max_h = max(f.shape[0] for f in frames)
    max_w = max(f.shape[1] for f in frames)
    padded_frames: list[np.ndarray] = []
    for f in frames:
        pad = np.zeros((max_h, max_w), dtype=f.dtype)
        pad[: f.shape[0], : f.shape[1]] = f
        padded_frames.append(pad)

    # ------------------------------------------------------------------- #
    # 3. Stack -> (N, Y, X)                                               #
    # ------------------------------------------------------------------- #
    stacked = np.stack(padded_frames)
    logger.info("Stacked shape: %s  (N files, Y, X)", stacked.shape)

    # ------------------------------------------------------------------- #
    # 4. Extract patches                                                   #
    # ------------------------------------------------------------------- #
    all_patches: list[np.ndarray] = []
    patch_meta: list[tuple[int, int, tuple]] = []

    for file_idx, frame in enumerate(stacked):
        h, w = frame.shape
        local_idx = 0
        # Use a strict non-overlapping grid so each spatial region is sampled once.
        for r0 in range(0, h, ps):
            for c0 in range(0, w, ps):
                r1 = min(r0 + ps, h)
                c1 = min(c0 + ps, w)

                patch = np.zeros((ps, ps), dtype=frame.dtype)
                patch[: r1 - r0, : c1 - c0] = frame[r0:r1, c0:c1]

                all_patches.append(patch)
                patch_meta.append((file_idx, local_idx, (r0, r1, c0, c1)))
                local_idx += 1

    all_patches_arr = np.stack(all_patches)
    logger.info(
        "Total patches: %d  (shape %s)", all_patches_arr.shape[0], all_patches_arr.shape
    )

    # ------------------------------------------------------------------- #
    # 5. Session management                                                #
    # ------------------------------------------------------------------- #
    os.makedirs(labelling_dir, exist_ok=True)

    latest_session = _find_latest_session(labelling_dir)
    if latest_session and os.path.exists(os.path.join(latest_session, "coords.npy")):
        coords = np.load(os.path.join(latest_session, "coords.npy"))
        logger.info("Loaded %d points from: %s", len(coords), latest_session)
    else:
        coords = np.empty((0, 3), dtype=np.float32)
        logger.info("No previous annotations found - starting fresh.")

    session_dir = _create_new_session(labelling_dir)
    logger.info("New session folder: %s", session_dir)

    # ------------------------------------------------------------------- #
    # 6. Napari annotation                                                 #
    # ------------------------------------------------------------------- #
    viewer = napari.Viewer()
    viewer.add_image(all_patches_arr, name="patches")

    points_layer = viewer.add_points(
        coords if len(coords) > 0 else None,
        name="selected",
        ndim=3,
        size=10,
        face_color="red",
    )

    # Auto-save coords on every frame change and every point addition
    def _autosave(event=None) -> None:  # noqa: ANN001
        pts_now = points_layer.data
        if len(pts_now) > 0:
            np.save(os.path.join(session_dir, "coords.npy"), pts_now)
            logger.debug("Auto-saved %d points.", len(pts_now))

    viewer.dims.events.current_step.connect(_autosave)
    points_layer.events.data.connect(_autosave)

    logger.info("Napari is open. Click points on the 'selected' layer, then close the viewer.")
    napari.run()

    # ------------------------------------------------------------------- #
    # 7. Save points                                                       #
    # ------------------------------------------------------------------- #
    # When napari closes, points_layer.data can return empty even if points
    # were annotated (layer cleaned up during shutdown).  Guard against
    # overwriting a valid autosaved file with an empty array.
    pts = points_layer.data
    autosaved_coords = os.path.join(session_dir, "coords.npy")
    if len(pts) > 0:
        np.save(autosaved_coords, pts)
        logger.info("Saved %d points from layer.", len(pts))
    elif os.path.exists(autosaved_coords):
        pts = np.load(autosaved_coords)
        logger.info(
            "Points layer was empty on close; kept autosaved %d point(s) from %s.",
            len(pts),
            autosaved_coords,
        )
    else:
        np.save(autosaved_coords, pts)  # empty – nothing to annotate
        logger.warning("No points annotated in this session.")
    np.save(os.path.join(session_dir, "frames.npy"), all_patches_arr)

    # ------------------------------------------------------------------- #
    # 8. Generate Gaussian-blob labels                                     #
    # ------------------------------------------------------------------- #
    n_patches = all_patches_arr.shape[0]
    if args.max_patches is not None:
        n_patches = min(n_patches, args.max_patches)

    patch_points: dict[int, list[list[float]]] = defaultdict(list)
    for row in pts:
        pidx = int(round(row[0]))
        y, x = float(row[1]), float(row[2])
        if 0 <= pidx < n_patches:
            patch_points[pidx].append([y, x])

    label_images = np.zeros((n_patches, ps, ps), dtype=np.float32)
    for pidx, points in patch_points.items():
        label_images[pidx] = centroids2images(
            points, ps, ps, g_radius=args.gaussian_radius
        )

    np.save(os.path.join(session_dir, "labels.npy"), label_images)
    nonzero_total = int((label_images.max(axis=(1, 2)) > 0).sum())
    logger.info(
        "Saved label images: %s  (non-zero: %d / %d)",
        label_images.shape,
        nonzero_total,
        n_patches,
    )

    # ------------------------------------------------------------------- #
    # 9. Split into sub-patches                                            #
    # ------------------------------------------------------------------- #
    work_patches = all_patches_arr[:n_patches, ...]

    sub_frames: list[np.ndarray] = []
    sub_labels: list[np.ndarray] = []
    for i in range(work_patches.shape[0]):
        for r in range(0, ps, sps):
            for c in range(0, ps, sps):
                sub_frames.append(work_patches[i, r : r + sps, c : c + sps])
                sub_labels.append(label_images[i, r : r + sps, c : c + sps])

    sub_frames_arr = np.stack(sub_frames)
    sub_labels_arr = np.stack(sub_labels)
    logger.info("%dx%d sub-patches: %s", sps, sps, sub_frames_arr.shape)

    # ------------------------------------------------------------------- #
    # 10. Downsample zero-label patches                                    #
    # ------------------------------------------------------------------- #
    nonzero_mask = sub_labels_arr.max(axis=(1, 2)) > 0
    nonzero_idx = np.where(nonzero_mask)[0]
    zero_idx = np.where(~nonzero_mask)[0]

    n_keep_zero = max(1, int(round(len(nonzero_idx) * args.zero_keep_ratio)))

    rng = np.random.default_rng(args.seed)
    kept_zero_idx = rng.choice(
        zero_idx, size=min(n_keep_zero, len(zero_idx)), replace=False
    )

    final_idx = np.sort(np.concatenate([nonzero_idx, kept_zero_idx]))
    sub_frames_final = sub_frames_arr[final_idx]
    sub_labels_final = sub_labels_arr[final_idx]

    logger.info("Non-zero patches : %d", len(nonzero_idx))
    logger.info("Zero patches kept: %d  (~%.0f%%)", len(kept_zero_idx), args.zero_keep_ratio * 100)
    logger.info("Final dataset    : %d patches", len(final_idx))

    # ------------------------------------------------------------------- #
    # 11. Save training set                                                #
    # ------------------------------------------------------------------- #
    os.makedirs(train_input_dir, exist_ok=True)
    os.makedirs(train_target_dir, exist_ok=True)

    for save_idx, (img_patch, lbl) in enumerate(
        zip(sub_frames_final, sub_labels_final)
    ):
        fname = f"{save_idx}.tif"
        tifffile.imwrite(
            os.path.join(train_input_dir, fname), img_patch.astype(np.float32)
        )
        tifffile.imwrite(
            os.path.join(train_target_dir, fname), lbl.astype(np.float32)
        )

    logger.info("Saved %d pairs to:", len(final_idx))
    logger.info("  input  -> %s", train_input_dir)
    logger.info("  target -> %s", train_target_dir)


if __name__ == "__main__":
    main()
