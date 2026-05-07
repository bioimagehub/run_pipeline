# %% Imports
import sys
import os
import importlib

import numpy as np
import matplotlib.pyplot as plt
from bioio import BioImage
from scipy.ndimage import label, center_of_mass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sam2_utils
importlib.reload(sam2_utils)
from sam2_utils import load_sam2_image_predictor, find_similar_object


# %% Input paths
video_path = r"E:\Oyvind\BIP-hub-scratch\train_macropinosome_model\input_drift_corrected\Input_crest__KiaWee__250225_RPE-mNG-Phafin2_BSD_10ul_001_1_drift.ome.tif"
mask_path  = r"E:\Oyvind\BIP-hub-scratch\train_macropinosome_model\deep_learning_output_2\masks\Input_crest__KiaWee__250225_RPE-mNG-Phafin2_BSD_10ul_001_1_drift.ome_mask.tif"
output_folder = r"C:\Users\oyvinode\Desktop\del"

mask_time = 0     # frame that contains the reference mask
size      = 2000  # spatial crop (YX) to keep memory manageable
channel   = 0
z_slice   = 0
half_size = 60    # half-width of the per-label tracking window (window = 2*half_size px)


# %% Load image / mask metadata
img  = BioImage(video_path)
mask = BioImage(mask_path)
print("Image shape:", img.shape, img.dtype)
print("Mask  shape:", mask.shape, mask.dtype)
print("Total frames:", img.dims.T)


# %% Get the initial labeled mask at mask_time
mask_raw = mask.get_image_data("YX", T=mask_time, C=0, Z=0)[:size, :size]

# Use the raw label values from the mask directly (keep original IDs)
labeled_mask = mask_raw.astype(np.int32)

# Fall back to scipy label() if the mask is binary
unique_vals = np.unique(labeled_mask)
unique_vals = unique_vals[unique_vals != 0]
if len(unique_vals) == 1 and unique_vals[0] == 1:
    print("Mask appears binary — running connected-component labeling.")
    labeled_mask, _ = label((mask_raw > 0).astype(np.uint8))
    labeled_mask = labeled_mask.astype(np.int32)
    unique_vals = np.unique(labeled_mask)
    unique_vals = unique_vals[unique_vals != 0]

num_labels = len(unique_vals)
print(f"Labels in initial mask: {num_labels}  (IDs: {unique_vals[:10]}{'...' if num_labels > 10 else ''})")


# %% Quick sanity-check: show initial mask overlaid on source frame
src_img = img.get_image_data("YX", T=mask_time, C=channel, Z=z_slice)[:size, :size]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].set_title(f"Source image  T={mask_time}")
axes[0].imshow(src_img, cmap="gray")
axes[0].axis("off")

axes[1].set_title(f"Initial labeled mask  ({num_labels} labels)")
axes[1].imshow(src_img, cmap="gray")
label_overlay = np.zeros((*src_img.shape, 4), dtype=np.float32)
for lbl in unique_vals:
    color = plt.cm.tab20(int(lbl) % 20 / 20.0)
    label_overlay[labeled_mask == lbl] = [color[0], color[1], color[2], 0.6]
axes[1].imshow(label_overlay)
axes[1].axis("off")

plt.tight_layout()
plt.show()


# %% Load SAM2 video predictor
predictor = load_sam2_image_predictor()


# %% Pre-load all frames into RAM as (n_total, H, W) once to avoid repeated disk I/O
H, W     = src_img.shape
n_total  = img.dims.T
n_future = n_total - mask_time - 1
print(f"Loading all {n_total} frames …")
all_frames = np.stack(
    [img.get_image_data("YX", T=t, C=channel, Z=z_slice)[:size, :size] for t in range(n_total)]
)  # (n_total, H, W)
print("All frames shape:", all_frames.shape)


# %% Compute per-label centroids and crop bounds
label_info: list[dict] = []  # {lbl, y0, y1, x0, x1}
for lbl in unique_vals:
    ys, xs = np.where(labeled_mask == lbl)
    cy, cx = int(ys.mean()), int(xs.mean())
    y0 = max(cy - half_size, 0)
    y1 = min(cy + half_size, H)
    x0 = max(cx - half_size, 0)
    x1 = min(cx + half_size, W)
    label_info.append({"lbl": int(lbl), "y0": y0, "y1": y1, "x0": x0, "x1": x1})

print(f"Centroids computed for {len(label_info)} labels.")

# Quick overview: source frame with bounding boxes
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_title(f"Source T={mask_time} — {len(label_info)} label windows")
ax.imshow(src_img, cmap="gray")
for info in label_info:
    rect = plt.Rectangle(
        (info["x0"], info["y0"]),
        info["x1"] - info["x0"],
        info["y1"] - info["y0"],
        edgecolor="cyan", facecolor="none", linewidth=0.8,
    )
    ax.add_patch(rect)
    ax.plot(
        (info["x0"] + info["x1"]) / 2,
        (info["y0"] + info["y1"]) / 2,
        "r.", markersize=3,
    )
ax.axis("off")
plt.tight_layout()
plt.show()


# %% Track each label independently in its own 40×40 window across ALL frames
#
# For each label A:
#   1. Crop the 40×40 source image and the FULL labeled_mask for that window.
#      Any neighbours B, C … present in the window are registered as separate
#      SAM2 objects, giving the model explicit "these are other things" context.
#   2. Build a (n_future, 40, 40) stack of the same crop from every future frame.
#   3. Call find_similar_object → (n_future, 40, 40) int32.
#   4. Write ONLY pixels where tracked == lbl back into all_masks; neighbour
#      predictions are discarded (they will be written in their own pass).

all_masks = np.zeros((n_total, H, W), dtype=np.int32)
all_masks[mask_time] = labeled_mask

print(f"Tracking {len(label_info)} labels across {n_future} frames …")
for idx, info in enumerate(label_info):
    lbl   = info["lbl"]
    y0, y1, x0, x1 = info["y0"], info["y1"], info["x0"], info["x1"]

    src_crop  = all_frames[mask_time, y0:y1, x0:x1]
    # Pass ALL labels visible in this window so SAM2 can distinguish neighbours.
    mask_crop = labeled_mask[y0:y1, x0:x1].copy()

    future_crops = all_frames[mask_time + 1 :, y0:y1, x0:x1]  # (n_future, ch, cw)

    neighbour_ids = [v for v in np.unique(mask_crop) if v != 0 and v != lbl]
    print(f"  [{idx+1}/{len(label_info)}] label={lbl}  crop=({y0}:{y1}, {x0}:{x1})  "
          f"neighbours in window={neighbour_ids}")

    tracked_crop = find_similar_object(
        source_image=src_crop,
        source_mask=mask_crop,
        target=future_crops,
        predictor=predictor,
    )  # (n_future, ch, cw) int32

    # Write back only the primary label; neighbour predictions are discarded here
    # (each neighbour will write its own result in its dedicated pass).
    for fi in range(n_future):
        t = mask_time + 1 + fi
        hit = tracked_crop[fi] == lbl
        all_masks[t, y0:y1, x0:x1][hit] = lbl

print("Tracking complete.")


# %% Per-frame label counts
counts = [(t, int((all_masks[t] > 0).sum()), len(np.unique(all_masks[t])) - 1)
          for t in range(n_total)]
print(f"{'Frame':>6}  {'Foreground px':>14}  {'Labels found':>13}")
for t, px, lbls in counts:
    print(f"{t:>6}  {px:>14}  {lbls:>13}")


# %% Visualize a grid of evenly-spaced frames
n_show     = min(8, n_total)
step       = max(1, n_total // n_show)
show_frames = list(range(0, n_total, step))[:n_show]

fig, axes = plt.subplots(1, len(show_frames), figsize=(4 * len(show_frames), 4))
if len(show_frames) == 1:
    axes = [axes]

for ax, t in zip(axes, show_frames):
    frame = img.get_image_data("YX", T=t, C=channel, Z=z_slice)[:size, :size]
    ax.imshow(frame, cmap="gray")

    tracked = all_masks[t]
    if tracked.max() > 0:
        overlay = np.zeros((*frame.shape, 4), dtype=np.float32)
        for lbl in np.unique(tracked):
            if lbl == 0:
                continue
            color = plt.cm.tab20(int(lbl) % 20 / 20.0)
            overlay[tracked == lbl] = [color[0], color[1], color[2], 0.5]
        ax.imshow(overlay)

    label_cnt = len(np.unique(tracked)) - 1
    prefix = "ref" if t == mask_time else "tracked"
    ax.set_title(f"T={t} ({prefix})\n{label_cnt} labels", fontsize=8)
    ax.axis("off")

plt.suptitle("SAM2 tracking — all labels, all frames", fontsize=10)
plt.tight_layout()
plt.show()


# %% Save 2-channel TIFF: C=0 image, C=1 tracked mask  shape=(T, C, Y, X)
import tifffile

os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "tracked_image_and_mask.tif")

# ImageJ format supports float32; cast both channels to float32.
combined = np.stack(
    [all_frames.astype(np.float32), all_masks.astype(np.float32)],
    axis=1,
)  # (T, 2, H, W)

tifffile.imwrite(
    output_path,
    combined,
    imagej=True,
    metadata={"axes": "TCYX"},
)
print(f"Saved 2-channel TIFF to: {output_path}  shape={combined.shape}")

# %%
