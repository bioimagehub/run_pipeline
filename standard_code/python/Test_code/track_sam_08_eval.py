# %% track_sam_08_eval.py
# Evaluation of global_mask_v8.tif: track lengths, spatial distribution,
# per-frame coverage, representative track overlays.

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive — saves to file
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tifffile
from bioio import BioImage
from skimage.measure import regionprops_table

# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------
mask_path   = r"C:\Users\oyvinode\Desktop\del\global_mask_v8.tif"
video_path  = r"E:\Oyvind\BIP-hub-scratch\train_macropinosome_model\input_drift_corrected\Input_crest__KiaWee__250225_RPE-mNG-Phafin2_BSD_10ul_001_1_drift.ome.tif"
out_dir     = r"C:\Users\oyvinode\Desktop\del\eval_v8"
os.makedirs(out_dir, exist_ok=True)

channel = 0
z_slice = 0

# -------------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------------
print("Loading mask …")
mask = tifffile.imread(mask_path).astype(np.int32)   # (T, H, W)
T, H, W = mask.shape
print(f"  mask shape: {mask.shape}  unique IDs: {mask.max()}")

print("Loading raw frames …")
img = BioImage(video_path)
n_total = min(T, img.dims.T)
raw = np.stack([
    img.get_image_data("YX", T=t, C=channel, Z=z_slice)
    for t in range(n_total)
])  # (T, H, W)
print(f"  raw shape: {raw.shape}  dtype: {raw.dtype}")

# -------------------------------------------------------------------------
# Track statistics — fast per-frame regionprops pass
# -------------------------------------------------------------------------
print("Computing track statistics …")

track_frames: dict[int, list[int]] = {}
track_areas:  dict[int, list[int]] = {}
track_cents:  dict[int, list[tuple[float, float]]] = {}

for t in range(T):
    props = regionprops_table(mask[t], properties=("label", "area", "centroid"))
    for tid, area, cy, cx in zip(
        props["label"], props["area"], props["centroid-0"], props["centroid-1"]
    ):
        if tid == 0:
            continue
        tid = int(tid)
        if tid not in track_frames:
            track_frames[tid] = []
            track_areas[tid]  = []
            track_cents[tid]  = []
        track_frames[tid].append(t)
        track_areas[tid].append(int(area))
        track_cents[tid].append((float(cy), float(cx)))

track_lengths = {tid: len(flist) for tid, flist in track_frames.items()}
lengths = np.array(list(track_lengths.values()))
n_tracks = len(track_lengths)

print(f"  Track length — min: {lengths.min()}  median: {np.median(lengths):.0f}"
      f"  mean: {lengths.mean():.1f}  max: {lengths.max()}")
print(f"  Tracks spanning all {T} frames: {(lengths == T).sum()}")
print(f"  Tracks >= 10 frames: {(lengths >= 10).sum()}")
print(f"  Tracks >= 50 frames: {(lengths >= 50).sum()}")

# Per-frame: number of active tracks and total masked pixels
n_active  = np.array([(mask[t] > 0).any() and len(np.unique(mask[t])) - 1 for t in range(T)])
n_active  = np.array([int(len(np.unique(mask[t])) - (1 if 0 in mask[t] else 0)) for t in range(T)])
px_per_t  = np.array([(mask[t] > 0).sum() for t in range(T)])

# -------------------------------------------------------------------------
# Plot 1: Track length histogram
# -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(lengths, bins=np.arange(1, lengths.max() + 2) - 0.5,
             color="steelblue", edgecolor="white", linewidth=0.4)
axes[0].set_xlabel("Track length (frames)")
axes[0].set_ylabel("Number of tracks")
axes[0].set_title(f"Track length distribution  (n={n_tracks})")
axes[0].axvline(T, color="red", linestyle="--", label=f"Full video ({T} frames)")
axes[0].legend()

# Log-scale version for the same
axes[1].hist(lengths, bins=np.arange(1, lengths.max() + 2) - 0.5,
             color="coral", edgecolor="white", linewidth=0.4)
axes[1].set_yscale("log")
axes[1].set_xlabel("Track length (frames)")
axes[1].set_ylabel("Number of tracks (log scale)")
axes[1].set_title("Track length distribution — log y")
axes[1].axvline(T, color="red", linestyle="--", label=f"Full video ({T} frames)")
axes[1].legend()

plt.suptitle("track_sam_08 — Track length distribution", fontsize=12)
plt.tight_layout()
out1 = os.path.join(out_dir, "01_track_length_histogram.png")
plt.savefig(out1, dpi=150)
plt.close()
print(f"  Saved {out1}")

# -------------------------------------------------------------------------
# Plot 2: Per-frame active track count and masked pixel count
# -------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

axes[0].plot(range(T), n_active, color="steelblue", linewidth=1.2)
axes[0].set_ylabel("Active tracks per frame")
axes[0].set_title("Per-frame track count")
axes[0].grid(alpha=0.3)

axes[1].fill_between(range(T), px_per_t, alpha=0.6, color="coral")
axes[1].plot(range(T), px_per_t, color="darkred", linewidth=0.8)
axes[1].set_xlabel("Frame (t)")
axes[1].set_ylabel("Masked pixels")
axes[1].set_title("Per-frame total masked area")
axes[1].grid(alpha=0.3)

plt.suptitle("track_sam_08 — Per-frame activity", fontsize=12)
plt.tight_layout()
out2 = os.path.join(out_dir, "02_per_frame_activity.png")
plt.savefig(out2, dpi=150)
plt.close()
print(f"  Saved {out2}")

# -------------------------------------------------------------------------
# Plot 3: Track lifetime map — colour each pixel by its track's total length
# -------------------------------------------------------------------------
# Build a per-pixel "track length" volume: value = length of whichever track owns it
# Build a lookup table: ID -> track_length
max_id = int(mask.max())
id_to_len = np.zeros(max_id + 1, dtype=np.float32)
for tid, l in track_lengths.items():
    if tid <= max_id:
        id_to_len[tid] = l

len_vol = np.zeros((T, H, W), dtype=np.float32)
for t in range(T):
    len_vol[t] = id_to_len[mask[t].clip(0, max_id)]

# Show middle frame
mid = T // 2
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(raw[mid], cmap="gray")
lifetime_img = np.ma.masked_where(mask[mid] == 0, len_vol[mid])
im = ax.imshow(lifetime_img, cmap="plasma", alpha=0.7,
               vmin=1, vmax=T)
plt.colorbar(im, ax=ax, fraction=0.03, label="Track length (frames)")
ax.set_title(f"Frame {mid} — pixels coloured by track lifetime")
ax.axis("off")
plt.tight_layout()
out3 = os.path.join(out_dir, "03_lifetime_map_mid_frame.png")
plt.savefig(out3, dpi=150)
plt.close()
print(f"  Saved {out3}")

# -------------------------------------------------------------------------
# Plot 4: Long-track overlay — colour by track ID for tracks >= 20 frames
# -------------------------------------------------------------------------
LONG_THRESH = 20
long_ids = {tid for tid, l in track_lengths.items() if l >= LONG_THRESH}
print(f"  Long tracks (>= {LONG_THRESH} frames): {len(long_ids)}")

# Build per-track frame sets for fast membership test
track_frame_sets = {tid: set(flist) for tid, flist in track_frames.items()}

# Show first, middle, last frames
show_frames = [0, T // 4, T // 2, 3 * T // 4, T - 1]
cmap_ids = cm.get_cmap("tab20", max(len(long_ids), 2))
id_to_col = {tid: i for i, tid in enumerate(sorted(long_ids))}

fig, axes = plt.subplots(1, len(show_frames), figsize=(5 * len(show_frames), 5))
for ax, t in zip(axes, show_frames):
    ax.imshow(raw[t], cmap="gray")
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    for tid in long_ids:
        if t not in track_frame_sets.get(tid, set()):
            continue
        pix_mask = mask[t] == tid
        col = np.array(cmap_ids(id_to_col[tid]))
        overlay[pix_mask] = [col[0], col[1], col[2], 0.7]
    ax.imshow(overlay)
    n_long_t = sum(1 for tid in long_ids if t in track_frame_sets.get(tid, set()))
    ax.set_title(f"t={t}  ({n_long_t} long tracks)", fontsize=8)
    ax.axis("off")
plt.suptitle(f"Tracks ≥ {LONG_THRESH} frames — consistent colour across panels", fontsize=11)
plt.tight_layout()
out4 = os.path.join(out_dir, "04_long_track_overlay.png")
plt.savefig(out4, dpi=150)
plt.close()
print(f"  Saved {out4}")

# -------------------------------------------------------------------------
# Plot 5: Centroid trajectories of the 20 longest tracks
# -------------------------------------------------------------------------
top20_ids = sorted(track_lengths, key=track_lengths.get, reverse=True)[:20]

fig, ax = plt.subplots(figsize=(10, 10))
# Show last raw frame as background
ax.imshow(raw[-1], cmap="gray")
cmap20 = cm.get_cmap("tab20", 20)
for i, tid in enumerate(top20_ids):
    cents = track_cents[tid]
    frs   = track_frames[tid]
    ys = [c[0] for c in cents]
    xs = [c[1] for c in cents]
    col = cmap20(i)
    ax.plot(xs, ys, "-", color=col, linewidth=1.5, alpha=0.8)
    # Mark first and last point
    ax.plot(xs[0],  ys[0],  "o", color=col, markersize=6)
    ax.plot(xs[-1], ys[-1], "x", color=col, markersize=8, markeredgewidth=2)
    ax.text(xs[0], ys[0], f" {tid}({track_lengths[tid]}f)", fontsize=6,
            color=col, va="center")
ax.set_title("Centroid trajectories — top 20 longest tracks  (o=first, x=last)")
ax.axis("off")
plt.tight_layout()
out5 = os.path.join(out_dir, "05_top20_trajectories.png")
plt.savefig(out5, dpi=150)
plt.close()
print(f"  Saved {out5}")

# -------------------------------------------------------------------------
# Plot 6: Area over time for the top 20 longest tracks
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))
for i, tid in enumerate(top20_ids):
    frs   = track_frames[tid]
    areas = track_areas[tid]
    col = cmap20(i)
    ax.plot(frs, areas, "-", color=col, linewidth=1.2, alpha=0.8,
            label=f"ID {tid} ({track_lengths[tid]}f)")
ax.set_xlabel("Frame (t)")
ax.set_ylabel("Track area (pixels)")
ax.set_title("Area over time — top 20 longest tracks")
ax.legend(loc="upper right", fontsize=6, ncol=2)
ax.grid(alpha=0.3)
plt.tight_layout()
out6 = os.path.join(out_dir, "06_top20_area_over_time.png")
plt.savefig(out6, dpi=150)
plt.close()
print(f"  Saved {out6}")

# -------------------------------------------------------------------------
# Plot 7: Kymograph strip for 3 representative long tracks
# (crop a region around centroid for each frame the track is alive)
# -------------------------------------------------------------------------
KYM_HALF = 30   # half-size of the crop in pixels
top3_ids = top20_ids[:3]

fig, big_axes = plt.subplots(3, 2, figsize=(16, 12))
for row_i, tid in enumerate(top3_ids):
    frs   = track_frames[tid]
    cents = track_cents[tid]
    n_fr  = len(frs)

    # Raw kymograph strip
    strip_raw  = np.zeros((KYM_HALF * 2, n_fr), dtype=np.float32)
    # Mask kymograph strip
    strip_mask = np.zeros((KYM_HALF * 2, n_fr), dtype=np.float32)

    for col_i, (t, (cy, cx)) in enumerate(zip(frs, cents)):
        y0 = int(max(0, cy - KYM_HALF))
        y1 = int(min(H, cy + KYM_HALF))
        x0 = int(max(0, cx - KYM_HALF))
        x1 = int(min(W, cx + KYM_HALF))
        # Take a vertical slice through centroid
        cx_local = int(cx) - x0
        slice_raw  = raw[t, y0:y1, x0 + cx_local] if x0 + cx_local < W else np.zeros(y1 - y0)
        slice_mask = (mask[t, y0:y1, x0 + cx_local] == tid).astype(np.float32) if x0 + cx_local < W else np.zeros(y1 - y0)
        h = y1 - y0
        strip_raw[:h, col_i]  = slice_raw
        strip_mask[:h, col_i] = slice_mask

    ax_raw  = big_axes[row_i, 0]
    ax_mask = big_axes[row_i, 1]
    ax_raw.imshow(strip_raw,  cmap="gray", aspect="auto")
    ax_raw.set_title(f"Track {tid} ({n_fr} frames) — raw kymograph", fontsize=9)
    ax_raw.set_xlabel("Frame index within track lifetime")
    ax_raw.set_ylabel("Y (px from centroid)")
    ax_mask.imshow(strip_mask, cmap="hot", aspect="auto", vmin=0, vmax=1)
    ax_mask.set_title(f"Track {tid} — mask kymograph", fontsize=9)
    ax_mask.set_xlabel("Frame index within track lifetime")

plt.suptitle("Kymograph strips — top 3 longest tracks", fontsize=12)
plt.tight_layout()
out7 = os.path.join(out_dir, "07_kymographs_top3.png")
plt.savefig(out7, dpi=150)
plt.close()
print(f"  Saved {out7}")

# -------------------------------------------------------------------------
# Summary text
# -------------------------------------------------------------------------
summary_path = os.path.join(out_dir, "summary.txt")
with open(summary_path, "w") as f:
    f.write(f"track_sam_08 evaluation summary\n")
    f.write(f"================================\n")
    f.write(f"Mask file   : {mask_path}\n")
    f.write(f"Video shape : {raw.shape}  (T, H, W)\n\n")
    f.write(f"Total unique track IDs : {n_tracks}\n")
    f.write(f"Track length min       : {lengths.min()}\n")
    f.write(f"Track length median    : {np.median(lengths):.0f}\n")
    f.write(f"Track length mean      : {lengths.mean():.1f}\n")
    f.write(f"Track length max       : {lengths.max()}\n")
    f.write(f"Tracks spanning all {T} frames : {(lengths == T).sum()}\n")
    f.write(f"Tracks >= 10 frames    : {(lengths >= 10).sum()}\n")
    f.write(f"Tracks >= 50 frames    : {(lengths >= 50).sum()}\n\n")
    f.write(f"Top 20 longest tracks (ID, length):\n")
    for tid in top20_ids:
        f.write(f"  ID {tid:5d}: {track_lengths[tid]:4d} frames\n")
print(f"  Saved {summary_path}")
print("\nEvaluation complete. All plots saved to:", out_dir)
# %%
