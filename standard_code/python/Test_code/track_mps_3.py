import sys

from bioio import BioImage
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import flood, expand_labels
from scipy.ndimage import binary_fill_holes, median_filter, binary_dilation, maximum_filter, minimum_filter
import pandas as pd
from skimage.measure import label, regionprops
from skimage.morphology import disk
import trackpy as tp
import cv2


def find_maxima(pred_dask, min_distance: int = 10, threshold_abs: float = 0.1) -> list:
    timepoints = pred_dask.shape[0]

    all_coordinates = []
    for t in range(timepoints):
        frame = np.asarray(pred_dask[t].compute())
        coordinates = peak_local_max(frame, min_distance=min_distance, threshold_abs=threshold_abs)
        napari_coordinates = [[int(t), 0, int(y), int(x)] for y, x in coordinates]
        all_coordinates.append(napari_coordinates)

    return all_coordinates


def build_validated_output_mask(
    filtered: np.ndarray,
    candidate_hole_mask: np.ndarray,
    center_y: int,
    center_x: int,
) -> tuple[np.ndarray | None, dict]:
    """Build a validated 0/1/2 mask from a candidate hole mask."""
    debug = {
        'touching_edge': True,
        'has_hole': False,
        'has_ring': False,
        'ring_brighter_than_hole': False,
    }

    hole_mask = remove_opposite_false_pixels(candidate_hole_mask, iterations=2)
    labels = label(hole_mask, connectivity=1)
    center_label = int(labels[center_y, center_x])
    if center_label == 0:
        return None, debug

    hole_mask = labels == center_label
    if not bool(hole_mask[center_y, center_x]):
        return None, debug

    touching_edge = (
        np.any(hole_mask[0, :])
        or np.any(hole_mask[-1, :])
        or np.any(hole_mask[:, 0])
        or np.any(hole_mask[:, -1])
    )
    debug['touching_edge'] = touching_edge

    dilated = np.asarray(binary_dilation(hole_mask, disk(5)), dtype=bool)
    ring_mask = np.asarray(np.logical_and(dilated, np.logical_not(hole_mask)), dtype=bool)

    has_hole = bool(np.any(hole_mask))
    has_ring = bool(np.any(ring_mask))
    debug['has_hole'] = has_hole
    debug['has_ring'] = has_ring
    if not has_hole or not has_ring:
        return None, debug

    ring_brighter_than_hole = float(np.median(filtered[ring_mask])) > float(np.median(filtered[hole_mask]))
    debug['ring_brighter_than_hole'] = ring_brighter_than_hole
    if touching_edge or not ring_brighter_than_hole:
        return None, debug

    output_mask = np.zeros_like(filtered, dtype=np.uint8)
    output_mask[hole_mask] = 1
    output_mask[ring_mask] = 2
    return output_mask, debug


def resolve_overlap_conflict(
    existing_region: np.ndarray,
    new_mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Resolve overlap between two 0/1/2 masks.

    Rule 1 – hole overlap (value 1 touches value 1): the two detections are
              the same object; merge by taking the union of both holes and rings.
    Rule 2 – ring-only overlap (value 2 touches value 2, holes do not touch):
              zero out the pixels covered by both rings.
    """
    debug = {'mode': 'no-overlap'}

    overlap = (existing_region > 0) & (new_mask > 0)
    if not np.any(overlap):
        return np.maximum(existing_region, new_mask), debug

    existing_hole = existing_region == 1
    new_hole = new_mask == 1

    if np.any(existing_hole & new_mask) or np.any(new_hole & existing_region):
        # Hole pixels from either object are touched — treat as one object.
        debug['mode'] = 'merge'
        resolved_region = np.zeros_like(existing_region)
        resolved_region[existing_hole | new_hole] = 1
        resolved_region[(existing_region == 2) | (new_mask == 2)] = 2
        # Hole takes priority over ring where they coincide.
        resolved_region[existing_hole | new_hole] = 1
        return resolved_region, debug

    # Only ring pixels overlap — zero out the shared zone.
    debug['mode'] = 'zero-ring-overlap'
    resolved_region = np.maximum(existing_region, new_mask)
    resolved_region[overlap] = 0
    return resolved_region, debug


def regrow_rings(frame_mask: np.ndarray, ring_width: int = 5) -> np.ndarray:
    """Regenerate ring pixels (value 2) for every hole (value 1) in a single frame.

    Each hole label is expanded outward by ring_width pixels using expand_labels.
    Pixels where two different hole expansions would meet are detected as conflict
    zones and set to 0, creating a natural zero-valued gap between adjacent objects.
    """
    hole_mask = frame_mask == 1
    if not np.any(hole_mask):
        return frame_mask.copy()

    hole_labels = label(hole_mask, connectivity=1)
    expanded = expand_labels(hole_labels, distance=ring_width)

    # Pixels where the 3x3 neighbourhood contains more than one label value are
    # the conflict zone -- the boundary between two expanding regions.
    max_neigh = maximum_filter(expanded, size=3)
    min_neigh = minimum_filter(expanded, size=3)
    conflict = (max_neigh != min_neigh) & (expanded > 0)

    result = np.zeros_like(frame_mask)
    result[(expanded > 0) & ~hole_mask & ~conflict] = 2
    result[hole_mask] = 1  # holes always overwrite rings
    return result


def find_macropinosomes(all_coordinates, img_dask, filter_patch_size=75, min_size=20, max_size=1000,
                        median_filter_size=3, initial_tolerance=0.1, show_plot=False):
    frame_shape = img_dask[0].compute().shape
    mask_out = np.zeros((img_dask.shape[0], frame_shape[0], frame_shape[1]), dtype=np.uint8)
    accepted_seed_points_by_frame = [[] for _ in range(img_dask.shape[0])]

    # t, frame_coordinates = 0, all_coordinates[0] # DO not delete - for debugging

    for t, frame_coordinates in enumerate(all_coordinates):
        frame = np.asarray(img_dask[t].compute())
        
        #_, _, y, x = frame_coordinates[0] # DO not delete - for debugging
        for _, _, y, x in frame_coordinates:
            y0 = max(0, y - filter_patch_size // 2)
            x0 = max(0, x - filter_patch_size // 2)
            y1 = min(frame.shape[0], y + filter_patch_size // 2 + 1)
            x1 = min(frame.shape[1], x + filter_patch_size // 2 + 1)

            patch = frame[y0:y1, x0:x1]
            cy, cx = np.array(patch.shape) // 2
            filtered = median_filter(patch, size=median_filter_size)

            tolerance = float(initial_tolerance) * (float(filtered.max()) - float(filtered.min()))
            output_mask = None
            while True:
                grown = flood(filtered, (cy, cx), tolerance=tolerance)
                if grown is None:
                    break
                mask = np.asarray(grown, dtype=bool)
                candidate_hole_mask = np.asarray(binary_fill_holes(mask), dtype=bool)
                output_mask, qc_debug = build_validated_output_mask(
                    filtered,
                    candidate_hole_mask,
                    center_y=cy,
                    center_x=cx,
                )
                if output_mask is not None:
                    break
                if not qc_debug['touching_edge']:
                    break
                tolerance *= 0.9

            if output_mask is None:
                continue

            existing_region = mask_out[t, y0:y1, x0:x1]
            overlap = (existing_region > 0) & (output_mask > 0)
            resolved_region, overlap_info = resolve_overlap_conflict(existing_region, output_mask)
            # if np.any(overlap):
            #     # Regrow rings on a temp full-frame copy for the debug plot.
            #     _temp_frame = mask_out[t].copy()
            #     _temp_frame[y0:y1, x0:x1] = resolved_region
            #     _regrown_patch = regrow_rings(_temp_frame, ring_width=5)[y0:y1, x0:x1]
            #     fig, axes = plt.subplots(2, 3, figsize=(14, 9))
            #     axes[0, 0].imshow(patch, cmap='gray')
            #     axes[0, 0].set_title(f"New patch t={t}, y={y}, x={x}")
            #     axes[0, 1].imshow(frame[y0:y1, x0:x1], cmap='gray')
            #     axes[0, 1].set_title("Existing global region")
            #     axes[0, 2].imshow(overlap, cmap='gray')
            #     axes[0, 2].set_title(f"Overlap pixels\n{overlap_info['mode']}")
            #     axes[1, 0].imshow(existing_region, vmin=0, vmax=2, cmap='viridis')
            #     axes[1, 0].set_title("Existing mask")
            #     axes[1, 1].imshow(output_mask, vmin=0, vmax=2, cmap='viridis')
            #     axes[1, 1].set_title("New mask")
            #     axes[1, 2].imshow(_regrown_patch, vmin=0, vmax=2, cmap='viridis')
            #     axes[1, 2].set_title("Resolved mask (regrown)")
            #     for axis in axes.ravel():
            #         axis.axis('off')
            #     plt.tight_layout()
            #     plt.show()

            mask_out[t, y0:y1, x0:x1] = resolved_region
            accepted_seed_points_by_frame[t].append([int(t), 0, int(y), int(x)])

    # Regenerate all rings globally from holes so adjacent objects always get a clean gap.
    for t in range(mask_out.shape[0]):
        mask_out[t] = regrow_rings(mask_out[t], ring_width=5)

    return mask_out, accepted_seed_points_by_frame


def remove_opposite_false_pixels(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Remove true pixels whose opposite 4-neighbors are false.

    Rule (4-neighborhood only): remove a center-true pixel when either
    (up and down are true, left and right are false) or
    (left and right are true, up and down are false).
    """
    out = np.asarray(mask, dtype=bool).copy()
    if iterations < 1:
        return out

    for _ in range(iterations):
        padded = np.pad(out, 1, mode='constant', constant_values=False)
        center = padded[1:-1, 1:-1]
        up = padded[:-2, 1:-1]
        down = padded[2:, 1:-1]
        left = padded[1:-1, :-2]
        right = padded[1:-1, 2:]

        vertical_true_horizontal_false = up & down & (~left) & (~right)
        horizontal_true_vertical_false = left & right & (~up) & (~down)
        to_remove = center & (vertical_true_horizontal_false | horizontal_true_vertical_false)

        if not np.any(to_remove):
            break
        out[to_remove] = False

    return out


def process_file(img_path, pred_path, channel=0, min_distance=10, threshold_abs=0.1, filter_patch_size=45,
                 search_range=15, min_size=20, max_size=1000, time_search_memory=3, min_timepoints=5,
                 initial_tolerance=0.1, median_filter_size=3):

    print("Loading images...")
    img_input = BioImage(img_path)
    print(f"input shape: {img_input.shape}, dtype: {img_input.dtype}")
    img_pred = BioImage(pred_path)
    print(f"pred shape: {img_pred.shape}, dtype: {img_pred.dtype}")

    img_dask = img_input.get_image_dask_data()       # TCZYX
    img_dask = img_dask[:, channel, :, :, :]        # select channel -> TZYX
    img_dask = img_dask.max(axis=1)                   # max projection along Z -> TYX

    if img_pred.dims.C > 1:
        raise ValueError("Prediction image should have one channel")
    pred_dask = img_pred.get_image_dask_data()        # TCZYX
    pred_dask = pred_dask[:, 0, :, :, :]            # select channel -> TZYX
    pred_dask = pred_dask.max(axis=1)                 # max projection along Z -> TYX

    print("Finding maxima in prediction...")
    all_coordinates = find_maxima(pred_dask, min_distance=min_distance, threshold_abs=threshold_abs)

    if all_coordinates is None or len(all_coordinates) == 0 or all(len(c) == 0 for c in all_coordinates):
        print("No coordinates found in prediction. Exiting.")
        return

    print("Filtering coordinates by hole score and size ...")
    mask, filtered_coordinates = find_macropinosomes(
        all_coordinates, img_dask,
        filter_patch_size=filter_patch_size,
        min_size=min_size,
        max_size=max_size,
        median_filter_size=median_filter_size,
        initial_tolerance=initial_tolerance,
    )

    import napari
    viewer = napari.Viewer()
    viewer.add_image(img_dask, name='input')
    viewer.add_image(pred_dask, name='prediction')
    viewer.add_image(mask, name='validated_mask', colormap='viridis', opacity=0.5)
    for t, frame_coords in enumerate(filtered_coordinates):
        if frame_coords:
            coords_array = np.array(frame_coords)
            viewer.add_points(coords_array, name=f'seed_points_t{t}', size=10, face_color='red', edge_color='yellow')
    napari.run()


    import tifffile
    out_path = "E:/Oyvind/BIP-hub-scratch/train_macropinosome_model/deep_learning_output/predictions/Input_crest__KiaWee__250225_RPE-mNG-Phafin2_BSD_10ul_001_1_drift_pred_mask.ome.tif"
    tifffile.imwrite(out_path, mask.astype(np.uint8))


def main():
    img_path = "E:/Oyvind/BIP-hub-scratch/train_macropinosome_model/input_drift_corrected/Input_crest__KiaWee__250225_RPE-mNG-Phafin2_BSD_10ul_001_1_drift.ome.tif"
    pred_path = "E:/Oyvind/BIP-hub-scratch/train_macropinosome_model/deep_learning_output/predictions/Input_crest__KiaWee__250225_RPE-mNG-Phafin2_BSD_10ul_001_1_drift_pred.ome.tif"

    channel = 0
    min_distance = 10
    threshold_abs = 0.1
    filter_patch_size = 45
    search_range = 35
    time_search_memory = 2
    min_timepoints = 5
    min_size = 20
    max_size = 2000
    median_filter_size = 3
    initial_tolerance = 0.1

    process_file(
        img_path, pred_path,
        channel=channel,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        filter_patch_size=filter_patch_size,
        search_range=search_range,
        time_search_memory=time_search_memory,
        min_timepoints=min_timepoints,
        min_size=min_size,
        max_size=max_size,
        median_filter_size=median_filter_size,
        initial_tolerance=initial_tolerance,
    )


if __name__ == "__main__":
    main()
