

import sys

from bioio import BioImage
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import flood, watershed
from scipy.ndimage import binary_fill_holes, median_filter, binary_dilation, distance_transform_edt
import pandas as pd
from skimage.measure import label, regionprops
from skimage.morphology import reconstruction, disk
import trackpy as tp
import numpy as np
import cv2


def find_maxima(pred_dask, min_distance: int = 10, threshold_abs: float = 0.1) -> list:
    timepoints = pred_dask.shape[0]

    all_coordinates = []
    # Find maxima in prediction
    for t in range(timepoints):
        # compute the dask array for the current timepoint and convert to numpy
        frame = np.asarray(pred_dask[t].compute())

        # Find peaks in the current frame
        coordinates = peak_local_max(frame, min_distance=min_distance, threshold_abs=threshold_abs)

        # Keep detections in napari point format [T, Z, Y, X] from the start.
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


def get_mask_centroid(mask: np.ndarray) -> tuple[int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    center = np.rint(coords.mean(axis=0)).astype(int)
    return int(center[0]), int(center[1])


def get_nearest_true_pixel(mask: np.ndarray, target_y: float, target_x: float) -> tuple[int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    distances = (coords[:, 0] - target_y) ** 2 + (coords[:, 1] - target_x) ** 2
    nearest = coords[int(np.argmin(distances))]
    return int(nearest[0]), int(nearest[1])


def build_separator_line_mask(
    shape: tuple[int, int],
    start_point: tuple[int, int] | None,
    end_point: tuple[int, int] | None,
) -> np.ndarray:
    line_mask = np.zeros(shape, dtype=bool)
    if start_point is None or end_point is None:
        return line_mask

    point_count = int(max(abs(start_point[0] - end_point[0]), abs(start_point[1] - end_point[1]))) + 1
    ys = np.rint(np.linspace(start_point[0], end_point[0], point_count)).astype(int)
    xs = np.rint(np.linspace(start_point[1], end_point[1], point_count)).astype(int)
    valid = (ys >= 0) & (ys < shape[0]) & (xs >= 0) & (xs < shape[1])
    line_mask[ys[valid], xs[valid]] = True
    return line_mask


def resolve_overlap_conflict(
    filtered: np.ndarray,
    existing_region: np.ndarray,
    new_mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    debug = {
        'mode': 'no-overlap',
        'merge_debug': None,
    }

    overlap = (existing_region > 0) & (new_mask > 0)
    if not np.any(overlap):
        return np.maximum(existing_region, new_mask), debug

    existing_hole = existing_region == 1
    new_hole = new_mask == 1
    existing_ring = existing_region == 2
    new_ring = new_mask == 2
    four_connected = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

    hole_related_overlap = (
        np.any(existing_hole & new_hole)
        or np.any(existing_hole & new_ring)
        or np.any(new_hole & existing_ring)
    )
    hole_touching = (
        np.any(binary_dilation(existing_hole, structure=four_connected) & new_hole)
        or np.any(binary_dilation(new_hole, structure=four_connected) & existing_hole)
    )

    if hole_related_overlap or hole_touching:
        debug['mode'] = 'merge-holes'
        merged_hole = existing_hole | new_hole
        existing_center = get_mask_centroid(existing_hole)
        new_center = get_mask_centroid(new_hole)
        if existing_center is not None and new_center is not None:
            midpoint_y = (existing_center[0] + new_center[0]) / 2.0
            midpoint_x = (existing_center[1] + new_center[1]) / 2.0
        elif existing_center is not None:
            midpoint_y, midpoint_x = existing_center
        elif new_center is not None:
            midpoint_y, midpoint_x = new_center
        else:
            return existing_region.copy(), debug

        merged_center = get_nearest_true_pixel(merged_hole, midpoint_y, midpoint_x)
        if merged_center is None:
            return existing_region.copy(), debug

        merged_candidate, merge_debug = build_validated_output_mask(
            filtered,
            merged_hole,
            center_y=merged_center[0],
            center_x=merged_center[1],
        )
        debug['merge_debug'] = merge_debug
        if merged_candidate is not None:
            return merged_candidate, debug
        return existing_region.copy(), debug

    resolved_region = np.maximum(existing_region, new_mask)
    if np.any(existing_ring & new_ring):
        debug['mode'] = 'ring-only-overlap'
        existing_center = get_mask_centroid(existing_hole)
        new_center = get_mask_centroid(new_hole)

        ring_union = resolved_region == 2

        if existing_center is None or new_center is None:
            debug['mode'] = 'ring-only-overlap-missing-seed'
            return resolved_region, debug

        seed_a = get_nearest_true_pixel(ring_union, existing_center[0], existing_center[1])
        seed_b = get_nearest_true_pixel(ring_union, new_center[0], new_center[1])

        if seed_a is None or seed_b is None or seed_a == seed_b:
            debug['mode'] = 'ring-only-overlap-invalid-seed'
            return resolved_region, debug

        markers = np.zeros(resolved_region.shape, dtype=np.int32)
        markers[seed_a] = 1
        markers[seed_b] = 2

        # Watershed from hole-center seeds; watershed_line=True creates explicit 0-valued divider.
        topo = -distance_transform_edt(ring_union)
        ring_labels = watershed(topo, markers=markers, mask=ring_union, watershed_line=True)

        resolved_region[ring_union] = 0
        resolved_region[ring_labels > 0] = 2
    else:
        debug['mode'] = 'generic-overlap'

    return resolved_region, debug


def find_macropinosomes(all_coordinates, img_dask, filter_patch_size=75, min_size=20, max_size=1000,
                        median_filter_size=3, initial_tolerance=0.1, show_plot=False):
    # Create mask with time dimension (T, Y, X)
    frame_shape = img_dask[0].compute().shape
    mask_out = np.zeros((img_dask.shape[0], frame_shape[0], frame_shape[1]), dtype=np.uint8)
    accepted_seed_points_by_frame = [[] for _ in range(img_dask.shape[0])]

    # t, frame_coordinates = 0, all_coordinates[0]
    for t, frame_coordinates in enumerate(all_coordinates):
        frame = np.asarray(img_dask[t].compute())
        # plt.figure(figsize=(12, 6))
        # plt.imshow(frame, cmap='gray')
        # plt.show()
      

        #  _, _, y, x = frame_coordinates[0]
        for _, _, y, x in frame_coordinates:
            y0 = max(0, y - filter_patch_size // 2)
            x0 = max(0, x - filter_patch_size // 2)
            y1 = min(frame.shape[0], y + filter_patch_size // 2 + 1)
            x1 = min(frame.shape[1], x + filter_patch_size // 2 + 1)

            patch = frame[y0:y1, x0:x1]
            # plt.figure(figsize=(6, 6))
            # plt.imshow(patch, cmap='gray')
            # plt.show()

            cy, cx = np.array(patch.shape) // 2

            filtered = median_filter(patch, size=median_filter_size)
            # plt.figure(figsize=(6, 6))
            # plt.imshow(filtered, cmap='gray')
            # plt.show()


            mp_qc = True
            tolerance = float(initial_tolerance) * (float(filtered.max()) - float(filtered.min()))
            output_mask = None
            while mp_qc:
                grown = flood(filtered, (cy, cx), tolerance=tolerance)
                if grown is None:
                    break
                mask = np.asarray(grown, dtype=bool)

                # fill holes in the mask
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
            resolved_region, overlap_info = resolve_overlap_conflict(filtered, existing_region, output_mask)
            if np.any(overlap):
                fig, axes = plt.subplots(2, 3, figsize=(14, 9))
                axes[0, 0].imshow(patch, cmap='gray')
                axes[0, 0].set_title(f"New patch t={t}, y={y}, x={x}")
                axes[0, 1].imshow(frame[y0:y1, x0:x1], cmap='gray')
                axes[0, 1].set_title("Existing global region")
                axes[0, 2].imshow(overlap, cmap='gray')
                axes[0, 2].set_title(f"Overlap pixels\n{overlap_info['mode']}")
                axes[1, 0].imshow(existing_region, vmin=0, vmax=2, cmap='viridis')
                axes[1, 0].set_title("Existing mask")
                axes[1, 1].imshow(output_mask, vmin=0, vmax=2, cmap='viridis')
                axes[1, 1].set_title("New mask")
                axes[1, 2].imshow(resolved_region, vmin=0, vmax=2, cmap='viridis')
                axes[1, 2].set_title("Resolved mask")
                for axis in axes.ravel():
                    axis.axis('off')
                plt.tight_layout()
                plt.show()

            mask_out[t, y0:y1, x0:x1] = resolved_region
            accepted_seed_points_by_frame[t].append([int(t), 0, int(y), int(x)])

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






def process_file(img_path, pred_path, channel=0, min_distance = 10, threshold_abs = 0.1, filter_patch_size=45,
                 search_range=15, min_size=20, max_size=1000, time_search_memory = 3, min_timepoints=5, initial_tolerance=0.1, median_filter_size=3):
    
    print("Loading images...")
    img_input = BioImage(img_path)
    print(f"input shape: {img_input.shape}, dtype: {img_input.dtype}")
    img_pred = BioImage(pred_path)
    print(f"pred shape: {img_pred.shape}, dtype: {img_pred.dtype}")

    img_dask = img_input.get_image_dask_data() # TCZYX
    img_dask = img_dask[:20, channel, :, :, :]# select channel -> TZYX    
    img_dask = img_dask.max(axis=1) # # Max projection along Z -> TYX
    #print(f"input max projection shape: {img_dask.shape}, dtype: {img_dask.dtype}")

    if img_pred.dims.C > 1:
        raise ValueError("Prediction image should have one channel")
    pred_dask = img_pred.get_image_dask_data() # TCZYX
    pred_dask = pred_dask[:20, 0, :, :, :]# select channel -> TZYX    
    pred_dask = pred_dask.max(axis=1) # # Max projection along Z -> TYX
    #print(f"pred max projection shape: {pred_dask.shape}, dtype: {pred_dask.dtype}")


    print("Finding maxima in prediction...")
    all_coordinates = find_maxima(pred_dask, min_distance=min_distance, threshold_abs=threshold_abs)
    # BioImage.show_napari(img_input, coordinates=all_coordinates)

    if all_coordinates is None or len(all_coordinates) == 0 or all(len(frame_coords) == 0 for frame_coords in all_coordinates):
        print("No coordinates found in prediction. Exiting.")
        return

    print("Filtering coordinates by hole score and size ...")
    # Filter coordinates based on hole score in the input image

    mask, filtered_coordinates = find_macropinosomes(all_coordinates, img_dask, filter_patch_size=filter_patch_size, min_size=min_size, max_size=max_size, median_filter_size=median_filter_size, initial_tolerance=initial_tolerance, show_plot=False)
    
    
    #filtered_coordinates, mask = validate_macropinosome(all_coordinates, img_dask, filter_patch_size=filter_patch_size, min_size=min_size, max_size=max_size)
    
    # plt.figure(figsize=(12, 6))
    # plt.imshow(img_dask[0], cmap='gray')
    # plt.show()

    # # Tracking    
    # print("Tracking coordinates across frames...")
    # tracked_coordinates = track_coordinates(filtered_coordinates, search_range=search_range, time_search_memory=time_search_memory, min_timepoints= min_timepoints, )
    
    # # Label mask regions by track ID for visualization
    # print("Labeling tracked regions in mask...")
    # labelled_mask = label_tracks_in_mask(mask, tracked_coordinates)
    # labelled_mask[labelled_mask == 1] = 0 # set untracked to background for visualization


    # show_napari(
    #     img_dask,
    #     labelled_mask,
    #     titles=['Input (ch0, max-Z)', 'Tracked mask'],
    # )

    
    




def main():

        
    # input 
    img_path = "E:/Oyvind/BIP-hub-scratch/train_macropinosome_model/input_drift_corrected/Input_crest__KiaWee__250225_RPE-mNG-Phafin2_BSD_10ul_001_1_drift.ome.tif"
    pred_path = "E:/Oyvind/BIP-hub-scratch/train_macropinosome_model/deep_learning_output/predictions/Input_crest__KiaWee__250225_RPE-mNG-Phafin2_BSD_10ul_001_1_drift_pred.ome.tif"
    
    channel = 0
    min_distance = 10
    threshold_abs = 0.1
    filter_patch_size=45
    search_range=35
    time_search_memory = 2
    min_timepoints = 5
    min_size = 20
    max_size = 2000
    median_filter_size = 3
    initial_tolerance=0.1


    process_file(img_path, pred_path, channel=channel, min_distance=min_distance,
                 threshold_abs=threshold_abs, filter_patch_size=filter_patch_size, search_range=search_range, time_search_memory=time_search_memory, min_timepoints=min_timepoints, min_size=min_size, max_size=max_size, median_filter_size=median_filter_size, initial_tolerance=initial_tolerance)




if __name__ == "__main__":
    main()