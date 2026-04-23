

import sys

from bioio import BioImage
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import flood
from scipy.ndimage import binary_fill_holes, median_filter
import pandas as pd
from skimage.measure import label, regionprops
from skimage.morphology import reconstruction
import trackpy as tp


def _show_napari(self, dims="TCZYX", coordinates=None):
    import napari

    # Get lazy data (BioImage already returns dask if possible)
    data = self.get_image_data(dims)

    viewer = napari.Viewer()

    kwargs = {}

    # Handle channel axis explicitly
    if "C" in dims:
        kwargs["channel_axis"] = dims.index("C")

    viewer.add_image(data, **kwargs)

    if coordinates is not None:
        # Flatten nested list-of-lists (e.g. grouped by timepoint) to a flat list of points.
        if coordinates and isinstance(coordinates[0], list) and isinstance(coordinates[0][0], list):
            coordinates = [pt for pts in coordinates for pt in pts]
        viewer.add_points(
            coordinates,
            size=8,
            face_color='red',
            name='Points',
        )

    napari.run()

BioImage.show_napari = _show_napari

def show_napari(*images, titles=None, coordinates=None):
    """Show one or more images in napari.

    Each image can be a numpy array, a dask array, or a BioImage object.
    BioImage objects are displayed with a channel axis so each channel
    becomes a separate layer.  Plain arrays are shown as-is.

    Parameters
    ----------
    *images:
        One or more images to display.
    titles:
        Optional list of layer name strings, one per image.  Falls back to
        'Image 0', 'Image 1', … when not provided.
    coordinates:
        Optional list of napari-format points [T, Z, Y, X] to overlay.
    """
    import napari
    import dask.array as da

    if titles is None:
        titles = [f"Image {i}" for i in range(len(images))]

    viewer = napari.Viewer()

    for img, title in zip(images, titles):
        if isinstance(img, BioImage):
            data = img.get_image_dask_data("TCZYX")
            viewer.add_image(data, channel_axis=1, name=title)
        elif isinstance(img, (np.ndarray, da.Array)):
            viewer.add_image(img, name=title)
        else:
            # Last-resort: try to wrap as numpy
            viewer.add_image(np.asarray(img), name=title)

    if coordinates is not None:
        if coordinates and isinstance(coordinates[0], list) and isinstance(coordinates[0][0], list):
            coordinates = [pt for pts in coordinates for pt in pts]
        viewer.add_points(
            coordinates,
            size=8,
            face_color='red',
            name='Points',
        )
    napari.run()

def greyscale_fill_holes_2d_cpu(image: np.ndarray) -> np.ndarray:
    '''
    Fill holes in a 2D grayscale image using morphological reconstruction.
    returns a score map where higher values indicate more likely holes.
    
    '''
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")

    if image.size == 0 or np.all(image == image.flat[0]):
        return image.copy()

    image_max = image.max()
    inverted = image_max - image

    seed = inverted.copy()
    seed[1:-1, 1:-1] = inverted.min()

    reconstructed = reconstruction(seed, inverted, method='dilation')
    return reconstructed

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

def filter_coordinates_by_hole_score(all_coordinates, img_dask, filter_patch_size=15, min_size=20, max_size=1000):
    # Create mask with time dimension (T, Y, X)
    frame_shape = img_dask[0].compute().shape
    mask_out = np.zeros((img_dask.shape[0], frame_shape[0], frame_shape[1]), dtype=bool)
    accepted_seed_points_by_frame = [[] for _ in range(img_dask.shape[0])]

    for t, frame_coordinates in enumerate(all_coordinates):
        frame = np.asarray(img_dask[t].compute())

        for _, _, y, x in frame_coordinates:
            y0 = max(0, y - filter_patch_size // 2)
            x0 = max(0, x - filter_patch_size // 2)
            y1 = min(frame.shape[0], y + filter_patch_size // 2 + 1)
            x1 = min(frame.shape[1], x + filter_patch_size // 2 + 1)

            patch = frame[y0:y1, x0:x1]

            # Median filtering reduces local noise before hole filling.
            patch = median_filter(patch, size=2)
            filled = greyscale_fill_holes_2d_cpu(patch)

            # Local seed coordinate in patch, robust for border-clipped patches.
            cy = int(y - y0)
            cx = int(x - x0)

            component_mask = flood(filled, (cy, cx), tolerance=0, connectivity=1)
            component_mask = binary_fill_holes(component_mask)

            size = int(component_mask.sum())
            full_size = filter_patch_size * filter_patch_size

            if size < min_size or size > max_size or size == full_size:
                print(f"\tTimepoint {t}, coordinate ({y}, {x}): size {size} out of range, skipping")
                continue

            # Only accepted detections contribute to the output mask.
            mask_out[t, y0:y1, x0:x1] |= component_mask
            accepted_seed_points_by_frame[t].append((int(y), int(x)))
            print(f"Timepoint {t}, coordinate ({y}, {x}): size {size} accepted")

    filtered_coordinates = []

    # Merge duplicate detections: if multiple accepted seeds fall in the same
    # connected component, keep one representative coordinate for that component.
    for t in range(mask_out.shape[0]):
        frame_mask = mask_out[t]
        if not frame_mask.any():
            continue

        component_map = label(frame_mask, connectivity=1)
        if component_map.max() == 0:
            continue

        props_by_label = {prop.label: prop for prop in regionprops(component_map)}
        component_to_points: dict[int, list[tuple[int, int]]] = {}
        for y, x in accepted_seed_points_by_frame[t]:
            component_id = int(component_map[y, x])
            if component_id == 0:
                continue
            component_to_points.setdefault(component_id, []).append((y, x))

        for component_id, seed_points in component_to_points.items():
            centroid_y, centroid_x = props_by_label[component_id].centroid
            representative_y, representative_x = min(
                seed_points,
                key=lambda pt: (pt[0] - centroid_y) ** 2 + (pt[1] - centroid_x) ** 2,
            )
            component_size = props_by_label[component_id].area
            filtered_coordinates.append([int(t), 0, int(representative_y), int(representative_x), component_size])

    # Sort by time, then by component size descending (largest holes first within each frame).
    filtered_coordinates.sort(key=lambda c: (c[0], -c[4]))
    # Strip the size column — downstream consumers expect [t, z, y, x] only.
    filtered_coordinates = [[t, z, y, x] for t, z, y, x, _ in filtered_coordinates]

    return filtered_coordinates, mask_out

def track_coordinates(coordinates, search_range=15, time_search_memory=3, min_timepoints=5):
            
    # Convert napari coordinates [t, z, y, x] to trackpy format [frame, x, y]
    # Note: trackpy uses Cartesian X,Y order (not matrix Y,X)
    df_data = [[t, x, y] for t, z, y, x in coordinates]
    df = pd.DataFrame(df_data, columns=['frame', 'x', 'y'])
    
    if len(df) == 0:
        print("No coordinates to track")
        return 
        
    try:
        linked_df = tp.link(df, search_range=search_range, memory=time_search_memory)
        print(f"Tracking complete. Found {linked_df['particle'].max() + 1} unique tracks")
        
        # Filter out particles that only appear in 1 frame (likely noise)
        particle_counts = linked_df.groupby('particle').size()
        long_tracks = particle_counts[particle_counts >= min_timepoints].index
        linked_df_filtered = linked_df[linked_df['particle'].isin(long_tracks)]
        
        print(f"After filtering: {len(long_tracks)} tracks with >= {min_timepoints} timepoints")
        
        return linked_df_filtered
    except Exception as e:
        print(f"Tracking failed: {e}")
        return None

def label_tracks_in_mask(mask: np.ndarray, tracked_coordinates: pd.DataFrame | None) -> np.ndarray:
    """Label mask regions by track ID.

    Label convention:
    - 0: background
    - 1: mask regions with no valid multi-frame track assignment
    - >=2: tracked regions, where label = particle_id + 2

    Strategy: for each frame, look up the connected-component ID at every track
    point position in one vectorized array index, build a small LUT, then apply
    it to the whole frame in a single numpy fancy-index — no per-component loop.
    """
    if mask.ndim != 3:
        raise ValueError(f"Expected mask with shape (T, Y, X), got shape {mask.shape}")

    mask_bool = mask.astype(bool)
    # Start with every foreground pixel labeled as untracked (1); background stays 0.
    labeled_mask = mask_bool.astype(np.uint32)

    if tracked_coordinates is None or len(tracked_coordinates) == 0:
        return labeled_mask

    required_columns = {"frame", "x", "y", "particle"}
    if not required_columns.issubset(set(tracked_coordinates.columns)):
        missing = required_columns - set(tracked_coordinates.columns)
        raise ValueError(f"tracked_coordinates missing required columns: {sorted(missing)}")

    # Pre-convert track columns to numpy for fast groupby-frame iteration.
    frames_arr = tracked_coordinates["frame"].to_numpy(dtype=np.int32)
    ys_arr = np.round(tracked_coordinates["y"].to_numpy(dtype=float)).astype(np.int32)
    xs_arr = np.round(tracked_coordinates["x"].to_numpy(dtype=float)).astype(np.int32)
    labels_arr = (tracked_coordinates["particle"].to_numpy(dtype=np.int32) + 2)

    H, W = mask_bool.shape[1], mask_bool.shape[2]

    for t in range(mask_bool.shape[0]):
        frame_mask = mask_bool[t]
        if not frame_mask.any():
            continue

        component_map = label(frame_mask, connectivity=1)
        n_components = int(component_map.max())
        if n_components == 0:
            continue

        # Select track points belonging to this frame.
        sel = frames_arr == t
        if not sel.any():
            continue  # no tracked points this frame; all foreground stays 1

        pt_ys = ys_arr[sel]
        pt_xs = xs_arr[sel]
        pt_labels = labels_arr[sel]

        # Clamp coordinates to valid image bounds.
        pt_ys = np.clip(pt_ys, 0, H - 1)
        pt_xs = np.clip(pt_xs, 0, W - 1)

        # Vectorized: look up which component each track point falls in.
        comp_ids_at_points = component_map[pt_ys, pt_xs]  # shape (n_points,)

        # Build LUT: component_id -> track_label (default 1 = untracked).
        lut = np.ones(n_components + 1, dtype=np.uint32)
        lut[0] = 0  # background component

        # Process points in order; last writer wins for collisions (rare).
        # We handle collisions by keeping the label of the nearest point to the
        # component centroid only when two distinct tracks hit the same component.
        comp_to_label: dict[int, int] = {}
        comp_to_collision: dict[int, bool] = {}
        for comp_id, track_label in zip(comp_ids_at_points.tolist(), pt_labels.tolist()):
            if comp_id == 0:
                continue  # point landed outside mask
            if comp_id not in comp_to_label:
                comp_to_label[comp_id] = int(track_label)
            elif comp_to_label[comp_id] != int(track_label):
                comp_to_collision[comp_id] = True  # two different tracks in one blob

        # Resolve collisions: pick track closest to component centroid.
        if comp_to_collision:
            from skimage.measure import regionprops
            props = {r.label: r.centroid for r in regionprops(component_map)}
            for comp_id in comp_to_collision:
                cy, cx = props[comp_id]
                best_label, best_dist2 = None, None
                for py, px, pl in zip(pt_ys.tolist(), pt_xs.tolist(), pt_labels.tolist()):
                    if component_map[py, px] != comp_id:
                        continue
                    d2 = (float(py) - cy) ** 2 + (float(px) - cx) ** 2
                    if best_dist2 is None or d2 < best_dist2:
                        best_dist2, best_label = d2, int(pl)
                if best_label is not None:
                    comp_to_label[comp_id] = best_label

        for comp_id, track_label in comp_to_label.items():
            lut[comp_id] = track_label

        # Apply LUT to entire frame in one vectorized operation.
        labeled_mask[t] = lut[component_map]

    return labeled_mask
    
def process_file(img_path, pred_path, channel=0, min_distance = 10, threshold_abs = 0.1, search_range=15, min_size=20, max_size=1000, time_search_memory = 3, min_timepoints=5):

    img_input = BioImage(img_path)
    print(f"input shape: {img_input.shape}, dtype: {img_input.dtype}")
    img_pred = BioImage(pred_path)
    print(f"pred shape: {img_pred.shape}, dtype: {img_pred.dtype}")

    img_dask = img_input.get_image_dask_data() # TCZYX
    img_dask = img_dask[:, channel, :, :, :]# select channel -> TZYX    
    img_dask = img_dask.max(axis=1) # # Max projection along Z -> TYX
    #print(f"input max projection shape: {img_dask.shape}, dtype: {img_dask.dtype}")

    if img_pred.dims.C > 1:
        raise ValueError("Prediction image should have one channel")
    pred_dask = img_pred.get_image_dask_data() # TCZYX
    pred_dask = pred_dask[:, 0, :, :, :]# select channel -> TZYX    
    pred_dask = pred_dask.max(axis=1) # # Max projection along Z -> TYX
    #print(f"pred max projection shape: {pred_dask.shape}, dtype: {pred_dask.dtype}")

    all_coordinates = find_maxima(pred_dask, min_distance=min_distance, threshold_abs=threshold_abs)
    # BioImage.show_napari(img_input, coordinates=all_coordinates)

    # Filter coordinates based on hole score in the input image
    filtered_coordinates, mask = filter_coordinates_by_hole_score(all_coordinates, img_dask, filter_patch_size=search_range, min_size=min_size, max_size=max_size)
  

    # Tracking    
    tracked_coordinates = track_coordinates(filtered_coordinates, search_range=search_range, time_search_memory=time_search_memory, min_timepoints= min_timepoints, )
    labelled_mask = label_tracks_in_mask(mask, tracked_coordinates)
    labelled_mask[labelled_mask == 1] = 0 # set untracked to background for visualization


    show_napari(
        img_dask,
        labelled_mask,
        titles=['Input (ch0, max-Z)', 'Tracked mask'],
    )

    
    




def main():

        
    # input 
    img_path = "E:/Oyvind/BIP-hub-scratch/train_macropinosome_model/input_drift_corrected/Input_crest__KiaWee__250225_RPE-mNG-Phafin2_BSD_10ul_001_1_drift.ome.tif"
    pred_path = "E:/Oyvind/BIP-hub-scratch/train_macropinosome_model/deep_learning_output/predictions/Input_crest__KiaWee__250225_RPE-mNG-Phafin2_BSD_10ul_001_1_drift_pred.ome.tif"
    
    channel = 0
    min_distance = 10
    threshold_abs = 0.1
    search_range=35
    time_search_memory = 2
    min_timepoints = 5
    min_size = 20
    max_size = 2000



    process_file(img_path, pred_path, channel=channel, min_distance=min_distance,
                 threshold_abs=threshold_abs, search_range=search_range, time_search_memory=time_search_memory, min_timepoints=min_timepoints, min_size=min_size, max_size=max_size)




if __name__ == "__main__":
    main()