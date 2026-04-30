

import sys

from bioio import BioImage
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import flood
from scipy.ndimage import binary_fill_holes, median_filter, binary_dilation
import pandas as pd
from skimage.measure import label, regionprops
from skimage.morphology import reconstruction, disk
import trackpy as tp
import numpy as np
import cv2


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

def estimate_hole_radius(patch, cy, cx, r_min=3, r_max=None):


    h, w = patch.shape
    if r_max is None:
        r_max = min(h, w) // 2

    # smooth to reduce noise
    patch_smooth = cv2.GaussianBlur(patch, (5, 5), 0)

    radii = np.arange(r_min, r_max)
    profile = []

    for r in radii:
        angles = np.linspace(0, 2*np.pi, 180)

        xs = (cx + r * np.cos(angles)).astype(int)
        ys = (cy + r * np.sin(angles)).astype(int)

        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        vals = patch_smooth[ys[valid], xs[valid]]

        # robust to non-circular shape
        profile.append(np.percentile(vals, 70))

    profile = np.array(profile)

    # smooth profile
    profile = cv2.GaussianBlur(profile.reshape(-1,1), (5,1), 0).ravel()

    gradient = np.gradient(profile)

    best_r = radii[np.argmax(gradient)]
    return best_r, profile, gradient


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

def validate_macropinosome(all_coordinates, img_dask, filter_patch_size=75, min_size=20, max_size=1000):
    # Create mask with time dimension (T, Y, X)
    frame_shape = img_dask[0].compute().shape
    mask_out = np.zeros((img_dask.shape[0], frame_shape[0], frame_shape[1]), dtype=bool)
    accepted_seed_points_by_frame = [[] for _ in range(img_dask.shape[0])]

    # t, frame_coordinates = 0, all_coordinates[0]
    for t, frame_coordinates in enumerate(all_coordinates):
        frame = np.asarray(img_dask[t].compute())

        #  _, _, y, x = frame_coordinates[0]
        for _, _, y, x in frame_coordinates:
            y0 = max(0, y - filter_patch_size // 2)
            x0 = max(0, x - filter_patch_size // 2)
            y1 = min(frame.shape[0], y + filter_patch_size // 2 + 1)
            x1 = min(frame.shape[1], x + filter_patch_size // 2 + 1)

            patch = frame[y0:y1, x0:x1]


            mask = get_macropinosome_mask(patch, median_filter_size=3, initial_tolerance=0.1, show_plot=False)




  
    # Deprecaed
    #         radius, profile, gradient = estimate_hole_radius(patch, cy, cx)

    #         yy, xx = np.ogrid[:patch.shape[0], :patch.shape[1]]
    #         dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)

    #         component_mask = dist <= radius
            
    #         import matplotlib.pyplot as plt

    #         plt.figure()
    #         plt.plot(profile, label="intensity")
    #         plt.plot(gradient, label="gradient")
    #         plt.axvline(radius, color='r')
    #         plt.legend()
    #         plt.title(f"Radius = {radius}")
    #         plt.show()

    #         size = int(component_mask.sum())
    #         full_size = filter_patch_size * filter_patch_size

    #         if size < min_size or size > max_size or size == full_size:
    #             print(f"\tTimepoint {t}, coordinate ({y}, {x}): size {size} out of range, skipping")
    #             continue

    #         # Only accepted detections contribute to the output mask.
    #         mask_out[t, y0:y1, x0:x1] |= component_mask
    #         accepted_seed_points_by_frame[t].append((int(y), int(x)))
    #         print(f"Timepoint {t}, coordinate ({y}, {x}): size {size} accepted")

    # filtered_coordinates = []

    # # Merge duplicate detections: if multiple accepted seeds fall in the same
    # # connected component, keep one representative coordinate for that component.
    # for t in range(mask_out.shape[0]):
    #     frame_mask = mask_out[t]
    #     if not frame_mask.any():
    #         continue

    #     component_map = label(frame_mask, connectivity=1)
    #     if component_map.max() == 0:
    #         continue

    #     props_by_label = {prop.label: prop for prop in regionprops(component_map)}
    #     component_to_points: dict[int, list[tuple[int, int]]] = {}
    #     for y, x in accepted_seed_points_by_frame[t]:
    #         component_id = int(component_map[y, x])
    #         if component_id == 0:
    #             continue
    #         component_to_points.setdefault(component_id, []).append((y, x))

    #     for component_id, seed_points in component_to_points.items():
    #         centroid_y, centroid_x = props_by_label[component_id].centroid
    #         representative_y, representative_x = min(
    #             seed_points,
    #             key=lambda pt: (pt[0] - centroid_y) ** 2 + (pt[1] - centroid_x) ** 2,
    #         )
    #         component_size = props_by_label[component_id].area
    #         filtered_coordinates.append([int(t), 0, int(representative_y), int(representative_x), component_size])

    # # Sort by time, then by component size descending (largest holes first within each frame).
    # filtered_coordinates.sort(key=lambda c: (c[0], -c[4]))
    # # Strip the size column — downstream consumers expect [t, z, y, x] only.
    # filtered_coordinates = [[t, z, y, x] for t, z, y, x, _ in filtered_coordinates]

    # return filtered_coordinates, mask_out

# def track_coordinates(coordinates, search_range=15, time_search_memory=3, min_timepoints=5):
            
#     # Convert napari coordinates [t, z, y, x] to trackpy format [frame, x, y]
#     # Note: trackpy uses Cartesian X,Y order (not matrix Y,X)
#     df_data = [[t, x, y] for t, z, y, x in coordinates]
#     df = pd.DataFrame(df_data, columns=['frame', 'x', 'y'])
    
#     if len(df) == 0:
#         print("No coordinates to track")
#         return 
        
#     try:
#         linked_df = tp.link(df, search_range=search_range, memory=time_search_memory)
#         print(f"Tracking complete. Found {linked_df['particle'].max() + 1} unique tracks")
        
#         # Filter out particles that only appear in 1 frame (likely noise)
#         particle_counts = linked_df.groupby('particle').size()
#         long_tracks = particle_counts[particle_counts >= min_timepoints].index
#         linked_df_filtered = linked_df[linked_df['particle'].isin(long_tracks)]
        
#         print(f"After filtering: {len(long_tracks)} tracks with >= {min_timepoints} timepoints")
        
#         return linked_df_filtered
#     except Exception as e:
#         print(f"Tracking failed: {e}")
#         return None

# def label_tracks_in_mask(mask: np.ndarray, tracked_coordinates: pd.DataFrame | None) -> np.ndarray:
#     """Label mask regions by track ID.

#     Label convention:
#     - 0: background
#     - 1: mask regions with no valid multi-frame track assignment
#     - >=2: tracked regions, where label = particle_id + 2

#     Strategy: for each frame, look up the connected-component ID at every track
#     point position in one vectorized array index, build a small LUT, then apply
#     it to the whole frame in a single numpy fancy-index — no per-component loop.
#     """
#     if mask.ndim != 3:
#         raise ValueError(f"Expected mask with shape (T, Y, X), got shape {mask.shape}")

#     mask_bool = mask.astype(bool)
#     # Start with every foreground pixel labeled as untracked (1); background stays 0.
#     labeled_mask = mask_bool.astype(np.uint32)

#     if tracked_coordinates is None or len(tracked_coordinates) == 0:
#         return labeled_mask

#     required_columns = {"frame", "x", "y", "particle"}
#     if not required_columns.issubset(set(tracked_coordinates.columns)):
#         missing = required_columns - set(tracked_coordinates.columns)
#         raise ValueError(f"tracked_coordinates missing required columns: {sorted(missing)}")

#     # Pre-convert track columns to numpy for fast groupby-frame iteration.
#     frames_arr = tracked_coordinates["frame"].to_numpy(dtype=np.int32)
#     ys_arr = np.round(tracked_coordinates["y"].to_numpy(dtype=float)).astype(np.int32)
#     xs_arr = np.round(tracked_coordinates["x"].to_numpy(dtype=float)).astype(np.int32)
#     labels_arr = (tracked_coordinates["particle"].to_numpy(dtype=np.int32) + 2)

#     H, W = mask_bool.shape[1], mask_bool.shape[2]

#     for t in range(mask_bool.shape[0]):
#         frame_mask = mask_bool[t]
#         if not frame_mask.any():
#             continue

#         component_map = label(frame_mask, connectivity=1)
#         n_components = int(component_map.max())
#         if n_components == 0:
#             continue

#         # Select track points belonging to this frame.
#         sel = frames_arr == t
#         if not sel.any():
#             continue  # no tracked points this frame; all foreground stays 1

#         pt_ys = ys_arr[sel]
#         pt_xs = xs_arr[sel]
#         pt_labels = labels_arr[sel]

#         # Clamp coordinates to valid image bounds.
#         pt_ys = np.clip(pt_ys, 0, H - 1)
#         pt_xs = np.clip(pt_xs, 0, W - 1)

#         # Vectorized: look up which component each track point falls in.
#         comp_ids_at_points = component_map[pt_ys, pt_xs]  # shape (n_points,)

#         # Build LUT: component_id -> track_label (default 1 = untracked).
#         lut = np.ones(n_components + 1, dtype=np.uint32)
#         lut[0] = 0  # background component

#         # Process points in order; last writer wins for collisions (rare).
#         # We handle collisions by keeping the label of the nearest point to the
#         # component centroid only when two distinct tracks hit the same component.
#         comp_to_label: dict[int, int] = {}
#         comp_to_collision: dict[int, bool] = {}
#         for comp_id, track_label in zip(comp_ids_at_points.tolist(), pt_labels.tolist()):
#             if comp_id == 0:
#                 continue  # point landed outside mask
#             if comp_id not in comp_to_label:
#                 comp_to_label[comp_id] = int(track_label)
#             elif comp_to_label[comp_id] != int(track_label):
#                 comp_to_collision[comp_id] = True  # two different tracks in one blob

#         # Resolve collisions: pick track closest to component centroid.
#         if comp_to_collision:
#             from skimage.measure import regionprops
#             props = {r.label: r.centroid for r in regionprops(component_map)}
#             for comp_id in comp_to_collision:
#                 cy, cx = props[comp_id]
#                 best_label, best_dist2 = None, None
#                 for py, px, pl in zip(pt_ys.tolist(), pt_xs.tolist(), pt_labels.tolist()):
#                     if component_map[py, px] != comp_id:
#                         continue
#                     d2 = (float(py) - cy) ** 2 + (float(px) - cx) ** 2
#                     if best_dist2 is None or d2 < best_dist2:
#                         best_dist2, best_label = d2, int(pl)
#                 if best_label is not None:
#                     comp_to_label[comp_id] = best_label

#         for comp_id, track_label in comp_to_label.items():
#             lut[comp_id] = track_label

#         # Apply LUT to entire frame in one vectorized operation.
#         labeled_mask[t] = lut[component_map]

#     return labeled_mask

# def get_macropinosome_mask(image: np.ndarray, median_filter_size: int = 3, initial_tolerance: float = 0.1, show_plot: bool = False) -> np.ndarray:
#     '''
#     From an input 2D grayscale image, find the macropinosome hole and ring mask.
#     Typically the input image is a cutout around (including entire ring)

#     The center pixel should be inside the hole

#     returns mask with 0 for background, 1 for hole, 2 for ring.
#     Returns all-zero mask if any QC/final check fails.
#     '''
        
#     def greyscale_fill_holes_2d_cpu(image: np.ndarray) -> np.ndarray:
#         '''
#         Fill holes in a 2D grayscale image using morphological reconstruction.
#         returns a score map where higher values indicate more likely holes.
        
#         '''
#         if image.ndim != 2:
#             raise ValueError(f"Expected 2D image, got shape {image.shape}")

#         if image.size == 0 or np.all(image == image.flat[0]):
#             return image.copy()
        
#         dtype = image.dtype

#         image_max = image.max()
#         inverted = image_max - image

#         seed = inverted.copy()
#         seed[1:-1, 1:-1] = inverted.min()

#         reconstructed = reconstruction(seed, inverted, method='dilation')
#         return reconstructed.astype(dtype)

#     def enforce_4connected_cleanup(mask: np.ndarray, center_y: int, center_x: int, iterations: int = 1) -> np.ndarray:
#         """Remove weak 1px bridges and keep only the center 4-connected component.

#         Bridge-removal rule (4-neighborhood only): remove a center-true pixel when either
#         (up and down are true, left and right are false) or
#         (left and right are true, up and down are false).

#         Then split corner-only links by extracting the connected component containing
#         the center pixel using strict 4-connectivity (up/down/left/right).
#         """
#         out = np.asarray(mask, dtype=bool).copy()
#         if iterations < 1:
#             iterations = 1

#         for _ in range(iterations):
#             padded = np.pad(out, 1, mode='constant', constant_values=False)
#             center = padded[1:-1, 1:-1]
#             up = padded[:-2, 1:-1]
#             down = padded[2:, 1:-1]
#             left = padded[1:-1, :-2]
#             right = padded[1:-1, 2:]

#             vertical_true_horizontal_false = up & down & (~left) & (~right)
#             horizontal_true_vertical_false = left & right & (~up) & (~down)
#             to_remove = center & (vertical_true_horizontal_false | horizontal_true_vertical_false)

#             if not np.any(to_remove):
#                 break
#             out[to_remove] = False

#         if not bool(out[center_y, center_x]):
#             return np.zeros_like(out, dtype=bool)

#         four_connected = np.array(
#             [[0, 1, 0],
#             [1, 1, 1],
#             [0, 1, 0]],
#             dtype=np.uint8,
#         )
#         labels, _ = label(out, structure=four_connected)
#         center_label = int(labels[center_y, center_x])
#         if center_label == 0:
#             return np.zeros_like(out, dtype=bool)

#         return labels == center_label



#     if image.ndim != 2:
#         raise ValueError(f"Expected 2D image, got shape {image.shape}")

#     cy, cx = np.array(image.shape) // 2

#     filtered = median_filter(image, size=median_filter_size)


#     mp_qc = True
#     tolerance = float(initial_tolerance) * (float(filtered.max()) - float(filtered.min()))
#     mask = np.zeros_like(filtered, dtype=bool)
#     touching_edge = True
#     while mp_qc:
#         grown = flood(filtered, (cy, cx), tolerance=tolerance)
#         if grown is None:
#             return np.zeros_like(filtered, dtype=np.uint8)
#         mask = np.asarray(grown, dtype=bool)

#         # fill holes in the mask
#         mask = np.asarray(binary_fill_holes(mask), dtype=bool)
        
#         # Remove thin bridges and split diagonal-only corner links.
#         mask = enforce_4connected_cleanup(mask, center_y=cy, center_x=cx, iterations=2)
#         if not bool(mask[cy, cx]):
#             return np.zeros_like(filtered, dtype=np.uint8)

#         # mask can not touch the edge of patch
#         touching_edge = np.any(mask[0, :]) or np.any(mask[-1, :]) or np.any(mask[:, 0]) or np.any(mask[:, -1])
#         if not touching_edge:
#             mp_qc = False

#         tolerance *= 0.9
    
#     # Build a ring band around the hole with a 5 px width (diffraction-limit constraint).
#     dilated = np.asarray(binary_dilation(mask, disk(5)), dtype=bool)
#     ring_mask = np.asarray(np.logical_and(dilated, np.logical_not(mask)), dtype=bool)

#     has_hole = bool(np.any(mask))
#     has_ring = bool(np.any(ring_mask))
#     if not has_hole or not has_ring:
#         return np.zeros_like(filtered, dtype=np.uint8)

#     ring_brighter_than_hole = float(np.median(filtered[ring_mask])) > float(np.median(filtered[mask]))

#     final_pass = has_hole and has_ring and ring_brighter_than_hole and (not touching_edge)
#     if not final_pass:
#         return np.zeros_like(filtered, dtype=np.uint8)

#     output_mask = np.zeros_like(filtered, dtype=np.uint8)
#     output_mask[mask] = 1
#     output_mask[ring_mask] = 2
#     return output_mask






def process_file(img_path, pred_path, channel=0, min_distance = 10, threshold_abs = 0.1, filter_patch_size=45,
                 search_range=15, min_size=20, max_size=1000, time_search_memory = 3, min_timepoints=5):
    
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

    print("Filtering coordinates by hole score and size ...")
    # Filter coordinates based on hole score in the input image
    filtered_coordinates, mask = validate_macropinosome(all_coordinates, img_dask, filter_patch_size=filter_patch_size, min_size=min_size, max_size=max_size)
  

    # Tracking    
    print("Tracking coordinates across frames...")
    tracked_coordinates = track_coordinates(filtered_coordinates, search_range=search_range, time_search_memory=time_search_memory, min_timepoints= min_timepoints, )
    
    # Label mask regions by track ID for visualization
    print("Labeling tracked regions in mask...")
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
    filter_patch_size=45
    search_range=35
    time_search_memory = 2
    min_timepoints = 5
    min_size = 20
    max_size = 2000



    # process_file(img_path, pred_path, channel=channel, min_distance=min_distance,
    #              threshold_abs=threshold_abs, filter_patch_size=filter_patch_size, search_range=search_range, time_search_memory=time_search_memory, min_timepoints=min_timepoints, min_size=min_size, max_size=max_size)




if __name__ == "__main__":
    main()