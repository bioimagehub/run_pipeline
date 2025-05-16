# from roifile import roiread
import numpy as np
import pandas as pd
# from skimage.morphology import skeletonize
# from skimage.measure import label
from scipy.ndimage import distance_transform_edt
import run_pipeline_helper_functions as rp

import yaml
import os
import matplotlib.pyplot as plt


from matplotlib.patches import Ellipse

import numpy as np
from collections import deque
import warnings
# from scipy.ndimage import binary_erosion



def plot_masks(*args, metadata=None):
    """
    Plots a variable number of masks with their corresponding titles and optionally overlays ROI metadata.

    Parameters:
    - args: A sequence of tuples, where each tuple is (mask, title).
    - metadata: Optional dict containing ROI information under ['Image metadata']['ROIs'].
    """
    num_masks = len(args)
    fig, ax = plt.subplots(1, num_masks, figsize=(num_masks * 6, 6))

    if num_masks == 1:
        ax = [ax]  # Make iterable if only one subplot

    for i, (mask, title) in enumerate(args):
        ax[i].imshow(mask, cmap='gray' if i == 0 else 'plasma')
        ax[i].set_title(title)
        ax[i].axis('off')

        if metadata is not None:
            try:
                rois = metadata.get("Image metadata", {}).get("ROIs", [])
                for roi in rois:
                    roi_data = roi.get("Roi", {})
                    pos = roi_data.get("Positions", {})
                    size = roi_data.get("Size", {})

                    x = pos.get("x", None)
                    y = pos.get("y", None)
                    sx = size.get("x", None)
                    sy = size.get("y", None)

                    if None not in (x, y, sx, sy):
                        ellipse = Ellipse(
                            (x, y),
                            width=sx,
                            height=sy,
                            edgecolor='red',
                            facecolor='none',
                            linewidth=2
                        )
                        ax[i].add_patch(ellipse)

            except Exception as e:
                print(f"Error drawing ROIs: {e}")

    # plt.tight_layout()
    plt.show()

def plot_distance_vs_intensity(df: pd.DataFrame, distance_column: str, intensity_column: str):
    # Plot distance vs intensity for each frame
    plt.figure(figsize=(10, 6))
    
    for frame in df['Frame'].unique():
        subset = df[df['Frame'] == frame]
        plt.scatter(subset[distance_column], subset[intensity_column], label=f'Frame {frame}', alpha=0.5)
    
    plt.title(f'Distance vs Intensity: {distance_column}')
    plt.xlabel(distance_column)
    plt.ylabel(intensity_column)
    plt.legend()
    plt.show()

def compute_distance_to_edge(indexed_mask_2d: np.ndarray) -> np.ndarray:

    # Create an output array initialized to the same shape as mask_2d
    distance_mask = np.ones_like(indexed_mask_2d) * np.nan  # Initialize with NaNs for unexplored pixels
    
    obj_ids =  np.unique(indexed_mask_2d)
    obj_ids = obj_ids[obj_ids>0]


    for obj_id in obj_ids:
        # Create a mask for the current object
        object_mask = (indexed_mask_2d == obj_id)

        # Compute the distance transform (distance to the nearest edge)
        dist_transform = distance_transform_edt(object_mask)  # Invert to get distance from background

        # Assign the distance values to the distance mask where the object is
        distance_mask[object_mask] = dist_transform[object_mask]

    return distance_mask

def compute_distance_to_rois(indexed_mask_2d: np.ndarray, metadata: dict) -> np.ndarray:
    # TODO Remove objects that do not have a point
    # Start with a mask full of NaNs
    distance_to_point_mask = np.full(indexed_mask_2d.shape, np.nan)

    # Make a binary mask of the objects
    object_mask = indexed_mask_2d > 0

    # Get the list of ROIs
    rois = metadata.get("Image metadata", {}).get("ROIs", [])

    if not rois:
        return distance_to_point_mask  # No ROIs, return all-NaN

    # Create a mask with zeros where ROI centers are, and inf elsewhere
    roi_centers_mask = np.full(indexed_mask_2d.shape, np.inf)

    for roi in rois:
        try:
            x = int(round(roi["Roi"]["Positions"].get("x", -1)))
            y = int(round(roi["Roi"]["Positions"].get("y", -1)))

            if 0 <= x < indexed_mask_2d.shape[1] and 0 <= y < indexed_mask_2d.shape[0]:
                roi_centers_mask[y, x] = 0  # mark the center of the ROI
            else:
                print(f"Invalid ROI position: ({x}, {y}). Skipping.")

        except (KeyError, TypeError) as e:
            print(f"Error processing ROI: {e}")

    # Compute Euclidean distance to nearest ROI center
    distance_map = distance_transform_edt(roi_centers_mask)

    # Keep distances only inside labeled objects
    distance_to_point_mask[object_mask] = distance_map[object_mask]

    return distance_to_point_mask

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.draw import disk
from collections import deque
import warnings

def compute_distance_along_edge_to_point_mask(indexed_mask_2d: np.ndarray, metadata: dict, edge_mask: np.ndarray) -> np.ndarray:
    output_mask = np.full(indexed_mask_2d.shape, np.nan)
    rois = metadata.get("Image metadata", {}).get("ROIs", [])
    
    if not rois:
        return output_mask

    # Precompute labeled object mask
    object_ids = np.unique(indexed_mask_2d)
    object_ids = object_ids[object_ids > 0]

    # Map object ID to ROI (should be one per object)
    roi_by_object = {}

    for roi in rois:
        pos = roi.get("Roi", {}).get("Positions", {})
        size = roi.get("Roi", {}).get("Size", {})
        x, y = int(round(pos.get("x", -1))), int(round(pos.get("y", -1)))
        r_x, r_y = size.get("x", 0) / 2, size.get("y", 0) / 2
        avg_radius = int(round((r_x + r_y) / 2))

        if x < 0 or y < 0 or y >= indexed_mask_2d.shape[0] or x >= indexed_mask_2d.shape[1]:
            print(f"Invalid ROI position: ({x}, {y}). Skipping.")
            continue

        obj_id = indexed_mask_2d[y, x]
        if obj_id == 0:
            print(f"ROI at ({x}, {y}) not inside an object. Skipping.")
            continue

        if obj_id in roi_by_object:
            warnings.warn(f"Multiple ROIs for object {obj_id} — skipping this object.", category=UserWarning)
            continue

        roi_by_object[obj_id] = {
            "center": (x, y),
            "radius": avg_radius
        }

    for obj_id, roi in roi_by_object.items():
        object_edge = (indexed_mask_2d == obj_id) & ~np.isnan(edge_mask)

        if not np.any(object_edge):
            continue

        x, y = roi["center"]
        radius = roi["radius"]

        # Create circular disk around ROI center
        rr, cc = disk((y, x), radius, shape=indexed_mask_2d.shape)
        roi_mask = np.zeros_like(indexed_mask_2d, dtype=bool)
        roi_mask[rr, cc] = True

        # The edge-pixels of the object that are in contact with the ROI
        start_mask = roi_mask & object_edge
        if not np.any(start_mask):
            print(f"No edge pixels found overlapping ROI for object {obj_id}. Skipping.")
            continue

        distance_map = np.full(indexed_mask_2d.shape, np.nan)
        visited = np.zeros_like(indexed_mask_2d, dtype=bool)

        # Breadth-First Search queue: each entry is (y, x, distance)
        queue = deque([(yy, xx, 0) for yy, xx in zip(*np.where(start_mask))])
        for yy, xx in zip(*np.where(start_mask)):
            visited[yy, xx] = True
            distance_map[yy, xx] = 0

        # 4-connected neighbors
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            y0, x0, dist = queue.popleft()
            for dy, dx in neighbors:
                y1, x1 = y0 + dy, x0 + dx
                if (
                    0 <= y1 < indexed_mask_2d.shape[0] and
                    0 <= x1 < indexed_mask_2d.shape[1] and
                    not visited[y1, x1] and
                    object_edge[y1, x1]
                ):
                    visited[y1, x1] = True
                    distance_map[y1, x1] = dist + 1
                    queue.append((y1, x1, dist + 1))

        # Write result to output
        output_mask[object_edge] = distance_map[object_edge]

    return output_mask



def process_rois(metadata: dict, mask_2d: np.ndarray) -> dict:
    try:
        rois = metadata.get("Image metadata", {}).get("ROIs", [])
        print(f"Found {len(rois)} rois")

        # Check if mask data is accessible
        if mask_2d.size == 0:
            print("Mask data is empty.")
            return metadata

        for roi in rois:
            try:
                # Extract ROI positions safely
                x = int(roi["Roi"]["Positions"].get("x", -1))
                y = int(roi["Roi"]["Positions"].get("y", -1))

                # Check if coordinates are valid
                if x < 0 or y < 0 or y >= mask_2d.shape[0] or x >= mask_2d.shape[1]:
                    print(f"Invalid ROI position: ({x}, {y}). Skipping.")
                    continue

                # Get nucleus ID
                nucleus_id = mask_2d[y, x]

                # Store the object ID back in the ROI
                roi['object_id_in_mask'] = int(nucleus_id)
                # print(f"Stored object_id {nucleus_id} for ROI at ({x}, {y})")

            except (KeyError, IndexError) as e:
                print(f"Error processing ROI: {e}")

    except Exception as e:
        print(f"Error processing ROIs: {e}")

    return metadata

import pandas as pd

def create_dataframe(mask_2d: np.ndarray, distance_to_edge_mask: np.ndarray, 
                     distance_to_point_mask: np.ndarray, 
                     distance_to_point_along_edge: np.ndarray, 
                     img_data_tyx: np.ndarray,) -> pd.DataFrame:
    
    rows = []
    for frame in range(img_data_tyx.shape[0]):
        frame_data = img_data_tyx[frame]
        for y in range(mask_2d.shape[0]):
            for x in range(mask_2d.shape[1]):
                if mask_2d[y, x] > 0:  # Only consider pixels in nucleus
                    row = {
                        "Frame": frame,
                        "Pixel": (y, x),
                        "Object Id": mask_2d[y, x],
                        "Intensity": frame_data[y, x],
                        "Distance to Edge": distance_to_edge_mask[y, x],
                        "Distance to Point": distance_to_point_mask[y, x],
                        "Distance to Point Along Edge": distance_to_point_along_edge[y, x]
                    }
                    rows.append(row)
    
    df = pd.DataFrame(rows)
    return df




def process_file(img_path:str,  yaml_path:str, mask_path:str, output_path:str) -> None:

    # TODO find a way to define this
    mask_initial_frame = 0
    mask_channel = 3   
    mask_z = 0

    img = rp.load_bioio(img_path)
    print(img.shape)
    with open(yaml_path, 'r') as f:
        metadata = yaml.safe_load(f)
    # print(metadata)
    mask = rp.load_bioio(mask_path)
    print(mask.shape)

    # Find the ID of this nucleus in the roi metadata
    mask_2d = mask.data[mask_initial_frame, mask_channel, mask_z, :, :]  # Adjust this slicing as necessary  
    
    updated_metadata = process_rois(metadata, mask_2d)

    # Make a distance matrix that tells how far it is to the nearest edge of the object
    distance_to_edge_mask = compute_distance_to_edge(mask_2d)
    
    edge_mask = np.where(distance_to_edge_mask <= 3, distance_to_edge_mask, np.nan)
    
    distance_to_point_along_edge = compute_distance_along_edge_to_point_mask(mask_2d, metadata, edge_mask)

    distance_to_point_mask = compute_distance_to_rois(mask_2d, metadata)
    
    # Call the plotting function with multiple masks and titles
    # plot_masks(
    # (mask_2d, 'Original Mask'), 
    # (distance_to_edge_mask, 'Distance to Edge'),
    # (distance_to_point_mask, "Distance to Point"),
    # (edge_mask, "Edge"),
    # (distance_to_point_along_edge, "Distance along Edge"),
    # metadata=updated_metadata
    #  )    
    

    # Add this after processing the masks in process_file function
    photoconversion_ch = 2
    df = create_dataframe(mask_2d, distance_to_edge_mask, distance_to_point_mask, distance_to_point_along_edge, img.data[:,photoconversion_ch,0])

    # You can save the DataFrame to a CSV file if desired
    df.to_csv(output_path, index=False)

    # TODO what I want to extract from img_data and save in a pandas dataframe
    # list all the pixels >0 in mask_2d, 
    # This should be done for all frames in img_data. and we need to store the frame info
    # 1) assign the distance to the edge of the nucleus from distance_to_edge_mask
    # 2) assign the distance to the point from the distance_to_point_mask
    # 3) for the pixels that are on the edge (defined in distance_to_point_along_edge) assign the distance to point (trough the cell), the rest should be na or something
    # 4) for the pixels that are on the edge (defined in distance_to_point_along_edge) assign the distance to point (along the edge), the rest should be na or something
    
    # make plots 
    # distance to edge on x axis and intensity on y axis. frame should be color
    # eucledean distance to point on x axis and intensity on y axis. frame should be color
    # distance to point along edge on x axis and intensity on y axis. frame should be color

    # Generate plots for distance vs intensity
    # plot_distance_vs_intensity(df, 'Distance to Edge', 'Intensity')
    # plot_distance_vs_intensity(df, 'Distance to Point', 'Intensity')
    # plot_distance_vs_intensity(df, 'Distance to Point Along Edge', 'Intensity')

img_path = r"Z:\Schink\Oyvind\biphub_user_data\6849908 - IMB - Coen - Sarah - Photoconv\input\LDC20250314_1321N1_BANF1-V5-mEos4b_WT001.nd2"
yaml_path = os.path.splitext(img_path)[0] + "_metadata.yaml"

mask_path = r"Z:\Schink\Oyvind\biphub_user_data\6849908 - IMB - Coen - Sarah - Photoconv\output_nuc_mask\LDC20250314_1321N1_BANF1-V5-mEos4b_WT001.tif"

output_path = r"Z:\Schink\Oyvind\biphub_user_data\6849908 - IMB - Coen - Sarah - Photoconv\output_data.csv"

process_file(img_path=img_path, yaml_path=yaml_path, mask_path=mask_path, output_path=output_path)
