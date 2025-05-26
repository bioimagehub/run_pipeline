import os
import sys
import warnings
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage.draw import disk
from collections import deque
from matplotlib.patches import Ellipse  # Uncomment if needed




import warnings
from collections import deque
from typing import Any, Dict, List, Tuple

import numpy as np
from skimage.draw import disk


import run_pipeline_helper_functions as rp
# from roifile import roiread  # Uncomment if needed
# from skimage.morphology import skeletonize  # Uncomment if needed
# from skimage.measure import label  # Uncomment if needed
# from scipy.ndimage import binary_erosion  # Uncomment if needed
    
def plot_masks(*args, metadata=None, save_image_to_path = None):
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
    # Save the plot if a save folder is provided
    if save_image_to_path:
        fig.savefig(save_image_to_path)
        plt.close(fig)  # Close the figure to free memory
    else:
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

def compute_euclidean_distance_to_point(indexed_mask_2d: np.ndarray, metadata: dict) -> np.ndarray:
    # Initialize a mask full of NaNs
    distance_to_point_mask = np.full(indexed_mask_2d.shape, np.nan)

    # Make a binary mask of the objects
    # object_mask = indexed_mask_2d > 0

    # Get the list of ROIs
    rois = metadata.get("Image metadata", {}).get("ROIs", [])
    
    if not rois:
        return distance_to_point_mask  # No ROIs, return all-NaN

    # Precompute labeled object mask
    object_ids = np.unique(indexed_mask_2d)
    object_ids = object_ids[object_ids > 0]

    # Map object ID to ROI
    roi_by_object = {}

    for roi in rois:
        pos_adjusted = roi.get("Roi", {}).get("Positions in mask", {})        
        pos_from_metadata = roi.get("Roi", {}).get("Positions", {})
        if pos_adjusted:
            pos = pos_adjusted
        elif pos_from_metadata:
            pos = pos_from_metadata
        else:
            print("No valid position found in ROI metadata. Skipping this ROI.")
            continue  # No ROIs, return all-NaN
        
        x = int(round(pos.get("x", -1)))
        y = int(round(pos.get("y", -1)))

        if 0 <= x < indexed_mask_2d.shape[1] and 0 <= y < indexed_mask_2d.shape[0]:
            obj_id = roi.get("object_id_in_mask", -1)  #

            if obj_id > 0:
                roi_by_object[obj_id] = (x, y)  # Record the ROI center
        else:
            print(f"Invalid ROI position outside: ({x}, {y}). Skipping.")
            continue

    if not roi_by_object:
        return distance_to_point_mask  # No valid ROIs for any objects, return all-NaN

    # Compute distances only for objects with ROIs
    for obj_id, (x, y) in roi_by_object.items():
        # Create a mask with zeros where ROI center is
        roi_centers_mask = np.full(indexed_mask_2d.shape, np.inf)
        roi_centers_mask[y, x] = 0  # mark the center of the ROI

        # Compute Euclidean distance to the ROI center
        distance_map = distance_transform_edt(roi_centers_mask)

        # Keep distances only inside the labeled object
        distance_to_point_mask[indexed_mask_2d == obj_id] = distance_map[indexed_mask_2d == obj_id]

    return distance_to_point_mask

import numpy as np
import warnings
from typing import Dict
from collections import deque
from skimage.draw import disk

def compute_distance_along_edge_to_point(
    edge_mask: np.ndarray, 
    metadata: Dict
) -> np.ndarray:
    """
    Computes the shortest distance along the edge mask to each ROI center using BFS.
    Processes all ROIs individually, even if they share object IDs.
    
    Args:
        edge_mask (np.ndarray): 2D binary or boolean edge mask.
        metadata (dict): Metadata dictionary containing ROI info.

    Returns:
        np.ndarray: Distance map where each pixel on the edge contains the shortest
                    path distance to any ROI within its circular neighborhood.
    """
    output_mask: np.ndarray = np.full(edge_mask.shape, np.nan, dtype=np.float32)
    rois: list = metadata.get("Image metadata", {}).get("ROIs", [])

    if not rois:
        return output_mask

    for i, roi in enumerate(rois):
        pos_adjusted = roi.get("Roi", {}).get("Positions in mask", {})        
        pos_from_metadata = roi.get("Roi", {}).get("Positions", {})
        pos = pos_adjusted or pos_from_metadata

        if not pos:
            print(f"ROI {i}: No valid position found in metadata. Skipping.")
            continue

        size = roi.get("Roi", {}).get("Size", {})
        x, y = int(round(pos.get("x", -1))), int(round(pos.get("y", -1)))

        if not (0 <= y < edge_mask.shape[0] and 0 <= x < edge_mask.shape[1]):
            print(f"ROI {i}: Invalid ROI position outside image: ({x}, {y}). Skipping.")
            continue

        if edge_mask[y, x] == 0:
            # ROI is not on the edge — just log and proceed
            print(f"ROI {i}: Position ({x}, {y}) not on edge. Using anyway.")
        
        # Use max(size_x, size_y) to define circular neighborhood
        radius = max(size.get("x", 0), size.get("y", 0))

        # Create ROI mask as disk
        rr, cc = disk((y, x), radius, shape=edge_mask.shape)
        roi_mask: np.ndarray = np.zeros_like(edge_mask, dtype=bool)
        roi_mask[rr, cc] = True

        # Starting points must lie both in the ROI and on the edge
        start_mask: np.ndarray = roi_mask & edge_mask.astype(bool)

        if not np.any(start_mask):
            print(f"ROI {i}: No overlap between ROI disk and edge. Skipping.")
            continue

        distance_map: np.ndarray = np.full(edge_mask.shape, np.nan, dtype=np.float32)
        visited: np.ndarray = np.zeros_like(edge_mask, dtype=bool)

        # Initialize BFS queue with edge pixels inside the ROI
        queue: deque = deque([(yy, xx, 0) for yy, xx in zip(*np.where(start_mask))])
        for yy, xx in zip(*np.where(start_mask)):
            visited[yy, xx] = True
            distance_map[yy, xx] = 0

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-connectivity

        while queue:
            y0, x0, dist = queue.popleft()
            for dy, dx in neighbors:
                y1, x1 = y0 + dy, x0 + dx
                if (
                    0 <= y1 < edge_mask.shape[0] and
                    0 <= x1 < edge_mask.shape[1] and
                    not visited[y1, x1] and
                    edge_mask[y1, x1]
                ):
                    visited[y1, x1] = True
                    distance_map[y1, x1] = dist + 1
                    queue.append((y1, x1, dist + 1))

        # Combine distances with existing output: take minimum per pixel
        edge_pixels = edge_mask.astype(bool)
        mask_valid = edge_pixels & ~np.isnan(distance_map)
        if np.any(mask_valid):
            if np.isnan(output_mask).all():
                output_mask[mask_valid] = distance_map[mask_valid]
            else:
                output_mask[mask_valid] = np.fmin(output_mask[mask_valid], distance_map[mask_valid])

    return output_mask



def find_nearest_non_zero_value(mask, point):
    # Get the coordinates of non-zero elements in the mask
    non_zero_indices = np.argwhere(mask != 0)
    
    if len(non_zero_indices) == 0:
        raise ValueError("No non-zero values found in the mask.")

    # Calculate the distance from the point to each non-zero point
    distances = np.linalg.norm(non_zero_indices - np.array(point), axis=1)

    # Get the index of the minimum distance
    min_index = np.argmin(distances)

    # Minimum distance
    min_distance = int(distances[min_index])  # Convert to Python int
    
    # Corresponding non-zero value
    closest_value = int(mask[tuple(non_zero_indices[min_index])])  # Convert to Python int

    # Get the coordinates of the closest non-zero value
    closest_coordinates = non_zero_indices[min_index].astype(int)  # Ensure it's an integer array
    closest_coordinates = tuple(int(coord) for coord in closest_coordinates)  # Convert to tuple of Python ints

    return min_distance, closest_value, closest_coordinates

def process_rois(metadata: dict, mask_2d: np.ndarray) -> dict:
    try:
        rois = metadata.get("Image metadata", {}).get("ROIs", [])
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
                    print(f"Invalid ROI position (outside image): ({x}, {y}). Skipping.")
                    continue

                # Get nucleus ID
                object_id = mask_2d[y, x]

                if object_id == 0:
                    # Loop for the closest pixel with values in
                    min_distance, new_object_id, closest_coordinates = find_nearest_non_zero_value(mask_2d, (y, x))

                    if min_distance < 20:  # TODO make this a parameter
                        object_id = new_object_id
                        print(f"Roi not inside mask, using the closest object {min_distance} pixels away. Id: {object_id}")

                        # Store the closest coordinates in the roi dictionary with separate x and y keys
                        roi["Roi"]["Positions in mask"] = {
                            "x": closest_coordinates[1],  # Use closest x coordinate
                            "y": closest_coordinates[0]   # Use closest y coordinate
                        }
                        # print(yaml.dump(metadata))  # Print the updated metadata

                    else:
                        print(f"Roi not inside mask, and is more than too far away {min_distance} pixels away. Id: {object_id}")

                # Store the object ID back in the ROI
                roi['object_id_in_mask'] = int(object_id)
                # print(f"Stored object_id {object_id} for ROI at ({x}, {y})")

            except Exception as e:
                print(f"Error processing ROI: {e}")

    except Exception as e:
        print(f"Error processing ROIs: {e}")

    return metadata

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

def process_file(img_path:str, yaml_path:str, mask_path:str, output_folder_path:str, mask_frame:int = 0, mask_channel = 0, mask_zslice = 0) -> None:
    print(img_path)
    output_file_basename = os.path.join(output_folder_path , os.path.splitext(os.path.basename(img_path))[0]) 
    
    img = rp.load_bioio(img_path)

    with open(yaml_path, 'r') as f:
        metadata = yaml.safe_load(f)
    # print(metadata)
    mask = rp.load_bioio(mask_path)

    # Compute distances
    mask_2d = mask.data[mask_frame, mask_channel, mask_zslice, :, :]  # Adjust this slicing as necessary  
    updated_metadata = process_rois(metadata, mask_2d)

    distance_to_edge_mask = compute_distance_to_edge(mask_2d)
    edge_mask = np.where(distance_to_edge_mask <= 3, distance_to_edge_mask, 0)
    distance_to_point_along_edge = compute_distance_along_edge_to_point(edge_mask, updated_metadata)
    distance_to_point_mask = compute_euclidean_distance_to_point(mask_2d, updated_metadata)

    # Save plots individually or all togheter
    if 1 != 1: # Change after debuging
        
        # Find the ID of this nucleus in the roi metadata
        plot_masks((mask_2d, 'Original Mask t=0'), metadata=updated_metadata, save_image_to_path= output_file_basename + "_mask_orig2d.png")

        # Make a distance matrix that tells how far it is to the nearest edge of the object
        plot_masks((distance_to_edge_mask, 'Distance to Edge t=0'), metadata=updated_metadata, save_image_to_path= output_file_basename + "_distance_to_edge.png")

        plot_masks((edge_mask, 'Edge mask t=0'), metadata=updated_metadata, save_image_to_path = output_file_basename + "_mask_edge.png")

        plot_masks((distance_to_point_along_edge, 'Distance along Edge t=0'), metadata=updated_metadata, save_image_to_path= output_file_basename + "_distance_along_edge.png")

        plot_masks((distance_to_point_mask, 'Eucledean Distance to point t=0'), metadata=updated_metadata, save_image_to_path= output_file_basename + "_distance_euc_to_edge.png")

    else:
        plot_masks((mask_2d, 'Original Mask t=0'),
                   (edge_mask, 'Edge mask t=0'),
                   (distance_to_edge_mask, 'Distance to Edge t=0'),
                   (distance_to_point_mask, 'Eucledean Distance to point t=0'),
                   (distance_to_point_along_edge, 'Distance along Edge t=0'),
                    metadata=updated_metadata,
                    save_image_to_path= output_file_basename + "_all_masks.png"
                   )
    
    # Add this after processing the masks in process_file function
    photoconversion_ch = 2
    df = create_dataframe(mask_2d, distance_to_edge_mask, distance_to_point_mask, distance_to_point_along_edge, img.data[:,photoconversion_ch,0])

    # You can save the DataFrame to a CSV file if desired

    df.to_csv(output_file_basename + "_output.csv", index=False)

    # Save the updated metadata if no errors occurred
    with open(os.path.join(output_folder_path, os.path.basename(yaml_path)), 'w') as out_f:
        yaml.safe_dump(metadata, out_f)  # Changed to dump the metadata
  
def process_folder(args: argparse.Namespace):
    files_to_process = rp.get_files_to_process(args.input_file_or_folder, args.input_file_extension, args.search_subfolders)

    os.makedirs(args.output_folder, exist_ok=True)


    #TODO REmove this



    for input_file_path in tqdm(files_to_process, desc="Processing files", unit="file"):
        yaml_path = os.path.splitext(input_file_path)[0] + args.yaml_file_extension
        # yaml_path_out = os.path.join(args.output_folder, os.path.basename(yaml_path))
        # if os.path.isfile(yaml_path_out):
        #     print(f"YAML file already exists, ski: {yaml_path_out}")
        #     continue
            
        mask_path = os.path.join(args.input_mask_or_folder, os.path.splitext(os.path.basename(input_file_path))[0] + args.mask_file_extension)


        process_file(img_path = input_file_path,
                     yaml_path = yaml_path,
                     mask_path = mask_path,
                     output_folder_path = args.output_folder,
                     mask_frame = args.mask_frame,
                     mask_channel = args.mask_channel,
                     mask_zslice = args.mask_zslice)

def main(parsed_args: argparse.Namespace):
    if os.path.isfile(parsed_args.input_file_or_folder) and os.path.isfile(parsed_args.input_mask_or_folder):
        print(f"Processing single file: {parsed_args.input_file_or_folder}")
        input_file_path = parsed_args.input_file_or_folder
        yaml_path = os.path.splitext(input_file_path)[0] + parsed_args.yaml_file_extension
        mask_path = os.path.join(parsed_args.input_mask_or_folder, os.path.splitext(os.path.basename(input_file_path))[0] + parsed_args.mask_file_extension)
        process_file(img_path = input_file_path,
                     yaml_path = yaml_path,
                     mask_path = mask_path,
                     output_folder_path = parsed_args.output_folder,
                     mask_frame = parsed_args.mask_frame,
                     mask_channel = parsed_args.mask_channel,
                     mask_zslice = parsed_args.mask_zslice)
        
    elif os.path.isdir(parsed_args.input_file_or_folder) and os.path.isdir(parsed_args.input_mask_or_folder):
        print(f"Processing folder: {parsed_args.input_file_or_folder}")
        process_folder(parsed_args)
    else:
        print("Error: The specified path is neither a file nor a folder. or one file and one folder was parsed ")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a folder (or file) of images and convert them to TIF format with optional drift correction.")
    parser.add_argument("--input-file-or-folder", type=str, help="Path to the file or folder to be processed.")
    parser.add_argument("--input-file-extension", type=str, default= ".tif", help="Extension of input image")
    parser.add_argument("--search-subfolders", action="store_true", help="Include subfolders in the search for files (Only applicable for directory input)")
    parser.add_argument("--yaml-file-extension", type=str, default= "_metadata.yaml", help="Extension relative to basename of input image name")

    parser.add_argument("--input-mask-or-folder", type=str, help="Path to the indexed mask or folder with indexed masks to be processed. (must be file if file, and folder if folder in --input-file-or-folder)")
    parser.add_argument("--mask-file-extension", type=str, default= ".tif", help="Extension relative to basename of input image")
    parser.add_argument("--measure-channel", type=int, default = 0, help="Zero based index, Which channel should be measured in the input file(s)")
    parser.add_argument("--output-folder", default= "output_fluctuations", type=str, help="Path to the file or folder to be processed.")

    parser.add_argument("--mask-frame", type=int, default = 0, help="Zero based index, defining which frame to find the indexed mask in")
    parser.add_argument("--mask-channel", type=int, default = 0, help="Zero based index, defining which channel to find the indexed mask in")
    parser.add_argument("--mask-zslice", type=int, default = 0, help="Zero based index, defining which zslice to find the indexed mask in")

    parsed_args = parser.parse_args()

    main(parsed_args)





