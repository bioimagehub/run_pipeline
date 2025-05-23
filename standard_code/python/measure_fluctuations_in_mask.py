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
    object_mask = indexed_mask_2d > 0

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
        pos = roi.get("Roi", {}).get("Positions", {})
        x = int(round(pos.get("x", -1)))
        y = int(round(pos.get("y", -1)))

        if 0 <= x < indexed_mask_2d.shape[1] and 0 <= y < indexed_mask_2d.shape[0]:
            obj_id = indexed_mask_2d[y, x]
            if obj_id > 0:
                roi_by_object[obj_id] = (x, y)  # Record the ROI center
        else:
            print(f"Invalid ROI position: ({x}, {y}). Skipping.")

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



def compute_distance_along_edge_to_point(indexed_mask_2d: np.ndarray, metadata: dict, edge_mask: np.ndarray) -> np.ndarray:
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

def find_nearest_non_zero_value(mask, point):
    # Get the coordinates of non-zero elements in the mask
    non_zero_indices = np.argwhere(mask != 0)
    
    if len(non_zero_indices) == 0:
        raise ValueError("No non-zero values found in the mask.")

    # Calculate the distance from the point to each non-zero point
    distances = np.linalg.norm(non_zero_indices - np.array(point), axis=1)
    print(distances)
    
    # Get the index of the minimum distance
    min_index = np.argmin(distances)
    print(min_index)

    # Minimum distance
    min_distance = distances[min_index]
    
    # Corresponding non-zero value
    closest_value = mask[tuple(non_zero_indices[min_index])]

    return min_distance, closest_value


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
                    print(f"Invalid ROI position: ({x}, {y}). Skipping.")
                    continue

                # Get nucleus ID

                nucleus_id = mask_2d[y, x]

                if nucleus_id == 0:
                    # Loop for the closes pixel with values in
                    distance, value = find_nearest_non_zero_value(mask_2d, (y, x))
                    
                    if distance < 20:
                        nucleus_id = value
                        print(f"Roi not inside mask, using the closest object {distance} pixels away. Id: {nucleus_id}")
                    else:
                        print(f"Roi not inside mask, and is more than too far away {distance} pixels away. Id: {nucleus_id}")


                        
                # Store the object ID back in the ROI
                roi['object_id_in_mask'] = int(nucleus_id)
                # print(f"Stored object_id {nucleus_id} for ROI at ({x}, {y})")

            except Exception as e:
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
    print(yaml.safe_dump(updated_metadata))
    
    with open(os.path.join(output_folder_path, os.path.basename(yaml_path)), 'w') as out_f:
        yaml.safe_dump(metadata, out_f)  # Changed to dump the metadata
    
    distance_to_edge_mask = compute_distance_to_edge(mask_2d)
    edge_mask = np.where(distance_to_edge_mask <= 3, distance_to_edge_mask, np.nan)
    distance_to_point_along_edge = compute_distance_along_edge_to_point(mask_2d, updated_metadata, edge_mask)
    distance_to_point_mask = compute_euclidean_distance_to_point(mask_2d, updated_metadata)

    # Save plots individually or all togheter
    # if 1 != 1: # Change after debuging
    #     # Find the ID of this nucleus in the roi metadata
    #     plot_masks((mask_2d, 'Original Mask t=0'), metadata=updated_metadata, save_image_to_path= output_file_basename + "_mask_orig2d.png")

    #     # Make a distance matrix that tells how far it is to the nearest edge of the object
    #     plot_masks((distance_to_edge_mask, 'Distance to Edge t=0'), metadata=updated_metadata, save_image_to_path= output_file_basename + "_distance_to_edge.png")

    #     plot_masks((edge_mask, 'Edge mask t=0'), metadata=updated_metadata, save_image_to_path = output_file_basename + "_mask_edge.png")

    #     plot_masks((distance_to_point_along_edge, 'Distance along Edge t=0'), metadata=updated_metadata, save_image_to_path= output_file_basename + "_distance_along_edge.png")

    #     plot_masks((distance_to_point_mask, 'Eucledean Distance to point t=0'), metadata=updated_metadata, save_image_to_path= output_file_basename + "_distance_euc_to_edge.png")

    # else:
    #     plot_masks((mask_2d, 'Original Mask t=0'),
    #                (edge_mask, 'Edge mask t=0'),
    #                (distance_to_edge_mask, 'Distance to Edge t=0'),
    #                (distance_to_point_mask, 'Eucledean Distance to point t=0'),
    #                (distance_to_point_along_edge, 'Distance along Edge t=0')
    #                )
    
    # Add this after processing the masks in process_file function
    photoconversion_ch = 2
    df = create_dataframe(mask_2d, distance_to_edge_mask, distance_to_point_mask, distance_to_point_along_edge, img.data[:,photoconversion_ch,0])

    # You can save the DataFrame to a CSV file if desired

    df.to_csv(output_file_basename + "_output.csv", index=False)


def process_folder(args: argparse.Namespace):
    files_to_process = rp.get_files_to_process(args.input_file_or_folder, args.input_file_extension, args.search_subfolders)

    os.makedirs(args.output_folder, exist_ok=True)


    for input_file_path in tqdm(files_to_process, desc="Processing files", unit="file"):
        yaml_path = os.path.splitext(input_file_path)[0] + args.yaml_file_extension
        
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
    parser.add_argument("--input-file-extension", type=str, default= ".nd2", help="Extension of input image")
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





