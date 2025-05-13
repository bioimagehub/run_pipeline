# from roifile import roiread
import os
import numpy as np
from skimage.measure import label
from skimage import measure, morphology
# from skimage.draw import polygon  # For rasterizing the polygons

from scipy.ndimage import center_of_mass
from scipy.spatial import ConvexHull

import numpy as np
import pandas as pd

import run_pipeline_helper_functions as rp

def imagej_roi_to_indexed_mask(mask_path, shape):
    raise NotImplementedError
    # rois = roiread(mask_path)
    # num_rois = len(rois)
    # indexed_mask = np.zeros((num_rois, *shape), dtype=np.int32)

    # for idx, roi in enumerate(rois):

    #     '''from roifile.py
    #         POLYGON = 0
    #         RECT = 1
    #         OVAL = 2
    #         LINE = 3
    #         FREELINE = 4
    #         POLYLINE = 5
    #         NOROI = 6
    #         FREEHAND = 7
    #         TRACED = 8
    #         ANGLE = 9
    #         POINT = 10        
    #     '''
        
    #     if roi.roitype == 7:
    #         # Assuming roi has an attribute 'integer_coordinates' or similar for freehand
    #         coord = roi.integer_coordinates
    #         rr, cc = coord[:, 0], coord[:, 1]  # Get x and y coordinates
            
    #         # Ensure coordinates are within bounds
    #         rr, cc = np.clip(rr, 0, shape[0] - 1), np.clip(cc, 0, shape[1] - 1)

    #         # Use polygon function for filling the region
    #         rr_poly, cc_poly = polygon(rr, cc, shape)
    #         indexed_mask[idx, rr_poly, cc_poly] = idx + 1  # +1 to avoid zero indexing

    #     elif roi.roitype == "ROI_TYPE.POLYGON":
    #         raise NotImplementedError("This has not been tested")
    #         # If there are polygons, handle them similarly to freehand
    #         # for polygon in roi.polygons:  # Ensure this is correctly defined in your roi object
    #         #     rr, cc = polygon.integer_coordinates[:, 0], polygon.integer_coordinates[:, 1]
    #         #     rr, cc = np.clip(rr, 0, shape[0] - 1), np.clip(cc, 0, shape[1] - 1)

    #         #     rr_poly, cc_poly = polygon(rr, cc, shape)
    #         #     indexed_mask[idx, rr_poly, cc_poly] = idx + 1
    #     else:
    #         print(f"ROI type '{roi.roitype}' not handled.")

    # # Combine into a single 2D mask
    # final_mask = np.max(indexed_mask, axis=0)  # Get the maximum value per pixel
    # return final_mask.astype(np.int32)  # Convert final mask to int32

# Note: Make sure to replace `roiread` with your actual function to read the ROIs

def compute_volume(mask: np.ndarray) -> int:
    """
    Compute the volume (number of foreground voxels) of the binary mask.
    
    Args:
        mask (np.ndarray): 3D binary numpy array.
    
    Returns:
        int: Volume (number of voxels).
    """
    return np.sum(mask)

def compute_bounding_box(mask: np.ndarray) -> tuple:
    """
    Get the bounding box of the binary mask.
    
    Args:
        mask (np.ndarray): 3D binary numpy array.
    
    Returns:
        tuple: Min and max coordinates (min_z, min_y, min_x, max_z, max_y, max_x).
    """
    coord = np.argwhere(mask)
    min_coords = coord.min(axis=0)
    max_coords = coord.max(axis=0)
    return (*min_coords, *max_coords)

def compute_principal_axes(mask: np.ndarray) -> tuple:
    """
    Compute the principal axes using eigenvalues and eigenvectors.
    
    Args:
        mask (np.ndarray): 3D binary numpy array.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Eigenvalues and eigenvectors.
    """
    coords = np.argwhere(mask)
    cov_matrix = np.cov(coords.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]

def get_masks(image_basename:str, mask_folders:str, mask_suffixes:str, mask_names:str = ""):
    # TODO Check that dtype is handled
    # TODO Check that diferent shapes    
    
    if not len(mask_folders) == len(mask_suffixes):
        raise ValueError("mask_folders, mask_suffixes and mask_names must be of equal length")
    if mask_names == "":
        mask_names = [""] * len(mask_folders)


    # Normalize inputs to lists if they are not already
    if isinstance(mask_folders, str):
        mask_folders = [mask_folders]
    if isinstance(mask_suffixes, str):
        mask_suffixes = [mask_suffixes]
    if isinstance(mask_names, str):
        mask_names = [mask_names]
    
    # Loop over all masks
    mask_channels  = [] #T_ZYX  where each list is a C
    mask_info = []
    for mask_folder, mask_suffix, mask_name in zip(mask_folders, mask_suffixes, mask_names):
        mask_path = os.path.join(mask_folder, os.path.splitext(image_basename)[0] + mask_suffix) 
        mask = rp.load_bioio(mask_path).data # Add this to masks as a new channel
        for ch in range(mask.shape[1]):
            mask_channels.append(mask[:,ch,:,:,:])
            mask_info.append([mask_folder, mask_suffix, mask_name])
    
    first_mask_shape = mask_channels[0].shape
    total_channels = len(mask_channels)
    combined_masks = np.zeros((first_mask_shape[0], total_channels, first_mask_shape[1], first_mask_shape[2], first_mask_shape[3]), dtype=mask_channels[0].dtype)

    for idx, single_mask in enumerate(mask_channels):
        combined_masks[:, idx, :, :, :] = single_mask  # Place TZYX into C dimension
        
        
    return combined_masks, mask_info  # Now returns masks of shape (T, total_channels, Z, Y, X)

        
def measure_masks(img_path, masks, mask_infolist):
    # measure on original image
    img = rp.load_bioio(img_path) #TCZYX
    
    T, C,_, _, _ = masks.shape

    if len(mask_infolist) != C:
        raise ValueError(f"Wrong dimensions beween len(mask_infolist) != {len(mask_infolist)}, {C}")
               
    # Ensure that mask is a indexed binary file
    if not is_indexed_binary(masks):
        mask_indexed = label_5d_mask(masks)
        print(np.unique(masks))
    else:
        mask_indexed = masks

    # process mask
    

    results = []    
    for t in range(T):
        for c in range(C):
            mask_indexed_slice = mask_indexed[t,c] # ZYX numpy array
            mask_values = np.unique(mask_indexed_slice)
            mask_info = mask_infolist[c]
            
            # loop over each 3D object
            for mask_id in mask_values[mask_values > 0]:  # Filter out non-positive values               
                mask_binary_slice = (mask_indexed_slice == mask_id)
                
                # Measurements on binary mask
                mask_size_pix = np.sum(mask_binary_slice)
                min_z, min_y, min_x, max_z, max_y, max_x = compute_bounding_box(mask_binary_slice)
                center_of_mass_z, center_of_mass_y, center_of_mass_x  = center_of_mass(mask_binary_slice)
                eigenvalues, eigenvectors = compute_principal_axes(mask_binary_slice)
                eigenvalues_1, eigenvalues_2, eigenvalues_3 = eigenvalues 
                results_tmp = {
                    'mask_folder': mask_info[0],
                    'mask_suffix': mask_info[1],
                    'mask_name':  mask_info[2],
                    'frame': t,
                    'channel': c,
                    'mask_id': mask_id,
                    'mask_size': mask_size_pix,
                    'bounding_box_min_z': min_z, 
                    'bounding_box_min_y': min_y,
                    'bounding_box_min_x': min_x,
                    'bounding_box_max_z': max_z,
                    'bounding_box_max_y': max_y,
                    'bounding_box_max_x': max_x,
                    'center_of_mass_z': center_of_mass_z,
                    'center_of_mass_y': center_of_mass_y,
                    'center_of_mass_x': center_of_mass_x,
                    'eigenvalues_1': eigenvalues_1,
                    'eigenvalues_2': eigenvalues_2,
                    'eigenvalues_3': eigenvalues_3,
                    'eigenvectors': ''.join(['(' + ','.join(map(str, ev)) + ')' for ev in eigenvectors]),

                    # Image channel statistics initialized to NaN
                    **{f'img_ch_{img_c}_min': np.nan for img_c in range(img.dims.C)},
                    **{f'img_ch_{img_c}_mean': np.nan for img_c in range(img.dims.C)},
                    **{f'img_ch_{img_c}_median': np.nan for img_c in range(img.dims.C)},
                    **{f'img_ch_{img_c}_max': np.nan for img_c in range(img.dims.C)},
                    **{f'img_ch_{img_c}_std': np.nan for img_c in range(img.dims.C)},
                    **{f'img_ch_{img_c}_var': np.nan for img_c in range(img.dims.C)},
                    **{f'img_ch_{img_c}_sum': np.nan for img_c in range(img.dims.C)},
                }
               

                # match img T with T from mask
                # Loop over all Cs in image for each C in mask
                # So that for example nuc in ch 3 in mask will measure values in all channels in input img.
                for img_c in range(img.dims.C):
                    values = img.data[t, img_c][mask_binary_slice] # -> ZYX
                    if values.size > 0:
                        results_tmp[f'img_ch_{img_c}_min'] = np.min(values)
                        results_tmp[f'img_ch_{img_c}_mean'] = np.mean(values)
                        results_tmp[f'img_ch_{img_c}_median'] = np.median(values)                    
                        results_tmp[f'img_ch_{img_c}_max'] = np.max(values)
                        results_tmp[f'img_ch_{img_c}_std'] = np.std(values)
                        results_tmp[f'img_ch_{img_c}_var'] = np.var(values)
                        results_tmp[f'img_ch_{img_c}_sum'] = np.sum(values)
                results.append(results_tmp)
    return pd.DataFrame(results)

def is_indexed_binary(mask):
    mask_values = np.unique(mask)
    
    # catch empty image
    if len(mask_values) == 1:
        return False
    
    # catch  if only two values
    # then its binary or only one mask and can be sent for indexing
    if len(mask_values) == 2:
        return False
    
    # 
    return True

def label_5d_mask(binary_mask):
    indexed_mask = np.zeros(binary_mask.shape)
    T, C,Z, _, _ = binary_mask.shape
    for t in range(T):
        for c in range(C):
            for z in range(Z):
                indexed_mask[t,c,z] = label(binary_mask[t,c,z])
    return indexed_mask 

def process_file(img_path:str,  mask_folders:str, mask_suffixes:str, mask_names:str, results_file_path:str) -> None:

    # Process binary mask
    if os.path.exists(results_file_path): 
        os.remove(results_file_path) # TODO Remove after testing

    if not os.path.exists(results_file_path): 
        # Make a pandas table with the IDs of all the masks
        combined_masks, mask_info = get_masks(image_basename=os.path.basename(img_path), mask_folders=mask_folders, mask_suffixes=mask_suffixes, mask_names=mask_names)

        # Do measurements on the binary masks
        results = measure_masks(img_path, combined_masks, mask_info)
        results.to_csv(results_file_path, sep = '\t', encoding='utf-8', index=False)   
        
    else:
        results = pd.read_csv(results_file_path, sep='\t', header=0)

    
    print(results)


img_path = r"Z:\Schink\Oyvind\colaboration_user_data\20250124_Viola\input_tif\230705_93_mNG-DFCP1_LT_LC3_CMvsLPDS__CM__CM_chol_2h__2023-07-07__230705_mNG-DFCP1_LT_LC3_CM_chol_2h_2.tif"
mask_folders = [r"Z:\Schink\Oyvind\colaboration_user_data\20250124_Viola\output_nuc_mask", r"Z:\Schink\Oyvind\colaboration_user_data\20250124_Viola\output_cellpose"]
results_file_path = r"Z:\Schink\Oyvind\colaboration_user_data\20250124_Viola\output.tsv"
mask_suffix_list = [".tif", "_cp_masks.tif"]
mask_names = ["threshold", "cytoplasm"]

process_file(img_path=img_path, mask_folders=mask_folders, mask_suffixes=mask_suffix_list, mask_names= mask_names, results_file_path=results_file_path)