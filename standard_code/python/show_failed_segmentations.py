#!/usr/bin/env python
"""
Show all failed segmentation cases one at a time.
Displays the image with ROI overlay to help diagnose why segmentation failed.
"""

import sys
import logging
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Add standard_code to path
sys.path.insert(0, str(Path(__file__).parent))

from bioimage_pipeline_utils import (
    get_grouped_files_to_process,
    load_tczyx_image,
    show_image
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_coordinates_from_metadata(yaml_path: str) -> tuple[int, int] | None:
    """Extract ROI coordinates from metadata YAML file."""
    try:
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        if 'Image metadata' in metadata and 'ROIs' in metadata['Image metadata']:
            rois = metadata['Image metadata']['ROIs']
            if isinstance(rois, list) and len(rois) > 0:
                roi = rois[0]
                if 'Roi' in roi and 'Positions' in roi['Roi']:
                    pos = roi['Roi']['Positions']
                    x = int(round(pos['x']))
                    y = int(round(pos['y']))
                    return (x, y)
    except Exception as e:
        logger.error(f"Error reading YAML {yaml_path}: {e}")
    
    return None


def show_failed_image(image_path: str, yaml_path: str, output_path: str, reason: str):
    """
    Display a failed segmentation case with ROI overlay.
    
    Args:
        image_path: Path to input image
        yaml_path: Path to YAML with ROI
        output_path: Expected output mask path (for display)
        reason: Failure reason
    """
    # Load image
    bio_img = load_tczyx_image(image_path)
    img_data = np.asarray(bio_img.data)
    
    # Ensure 5D
    while img_data.ndim < 5:
        img_data = img_data[np.newaxis, ...]
    
    T, C, Z, Y, X = img_data.shape
    
    # Get first timepoint, first channel, max Z projection
    if Z > 1:
        img_2d = np.max(img_data[0, 0, :, :, :], axis=0)
    else:
        img_2d = img_data[0, 0, 0, :, :]
    
    # Get ROI coordinates
    roi_coords = get_coordinates_from_metadata(yaml_path)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Display image
    ax.imshow(img_2d, cmap='gray')
    ax.set_title(f"FAILED: {Path(image_path).name}\n{reason}", fontsize=12, color='red')
    ax.axis('off')
    
    # Overlay ROI if available
    if roi_coords:
        roi_x, roi_y = roi_coords
        # Draw crosshair at ROI center
        ax.plot(roi_x, roi_y, 'r+', markersize=20, markeredgewidth=3, label='ROI Center')
        # Draw circle around ROI
        circle = patches.Circle((roi_x, roi_y), radius=50, fill=False, 
                               edgecolor='red', linewidth=2, linestyle='--', label='ROI Region')
        ax.add_patch(circle)
        ax.legend(loc='upper right', fontsize=10)
        ax.text(5, 20, f'ROI: ({roi_x}, {roi_y})', color='yellow', 
               fontsize=12, bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Add edge indicators if near edge
    h, w = img_2d.shape
    edge_margin = 20
    if roi_coords:
        roi_x, roi_y = roi_coords
        near_edges = []
        if roi_y < edge_margin:
            near_edges.append('TOP')
            ax.axhline(y=0, color='yellow', linewidth=3, linestyle='-', alpha=0.7)
        if roi_y > h - edge_margin:
            near_edges.append('BOTTOM')
            ax.axhline(y=h-1, color='yellow', linewidth=3, linestyle='-', alpha=0.7)
        if roi_x < edge_margin:
            near_edges.append('LEFT')
            ax.axvline(x=0, color='yellow', linewidth=3, linestyle='-', alpha=0.7)
        if roi_x > w - edge_margin:
            near_edges.append('RIGHT')
            ax.axvline(x=w-1, color='yellow', linewidth=3, linestyle='-', alpha=0.7)
        
        if near_edges:
            ax.text(5, h-20, f'NEAR EDGE: {", ".join(near_edges)}', color='yellow',
                   fontsize=12, bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function to loop through all failed segmentations."""
    
    # Define paths
    input_pattern = r'E:\Coen\Sarah\6849908-IMB-Coen-Sarah-Photoconv_global/cellprofiler_input/*_bleach_corrected.tif'
    yaml_pattern = r'E:\Coen\Sarah\6849908-IMB-Coen-Sarah-Photoconv_global/input_tif/*_metadata.yaml'
    output_folder = r'E:\Coen\Sarah\6849908-IMB-Coen-Sarah-Photoconv_global/output_masks_simple_threshold_AI'
    
    # Get grouped files
    logger.info("Finding file groups...")
    search_patterns = {
        'image': input_pattern,
        'yaml': yaml_pattern
    }
    
    grouped_files = get_grouped_files_to_process(search_patterns, search_subfolders=False)
    logger.info(f"Found {len(grouped_files)} file groups")
    
    # Find failed cases
    failed_cases = []
    for basename, file_dict in grouped_files.items():
        if 'image' not in file_dict or 'yaml' not in file_dict:
            continue
        
        image_path = file_dict['image']
        yaml_path = file_dict['yaml']
        
        # Expected output path
        image_filename = Path(image_path).stem
        output_path = Path(output_folder) / f"{image_filename}_mask.tif"
        
        # Check if mask was created
        if not output_path.exists():
            # Determine failure reason by checking YAML and image
            roi_coords = get_coordinates_from_metadata(yaml_path)
            
            if roi_coords is None:
                reason = "No ROI coordinates in YAML"
            else:
                roi_x, roi_y = roi_coords
                # Check if near edge
                if roi_x < 50 or roi_x > 462 or roi_y < 50 or roi_y > 462:
                    reason = "ROI near image edge (nucleus likely touches edge)"
                else:
                    reason = "Unknown failure (possibly too small or threshold failure)"
            
            failed_cases.append({
                'image': image_path,
                'yaml': yaml_path,
                'output': str(output_path),
                'reason': reason,
                'basename': basename
            })
    
    logger.info(f"Found {len(failed_cases)} failed segmentations out of {len(grouped_files)} total")
    
    if len(failed_cases) == 0:
        logger.info("No failures found!")
        return
    
    # Show each failed case
    for i, case in enumerate(failed_cases, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Showing failure {i}/{len(failed_cases)}")
        logger.info(f"Basename: {case['basename']}")
        logger.info(f"Image: {case['image']}")
        logger.info(f"YAML: {case['yaml']}")
        logger.info(f"Reason: {case['reason']}")
        logger.info(f"{'='*80}")
        
        try:
            show_failed_image(
                case['image'],
                case['yaml'],
                case['output'],
                case['reason']
            )
        except Exception as e:
            logger.error(f"Error displaying image: {e}")
            import traceback
            traceback.print_exc()
        
        # Ask to continue
        # if i < len(failed_cases):
        #     response = input(f"\nPress Enter to see next failure ({i+1}/{len(failed_cases)}), or 'q' to quit: ")
        #     if response.lower() == 'q':
        #         logger.info("Exiting viewer")
        #         break
    
    logger.info(f"\nFinished viewing {min(i, len(failed_cases))} of {len(failed_cases)} failed cases")


if __name__ == "__main__":
    main()
