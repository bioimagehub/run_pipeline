"""
Quick script to relabel masks with contiguous labels starting from 1.
Processes all *_segmentation.tif files in a folder.
"""
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add standard_code to path
sys.path.insert(0, r"c:\Users\cc_lab\Documents\git\run_pipeline\standard_code\python")
import bioimage_pipeline_utils as rp


def relabel_contiguous(mask_data: np.ndarray) -> np.ndarray:
    """Relabel mask so labels are contiguous starting from 1.
    
    Args:
        mask_data: 5D array in TCZYX order
        
    Returns:
        Relabeled mask with contiguous labels
    """
    output = np.copy(mask_data)
    
    for c in range(output.shape[1]):  
        unique_labels = np.unique(output[:, c, :, :, :])
        unique_labels = unique_labels[unique_labels > 0]  # Exclude background
        
        if len(unique_labels) == 0:
            continue
            
        # Check if already contiguous
        expected = np.arange(1, len(unique_labels) + 1)
        if np.array_equal(unique_labels, expected):
            print(f"  Channel {c}: Already contiguous (1-{len(unique_labels)})")
            continue
        
        print(f"  Channel {c}: Relabeling {len(unique_labels)} objects: {unique_labels} -> 1-{len(unique_labels)}")
        
        # Create mapping
        new_label = 1
        label_mapping = {}
        for ul in unique_labels:
            label_mapping[ul] = new_label
            new_label += 1
        
        # Apply mapping
        for old_label, new_label in label_mapping.items():
            output[:, c, :, :, :][output[:, c, :, :, :] == old_label] = new_label
    
    return output


def main():
    # Folder to process
    folder = Path(r"E:\Coen\Sarah\6849908-IMB-Coen-Sarah-Photoconv\SP20250627\output_masks")
    
    # Find all segmentation masks
    mask_files = list(folder.glob("*_segmentation.tif"))
    
    if not mask_files:
        print(f"No *_segmentation.tif files found in {folder}")
        return
    
    print(f"Found {len(mask_files)} mask files to process\n")
    
    for mask_file in tqdm(mask_files, desc="Relabeling masks"):
        print(f"\nProcessing: {mask_file.name}")
        
        # Load mask
        img = rp.load_tczyx_image(str(mask_file))
        
        # Check current labels
        unique_labels = np.unique(img.data)
        print(f"  Original labels: {unique_labels}")
        
        # Relabel
        relabeled = relabel_contiguous(img.data)
        
        # Check new labels
        unique_labels_new = np.unique(relabeled)
        print(f"  New labels: {unique_labels_new}")
        
        # Save back (overwrite original)
        rp.save_tczyx_image(relabeled, str(mask_file), dim_order="TCZYX")
        print(f"  ✓ Saved")
    
    print("\n✅ All masks relabeled successfully!")


if __name__ == "__main__":
    main()
