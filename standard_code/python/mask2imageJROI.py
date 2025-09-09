
import argparse
import os
from roifile import ImagejRoi, roiwrite
import numpy as np
import run_pipeline_helper_functions as rp



def mask_to_rois(mask: np.ndarray):
    """
    Convert a labeled mask (indexed image, TCZYX or lower) to a list of ImageJ ROI objects.
    Each unique label (except 0) in each (T, C, Z) plane is converted to a ROI.
    """
    from skimage import measure
    rois = []
    shape = mask.shape
    # Pad shape to 5D if needed
    while len(shape) < 5:
        mask = np.expand_dims(mask, axis=0)
        shape = mask.shape
    T, C, Z, Y, X = mask.shape
    for t in range(T):
        for c in range(C):
            for z in range(Z):
                plane = mask[t, c, z]
                labels = np.unique(plane)
                for label in labels:
                    if label == 0:
                        continue
                    mask_bin = (plane == label).astype(np.uint8)
                    contours = measure.find_contours(mask_bin, 0.5)
                    for contour in contours:
                        coords = np.fliplr(contour).astype(np.int16)
                        if len(coords) < 3:
                            continue
                        roi = ImagejRoi.frompoints(coords)
                        rois.append(roi)
    return rois


def main():
    parser = argparse.ArgumentParser(description="Convert indexed mask images to ImageJ ROI zip files.")
    parser.add_argument("-p", "--input-file-or-folder", type=str, required=True, help="Path to a single input mask file or a folder of mask files to process.")
    parser.add_argument("-e", "--extension", type=str, default=".tif", help="File extension or pattern to search for (default: .tif)")
    parser.add_argument("-R", "--search_subfolders", action="store_true", help="Search recursively in subfolders (used only if input is a folder).")
    parser.add_argument("-o", "--output-folder", type=str, help="Output folder for ROI zip files. Defaults to input folder.")
    parser.add_argument("--output_file_name_extension", type=str, default="_rois", help="Extension to append to output file name (default: '_rois').")
    args = parser.parse_args()

    if os.path.isfile(args.input_file_or_folder):
        files = [args.input_file_or_folder]
        input_base = os.path.dirname(args.input_file_or_folder)
    else:
        files = rp.get_files_to_process(args.input_file_or_folder, args.extension, args.search_subfolders)
        input_base = args.input_file_or_folder

    if not files:
        print("No files found to process.")
        return

    output_folder = args.output_folder if args.output_folder else input_base
    os.makedirs(output_folder, exist_ok=True)

    for file_path in files:
        img = rp.load_bioio(file_path)
        mask = img.data if hasattr(img, 'data') else np.array(img)
        rois = mask_to_rois(mask)
        base = os.path.splitext(os.path.basename(file_path))[0]
        roi_zip_path = os.path.join(output_folder, f"{base}{args.output_file_name_extension}.zip")
        roiwrite(roi_zip_path, rois)
        print(f"Saved {len(rois)} ROIs to {roi_zip_path}")

if __name__ == "__main__":
    main()
