
import argparse
import os
from roifile import ImagejRoi, roiwrite
import numpy as np
import bioimage_pipeline_utils as rp



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
    import concurrent.futures
    from tqdm import tqdm



    parser = argparse.ArgumentParser(description="Convert indexed mask images to ImageJ ROI zip files.")
    parser.add_argument("--input-search-pattern", type=str, required=True, help="Glob pattern for input mask images, e.g. 'folder/*.tif' or 'folder/somefile*.tif'. Use a single file path for one image.")
    parser.add_argument("--output-folder", type=str, help="Output folder for ROI zip files. Defaults to input folder.")
    parser.add_argument("--output-file-name-extension", type=str, default="_rois", help="Extension to append to output file name (default: '_rois').")
    parser.add_argument("--search-subfolders", action="store_true", help="Enable recursive search (only relevant if pattern does not already include '**')")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing (default: parallel enabled)")
    args = parser.parse_args()

    files = rp.get_files_to_process2(args.input_search_pattern, args.search_subfolders)
    if not files:
        print("No files found to process.")
        return

    input_base = os.path.dirname(args.input_search_pattern)
    output_folder = args.output_folder if args.output_folder else input_base
    os.makedirs(output_folder, exist_ok=True)

    def process_file(file_path):
        try:
            img = rp.load_tczyx_image(file_path)
            mask = img.data if hasattr(img, 'data') else np.array(img)
            rois = mask_to_rois(mask)
            base = os.path.splitext(os.path.basename(file_path))[0]
            roi_zip_path = os.path.join(output_folder, f"{base}{args.output_file_name_extension}.zip")
            roiwrite(roi_zip_path, rois)
            return (file_path, None)
        except Exception as e:
            return (file_path, str(e))

    if not args.no_parallel and len(files) > 1:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_file, f) for f in files]
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(files), desc='Converting masks to ROIs'):
                file_path, error = f.result()
                if error is not None:
                    print(f"Error processing {file_path}: {error}")
    else:
        for file_path in tqdm(files, desc='Converting masks to ROIs'):
            _, error = process_file(file_path)
            if error is not None:
                print(f"Error processing {file_path}: {error}")

if __name__ == "__main__":
    main()
