import argparse
import run_pipeline_helper_functions as rp



parser = argparse.ArgumentParser()
parser.add_argument("--image-folder",nargs='+',  type=str, help="Path to the input image folder(s) (E.g. ./input_tif ./input_tif2)")
parser.add_argument("--image-suffix",nargs='+', type=rp.split_comma_separated_strstring, help="The extension of the input images, e.g. --image-suffix .tif, nargs must match nargs of image-folder and if more than one suffix per folder use csv format, e.g. --image-suffix .tif,.png")
parser.add_argument("--mask-folders", nargs='+', type=str, help="List of input mask folder paths, e.g. --input-masks /path/to/mask1 /path/to/mask2")
parser.add_argument("--mask-suffixes",nargs='+', type=rp.split_comma_separated_strstring, help="Suffix to copy from input masks, must be provided for each input mask, e.g. --mask-suffixes _mask.tif, or if several mask folders --mask-suffixes _mask.tif _cp_mask.tif")

#parser.add_argument("--mask-names", type=rp.split_comma_separated_strstring, nargs='+', help="Names of the masks to be processed, e.g. --mask-names mask1 mask2, or of several ch in tif --mask-names protein1,protein2 cytoplasm,nucleus")
#parser.add_argument("--output-folder", type=str, help="Path to the output folder where csv will be saved")

parsed_args: argparse.Namespace = parser.parse_args()

print(f"Image folder: {parsed_args.image_folder}")
print(f"Image suffix: {parsed_args.image_suffix}") 
# print(f"Mask folders: {parsed_args.mask_folders}")
# print(f"Mask suffixes: {parsed_args.mask_suffixes}")
# print(f"Output folder: {parsed_args.output_folder}")
