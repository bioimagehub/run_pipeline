from bioio import BioImage
import bioio_bioformats
import yaml
import os
import argparse

def get_metadata(img):
    # Image dimensions
    t, c, z, y, x = img.dims.T, img.dims.C, img.dims.Z, img.dims.Y, img.dims.X

    # Physical dimensions
    z_um, y_um, x_um = img.physical_pixel_sizes.Z, img.physical_pixel_sizes.Y, img.physical_pixel_sizes.X

    # Channel info
    channel_info = [str(n) for n in img.channel_names]

    # Extract metadata
    image_metadata = {
        'Image metadata': {
            'Channels': [{'Name': f'Please fill in e.g. {name}'} for name in channel_info],
            'Image dimensions': {'C': c, 'T': t, 'X': x, 'Y': y, 'Z': z},
            'Physical dimensions': {'T_ms': None, 'X_um': x_um, 'Y_um': y_um, 'Z_um': None},
        }
    }
    return image_metadata

def process_image(input_file_path, output_file_path):

    # Image
    output_file_path_tif = os.path.splitext(output_file_path)[0] + ".tif" 
    imp = BioImage(input_file_path, reader=bioio_bioformats.Reader)
    imp.save(output_file_path_tif)
    
    # metadata
    output_file_path_yaml = os.path.splitext(output_file_path)[0] + "_metadata.yaml"
    metadata = get_metadata(imp)
    with open(output_file_path_yaml, 'w') as yaml_file:
        yaml.dump(metadata, yaml_file, sort_keys=False)


# path = r"C:\Users\oodegard\Desktop\bfconvert_example\input\230705_mNG-DFCP1_LT_LC3_CM_chol_2h.ims"
# out =  r"C:\Users\oodegard\Desktop\bfconvert_example\input\230705_mNG-DFCP1_LT_LC3_CM_chol_2h.tif"
# process_image(path, out)


def main():
    parser = argparse.ArgumentParser()
    # Arguments for folder processing (default mode)
    parser.add_argument("-i", "--input_file", type=str, help="Path file to be processed")
    parser.add_argument("-o", "--output_file", type=str, default=None, help="Path to the output file")
    args = parser.parse_args()

    process_image(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
