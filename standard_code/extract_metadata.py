from bioio import BioImage
import yaml


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


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Extract metadata from a BioImage file.")
    parser.add_argument("-i", "--input_file", type=str, required=True, help="Path to the input BioImage file")
    parser.add_argument("-o", "--output_file", type=str, required=False, help="Path to save the metadata YAML file")

    args = parser.parse_args()

    # Check if output file path is provided, if not set default
    if args.output_file is None:
        args.output_file = os.path.splitext(args.input_file)[0] + "_metadata.yaml"
        

    # Load the image
    img = BioImage(args.input_file)

    # Get metadata
    metadata = get_metadata(img)

    # Save metadata to YAML file
    with open(args.output_file, 'w') as f:
        yaml.dump(metadata, f)