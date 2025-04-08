from bioio import BioImage


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