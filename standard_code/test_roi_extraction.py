import nd2
import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def parse_roi_circles(nd2_file, metadata_parsed):
    """
    Parses circular ROI animations in an ND2 file and extracts positions and sizes.

    Args:
        nd2_file: The ND2File object.
        metadata_parsed: Metadata dictionary with parsed image dimensions and microns per pixel.

    Returns:
        dict: Dictionary containing 'positions', 'sizes', and 'type' for circular ROIs.
    """
    roi_data = {
        "positions": [],
        "sizes": [],  # Diameter for circle ROIs
        "type": "circle"
    }

    # Image dimensions from metadata
    image_width = metadata_parsed["width"] * metadata_parsed["pixel_microns"]
    image_height = metadata_parsed["height"] * metadata_parsed["pixel_microns"]

    for roi_id, roi in nd2_file.rois.items():
        # Check if ROI shape is a circle
        if roi.info.shapeType == nd2.structures.RoiShapeType.Circle:
            anim_params = roi.animParams[0] if roi.animParams else None

            if anim_params:
                # Calculate position, converted to list for YAML readability
                cx = getattr(anim_params, 'centerX', 0)
                cy = getattr(anim_params, 'centerY', 0)
                print(cx, cy)
                # print((1 + getattr(anim_params, 'centerX', 0)),  (1 + getattr(anim_params, 'centerY', 0)))
                position = [
                    float(0.5 * image_width * (1 + getattr(anim_params, 'centerX', 0))),
                    float(0.5 * image_height * (1 + getattr(anim_params, 'centerY', 0))),
                    float(getattr(anim_params, 'centerZ', 0))
                ]
                roi_data["positions"].append(position)

                # Calculate size (diameter of the circle)
                box_shape = getattr(anim_params, 'boxShape', None)
                if box_shape:
                    diameter_x = box_shape.sizeX * image_width
                    diameter_y = box_shape.sizeY * image_height
                    # Assuming symmetric circles, use average diameter
                    average_diameter = (diameter_x + diameter_y) / 2
                    roi_data["sizes"].append(float(average_diameter))
                else:
                    roi_data["sizes"].append(float(0))
        else:
            print(f"ROI ID: {roi_id} is not a circle. Other shapes are not implemented yet.")

    return roi_data

def plot_image_with_rois(file_path):
    with nd2.ND2File(file_path) as nd2_file:
        # Read image data and convert to a displayable format
        image_data = nd2_file.asarray()
        print(image_data.shape)

        # Example of parsed metadata input
        metadata_parsed = {
            "width": nd2_file.sizes['X'],
            "height": nd2_file.sizes['Y'],
            "pixel_microns": nd2_file.voxel_size().x  
        }

        # Parse ROI circular animations
        roi_data = parse_roi_circles(nd2_file, metadata_parsed)
        print(roi_data)
        # Plot the image
        plt.figure(figsize=(8, 8))
        plt.imshow(image_data[1, 2], cmap='gray', origin='lower')  # Assuming first time point and channel
        
        # Overlay circles
        for position, size in zip(roi_data['positions'], roi_data['sizes']):
            circle = Circle((position[0], position[1]), size / 2, color='red', fill=False)
            plt.gca().add_patch(circle)

        plt.title('ROI Circles Overlayed on Image')
        plt.axis('off')
        plt.show()

# Example file path
file_path = r"Z:\Schink\Oyvind\biphub_user_data\6849908 - IMB - Coen - Sarah - Photoconv\input\LDC20250314_1321N1_BANF1-V5-mEos4b_WT001.nd2"

# Plot the image with ROIs overlayed
plot_image_with_rois(file_path)
