from nd2reader import ND2Reader
import os


# file_path = r"\\SCHINKLAB-NAS\data1\Schink\Oyvind\biphub_user_data\6849908 - IMB - Coen - Sarah - Photoconv\input\LDC20250314_1321N1_BANF1-V5-mEos4b_3SA_KD002.nd2"
# file_path =r"\\SCHINKLAB-NAS\data1\Schink\Oyvind\biphub_user_data\6849908 - IMB - Coen - Sarah - Photoconv\input\SP20241212_PC_mEOS-BAF_PAR_3SA016.nd2"

# Read the ND2 file and access metadata


def get_roi_info(file_path):
    # Read the ND2 file and access metadata
    roi_info = []
    try:
        with ND2Reader(file_path) as images:
            metadata = images.metadata
            
            # Get the size of a pixel in micrometers
            pixel_microns = metadata.get('pixel_microns', 1)  # Default to 1 if not found
            
            # Access ROIs from metadata
            rois = metadata.get('rois', [])
            
            
            
            # Loop through each ROI to extract position and size information
            for roi in rois:
                positions = roi.get('positions', [])
                sizes = roi.get('sizes', [])
                shape = roi.get('shape', 'unknown')  # Extract shape if available
                roi_type = roi.get('type', 'unknown')  # Extract type if available
                
                # Convert position and size from micrometers to pixels
                for pos, size in zip(positions, sizes):
                    pos_pixels = [float(p / pixel_microns) for p in pos]
                    size_pixels = [float(s / pixel_microns) for s in size]

                    roi_pixels = {
                        "Roi": {
                            "Positions": {
                                "x": pos_pixels[0],
                                "y": pos_pixels[1]
                            },
                            "Size": {
                                "x": size_pixels[0],
                                "y": size_pixels[1]
                            },
                            "Shape": shape,
                            "Type": roi_type
                        }
                    }

                    roi_info.append(roi_pixels)
                    
                    # Print position and size in pixels
                    #print(f"Position in pixels: {pos_pixels}")
                    #print(f"Size in pixels: {size_pixels}")
    except Exception as e:
        print(f"Error processing: {e}")
    print("")

    return roi_info



folder_path = r"\\SCHINKLAB-NAS\data1\Schink\Oyvind\biphub_user_data\6849908 - IMB - Coen - Sarah - Photoconv\input"
filenames = os.listdir(folder_path)

for file in filenames:
    if file.endswith(".nd2"):
        print(f"Processing {file}")
        roi_info = get_roi_info(os.path.join(folder_path, file))
        print(roi_info)


