from nd2reader import ND2Reader

# Example file path
file_path = r"Z:\Schink\Oyvind\biphub_user_data\6849908 - IMB - Coen - Sarah - Photoconv\input\LDC20250314_1321N1_BANF1-V5-mEos4b_WT001.nd2"

# Read the ND2 file and access metadata
with ND2Reader(file_path) as images:
    metadata = images.metadata
    
    # Get the size of a pixel in micrometers
    pixel_microns = metadata.get('pixel_microns')
    
    # Access ROIs from metadata
    rois = metadata.get('rois', [])
    
    # Loop through each ROI to extract position and size information
    for roi in rois:
        print("ROI:", roi)
        positions = roi.get('positions', [])
        sizes = roi.get('sizes', [])
        
        # Convert position and size from micrometers to pixels
        for pos, size in zip(positions, sizes):
            pos_pixels = pos / pixel_microns
            size_pixels = size / pixel_microns
            
            # Print position and size in pixels
            print(f"Position in pixels: {pos_pixels}")
            print(f"Size in pixels: {size_pixels}")
