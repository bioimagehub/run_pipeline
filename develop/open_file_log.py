from bioio import BioImage
import bioio_bioformats
import tifffile
from xml.dom.minidom import parseString

def run_bioimage(file_path, save_tif_path=None):
    # Load the image using bioio with Bio-Formats backend
    img = BioImage(file_path, reader=bioio_bioformats.Reader)
    
    # Only use the top resolution level
    img_data = img.data # TCZYX numpy array

    # Extract metadata (includes OME-XML)
    print(dir(img))
    ome_xml = img.ome_metadata.to_xml()  # This is the raw OME-XML string
    #pretty_xml = parseString(ome_xml).toprettyxml()

    # Determine axis labels for tifffile (e.g., "TCZYX")
    axes = img.dims.order

    print("Saving image with shape:", img_data.shape)
    print("Axes order:", axes)

    if save_tif_path is not None:
        tifffile.imwrite(
            save_tif_path,
            img_data,
            photometric="minisblack",
            metadata={
                'axes': axes,
                'omexml': ome_xml  # explicitly embed the original OME-XML
            },
            imagej=False
        )
        print(f"Saved OME-TIFF to: {save_tif_path}")

    return img_data  # Optional return for further use


file_path = r"Z:\Schink\Oyvind\colaboration_user_data\20250124 - Viola\Input\230705_93_mNG-DFCP1_LT_LC3_CMvsLPDS\CM\CM_chol_2h\2023-07-07\230705_mNG-DFCP1_LT_LC3_CM_chol_2h.ims"

save_tif_path = r"C:\Users\oyvinode\Desktop\test\test.tif"    

run_bioimage(file_path, save_tif_path)
