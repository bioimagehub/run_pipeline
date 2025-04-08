from bioio import BioImage
import bioio_bioformats
import tifffile
from bioio.writers import OmeTiffWriter

def run_bioimage(file_path, save_tif_path=None):
    # Load the image using bioio with Bio-Formats backend
    img = BioImage(file_path, reader=bioio_bioformats.Reader)
    
    # Only use the top resolution level
    img_data = img.data # TCZYX numpy array

    OmeTiffWriter.save(img_data, save_tif_path, dim_order=img.dims.order, physical_pixel_sizes=img.physical_pixel_sizes, metadata=img.metadata)
   

    # if save_tif_path is not None:
    #     tifffile.imwrite(
    #         save_tif_path,
    #         img_data,
    #         photometric="minisblack",
    #         metadata={
    #             'axes': axes,
                                        
    #         },
    #         imagej=True
    #     )
    #     print(f"Saved OME-TIFF to: {save_tif_path}")

    # return img_data  # Optional return for further use


file_path = r"Z:\Schink\Oyvind\colaboration_user_data\20250124_Viola\Input\230705_93_mNG-DFCP1_LT_LC3_CMvsLPDS\CM\CM_chol_2h\2023-07-07\230705_mNG-DFCP1_LT_LC3_CM_chol_2h.ims"

save_tif_path = r"C:\Users\oyvinode\Desktop\test\test.tif"    

run_bioimage(file_path, save_tif_path)
