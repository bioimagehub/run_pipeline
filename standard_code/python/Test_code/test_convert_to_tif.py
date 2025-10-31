from bioio import BioImage
 
import bioio_ome_tiff, bioio_tifffile, bioio_nd2, bioio_lif, bioio_czi, bioio_dv


path = r"F:\BIPHUB-OYVIND\test_data\different_fileformats\CZI example\EW25-63c_HeLa-GFP-CETN1-mChTub_scr_si8_30hKD.czi"
path_out = r"F:\BIPHUB-OYVIND\test_data\different_fileformats\CZI example\EW25-63c_HeLa-GFP-CETN1-mChTub_scr_si8_30hKD_test.ome.tif"


img = BioImage(path, reader=bioio_czi.Reader)
img.save(path_out)
print(img.shape)
