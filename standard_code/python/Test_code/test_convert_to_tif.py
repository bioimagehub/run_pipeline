from bioio import BioImage
 
import bioio_ome_tiff, bioio_tifffile, bioio_nd2, bioio_lif, bioio_czi, bioio_dv


path = r"F:\BIPHUB-OYVIND\test_data\different_fileformats\29102025_Apsana-04.czi"
path_out = r"F:\BIPHUB-OYVIND\test_data\different_fileformats\29102025_Apsana-04.ome.tif"


img = BioImage(path, reader=bioio_czi.Reader)
img.save(path_out)
print(img.shape)
