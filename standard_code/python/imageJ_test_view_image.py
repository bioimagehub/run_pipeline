import imageJ_view_server as ij_server
import bioimage_pipeline_utils as rp

path = r"C:\Users\oodegard\Documents\BIPHub_files\biphub_user_data\6990985 - Meng Pan - Schink -IMB\input\HyperStackReg_MP - Copy.tif"

img = rp.load_tczyx_image(path)

data = img.data
print("Data shape:", data.shape)

ij_server.show_image(data, verbose=True)


