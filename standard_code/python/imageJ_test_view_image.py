import imageJ_view_server as ij_server
import bioimage_pipeline_utils as rp

# run with command 
# uv run --group convert-to-tif python C:\git\run_pipeline\standard_code\python\imageJ_test_view_image.py 
path = r"C:\Users\oodegard\Documents\BIPHub_files\biphub_user_data\6990985 - Meng Pan - Schink -IMB\input\HyperStackReg_MP - Copy.tif"

img = rp.load_tczyx_image(path)

data = img.data
print("Data shape:", data.shape)

ij_server.show_image(data, verbose=True)


