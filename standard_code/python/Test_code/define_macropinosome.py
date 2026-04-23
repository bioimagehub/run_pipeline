
from bioio import BioImage
from skimage.feature import peak_local_max


def _show_napari(self, dims="TCZYX", coordinates=None):
    import napari

    # Get lazy data (BioImage already returns dask if possible)
    data = self.get_image_data(dims)

    viewer = napari.Viewer()

    kwargs = {}

    # Handle channel axis explicitly
    if "C" in dims:
        kwargs["channel_axis"] = dims.index("C")

    viewer.add_image(data, **kwargs)

    if coordinates is not None:
        viewer.add_points(
            coordinates,
            size=15,
            face_color='red',
            name='Points',
        )

    napari.run()
BioImage.show_napari = _show_napari


path = r"C:\Users\oyvinode\Desktop\of_tmp_process\Merged_mp_pred.ome.tif"
pred_channel = 1

img = BioImage(path)
# The input image is in channel 1
# A gaussian for each macropinosome in ch2


# find local maxima in channel 2
# plot



T, C, Z, Y, X = img.shape

# for testing
T= 5
napari_points = []

for t in range(T):
    img_plane_pred = img.get_image_data("YX", Z=0, T=t, C=pred_channel)
    coordinates = peak_local_max(img_plane_pred, min_distance=10, threshold_rel=10)
    for y, x in coordinates:
        napari_points.append([t, 0, int(y), int(x)])

for point in napari_points:
    print(point)


# %%

img.show_napari(coordinates=napari_points)

