


from bioio import BioImage
import matplotlib.pyplot as plt

def _show_napari(self, dims="TCZYX"):
    import napari

    # Get lazy data (BioImage already returns dask if possible)
    data = self.get_image_data(dims)

    viewer = napari.Viewer()

    kwargs = {}

    # Handle channel axis explicitly
    if "C" in dims:
        kwargs["channel_axis"] = dims.index("C")

    viewer.add_image(data, **kwargs)

    napari.run()
BioImage.show_napari = _show_napari





path = r"C:\Users\oyvinode\Desktop\of_tmp_process\Merged_mp_pred.tif"

img = BioImage(path)


# Is it possible to add a new feature to BioImage that allows us to directly visualize the image using napari? For example, we could add a method called `show_napari()` that opens the image in napari for interactive visualization. This would be especially useful for large 3D images or multi-channel datasets, as napari provides powerful tools for exploring such data.
# us to open up an image in napari directly from the BioImage class. This would allow us to take advantage of napari's interactive visualization capabilities, especially for large 3D images or multi-channel datasets. We could implement a method called `show_napari()` that initializes a napari viewer and displays the image data. This would enhance our ability to explore and analyze complex bioimages more effectively.
img.show_napari()
# This currently gives AttributeError: 'BioImage' object has no attribute 'show'"
