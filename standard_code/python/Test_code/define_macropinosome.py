
# THis is a working prototype. do not delete or change


from glob import glob

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, distance_transform_edt
from skimage.filters import gaussian
from skimage.morphology import disk, remove_small_objects
from skimage.segmentation import flood
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from skimage.morphology import reconstruction
from scipy.ndimage import binary_fill_holes, median_filter


import tifffile

overlay_cmap = ListedColormap([
    (0.0, 0.0, 0.0, 0.0),   # 0 background: transparent
    (0.2, 1.0, 0.2, 0.55),  # 1 hole pass: green
    (1.0, 0.2, 0.2, 0.55),  # 2 hole fail: red
    (1.0, 0.0, 1.0, 0.65),  # 3 ring: magenta
])


def show_napari(*images, titles=None, coordinates=None):
    """Show one or more images in napari.

    Each image can be a numpy array, a dask array, or a BioImage object.
    BioImage objects are displayed with a channel axis so each channel
    becomes a separate layer.  Plain arrays are shown as-is.

    Parameters
    ----------
    *images:
        One or more images to display.
    titles:
        Optional list of layer name strings, one per image.  Falls back to
        'Image 0', 'Image 1', … when not provided.
    coordinates:
        Optional list of napari-format points [T, Z, Y, X] to overlay.
    """
    import napari
    import dask.array as da

    if titles is None:
        titles = [f"Image {i}" for i in range(len(images))]

    viewer = napari.Viewer()

    for img, title in zip(images, titles):
        if hasattr(img, "get_image_dask_data"):
            data = img.get_image_dask_data("TCZYX")
            viewer.add_image(data, channel_axis=1, name=title)
        elif isinstance(img, (np.ndarray, da.Array)):
            viewer.add_image(img, name=title)
        else:
            # Last-resort: try to wrap as numpy
            viewer.add_image(np.asarray(img), name=title)

    if coordinates is not None:
        if coordinates and isinstance(coordinates[0], list) and isinstance(coordinates[0][0], list):
            coordinates = [pt for pts in coordinates for pt in pts]
        viewer.add_points(
            coordinates,
            size=8,
            face_color='red',
            name='Points',
        )
    napari.run()

def greyscale_fill_holes_2d_cpu(image: np.ndarray) -> np.ndarray:
    '''
    Fill holes in a 2D grayscale image using morphological reconstruction.
    returns a score map where higher values indicate more likely holes.
    
    '''
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")

    if image.size == 0 or np.all(image == image.flat[0]):
        return image.copy()
    
    dtype = image.dtype

    image_max = image.max()
    inverted = image_max - image

    seed = inverted.copy()
    seed[1:-1, 1:-1] = inverted.min()

    reconstructed = reconstruction(seed, inverted, method='dilation')
    return reconstructed.astype(dtype)

def remove_opposite_false_pixels(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Remove true pixels whose opposite 4-neighbors are false.

    Rule (4-neighborhood only): remove a center-true pixel when either
    (up and down are true, left and right are false) or
    (left and right are true, up and down are false).
    """
    out = np.asarray(mask, dtype=bool).copy()
    if iterations < 1:
        return out

    for _ in range(iterations):
        padded = np.pad(out, 1, mode='constant', constant_values=False)
        center = padded[1:-1, 1:-1]
        up = padded[:-2, 1:-1]
        down = padded[2:, 1:-1]
        left = padded[1:-1, :-2]
        right = padded[1:-1, 2:]

        vertical_true_horizontal_false = up & down & (~left) & (~right)
        horizontal_true_vertical_false = left & right & (~up) & (~down)
        to_remove = center & (vertical_true_horizontal_false | horizontal_true_vertical_false)

        if not np.any(to_remove):
            break
        out[to_remove] = False

    return out






def process_image(img_path):
    img = tifffile.imread(img_path).astype(np.int32)

    cy, cx = np.array(img.shape) // 2


    filtered = median_filter(img, size=3)
    # fix, axes = plt.subplots(1, 2, figsize=(11, 4))
    # axes[0].imshow(img, cmap='gray')
    # axes[0].set_title("Original image")
    # axes[1].imshow(filtered, cmap='gray')
    # axes[1].set_title("Median filtered image")
    # plt.show()

    filled = greyscale_fill_holes_2d_cpu(filtered)
    
    # print(filtered.dtype, filled.dtype) # int32 int32
    # calculate score
    score = filled / filtered
    # arrange from 0 to 1
    score = (score - score.min()) / (score.max() - score.min())
    
    # plt.imshow(score, cmap='gray')
    # plt.title("Score map")
    # plt.show()

    # do not alow that the mask touches the edge. reduce the tolerance until it does not touch the edge
    mp_qc = True
    tolerance = 0.1 * (filtered.max() - filtered.min())
    mask = np.zeros_like(filtered, dtype=bool)
    touching_edge = True
    while mp_qc:
        grown = flood(filtered, (cy, cx), tolerance=tolerance)
        if grown is None:
            break
        mask = np.asarray(grown, dtype=bool)

        # fill holes in the mask
        mask = np.asarray(binary_fill_holes(mask), dtype=bool)
        
        
        mask = remove_opposite_false_pixels(mask, iterations=2)
        # keep only the connected component that contains the center
        if not bool(mask[cy, cx]):
            break
        center_component = flood(mask.astype(np.uint8), (cy, cx), tolerance=0)
        if center_component is None:
            break
        mask = np.asarray(center_component, dtype=bool)

        # mask can not touch the edge of patch
        touching_edge = np.any(mask[0, :]) or np.any(mask[-1, :]) or np.any(mask[:, 0]) or np.any(mask[:, -1])
        if not touching_edge:
            mp_qc = False

        tolerance *= 0.9
    
    # Build a ring band around the hole with a 5 px width (diffraction-limit constraint).
    dilated = np.asarray(binary_dilation(mask, disk(5)), dtype=bool)
    ring_mask = np.asarray(np.logical_and(dilated, np.logical_not(mask)), dtype=bool)

    has_hole = bool(np.any(mask))
    has_ring = bool(np.any(ring_mask))
    ring_brighter_than_hole = False
    if has_hole and has_ring:
        ring_brighter_than_hole = float(np.median(filtered[ring_mask])) > float(np.median(filtered[mask]))

    final_pass = has_hole and has_ring and ring_brighter_than_hole and (not touching_edge)

    if not final_pass:
        print(f"Image {img_path} failed final test.")

    hole_overlay_value = 1 if final_pass else 2
    overlay = np.zeros_like(filtered, dtype=np.uint8)
    overlay[ring_mask] = 3
    overlay[mask] = hole_overlay_value

    
    # final plot
    fix, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Input image (grayscale)")

    axes[1].imshow(img, cmap='gray')
    axes[1].imshow(overlay, cmap=overlay_cmap, vmin=0, vmax=3, interpolation='nearest')
    center_color = 'lime' if final_pass else 'magenta'
    axes[1].scatter([cx], [cy], s=42, c=center_color, marker='x')
    result_text = "pass" if final_pass else "fail"
    axes[1].set_title("Input + hole/ring overlay ({}, tol={:.2f})".format(result_text, tolerance))

    plt.show()



def process_folder(folder_path):
    images = glob(folder_path + "/*.tif")[:10]
    for img_path in images:
        process_image(img_path)



if __name__ == "__main__":
    path = r"C:\Users\oyvinode\Desktop\del"
    process_folder(path)

    

    


