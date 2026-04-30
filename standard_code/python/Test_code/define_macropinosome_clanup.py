
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

def find_macropinosome(image: np.ndarray, median_filter_size: int = 3, tolerance: float = 0.1, show_plot: bool = False) -> np.ndarray:
    '''
    From an input 2D grayscale image, find the macropinosome hole and ring mask.
    Typically the input image is a cutout around (including entire ring)

    The center pixel should be inside the hole

    returns mask with 0 for background, 1 for hole, 2 for ring.
    Returns all-zero mask if any QC/final check fails.
    '''
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")

    cy, cx = np.array(image.shape) // 2

    filtered = median_filter(image, size=median_filter_size)


    mp_qc = True
    tolerance = float(tolerance) * (float(filtered.max()) - float(filtered.min()))
    mask = np.zeros_like(filtered, dtype=bool)
    touching_edge = True
    while mp_qc:
        grown = flood(filtered, (cy, cx), tolerance=tolerance)
        if grown is None:
            return np.zeros_like(filtered, dtype=np.uint8)
        mask = np.asarray(grown, dtype=bool)

        # fill holes in the mask
        mask = np.asarray(binary_fill_holes(mask), dtype=bool)
        
        # remove objects joined by a single pixel
        mask = remove_opposite_false_pixels(mask, iterations=2)

        #
        # keep only the connected component that contains the center
        if not bool(mask[cy, cx]):
            return np.zeros_like(filtered, dtype=np.uint8)
        center_component = flood(mask.astype(np.uint8), (cy, cx), tolerance=0)
        if center_component is None:
            return np.zeros_like(filtered, dtype=np.uint8)
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
    if not has_hole or not has_ring:
        return np.zeros_like(filtered, dtype=np.uint8)

    ring_brighter_than_hole = float(np.median(filtered[ring_mask])) > float(np.median(filtered[mask]))

    final_pass = has_hole and has_ring and ring_brighter_than_hole and (not touching_edge)
    if not final_pass:
        return np.zeros_like(filtered, dtype=np.uint8)

    output_mask = np.zeros_like(filtered, dtype=np.uint8)
    output_mask[mask] = 1
    output_mask[ring_mask] = 2

    if show_plot:
        overlay = np.zeros_like(filtered, dtype=np.uint8)
        overlay[mask] = 1
        overlay[ring_mask] = 3

        fix, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title("Input image (grayscale)")

        axes[1].imshow(image, cmap='gray')
        axes[1].imshow(overlay, cmap=overlay_cmap, vmin=0, vmax=3, interpolation='nearest')
        axes[1].scatter([cx], [cy], s=42, c='lime', marker='x')
        axes[1].set_title("Input + hole/ring overlay (pass)")
        plt.show()

    return output_mask


def process_image(img_path):
    img = tifffile.imread(img_path).astype(np.int32)

    find_macropinosome(img, show_plot=True)



def process_folder(folder_path):
    images = glob(folder_path + "/*.tif")[10:20]
    for img_path in images:
        process_image(img_path)


if __name__ == "__main__":
 folder_path = r"C:\Users\oyvinode\Desktop\del"
 process_folder(folder_path)