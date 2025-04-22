import numpy as np
import dask.array as da
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
from dask import delayed, compute
from dask.diagnostics import ProgressBar
import multiprocessing
from tqdm import tqdm  # Make sure to import tqdm correctly
from concurrent.futures import ProcessPoolExecutor


def get_frame_shift(reference_frame: np.ndarray,
                    target_frame: np.ndarray,
                    upsample_factor: int = 10,
                    space: str = 'real',
                    disambiguate: bool = False,
                    overlap_ratio: float = 0.8
                    ) -> np.ndarray:
    """
    Must be 3D frames (Z, Y, X).
    Register the target frame to the reference frame using phase cross-correlation.
    Returns the shift in Z, Y, X coordinates.
    """
    # Ensure that reference_frame and target_frame are 3D arrays not dask arrays
    # This is not necessary anymore since defined in function definition
    if isinstance(reference_frame, da.Array):
        reference_frame = reference_frame.compute()
    if isinstance(target_frame, da.Array):
        target_frame = target_frame.compute()

    shift, error, diffphase = phase_cross_correlation(
        reference_frame, 
        target_frame, 
        upsample_factor=upsample_factor, 
        space=space, 
        disambiguate=disambiguate, 
        overlap_ratio=overlap_ratio
    )
    return shift


def get_frame_shifts(image: da.Array,
                     upsample_factor: int=10,
                     space: str = 'real',
                     disambiguate: bool = False,
                     overlap_ratio:float  = 0.8
                     ) -> np.ndarray:
    """
    Fast parallel computation of cumulative shifts for a 4D image stack (T, Z, Y, X).
    Returns a NumPy array of shape (T, 3).
    """
    frames = image.shape[0]
    
    # First shift is zero
    shifts = [delayed(lambda: np.zeros(3, dtype=np.float32))()]

    for f in range(1, frames):
        ref_frame = image[f - 1]
        target_frame = image[f]

        shifts.append(
            delayed(get_frame_shift)(
                ref_frame, target_frame,
                upsample_factor=upsample_factor,
                space=space,
                disambiguate=disambiguate,
                overlap_ratio=overlap_ratio
            )
        )

    with ProgressBar():
        computed_shifts = compute(*shifts, scheduler='threads', num_workers=multiprocessing.cpu_count())

    return np.cumsum(np.stack(computed_shifts), axis=0)

def get_image_shifts(image: da.Array, 
                     drift_correct_channel: int, 
                     upsample_factor: int = 10,
                     space: str = 'real', 
                     disambiguate: bool = False, 
                     overlap_ratio: float = 0.8
                     ) -> np.ndarray:
    """
    Register an image stack with the dimensions T, C, Z, Y, X.

    Compute shifts on one channel, then prepend a zero shift for the channel axis.
    Returns a NumPy array of shape (T, 4): [C, Z, Y, X] shifts.
    """
    shifts = get_frame_shifts(image[:, drift_correct_channel, :, :, :],
                              upsample_factor=upsample_factor,
                              space=space,
                              disambiguate=disambiguate,
                              overlap_ratio=overlap_ratio)
    


    return shifts


def apply_shift_to_frame(args):
    t, c, frame, shift_t = args
    return t, c, shift(frame, shift_t)

def register_image(image: np.ndarray, 
                   drift_correct_channel: int, 
                   upsample_factor: int = 10,
                   space: str = 'real', 
                   disambiguate: bool = False, 
                   overlap_ratio: float = 0.8
                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    Register an image stack with the dimensions T, C, Z, Y, X.
    Calculate shifts on one channel, then apply shifts to all channels.
    Apply shifts to the image stack and return the registered image.
    """
    shifts = get_image_shifts(image, drift_correct_channel,
                               upsample_factor=upsample_factor, 
                               space=space, 
                               disambiguate=disambiguate, 
                               overlap_ratio=overlap_ratio)

    T, C, Z, Y, X = image.shape
    registered_image = np.zeros_like(image)

    # Prepare tasks
    tasks = [(t, c, image[t, c], shifts[t]) for t in range(T) for c in range(C)]

    print(f"Applying shifts to {T} frames and {C} channels...")

    # Use ProcessPoolExecutor to apply shifts in parallel
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(apply_shift_to_frame, tasks), 
                            total=len(tasks), 
                            desc="Applying Shifts"))

    # Store results
    for t, c, shifted in results:
        registered_image[t, c] = shifted

    return registered_image, shifts



if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Drift correction for 5D image stacks.")
    parser.add_argument("-i", "--input_file", type=str, required=True, help="Path to the input image file")
    parser.add_argument("-o", "--output_file", type=str, required=True, help="Path to save the registered image")
    parser.add_argument("-c", "--drift_correct_channel", type=int, default=0, help="Channel to use for drift correction")
    
    args = parser.parse_args()

    from bioio import BioImage
    import bioio_bioformats
    from bioio.writers import OmeTiffWriter


    img = BioImage(args.input_file, reader=bioio_bioformats.Reader)
    image = img.data

    reg_imgage, shifts = register_image(image, args.drift_correct_channel)    
    print("Shifts:")
    print(shifts)
    print("Registered image shape:")
    print(reg_imgage) 


    OmeTiffWriter.save(reg_imgage, args.output_file, dim_order="TCZYX")





def test_drift():
    # Define dimensions
    T, C, Z, Y, X = 4, 2, 3, 4, 4

    # Create a numpy array filled with zeros (you can also use random values or ones)
    image = np.ones((T, C, Z, Y, X), dtype=np.uint16)

    # Example: set a specific voxel to a value
    image[0, 0, 1, 1, 1] = 200  # Set the voxel in timepoint 0, channel 1, z-plane 2, y=5, x=5
    image[0, 0, 1, 1, 2] = 201  # Set the voxel in timepoint 0, channel 1, z-plane 2, y=5, x=5
    image[0, 0, 1, 2, 1] = 202  # Set the voxel in timepoint 0, channel 1, z-plane 2, y=5, x=5
    image[0, 0, 1, 2, 2] = 203  # Set the voxel in timepoint 0, channel 1, z-plane 2, y=5, x=5

    # Example: set a specific voxel to a value

    image[1, 0, 1, 2, 1] = 204  # Set the voxel in timepoint 0, channel 1, z-plane 2, y=5, x=5
    image[1, 0, 1, 2, 2] = 205  # Set the voxel in timepoint 0, channel 1, z-plane 2, y=5, x=5
    image[1, 0, 1, 3, 1] = 206  # Set the voxel in timepoint 0, channel 1, z-plane 2, y=5, x=5
    image[1, 0, 1, 3, 2] = 207  # Set the voxel in timepoint 0, channel 1, z-plane 2, y=5, x=5

    image[2, 0, 1, 1, 1] = 208  # Set the voxel in timepoint 0, channel 1, z-plane 2, y=5, x=5
    image[2, 0, 1, 1, 2] = 209  # Set the voxel in timepoint 0, channel 1, z-plane 2, y=5, x=5
    image[2, 0, 1, 2, 1] = 210  # Set the voxel in timepoint 0, channel 1, z-plane 2, y=5, x=5
    image[2, 0, 1, 2, 2] = 211  # Set the voxel in timepoint 0, channel 1, z-plane 2, y=5, x=5

    image[3, 0, 1, 2, 2] = 212  # Set the voxel in timepoint 0, channel 1, z-plane 2, y=5, x=5
    image[3, 0, 1, 2, 3] = 213  # Set the voxel in timepoint 0, channel 1, z-plane 2, y=5, x=5
    image[3, 0, 1, 3, 2] = 214  # Set the voxel in timepoint 0, channel 1, z-plane 2, y=5, x=5
    image[3, 0, 1, 3, 3] = 215  # Set the voxel in timepoint 0, channel 1, z-plane 2, y=5, x=5
    # print("frame 0")
    # print( image[0])
    # print("frame 1")
    # print( image[1])

    chunk_size = (1, 1, 1, Y, X)  # Adjust chunk size based on your memory limits and preferences

    image = da.from_array(image, chunks=chunk_size)
    # test registering two frames
    # T, C, Z, Y, X

    # shift = get_frame_shift(image[0,0,:,:,:], image[1,0,:,:,:])
    # print(shift)

    # shifts = get_frame_shifts(image[:,0,:,:,:])
    # print(shifts)

    # shifts = get_image_shifts(image, 0)
    # print(shifts)

    registered_image, shifts = register_image(image, 0)
    print(shifts)
    print(registered_image)