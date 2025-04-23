from skimage.registration import phase_cross_correlation
from skimage.transform import SimilarityTransform, warp, rotate
import scipy as sp
import numpy as np
from bioio.writers import OmeTiffWriter

from pystackreg import StackReg
from skimage import io
from tqdm import tqdm


def correct_frame(template, moved):
    # A default registration pipeline with phase_cross_corr():

    # registration:
    shift, _, _ = phase_cross_correlation(template, moved, upsample_factor=30, 
                                        normalization="phase")
    shifts=[shift[1], shift[0]]
    print(f'Detected translational shift: {shifts}')
    tform = SimilarityTransform(translation=(-shift[1], -shift[0]))
    registered = warp(moved, tform, preserve_range=True)    

    # metrics:
    pearson_corr_R     = sp.stats.pearsonr(template.flatten(), moved.flatten())[0]
    pearson_corr_R_reg = sp.stats.pearsonr(template.flatten(), registered.flatten())[0]
    print(f"Pearson correlation coefficient image vs. moved: {pearson_corr_R}")
    print(f"Pearson correlation coefficient image vs. registration: {pearson_corr_R_reg}")

    return shifts, registered


def drift_correct(video, drift_correct_channel=0, gaussian_blur=-1):
    from skimage.filters import gaussian

    T, C, Z, Y, X = video.shape    
    corrected_video = np.zeros_like(video)

    sr = StackReg(StackReg.RIGID_BODY)

    # Create a stack of max-projected frames over time for the reference channel
    ref_stack = np.max(video[:, drift_correct_channel, :, :, :], axis=1)  # shape: (T, Y, X)

    if gaussian_blur > 0:
        # Apply Gaussian blur in YX (not time)
        ref_stack = gaussian(ref_stack, sigma=(0, gaussian_blur, gaussian_blur))

    # Register max projections over time
    tmats = sr.register_stack(ref_stack, reference='previous')

    # Apply transformation to all channels and all Z-slices
    for t in tqdm(range(T), desc="Drift correction", unit="frame"):
        for c in range(C):
            for z in range(Z):
                corrected_video[t, c, z, :, :] = sr.transform(video[t, c, z, :, :], tmats[t])

    return tmats, corrected_video

def test_case_1() -> tuple[np.ndarray, np.ndarray]:
    from skimage import data
    from skimage.transform import SimilarityTransform, warp

    # Create a test case with a known shift
    image:np.ndarray = data.cells3d()[30,1,:,:]

    shift_known = (-52, 23)
    tform = SimilarityTransform(translation=(shift_known))
    image_moved:np.ndarray = warp(image, tform, preserve_range=True)
    return image, image_moved


def test_case_2() -> tuple[np.ndarray, np.ndarray]:
    from bioio import BioImage
    # import bioio_bioformats
    import bioio_nd2
    
    img = BioImage(r"C:\Users\oodegard\Desktop\collection_of_different_file_formats\input_nd2\1.nd2", reader=bioio_nd2.Reader)

    image:np.ndarray = img.data[10,0,:,:,:]  # "TCZYX "
    image_moved:np.ndarray = img.data[11,0,:,:,:]  # "TCZYX "
    del img  # Free up memory
    return image, image_moved

def test_case_3() -> tuple[np.ndarray, np.ndarray]:
    from bioio import BioImage
    # import bioio_bioformats
    import bioio_nd2
    
    img = BioImage(r"C:\Users\oodegard\Desktop\collection_of_different_file_formats\input_nd2\1.nd2", reader=bioio_nd2.Reader)

    image_np = img.data
    del img  # Free up memory
    return image_np


def plot_registered_image(image, image_moved, registered):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')  # Hide axes
    
    axes[1].imshow(image_moved, cmap='gray')
    axes[1].set_title('Shifted Image')
    axes[1].axis('off')  # Hide axes

    axes[2].imshow(registered, cmap='gray')
    axes[2].set_title('Registered Image')
    axes[2].axis('off')  # Hide axes
    plt.show()

if __name__ == "__main__":
    
    # Test case 1: Known shift
    # image, image_moved = test_case_1()
    # shifts, registered = correct_frame(image, image_moved)
    # plot_registered_image(image, image_moved, registered)
    # print(f"Shifts: {shifts}")
    # print(f"Registered image shape: {registered.shape}")
    
    # # Test case 2: Real data
    # image, image_moved = test_case_2()  
    # path = r"C:\Users\oodegard\Desktop\collection_of_different_file_formats\input_nd2_tif\1_not_registered.tif"
    
    # # Save the original image to the specified output file path
    # movie = np.concatenate((image, image_moved), axis=0)  # Appends registered as a new time frame
    # movie = movie[np.newaxis, np.newaxis, :, :, :]  # Change image to shape (1, 1, Y, X)
    # OmeTiffWriter.save(movie, path, dim_order="TCZYX")  # Save the image to the specified output file path
    
    
    
    # shifts, registered = correct_frame(image, image_moved)
    # plot_registered_image(image[0], image_moved[0], registered[0])
    # print(f"Shifts: {shifts}")
    # print(f"Registered image shape: {registered.shape}")

    # image = image[  np.newaxis, np.newaxis, :, :, :]  # Change image to shape (1, 1, Y, X)
    # registered = registered[np.newaxis, np.newaxis, :, :, :]  # Change registered to shape (1, 1, Y, X
    # movie = np.concatenate((image, registered), axis=0)  # Appends registered as a new time frame

    # # save
    # path = r"C:\Users\oodegard\Desktop\collection_of_different_file_formats\input_nd2_tif\1_registered.tif"
    # OmeTiffWriter.save(movie, path, dim_order="TCZYX")  # Save the image to the specified output file path


    # Test case 3: Real data 5D stack

    print("loading images")
    image = test_case_3()

    print("correcting drift")
    tmats, corrected_image,  = drift_correct(image, drift_correct_channel=0, gaussian_blur = -1)
    print(f"Corrected image shape: {corrected_image.shape}")
    
    path = r"C:\Users\oodegard\Desktop\collection_of_different_file_formats\input_nd2_tif\1_registered.tif"
    OmeTiffWriter.save(corrected_image, path, dim_order="TCZYX")  # Save the image to the specified output file path
    print(f"Saved corrected image to {path}")



