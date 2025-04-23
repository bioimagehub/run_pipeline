import numpy as np
from bioio import BioImage # Spessific readers imported before ussage
from bioio.writers import OmeTiffWriter
from pystackreg import StackReg
from tqdm import tqdm
import argparse
from joblib import Parallel, delayed


def drift_correct_xy(video:np.ndarray , drift_correct_channel:int=0) -> tuple[np.ndarray, np.ndarray]:
    T, C, Z, Y, X = video.shape    
    corrected_video:np.ndarray = np.zeros_like(video)

    sr = StackReg(StackReg.RIGID_BODY)

    # Create a stack of max-projected frames over time for the reference channel
    ref_stack:np.ndarray = np.max(video[:, drift_correct_channel, :, :, :], axis=1)  # shape: (T, Y, X)

    # Register max projections over time
    print("Finding shifts: ")
    tmats = sr.register_stack(ref_stack, reference='previous', verbose=True)

    # Apply transformation to all channels and all Z-slices
    for t in tqdm(range(T), desc="Applying shifts", unit="frame"):
        for c in range(C):
            for z in range(Z):
                corrected_video[t, c, z, :, :] = sr.transform(video[t, c, z, :, :], tmats[t])
    return corrected_video, tmats

from pystackreg import StackReg
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np

def drift_correct_xy_parallel(video: np.ndarray, drift_correct_channel: int = 0) -> tuple[np.ndarray, np.ndarray]:
    T, C, Z, Y, X = video.shape    
    corrected_video = np.zeros_like(video)
    
    sr = StackReg(StackReg.RIGID_BODY)

    # Max-projection along Z for drift correction
    ref_stack = np.max(video[:, drift_correct_channel, :, :, :], axis=1)  # shape: (T, Y, X)

    # Register max projections over time
    print("Finding shifts:")
    tmats = sr.register_stack(ref_stack, reference='previous', verbose=True)

    # Parallel transform function
    def apply_shift(t):
        local_sr = StackReg(StackReg.RIGID_BODY)  # Create separate instance to avoid shared state
        out = np.empty_like(video[t])
        for c in range(C):
            for z in range(Z):
                out[c, z] = local_sr.transform(video[t, c, z], tmats[t])
        return out

    # Parallel processing across frames
    corrected_list = Parallel(n_jobs=-1)(
        delayed(apply_shift)(t) for t in tqdm(range(T), desc="Applying shifts")
    )

    corrected_video = np.stack(corrected_list)
    return corrected_video, tmats


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Drift correction for 5D image stacks.")
    parser.add_argument("-i", "--input_file", type=str, required=True, help="Path to the input image file")
    parser.add_argument("-o", "--output_file", type=str, required=True, help="Path to save the registered image")
    parser.add_argument("-c", "--drift_correct_channel", type=int, default=0, help="Channel to use for drift correction")
    
    args = parser.parse_args()
    
    if args.input_file.endswith(".nd2"):
        import bioio_nd2
        img = BioImage(args.input_file, reader=bioio_nd2.Reader)
    else:
        # Bioformats has this annoying printout so I prefer to use a different reader 
        import bioio_bioformats
        img = BioImage(args.input_file, reader=bioio_bioformats.Reader)


    import time
    start_time = time.time()
    image = img.data
    

    reg_imgage, shifts = drift_correct_xy_parallel(image, args.drift_correct_channel)    
    end_time = time.time()
    print(f"Drift correction took {end_time - start_time:.2f} seconds")
    
    
    print("Registered image shape:")
    print(reg_imgage.shape) 

    OmeTiffWriter.save(reg_imgage, args.output_file, dim_order="TCZYX")

