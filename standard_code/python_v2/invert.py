

import run_pipeline_utils as rpu
import numpy as np

# The only part that is really specific to this module should be 
def local_invert(plane):
    return np.invert(plane)    


def main():
    file_list = [r"E:\Oyvind\BIP-hub-scratch\train_macropinosome_model\input_drift_corrected\Input_crest__KiaWee__input__250225_RPE-mNG-Phafin2_BSD_10ul_001_drift_1.ome.tif"]
    
    
    rpu.process_files(file_list, local_invert, chunk_type="YX")


if __name__ == "__main__":
    main()

