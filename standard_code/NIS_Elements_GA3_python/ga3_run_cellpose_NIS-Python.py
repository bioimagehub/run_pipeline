# IMPORTANT: 'limnode' must be imported like this (not from nor as)
import limnode



# ============================================================================
# USER CONFIGURATION
# ============================================================================
# Available Cellpose models:
AVAILABLE_MODELS = [
    'cyto3',      # Generalist cytoplasm model (default, recommended)
    'cyto2',      # Cytoplasm model (previous version)
    'cyto',       # Original cytoplasm model
    'nuclei',     # Nuclei model
    'tissuenet',  # Tissue model
    'livecell',   # Live cell model
    'cyto2torch', # PyTorch version of cyto2
    'CP',         # Custom pretrained model
    'CPx',        # Custom pretrained model (extended)
    'TN1',        # TissueNet v1
    'TN2',        # TissueNet v2
    'TN3',        # TissueNet v3
    'LC1',        # LiveCell v1
    'LC2',        # LiveCell v2
    'LC3',        # LiveCell v3
    'LC4',        # LiveCell v4
]

# Select model by changing the index (0 = first model = 'cyto3')
SELECTED_MODEL = AVAILABLE_MODELS[0]  # Currently: 'cyto3'

# ============================================================================
# SEGMENTATION PARAMETERS
# ============================================================================
# Cell diameter in pixels (most important parameter!)
# Set to None to use automatic diameter estimation
# Typical values: 30 (default), 15-100 depending on cell size
DIAMETER = 30.0  # Set to None for automatic estimation

# Cell probability threshold (0.0 to 1.0)
# All pixels with value above threshold kept for masks
# Decrease to find more and larger masks
CELLPROB_THRESHOLD = 0.0  # Default: 0.0

# Flow error threshold (0.0 to 1.0) - not used for 3D
# All cells with errors below threshold are kept
# Increase to be more lenient, decrease to be more strict
FLOW_THRESHOLD = 0.4  # Default: 0.4

# Minimum mask size in pixels
# All ROIs below this size will be discarded
MIN_SIZE = 15  # Default: 15

# GPU acceleration (True/False)
# Set to True if GPU is available for faster processing
USE_GPU = False  # Set to True if you have CUDA-capable GPU

# Additional parameters (advanced)
DO_3D = False  # Set to True for 3D segmentation
NORMALIZE = True  # Normalize image intensities
INVERT = False  # Invert image (if cells are dark instead of bright)
RESAMPLE = True  # Run dynamics at original image size (slower but more accurate)
# ============================================================================

model = None

# defines output parameter properties
def output(inp: tuple[limnode.AnyInDef], out: tuple[limnode.AnyOutDef]) -> None:
    out[0].makeNew("cell", (0, 255, 255))

# return Program for dimension reduction or two-pass processing
def build(loops: list[limnode.LoopDef]) -> limnode.Program|None:
    return None

# called for each frame/volume
def run(inp: tuple[limnode.AnyInData], out: tuple[limnode.AnyOutData], ctx: limnode.RunContext) -> None:
    import sys
    import numpy
    from cellpose import models
    
    # Debug: Print versions to NIS-Elements console
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {numpy.__version__}")
    print(f"NumPy location: {numpy.__file__}")
    
    try:
        import cellpose
        print(f"Cellpose version: {cellpose.__version__}")
    except AttributeError:
        print("Cellpose version: unknown")
    
    global model
    if model is None:
        print(f"Initializing Cellpose model: {SELECTED_MODEL}")
        model = models.Cellpose(model_type=SELECTED_MODEL, gpu=USE_GPU)
    
    # Run Cellpose segmentation with configured parameters
    masks, flows, styles, diams = model.eval(
        inp[0].data[0, :, :, 0],
        diameter=DIAMETER,
        cellprob_threshold=CELLPROB_THRESHOLD,
        flow_threshold=FLOW_THRESHOLD,
        min_size=MIN_SIZE,
        do_3D=DO_3D,
        normalize=NORMALIZE,
        invert=INVERT,
        resample=RESAMPLE
    )
    out[0].data[0, :, :, 0] = masks.astype(numpy.uint8)

# child process initialization (when outproc is set) 
if __name__ == '__main__':
    limnode.child_main(run, output, build)
