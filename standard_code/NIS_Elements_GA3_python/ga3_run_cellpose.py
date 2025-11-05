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

model = None

# defines output parameter properties
def output(inp: tuple[limnode.AnyInDef], out: tuple[limnode.AnyOutDef]) -> None:
    out[0].makeNew("cell", (0, 255, 255))

# return Program for dimension reduction or two-pass processing
def build(loops: list[limnode.LoopDef]) -> limnode.Program|None:
    return None

# called for each frame/volume
def run(inp: tuple[limnode.AnyInData], out: tuple[limnode.AnyOutData], ctx: limnode.RunContext) -> None:
    import numpy
    from cellpose import models
    
    global model
    if model is None:
        model = models.Cellpose(model_type=SELECTED_MODEL)
    
    masks, flows, styles, diams = model.eval(inp[0].data[0, :, :, 0])
    out[0].data[0, :, :, 0] = masks.astype(numpy.uint8)

# child process initialization (when outproc is set) 
if __name__ == '__main__':
    limnode.child_main(run, output, build)
