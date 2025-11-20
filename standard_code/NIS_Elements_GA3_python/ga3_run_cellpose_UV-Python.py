# IMPORTANT: 'limnode' must be imported like this (not from nor as)
import limnode
import subprocess
import sys
import os
import struct

# ============================================================================
# USER CONFIGURATION
# ============================================================================
# Available Cellpose models:
AVAILABLE_MODELS = [
    'cyto3',      # Generalist cytoplasm model (default, recommended)
    # ...existing code...
]

SELECTED_MODEL = AVAILABLE_MODELS[0]
DIAMETER = 30.0
CELLPROB_THRESHOLD = 0.0
FLOW_THRESHOLD = 0.4
MIN_SIZE = 15
USE_GPU = False
DO_3D = False
NORMALIZE = True
INVERT = False
RESAMPLE = True

# Path to UV environment with cellpose
NIS_ENVS_ROOT = r"D:\biphub\NIS-Python-envs"
UV_VENV = os.path.join(NIS_ENVS_ROOT, "venv_cellpose3")
UV_PYTHON = os.path.join(UV_VENV, "Scripts", "python.exe")

# ============================================================================

# Global subprocess handle for persistent process
_cellpose_process = None

def _ensure_cellpose_process():
    """Start persistent cellpose subprocess if not already running."""
    global _cellpose_process
    
    if _cellpose_process is None or _cellpose_process.poll() is not None:
        # Create helper script inline
        helper_script = f"""
import sys
import struct
import numpy as np
from cellpose import models

# Initialize model once
model = models.Cellpose(model_type='{SELECTED_MODEL}', gpu={USE_GPU})

while True:
    # Read image dimensions (height, width)
    header = sys.stdin.buffer.read(8)
    if len(header) < 8:
        break
    h, w = struct.unpack('II', header)
    
    # Read image data
    img_bytes = sys.stdin.buffer.read(h * w * 8)  # float64
    img = np.frombuffer(img_bytes, dtype=np.float64).reshape(h, w)
    
    # Run cellpose
    masks, _, _, _ = model.eval(
        img,
        diameter={DIAMETER},
        cellprob_threshold={CELLPROB_THRESHOLD},
        flow_threshold={FLOW_THRESHOLD},
        min_size={MIN_SIZE},
        do_3D={DO_3D},
        normalize={NORMALIZE},
        invert={INVERT},
        resample={RESAMPLE}
    )
    
    # Write result
    result = masks.astype(np.uint8).tobytes()
    sys.stdout.buffer.write(result)
    sys.stdout.buffer.flush()
"""
        
        _cellpose_process = subprocess.Popen(
            [UV_PYTHON, "-c", helper_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )

def output(inp: tuple[limnode.AnyInDef], out: tuple[limnode.AnyOutDef]) -> None:
    out[0].makeNew("cell", (0, 255, 255))

def build(loops: list[limnode.LoopDef]) -> limnode.Program|None:
    return None

def run(inp: tuple[limnode.AnyInData], out: tuple[limnode.AnyOutData], ctx: limnode.RunContext) -> None:
    import numpy as np
    
    _ensure_cellpose_process()
    
    # Get input image
    img = inp[0].data[0, :, :, 0].astype(np.float64)
    h, w = img.shape
    
    # Send dimensions
    _cellpose_process.stdin.write(struct.pack('II', h, w))
    _cellpose_process.stdin.flush()
    
    # Send image data
    _cellpose_process.stdin.write(img.tobytes())
    _cellpose_process.stdin.flush()
    
    # Read result
    result_bytes = _cellpose_process.stdout.read(h * w)
    masks = np.frombuffer(result_bytes, dtype=np.uint8).reshape(h, w)
    
    # Write output
    out[0].data[0, :, :, 0] = masks

if __name__ == '__main__':
    limnode.child_main(run, output, build)