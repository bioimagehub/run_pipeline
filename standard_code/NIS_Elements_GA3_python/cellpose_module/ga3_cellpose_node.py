"""
GA3 Cellpose Node - Minimal wrapper using GA3Adapter

PASTE THIS INTO NIS-ELEMENTS GA3 PYTHON NODE

Instructions:
1. Open NIS-Elements GA3 editor
2. Add Python node (ND Processing & Conversions > Python Scripting > Python)
3. Configure: 1 Channel input, 1 Binary output
4. Paste this entire script
5. Update ADAPTER_PATH to your installation location
6. Enable "Run out of process"
7. Run your workflow!

Author: BIPHUB Team
License: MIT
"""

import limnode
import sys
from pathlib import Path

# ============================================================================
# CONFIGURATION - Update this path to your installation
# ============================================================================
ADAPTER_PATH = r'C:\git\run_pipeline\standard_code\NIS_Elements_GA3_python'

# Cellpose parameters (adjust as needed)
MODEL = "cyto3"          # Options: cyto, cyto2, cyto3, nuclei
DIAMETER = None          # None for auto-detect, or specify in pixels (e.g., 30)
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = 0.0

# ============================================================================
# Setup adapter (runs once on first import)
# ============================================================================
sys.path.insert(0, ADAPTER_PATH)
from ga3adapter import GA3Adapter

adapter = GA3Adapter(
    module_dir="cellpose_module",
    worker_script="cellpose_worker.py"
)

# ============================================================================
# GA3 Node Interface
# ============================================================================

def output(inp, out):
    """Define output properties."""
    out[0].makeNew("Cellpose Masks", (0, 255, 255)).makeInt32()


def build(loops):
    """Define execution strategy (default: once per frame)."""
    return None


def run(inp, out, ctx):
    """Process each frame through Cellpose."""
    # Extract 2D image from GA3's 4D structure (z, y, x, channels)
    image = inp[0].data[0, :, :, 0]
    
    # Call Cellpose via adapter
    masks = adapter.process(
        image,
        model=MODEL,
        diameter=DIAMETER,
        flow_threshold=FLOW_THRESHOLD,
        cellprob_threshold=CELLPROB_THRESHOLD
    )
    
    # Write result back to GA3
    out[0].data[0, :, :, 0] = masks.astype('int32')


# Required for out-of-process execution
if __name__ == '__main__':
    limnode.child_main(run, output, build)
