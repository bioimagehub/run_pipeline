"""
GA3 Cellpose Node - Coordinator for UV-managed Cellpose environment

This script runs inside NIS-Elements' built-in Python and coordinates
calls to the external UV-managed Cellpose worker process.

Usage in GA3:
1. Add this as a Python node in GA3 editor
2. Configure input (1 channel) and output (1 binary)
3. Set "Run out of process" to avoid conflicts
4. The node will auto-bootstrap the Cellpose environment on first use

Author: BIPHUB Team
License: MIT
"""

# IMPORTANT: 'limnode' must be imported like this (not from nor as)
import limnode
import numpy as np
import subprocess
import tempfile
import logging
import sys
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - GA3-Cellpose - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MODULE_DIR = Path(__file__).parent
VENV_DIR = MODULE_DIR / ".venv"
WORKER_SCRIPT = MODULE_DIR / "cellpose_worker.py"

# Cellpose parameters (can be exposed to GA3 UI)
CELLPOSE_MODEL = "cyto3"  # Options: cyto, cyto2, cyto3, nuclei
CELL_DIAMETER = None  # None for auto-estimate, or specify in pixels
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = 0.0


def ensure_cellpose_environment() -> Path:
    """
    Ensure UV-managed Cellpose environment exists.
    
    Creates the environment on first use using UV for fast, reproducible setup.
    
    Returns:
        Path to Python executable in the virtual environment
    """
    python_exe = VENV_DIR / "Scripts" / "python.exe"
    
    if python_exe.exists():
        logger.info(f"Cellpose environment found: {VENV_DIR}")
        return python_exe
    
    logger.info("=" * 60)
    logger.info("FIRST-TIME SETUP: Creating Cellpose environment...")
    logger.info("This will take 1-2 minutes. Subsequent runs will be instant.")
    logger.info("=" * 60)
    
    try:
        # Check if UV is available
        uv_check = subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            text=True
        )
        
        if uv_check.returncode != 0:
            raise RuntimeError(
                "UV package manager not found! Please install from: https://github.com/astral-sh/uv\n"
                "Or use: pip install uv"
            )
        
        logger.info(f"UV version: {uv_check.stdout.strip()}")
        
        # Create virtual environment
        logger.info(f"Creating virtual environment at: {VENV_DIR}")
        subprocess.run(
            ["uv", "venv", str(VENV_DIR)],
            check=True,
            cwd=MODULE_DIR
        )
        
        # Install dependencies from pyproject.toml
        logger.info("Installing Cellpose and dependencies (this may take a minute)...")
        subprocess.run(
            ["uv", "pip", "install", "-e", str(MODULE_DIR)],
            check=True,
            cwd=MODULE_DIR
        )
        
        logger.info("=" * 60)
        logger.info("✓ Cellpose environment ready!")
        logger.info("=" * 60)
        
        return python_exe
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create Cellpose environment: {e}")
        raise RuntimeError(
            f"Environment setup failed. Please check the logs above.\n"
            f"Manual setup: cd {MODULE_DIR} && uv venv && uv pip install -e ."
        )


def call_cellpose_worker(
    image: np.ndarray,
    model_type: str = "cyto3",
    diameter: Optional[float] = None,
) -> np.ndarray:
    """
    Call external Cellpose worker process with image data.
    
    Args:
        image: 2D numpy array (single channel image)
        model_type: Cellpose model to use
        diameter: Expected cell diameter (None for auto)
        
    Returns:
        Segmentation masks as 2D numpy array
    """
    python_exe = ensure_cellpose_environment()
    
    with tempfile.TemporaryDirectory(prefix="ga3_cellpose_") as tmpdir:
        tmpdir = Path(tmpdir)
        input_path = tmpdir / "input.npy"
        output_path = tmpdir / "output.npy"
        
        # Save input image
        logger.info(f"Saving input image (shape={image.shape}) to temp file...")
        np.save(input_path, image)
        
        # Prepare command
        cmd = [
            str(python_exe),
            str(WORKER_SCRIPT),
            "--input", str(input_path),
            "--output", str(output_path),
            "--model", model_type,
            "--flow-threshold", str(FLOW_THRESHOLD),
            "--cellprob-threshold", str(CELLPROB_THRESHOLD),
        ]
        
        if diameter is not None:
            cmd.extend(["--diameter", str(diameter)])
        
        # Call worker
        logger.info(f"Calling Cellpose worker with model: {model_type}")
        logger.info(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Log output
        if result.stdout:
            logger.info(f"Worker stdout:\n{result.stdout}")
        
        if result.returncode != 0:
            error_msg = f"Cellpose worker failed with code {result.returncode}"
            if result.stderr:
                error_msg += f"\nError: {result.stderr}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Load result
        if not output_path.exists():
            raise RuntimeError(f"Worker did not create output file: {output_path}")
        
        logger.info("Loading segmentation masks...")
        masks = np.load(output_path)
        logger.info(f"Masks loaded. Shape: {masks.shape}, max ID: {masks.max()}")
        
        return masks


# ============================================================================
# GA3 Node Interface (limnode)
# ============================================================================

def output(inp: tuple[limnode.AnyInDef], out: tuple[limnode.AnyOutDef]) -> None:
    """Define output parameter properties."""
    # Output will be a binary mask with cell IDs
    out[0].makeNew("Cellpose Masks", (0, 255, 255)).makeInt32()


def build(loops: list[limnode.LoopDef]) -> limnode.Program | None:
    """Define how run() is called (default: once per frame)."""
    return None


def run(inp: tuple[limnode.AnyInData], out: tuple[limnode.AnyOutData], ctx: limnode.RunContext) -> None:
    """
    Process each frame/volume.
    
    Called by GA3 for each frame. Extracts image, calls external worker,
    and loads results back into GA3's data structures.
    """
    try:
        # Extract 2D image from GA3's 4D structure (z, y, x, channels)
        # For 2D images: z=1, channels=1
        image = inp[0].data[0, :, :, 0]  # Take first z-plane, first channel
        
        logger.info(f"Processing frame: {ctx.outCoordinates}")
        logger.info(f"Image shape: {image.shape}, dtype: {image.dtype}, "
                   f"range: [{image.min()}, {image.max()}]")
        
        # Call external Cellpose worker
        masks = call_cellpose_worker(
            image=image,
            model_type=CELLPOSE_MODEL,
            diameter=CELL_DIAMETER,
        )
        
        # Write result back to GA3
        out[0].data[0, :, :, 0] = masks.astype(np.int32)
        
        logger.info(f"✓ Frame processed successfully. Found {masks.max()} cells.")
        
    except Exception as e:
        logger.error(f"✗ Frame processing failed: {str(e)}", exc_info=True)
        # Re-raise to let GA3 know something went wrong
        raise


# Child process initialization (when GA3 runs this out-of-process)
if __name__ == '__main__':
    limnode.child_main(run, output, build)
