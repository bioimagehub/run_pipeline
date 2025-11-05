"""
Base classes for UV-managed external Python modules in GA3

This module provides reusable infrastructure for integrating external Python
packages into NIS-Elements GA3 using isolated UV-managed environments.

Author: BIPHUB Team
License: MIT

Usage Example:
    class MyCellposeNode(ExternalModuleNode):
        MODULE_NAME = "cellpose"
        WORKER_SCRIPT = "cellpose_worker.py"
        
        def process_image(self, image: np.ndarray) -> np.ndarray:
            return self.call_worker(
                self.WORKER_SCRIPT,
                image=image,
                model="cyto3"
            )
"""

import subprocess
import tempfile
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExternalModuleNode(ABC):
    """
    Base class for GA3 nodes that use UV-managed external Python environments.
    
    This class handles:
    - Environment detection and creation
    - Data serialization to/from subprocess
    - Error handling and logging
    - Subprocess management
    
    Subclasses must implement:
    - MODULE_NAME: Name of the module directory
    - WORKER_SCRIPT: Name of the worker Python script
    - process_image(): Image processing logic
    """
    
    # Subclasses must override these
    MODULE_NAME: str = None
    WORKER_SCRIPT: str = None
    
    def __init__(self, module_dir: Optional[Path] = None):
        """
        Initialize external module node.
        
        Args:
            module_dir: Path to module directory (auto-detected if None)
        """
        if self.MODULE_NAME is None or self.WORKER_SCRIPT is None:
            raise NotImplementedError(
                "Subclass must define MODULE_NAME and WORKER_SCRIPT"
            )
        
        # Detect module directory
        if module_dir is None:
            # Assume we're in standard_code/python/
            module_dir = Path(__file__).parent / f"{self.MODULE_NAME}_module"
        
        self.module_dir = module_dir
        self.venv_dir = module_dir / ".venv"
        self.worker_path = module_dir / self.WORKER_SCRIPT
        
        # Ensure environment exists
        self.python_exe = self._ensure_environment()
        
        logger.info(f"Initialized {self.MODULE_NAME} module at: {self.module_dir}")
    
    def _get_uv_path(self) -> Path:
        """
        Get path to bundled UV executable.
        
        Returns:
            Path to uv.exe in external/UV/
        """
        # Start from this file's location and navigate to external/UV/
        repo_root = Path(__file__).parent.parent.parent
        uv_exe = repo_root / "external" / "UV" / "uv.exe"
        
        if not uv_exe.exists():
            raise RuntimeError(
                f"Bundled UV not found at: {uv_exe}\n"
                f"Expected location: <repo_root>/external/UV/uv.exe"
            )
        
        return uv_exe
    
    def _ensure_environment(self) -> Path:
        """
        Ensure UV-managed environment exists, create if missing.
        
        Returns:
            Path to Python executable in virtual environment
        """
        python_exe = self.venv_dir / "Scripts" / "python.exe"
        
        if python_exe.exists():
            logger.info(f"Environment found: {self.venv_dir}")
            return python_exe
        
        logger.info("="*60)
        logger.info(f"FIRST-TIME SETUP: Creating {self.MODULE_NAME} environment...")
        logger.info("This will take 1-2 minutes. Subsequent runs will be instant.")
        logger.info("="*60)
        
        try:
            # Get bundled UV
            uv_exe = self._get_uv_path()
            logger.info(f"Using bundled UV: {uv_exe}")
            
            # Check UV availability
            uv_check = subprocess.run(
                [str(uv_exe), "--version"],
                capture_output=True,
                text=True
            )
            
            if uv_check.returncode != 0:
                raise RuntimeError(
                    f"UV executable failed: {uv_exe}"
                )
            
            logger.info(f"UV version: {uv_check.stdout.strip()}")
            
            # Create venv
            logger.info(f"Creating virtual environment at: {self.venv_dir}")
            subprocess.run(
                [str(uv_exe), "venv", str(self.venv_dir)],
                check=True,
                cwd=self.module_dir
            )
            
            # Install dependencies
            logger.info("Installing dependencies...")
            subprocess.run(
                [str(uv_exe), "pip", "install", "-e", str(self.module_dir)],
                check=True,
                cwd=self.module_dir
            )
            
            logger.info("="*60)
            logger.info(f"✓ {self.MODULE_NAME} environment ready!")
            logger.info("="*60)
            
            return python_exe
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Environment setup failed: {e}\n"
                f"Manual setup: cd {self.module_dir} && uv venv && uv pip install -e ."
            )
    
    def call_worker(
        self,
        worker_script: str,
        input_image: np.ndarray,
        output_name: str = "output",
        timeout: int = 300,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Call external worker process with standardized interface.
        
        Args:
            worker_script: Name of worker script (relative to module_dir)
            input_image: Input image as numpy array
            output_name: Base name for output file
            timeout: Subprocess timeout in seconds
            **kwargs: Additional arguments passed as CLI flags to worker
        
        Returns:
            Output image as numpy array
        
        Example:
            result = self.call_worker(
                "cellpose_worker.py",
                input_image=img,
                model="cyto3",
                diameter=30
            )
        """
        worker_path = self.module_dir / worker_script
        
        if not worker_path.exists():
            raise FileNotFoundError(f"Worker script not found: {worker_path}")
        
        with tempfile.TemporaryDirectory(prefix=f"ga3_{self.MODULE_NAME}_") as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "input.npy"
            output_path = tmpdir / f"{output_name}.npy"
            
            # Save input
            logger.info(f"Saving input image (shape={input_image.shape})")
            np.save(input_path, input_image)
            
            # Build command
            cmd = [
                str(self.python_exe),
                str(worker_path),
                "--input", str(input_path),
                "--output", str(output_path),
            ]
            
            # Add additional arguments
            for key, value in kwargs.items():
                flag = f"--{key.replace('_', '-')}"
                cmd.extend([flag, str(value)])
            
            # Call worker
            logger.info(f"Calling worker: {worker_script}")
            logger.debug(f"Command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # Log output
            if result.stdout:
                logger.info(f"Worker stdout:\n{result.stdout}")
            
            if result.returncode != 0:
                error_msg = f"Worker failed with code {result.returncode}"
                if result.stderr:
                    error_msg += f"\nError: {result.stderr}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Load result
            if not output_path.exists():
                raise RuntimeError(f"Worker did not create output: {output_path}")
            
            logger.info("Loading result...")
            output = np.load(output_path)
            logger.info(f"Result loaded. Shape: {output.shape}")
            
            return output
    
    @abstractmethod
    def process_image(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Process image using external module.
        
        Subclasses must implement this method.
        
        Args:
            image: Input image as numpy array
            **params: Module-specific parameters
        
        Returns:
            Processed image as numpy array
        """
        pass
    
    def validate_environment(self) -> Dict[str, Any]:
        """
        Validate that the environment is properly set up.
        
        Returns:
            Dictionary with validation results
        """
        checks = {
            "venv_exists": self.venv_dir.exists(),
            "python_exists": self.python_exe.exists(),
            "worker_exists": self.worker_path.exists(),
        }
        
        if checks["python_exists"]:
            # Check Python version
            try:
                result = subprocess.run(
                    [str(self.python_exe), "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                checks["python_version"] = result.stdout.strip()
            except Exception as e:
                checks["python_version"] = f"Error: {e}"
        
        checks["all_valid"] = all([
            checks["venv_exists"],
            checks["python_exists"],
            checks["worker_exists"],
        ])
        
        return checks


class GA3NodeMixin:
    """
    Mixin for GA3-specific limnode interface.
    
    Provides standard GA3 node methods that can be mixed into
    ExternalModuleNode subclasses.
    
    Example:
        class MyCellposeNode(ExternalModuleNode, GA3NodeMixin):
            # ... implementation ...
    """
    
    @staticmethod
    def extract_2d_image(inp_data) -> np.ndarray:
        """
        Extract 2D image from GA3's 4D data structure.
        
        GA3 stores images as (z, y, x, channels). For 2D single-channel:
        - z = 1 (single plane)
        - channels = 1 (single channel)
        
        Args:
            inp_data: GA3 input data object
        
        Returns:
            2D numpy array (y, x)
        """
        # Extract first z-plane, first channel
        return inp_data.data[0, :, :, 0]
    
    @staticmethod
    def insert_2d_result(out_data, result: np.ndarray) -> None:
        """
        Insert 2D result into GA3's 4D output structure.
        
        Args:
            out_data: GA3 output data object
            result: 2D numpy array to insert
        """
        out_data.data[0, :, :, 0] = result


# Example usage template
"""
Example: Creating a Cellpose node using the base class

# In cellpose_module/ga3_cellpose_node.py:

import limnode
from external_module_base import ExternalModuleNode, GA3NodeMixin

class CellposeNode(ExternalModuleNode, GA3NodeMixin):
    MODULE_NAME = "cellpose"
    WORKER_SCRIPT = "cellpose_worker.py"
    
    def process_image(self, image: np.ndarray, **params) -> np.ndarray:
        return self.call_worker(
            self.WORKER_SCRIPT,
            input_image=image,
            model=params.get("model", "cyto3"),
            diameter=params.get("diameter", None),
        )

# Global instance
node = CellposeNode()

# GA3 interface
def output(inp, out):
    out[0].makeNew("Cellpose Masks", (0, 255, 255)).makeInt32()

def build(loops):
    return None

def run(inp, out, ctx):
    try:
        # Extract image
        image = node.extract_2d_image(inp[0])
        
        # Process
        masks = node.process_image(
            image,
            model="cyto3",
            diameter=30
        )
        
        # Insert result
        node.insert_2d_result(out[0], masks.astype(np.int32))
        
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    limnode.child_main(run, output, build)
"""
