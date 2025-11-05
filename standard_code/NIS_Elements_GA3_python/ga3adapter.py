"""
Universal GA3 Adapter - Orchestrate external UV-managed Python environments

This adapter allows NIS-Elements GA3 to call external Python workers
running in isolated UV-managed environments, avoiding dependency conflicts.

Usage Example:
    from ga3adapter import GA3Adapter
    
    adapter = GA3Adapter(
        module_dir="cellpose_module",
        worker_script="cellpose_worker.py"
    )
    
    result = adapter.process(
        input_image,
        model="cyto3",
        diameter=30
    )

MIT License - BIPHUB 2025
"""

import numpy as np
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)


class GA3Adapter:
    """
    Universal adapter for calling external Python workers from GA3.
    
    Handles environment setup, subprocess orchestration, and data transfer
    via temporary numpy files.
    """
    
    def __init__(
        self,
        module_dir: str,
        worker_script: str,
        base_path: Optional[str] = None
    ):
        """
        Initialize adapter.
        
        Args:
            module_dir: Directory containing the worker module (relative to base_path)
            worker_script: Name of the worker script to execute
            base_path: Base path for modules (defaults to this file's parent)
        """
        if base_path is None:
            base_path = Path(__file__).parent
        else:
            base_path = Path(base_path)
        
        self.module_dir = base_path / module_dir
        self.worker_script = self.module_dir / worker_script
        self.venv_python = self.module_dir / ".venv" / "Scripts" / "python.exe"
        
        # Validate paths
        if not self.module_dir.exists():
            raise ValueError(f"Module directory not found: {self.module_dir}")
        if not self.worker_script.exists():
            raise ValueError(f"Worker script not found: {self.worker_script}")
        
        # Ensure environment exists
        self._ensure_environment()
    
    def _ensure_environment(self) -> None:
        """Ensure UV-managed environment exists, create if needed."""
        if self.venv_python.exists():
            logger.info(f"Environment found: {self.venv_python}")
            return
        
        logger.info("=" * 60)
        logger.info("FIRST-TIME SETUP: Creating environment...")
        logger.info("This will take 1-2 minutes.")
        logger.info("=" * 60)
        
        try:
            # Find bundled UV
            uv_exe = self._find_uv()
            
            # Create venv
            logger.info("Creating virtual environment...")
            subprocess.run(
                [str(uv_exe), "venv", str(self.module_dir / ".venv")],
                check=True,
                cwd=self.module_dir,
                capture_output=True
            )
            
            # Install dependencies
            logger.info("Installing dependencies...")
            subprocess.run(
                [str(uv_exe), "pip", "install", "-e", str(self.module_dir)],
                check=True,
                cwd=self.module_dir,
                capture_output=True
            )
            
            logger.info("✓ Environment ready!")
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Environment setup failed: {e}"
            if e.stderr:
                error_msg += f"\nStderr: {e.stderr.decode()}"
            raise RuntimeError(error_msg)
    
    def _find_uv(self) -> Path:
        """Find bundled UV executable."""
        # Try bundled UV first (3 levels up: ga3adapter.py -> NIS_Elements_GA3_python -> standard_code -> repo_root)
        repo_root = Path(__file__).parent.parent.parent
        bundled_uv = repo_root / "external" / "UV" / "uv.exe"
        
        if bundled_uv.exists():
            logger.info(f"Using bundled UV: {bundled_uv}")
            return bundled_uv
        
        # Fall back to system UV
        result = subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("Using system UV")
            return Path("uv")
        
        raise RuntimeError(
            "UV not found! Expected bundled UV at:\n"
            f"  {bundled_uv}\n"
            "Or install system UV from: https://github.com/astral-sh/uv"
        )
    
    def process(
        self,
        input_data: np.ndarray,
        timeout: int = 300,
        **worker_kwargs: Any
    ) -> np.ndarray:
        """
        Process data through external worker.
        
        Args:
            input_data: Input numpy array
            timeout: Maximum execution time in seconds
            **worker_kwargs: Arguments passed to worker as --key=value
                           (underscores converted to hyphens)
        
        Returns:
            Output numpy array from worker
        
        Example:
            masks = adapter.process(
                image,
                model="cyto3",
                diameter=30,
                flow_threshold=0.4
            )
            # Calls: worker.py --model=cyto3 --diameter=30 --flow-threshold=0.4
        """
        with tempfile.TemporaryDirectory(prefix="ga3_adapter_") as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "input.npy"
            output_path = tmpdir / "output.npy"
            
            # Save input
            logger.info(f"Saving input (shape={input_data.shape}) to temp file...")
            np.save(input_path, input_data)
            
            # Build command
            cmd = [
                str(self.venv_python),
                str(self.worker_script),
                "--input", str(input_path),
                "--output", str(output_path),
            ]
            
            # Add worker-specific arguments
            for key, value in worker_kwargs.items():
                if value is not None:  # Skip None values
                    # Convert underscores to hyphens for CLI style
                    cli_key = key.replace('_', '-')
                    cmd.extend([f"--{cli_key}", str(value)])
            
            # Execute worker
            logger.info(f"Calling worker: {self.worker_script.name}")
            logger.info(f"Command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # Log output
            if result.stdout:
                logger.info(f"Worker output:\n{result.stdout}")
            
            if result.returncode != 0:
                error_msg = f"Worker failed with code {result.returncode}"
                if result.stderr:
                    error_msg += f"\nError: {result.stderr}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Load result
            if not output_path.exists():
                raise RuntimeError(f"Worker did not create output: {output_path}")
            
            logger.info("Loading results...")
            output_data = np.load(output_path)
            logger.info(f"Output loaded: shape={output_data.shape}, dtype={output_data.dtype}")
            
            return output_data
