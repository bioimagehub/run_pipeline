"""
Test script for Cellpose module - Run this to verify installation

This script tests the Cellpose worker in isolation before trying in GA3.

Usage:
    python test_cellpose_module.py
"""

import numpy as np
from pathlib import Path
import subprocess
import sys

MODULE_DIR = Path(__file__).parent
VENV_PYTHON = MODULE_DIR / ".venv" / "Scripts" / "python.exe"
WORKER_SCRIPT = MODULE_DIR / "cellpose_worker.py"


def create_test_image() -> np.ndarray:
    """Create a synthetic test image with circular 'cells'."""
    print("Creating synthetic test image...")
    
    img = np.zeros((512, 512), dtype=np.float32)
    
    # Add some circular "cells"
    from scipy import ndimage
    np.random.seed(42)
    
    # Create blobs
    for _ in range(10):
        x, y = np.random.randint(50, 462, 2)
        radius = np.random.randint(15, 30)
        
        y_grid, x_grid = np.ogrid[:512, :512]
        mask = (x_grid - x)**2 + (y_grid - y)**2 <= radius**2
        img[mask] = np.random.rand() * 0.5 + 0.5
    
    # Add some noise
    img += np.random.rand(512, 512) * 0.1
    
    return img


def get_uv_path() -> Path:
    """Get path to bundled UV executable."""
    repo_root = Path(__file__).parent.parent.parent.parent
    uv_exe = repo_root / "external" / "UV" / "uv.exe"
    
    if not uv_exe.exists():
        raise RuntimeError(
            f"Bundled UV not found at: {uv_exe}\n"
            f"Expected location: <repo_root>/external/UV/uv.exe"
        )
    
    return uv_exe


def test_environment_setup():
    """Test that UV environment exists or can be created."""
    print("\n" + "="*60)
    print("TEST 1: Environment Setup")
    print("="*60)
    
    if VENV_PYTHON.exists():
        print(f"✓ Virtual environment found at: {VENV_PYTHON}")
        return True
    
    print("✗ Virtual environment not found")
    print("Creating environment... (this will take 1-2 minutes)")
    
    try:
        # Get bundled UV
        uv_exe = get_uv_path()
        print(f"✓ Using bundled UV: {uv_exe}")
        
        # Check UV is available
        result = subprocess.run([str(uv_exe), "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"✗ UV failed: {uv_exe}")
            return False
        
        print(f"✓ UV version: {result.stdout.strip()}")
        
        # Create venv
        print("Creating virtual environment...")
        subprocess.run([str(uv_exe), "venv", str(MODULE_DIR / ".venv")], check=True, cwd=MODULE_DIR)
        
        # Install dependencies
        print("Installing dependencies...")
        subprocess.run(
            [str(uv_exe), "pip", "install", "-e", str(MODULE_DIR)],
            check=True,
            cwd=MODULE_DIR
        )
        
        print("✓ Environment created successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Environment setup failed: {e}")
        return False


def test_worker_script():
    """Test that worker script can run."""
    print("\n" + "="*60)
    print("TEST 2: Worker Script")
    print("="*60)
    
    try:
        # Try scipy for test image generation
        try:
            test_img = create_test_image()
        except ImportError:
            print("Note: scipy not available, using simpler test image")
            test_img = np.random.rand(512, 512).astype(np.float32) * 0.5
        
        # Save test image
        test_input = MODULE_DIR / "test_input.npy"
        test_output = MODULE_DIR / "test_output.npy"
        
        print(f"Saving test image to: {test_input}")
        np.save(test_input, test_img)
        
        # Run worker
        print("Running Cellpose worker...")
        cmd = [
            str(VENV_PYTHON),
            str(WORKER_SCRIPT),
            "--input", str(test_input),
            "--output", str(test_output),
            "--model", "cyto3",
            "--diameter", "30"
        ]
        
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        print("\n--- Worker Output ---")
        print(result.stdout)
        
        if result.returncode != 0:
            print("\n--- Worker Errors ---")
            print(result.stderr)
            print(f"\n✗ Worker failed with exit code {result.returncode}")
            return False
        
        # Check output exists
        if not test_output.exists():
            print(f"✗ Output file not created: {test_output}")
            return False
        
        # Load and validate output
        masks = np.load(test_output)
        n_cells = int(masks.max())
        
        print(f"\n✓ Segmentation successful!")
        print(f"  - Input shape: {test_img.shape}")
        print(f"  - Output shape: {masks.shape}")
        print(f"  - Cells found: {n_cells}")
        print(f"  - Mask dtype: {masks.dtype}")
        
        # Cleanup
        test_input.unlink()
        test_output.unlink()
        
        return True
        
    except Exception as e:
        print(f"\n✗ Worker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_availability():
    """Test if GPU is available for Cellpose."""
    print("\n" + "="*60)
    print("TEST 3: GPU Availability")
    print("="*60)
    
    try:
        result = subprocess.run(
            [
                str(VENV_PYTHON),
                "-c",
                "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); "
                "print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); "
                "print(f'Device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        print(result.stdout)
        
        if "CUDA available: True" in result.stdout:
            print("✓ GPU acceleration available")
        else:
            print("⚠ GPU not available - will use CPU (slower)")
        
        return True
        
    except Exception as e:
        print(f"⚠ Could not check GPU status: {e}")
        return True  # Non-critical


def main():
    """Run all tests."""
    print("""
╔════════════════════════════════════════════════════════════╗
║  Cellpose Module Test Suite                               ║
║  Testing UV-managed environment and worker script          ║
╚════════════════════════════════════════════════════════════╝
""")
    
    results = []
    
    # Test 1: Environment setup
    results.append(("Environment Setup", test_environment_setup()))
    
    if not results[0][1]:
        print("\n✗ Cannot proceed without working environment")
        return 1
    
    # Test 2: Worker script
    results.append(("Worker Script", test_worker_script()))
    
    # Test 3: GPU availability (non-critical)
    test_gpu_availability()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} - {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n✓ All tests passed! Module is ready for use in GA3.")
        print("\nNext steps:")
        print("  1. Open NIS-Elements GA3 editor")
        print("  2. Insert Python node")
        print("  3. Copy contents of ga3_cellpose_node.py")
        print("  4. Enable 'Run out of process'")
        print("  5. Connect to workflow and run!")
        return 0
    else:
        print("\n✗ Some tests failed. Please check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
