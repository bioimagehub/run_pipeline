"""
Test script for Cellpose module using GA3Adapter

This script tests the complete GA3 adapter + Cellpose worker pipeline.

Usage:
    python test_cellpose_module.py
"""

import numpy as np
import sys
from pathlib import Path

# Add adapter to path
MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(MODULE_DIR.parent))

from ga3adapter import GA3Adapter


def create_test_image() -> np.ndarray:
    """Create a synthetic test image with circular 'cells'."""
    print("Creating synthetic test image...")
    
    img = np.zeros((512, 512), dtype=np.float32)
    
    try:
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
    except ImportError:
        print("Note: scipy not available, using simple test image")
        img = np.random.rand(512, 512).astype(np.float32) * 0.5
    
    return img


def test_adapter_initialization():
    """Test that the GA3Adapter initializes correctly."""
    print("\n" + "="*60)
    print("TEST 1: Adapter Initialization")
    print("="*60)
    
    try:
        print("Creating GA3Adapter instance...")
        adapter = GA3Adapter(
            module_dir="cellpose_module",
            worker_script="cellpose_worker.py"
        )
        print(f"✓ Adapter initialized successfully")
        print(f"  - Module dir: {adapter.module_dir}")
        print(f"  - Worker script: {adapter.worker_script}")
        print(f"  - Python executable: {adapter.venv_python}")
        
        if adapter.venv_python.exists():
            print(f"✓ Virtual environment found")
        else:
            print(f"⚠ Virtual environment will be created on first use")
        
        return adapter
        
    except Exception as e:
        print(f"✗ Adapter initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_segmentation(adapter: GA3Adapter):
    """Test segmentation using the adapter."""
    print("\n" + "="*60)
    print("TEST 2: Cellpose Segmentation")
    print("="*60)
    
    try:
        # Create test image
        print("Creating test image...")
        test_img = create_test_image()
        print(f"✓ Test image created: shape={test_img.shape}, dtype={test_img.dtype}")
        
        # Run segmentation
        print("\nRunning Cellpose segmentation via adapter...")
        masks = adapter.process(
            test_img,
            model="cyto3",
            diameter=30
        )
        
        n_cells = int(masks.max())
        print(f"\n✓ Segmentation complete!")
        print(f"  - Input shape: {test_img.shape}")
        print(f"  - Output shape: {masks.shape}")
        print(f"  - Cells found: {n_cells}")
        print(f"  - Mask dtype: {masks.dtype}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Segmentation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("""
╔════════════════════════════════════════════════════════════╗
║  Cellpose Module Test Suite                               ║
║  Testing GA3Adapter + UV-managed Cellpose worker          ║
╚════════════════════════════════════════════════════════════╝
""")
    
    results = []
    
    # Test 1: Adapter initialization
    adapter = test_adapter_initialization()
    results.append(("Adapter Initialization", adapter is not None))
    
    if adapter is None:
        print("\n✗ Cannot proceed without working adapter")
        return 1
    
    # Test 2: Segmentation
    results.append(("Cellpose Segmentation", test_segmentation(adapter)))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} - {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nNext steps:")
        print("1. Open NIS-Elements GA3 editor")
        print("2. Add Python node (ND Processing & Conversions > Python)")
        print("3. Copy entire contents of ga3_cellpose_node.py")
        print("4. Update ADAPTER_PATH to your installation")
        print("5. Enable 'Run out of process'")
        print("6. Run your workflow!")
        return 0
    else:
        print("\n✗ Some tests failed. Please check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
