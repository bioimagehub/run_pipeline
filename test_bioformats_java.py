"""Test if bioformats works with Java on PATH"""
import sys
import traceback

print("=" * 70)
print("Testing Bio-Formats with Java")
print("=" * 70)
print()

print("Step 1: Importing jpype...")
try:
    import jpype
    print("✓ jpype imported successfully")
except Exception as e:
    print(f"✗ jpype import failed!")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    traceback.print_exc()
    sys.exit(1)
    
    print("Step 2: Finding JVM...")
    jvm_path = jpype.getDefaultJVMPath()
    print(f"✓ JVM Path: {jvm_path}")
    print()
    
    print("Step 3: Starting JVM...")
    if not jpype.isJVMStarted():
        jpype.startJVM(jvm_path, "-Xmx512M", convertStrings=False)
        print("✓ JVM started successfully")
    else:
        print("✓ JVM already running")
    print()
    
    print("Step 4: Importing scyjava...")
    import scyjava
    print("✓ scyjava imported successfully")
    print()
    
    print("Step 5: Importing bioio_bioformats...")
    import bioio_bioformats
    print("✓ bioio_bioformats imported successfully")
    print()
    
    print("Step 6: Testing BioImage with bioformats reader...")
    from bioio import BioImage
    test_file = r"D:\biphub\test_files\2del.tif"
    print(f"  Loading: {test_file}")
    img = BioImage(test_file, reader=bioio_bioformats.Reader)
    print(f"✓ Image loaded via Bio-Formats: shape={img.shape}")
    print()
    
    print("=" * 70)
    print("SUCCESS! Bio-Formats is working correctly with Java")
    print("=" * 70)
    sys.exit(0)
    
except Exception as e:
    print()
    print("=" * 70)
    print("FAILED!")
    print("=" * 70)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
