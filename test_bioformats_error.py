"""Test that bioformats errors give helpful messages"""
import sys
import os

print("Step 1: Setting up paths...")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'standard_code', 'python'))

print("Step 2: Importing module...")
try:
    import bioimage_pipeline_utils as rp
    print("Module imported successfully")
except Exception as e:
    print(f"Failed to import: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Step 3: Testing bioformats error handling...")
print()

# Create a fake file with an exotic extension that requires bioformats
fake_exotic_file = os.path.join(os.path.dirname(__file__), "test_fake_exotic_format.oib")
print(f"Creating test file: {fake_exotic_file}")
with open(fake_exotic_file, "w") as f:
    f.write("fake")

try:
    print("Attempting to load exotic format...")
    img = rp.load_tczyx_image(fake_exotic_file)
    print("ERROR: Should have raised an exception!")
    sys.exit(1)
except RuntimeError as e:
    print("SUCCESS! Got helpful error message:")
    print()
    print(str(e))
    sys.exit(0)
except Exception as e:
    print(f"ERROR: Got unexpected exception type: {type(e).__name__}")
    print(str(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    # Clean up
    if os.path.exists(fake_exotic_file):
        os.remove(fake_exotic_file)
        print(f"Cleaned up test file")
