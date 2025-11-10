"""
Quick script to check the format of H5 files for Ilastik compatibility.
Usage: python check_h5_format.py <path_to_h5_file>
"""
import h5py
import json
import sys
import os

def check_h5_format(h5_path):
    """Check if an H5 file matches Ilastik's expected format (CTZYX)."""
    if not os.path.exists(h5_path):
        print(f"ERROR: File not found: {h5_path}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Checking: {os.path.basename(h5_path)}")
    print(f"{'='*60}")
    
    with h5py.File(h5_path, 'r') as f:
        # Check if '/data' dataset exists
        if 'data' not in f:
            print("❌ ERROR: No '/data' dataset found in H5 file")
            print(f"   Available datasets: {list(f.keys())}")
            return False
        
        dataset = f['data']
        shape = dataset.shape
        print(f"✓ Dataset shape: {shape}")
        
        # Check axistags
        if 'axistags' in dataset.attrs:
            axistags_json = dataset.attrs['axistags']
            if isinstance(axistags_json, bytes):
                axistags_json = axistags_json.decode('utf-8')
            axistags = json.loads(axistags_json)
            
            axes = axistags.get('axes', [])
            axis_keys = [ax['key'] for ax in axes]
            axis_string = ''.join(axis_keys)
            
            print(f"✓ Axis tags: {axis_string}")
            print(f"  Axis details:")
            for i, ax in enumerate(axes):
                type_flag = ax.get('typeFlags', 0)
                type_name = {1: 'Channel', 2: 'Space', 16: 'Time'}.get(type_flag, f'Unknown({type_flag})')
                print(f"    [{i}] {ax['key']} - {type_name} (size: {shape[i]})")
            
            # Check if format is correct for Ilastik
            if len(axis_keys) >= 1 and axis_keys[0] == 'c':
                print("✅ CORRECT: Channel axis is at position 0 (Ilastik/VIGRA compatible)")
            else:
                print(f"❌ ERROR: Channel axis should be at position 0, but found '{axis_keys[0] if axis_keys else 'none'}'")
                print(f"   Current order: {axis_string}")
                print(f"   Expected: c... (channel first)")
                return False
        else:
            print("⚠️  WARNING: No axistags metadata found")
        
        # Check element_size_um
        if 'element_size_um' in f.attrs:
            element_size = list(f.attrs['element_size_um'])
            print(f"✓ Physical pixel sizes: {element_size}")
        else:
            print("⚠️  No physical pixel size metadata")
    
    print(f"{'='*60}\n")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_h5_format.py <path_to_h5_file_or_directory>")
        print("\nExample:")
        print("  python check_h5_format.py data/image.h5")
        print("  python check_h5_format.py data/  # checks all .h5 files in directory")
        sys.exit(1)
    
    path = sys.argv[1]
    
    if os.path.isdir(path):
        # Check all H5 files in directory
        h5_files = [f for f in os.listdir(path) if f.endswith('.h5')]
        if not h5_files:
            print(f"No .h5 files found in {path}")
            sys.exit(1)
        
        print(f"\nFound {len(h5_files)} H5 file(s) in {path}\n")
        all_ok = True
        for h5_file in h5_files:
            h5_path = os.path.join(path, h5_file)
            if not check_h5_format(h5_path):
                all_ok = False
        
        if all_ok:
            print("\n✅ All files are Ilastik-compatible!")
        else:
            print("\n❌ Some files have incorrect format. Re-run merge_channels.py to fix.")
    else:
        # Check single file
        if check_h5_format(path):
            print("✅ File is Ilastik-compatible!")
        else:
            print("❌ File has incorrect format. Re-run merge_channels.py to fix.")
