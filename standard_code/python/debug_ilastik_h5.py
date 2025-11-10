"""
Deep debugging script to understand why Ilastik is seeing channelIndex != 0.
This mimics how Ilastik reads H5 files to diagnose the exact issue.
"""
import h5py
import json
import sys
import os
import numpy as np

def debug_h5_for_ilastik(h5_path):
    """Deep dive into H5 file structure to understand Ilastik's perspective."""
    print(f"\n{'='*70}")
    print(f"DEEP DEBUG: {os.path.basename(h5_path)}")
    print(f"{'='*70}")
    
    with h5py.File(h5_path, 'r') as f:
        # 1. Check dataset location
        print("\n1. DATASET STRUCTURE:")
        print(f"   Root keys: {list(f.keys())}")
        
        if 'data' not in f:
            print("   ❌ ERROR: No '/data' dataset found!")
            return False
        
        dataset = f['data']
        print(f"   ✓ Found '/data' dataset")
        
        # 2. Check raw shape
        print(f"\n2. RAW DATA SHAPE:")
        print(f"   Shape: {dataset.shape}")
        print(f"   Dtype: {dataset.dtype}")
        print(f"   Number of dimensions: {len(dataset.shape)}")
        
        # 3. Check axistags in detail
        print(f"\n3. AXISTAGS METADATA:")
        if 'axistags' not in dataset.attrs:
            print("   ❌ ERROR: No 'axistags' attribute found!")
            return False
        
        axistags_raw = dataset.attrs['axistags']
        print(f"   Raw type: {type(axistags_raw)}")
        
        if isinstance(axistags_raw, bytes):
            axistags_str = axistags_raw.decode('utf-8')
        else:
            axistags_str = axistags_raw
        
        print(f"   Raw JSON (first 200 chars): {axistags_str[:200]}")
        
        axistags = json.loads(axistags_str)
        axes = axistags.get('axes', [])
        
        print(f"\n   Number of axes defined: {len(axes)}")
        print(f"   Shape has {len(dataset.shape)} dimensions")
        
        if len(axes) != len(dataset.shape):
            print(f"   ⚠️  WARNING: Mismatch between axes metadata ({len(axes)}) and shape ({len(dataset.shape)})")
        
        # 4. Detailed axis analysis
        print(f"\n4. AXIS-BY-AXIS ANALYSIS:")
        for i, axis in enumerate(axes):
            print(f"\n   Axis {i}:")
            print(f"      key: '{axis.get('key', 'MISSING')}'")
            print(f"      typeFlags: {axis.get('typeFlags', 'MISSING')}")
            
            type_flag = axis.get('typeFlags', 0)
            type_names = {
                1: 'Channel',
                2: 'Space', 
                16: 'Time',
                0: 'Unknown/None'
            }
            type_name = type_names.get(type_flag, f'Custom({type_flag})')
            print(f"      Type: {type_name}")
            
            if i < len(dataset.shape):
                print(f"      Size: {dataset.shape[i]}")
            
            # Check for resolution info
            if 'resolution' in axis:
                print(f"      Resolution: {axis['resolution']}")
            if 'description' in axis:
                print(f"      Description: {axis['description']}")
        
        # 5. Channel axis check (THE CRITICAL PART)
        print(f"\n5. CHANNEL AXIS VALIDATION:")
        channel_positions = [i for i, ax in enumerate(axes) if ax.get('key') == 'c']
        
        if not channel_positions:
            print("   ❌ ERROR: No channel axis found!")
            return False
        
        if len(channel_positions) > 1:
            print(f"   ⚠️  WARNING: Multiple channel axes at positions {channel_positions}")
        
        channel_pos = channel_positions[0]
        print(f"   Channel axis position: {channel_pos}")
        print(f"   Channel dimension size: {dataset.shape[channel_pos]}")
        
        if channel_pos == 0:
            print("   ✅ CORRECT: Channel is at position 0")
        else:
            print(f"   ❌ ERROR: Channel at position {channel_pos}, should be 0!")
            print(f"   This is what causes 'assert vol.channelIndex == 0' to fail")
            
        # 6. Check typeFlags for channel axis
        print(f"\n6. CHANNEL AXIS TYPE FLAGS:")
        channel_axis = axes[channel_pos]
        channel_type_flag = channel_axis.get('typeFlags', -1)
        print(f"   Channel typeFlags value: {channel_type_flag}")
        
        if channel_type_flag == 1:
            print("   ✅ CORRECT: typeFlags = 1 (Channel)")
        else:
            print(f"   ❌ ERROR: typeFlags should be 1 for channel, got {channel_type_flag}")
        
        # 7. Sample a small slice to verify data layout
        print(f"\n7. DATA SAMPLING:")
        if len(dataset.shape) == 5:
            # Try to read a small 2D slice from each channel
            try:
                sample_slice = dataset[0, 0, 0, :10, :10]
                print(f"   Sample [c=0, t=0, z=0, y=0:10, x=0:10] shape: {sample_slice.shape}")
                print(f"   Sample values (first 5): {sample_slice.flatten()[:5]}")
                
                if dataset.shape[0] > 1:  # If there are multiple channels
                    sample_slice2 = dataset[1, 0, 0, :10, :10]
                    print(f"   Sample [c=1, t=0, z=0, y=0:10, x=0:10] shape: {sample_slice2.shape}")
                    print(f"   Sample values (first 5): {sample_slice2.flatten()[:5]}")
            except Exception as e:
                print(f"   ⚠️  Could not read sample data: {e}")
        
        # 8. Check for any other metadata
        print(f"\n8. OTHER METADATA:")
        print(f"   Dataset attributes: {list(dataset.attrs.keys())}")
        for key in dataset.attrs.keys():
            if key != 'axistags':  # Already printed
                val = dataset.attrs[key]
                if isinstance(val, (int, float, str)):
                    print(f"      {key}: {val}")
                elif isinstance(val, np.ndarray) and val.size < 10:
                    print(f"      {key}: {list(val)}")
                else:
                    print(f"      {key}: {type(val)}")
        
        print(f"\n   File-level attributes: {list(f.attrs.keys())}")
        for key in f.attrs.keys():
            val = f.attrs[key]
            if isinstance(val, (int, float, str)):
                print(f"      {key}: {val}")
            elif isinstance(val, np.ndarray) and val.size < 10:
                print(f"      {key}: {list(val)}")
            else:
                print(f"      {key}: {type(val)}")
    
    print(f"\n{'='*70}\n")
    return channel_pos == 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_ilastik_h5.py <path_to_h5_file>")
        print("\nThis script provides detailed debugging information about")
        print("H5 file structure to diagnose Ilastik compatibility issues.")
        sys.exit(1)
    
    h5_path = sys.argv[1]
    
    if not os.path.exists(h5_path):
        print(f"ERROR: File not found: {h5_path}")
        sys.exit(1)
    
    if os.path.isdir(h5_path):
        # Just check the first file in directory
        h5_files = [f for f in os.listdir(h5_path) if f.endswith('.h5')]
        if not h5_files:
            print(f"No .h5 files found in {h5_path}")
            sys.exit(1)
        h5_path = os.path.join(h5_path, h5_files[0])
        print(f"Checking first file in directory: {h5_files[0]}")
    
    if debug_h5_for_ilastik(h5_path):
        print("✅ File appears to be correctly formatted for Ilastik")
    else:
        print("❌ File has issues that may prevent Ilastik from loading it")
