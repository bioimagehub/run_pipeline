"""Quick test of the split channel export feature."""
import numpy as np
import h5py
import json
import tempfile
import os

# Create a simple test multi-channel H5 file in CTZYX format
test_data = np.random.randint(0, 255, (2, 37, 1, 512, 512), dtype=np.uint16)  # 2 channels, 37 timepoints

axis_configs = [
    {'key': 'c', 'typeFlags': 1, 'resolution': 0, 'description': ''},
    {'key': 't', 'typeFlags': 16, 'resolution': 0, 'description': ''},
    {'key': 'z', 'typeFlags': 2, 'resolution': 0, 'description': ''},
    {'key': 'y', 'typeFlags': 2, 'resolution': 0, 'description': ''},
    {'key': 'x', 'typeFlags': 2, 'resolution': 0, 'description': ''},
]

# Test with split channels
output_dir = "E:/Coen/Sarah/6849908-IMB-Coen-Sarah-Photoconv_global/ilastik_split_test/"
os.makedirs(output_dir, exist_ok=True)

# Simulate splitting channels
for channel_idx in range(2):
    h5_file_path = os.path.join(output_dir, f"test_c{channel_idx}.h5")
    
    # Extract single channel (keep as CTZYX with C=1)
    single_channel = test_data[channel_idx:channel_idx+1, :, :, :, :]
    
    print(f"Channel {channel_idx}: shape = {single_channel.shape}")
    
    with h5py.File(h5_file_path, 'w') as f:
        dset = f.create_dataset('data', data=single_channel, compression='gzip', compression_opts=4)
        dset.attrs['axistags'] = json.dumps({'axes': axis_configs})
        f.attrs['element_size_um'] = [1.0, 1.0, 1.0, 0.0713, 0.0713]
    
    print(f"  Saved to {os.path.basename(h5_file_path)}")

print("\n✅ Test files created successfully!")
print(f"\nVerify with:")
print(f'python standard_code/python/debug_ilastik_h5.py "{os.path.join(output_dir, "test_c0.h5")}"')
