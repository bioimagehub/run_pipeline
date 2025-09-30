"""
Debug Huang thresholding to understand what's happening.
"""
import sys
sys.path.append(r'e:\Oyvind\OF_git\run_pipeline\standard_code\python')
import bioimage_pipeline_utils as rp
import numpy as np
import matplotlib.pyplot as plt

try:
    from skimage.filters import threshold_otsu
    print("✓ scikit-image successfully imported")
except ImportError as e:
    print(f"❌ scikit-image import failed: {e}")
    threshold_otsu = None

def debug_otsu_threshold():
    """Debug the Otsu thresholding process."""
    
    # Load a sample frame
    input_file = r"E:\Oyvind\BIP-hub-test-data\drift\input\live_cells\1_Meng.nd2"
    img = rp.load_tczyx_image(input_file)
    sample_frame = img.data[0, 0, 0].astype(np.float32)  # First frame
    
    print(f"Sample frame shape: {sample_frame.shape}")
    print(f"Sample frame dtype: {sample_frame.dtype}")
    print(f"Sample frame range: [{sample_frame.min():.1f}, {sample_frame.max():.1f}]")
    print(f"Sample frame mean: {sample_frame.mean():.1f}")
    print(f"Sample frame std: {sample_frame.std():.1f}")
    
    if threshold_otsu is None:
        print("❌ Cannot test - threshold_otsu not available")
        return
    
    # Apply Otsu thresholding
    try:
        otsu_threshold = threshold_otsu(sample_frame)
        print(f"\n📊 Otsu threshold value: {otsu_threshold:.1f}")
        
        # Create mask
        mask = sample_frame > otsu_threshold
        pixels_above_threshold = np.sum(mask)
        total_pixels = mask.size
        percentage = (pixels_above_threshold / total_pixels) * 100
        
        print(f"📊 Pixels above threshold: {pixels_above_threshold:,} / {total_pixels:,} ({percentage:.1f}%)")
        
        # Apply mask
        masked_frame = sample_frame * mask
        print(f"📊 Masked frame range: [{masked_frame.min():.1f}, {masked_frame.max():.1f}]")
        print(f"📊 Masked frame non-zero pixels: {np.sum(masked_frame > 0):,}")
        
        # Compare histograms
        print(f"\n📈 HISTOGRAM COMPARISON:")
        original_nonzero = sample_frame[sample_frame > 0]
        masked_nonzero = masked_frame[masked_frame > 0]
        
        print(f"Original non-zero pixels: {len(original_nonzero):,}")
        print(f"Masked non-zero pixels: {len(masked_nonzero):,}")
        print(f"Reduction factor: {len(original_nonzero) / max(1, len(masked_nonzero)):.1f}x")
        
        # Check if masking makes a significant difference
        if percentage > 90:
            print(f"⚠️  WARNING: {percentage:.1f}% of pixels are above threshold")
            print(f"   Otsu masking may not significantly change the image")
        elif percentage > 50:
            print(f"✓ Moderate masking: {percentage:.1f}% of pixels retained")
        else:
            print(f"✓ Strong masking: Only {percentage:.1f}% of pixels retained")
            
        # Test different frames to see variation
        print(f"\n🔄 TESTING THRESHOLD VARIATION ACROSS FRAMES:")
        for t in [0, 5, 10, 15]:
            if t < img.data.shape[0]:
                frame = img.data[t, 0, 0].astype(np.float32)
                threshold = threshold_otsu(frame)
                mask = frame > threshold
                retention = (np.sum(mask) / mask.size) * 100
                print(f"  T={t}: Threshold={threshold:.1f}, Retention={retention:.1f}%")
                
    except Exception as e:
        print(f"❌ Otsu thresholding failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_otsu_threshold()