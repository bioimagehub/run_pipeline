"""
Quick visual check to see if there are actually large drifts in the raw data.
"""
import sys
sys.path.append(r'e:\Oyvind\OF_git\run_pipeline\standard_code\python')
import bioimage_pipeline_utils as rp
import numpy as np
import matplotlib.pyplot as plt

def check_raw_drift():
    """Check if there are visible drifts in the raw data by comparing frames."""
    
    # Load data
    input_file = r"E:\Oyvind\BIP-hub-test-data\drift\input\live_cells\1_Meng.nd2"
    img = rp.load_tczyx_image(input_file)
    img_data = img.data.astype(np.float32)
    
    # Crop to 20 frames for speed
    img_data = img_data[:20, 0, 0]  # Take first 20 frames, channel 0, Z=0
    T, H, W = img_data.shape
    print(f"Checking {T} frames of size {H}x{W}")
    
    # Simple cross-correlation to detect shifts (using numpy's built-in)
    reference = img_data[0]  # First frame as reference
    
    print("\n=== SIMPLE CROSS-CORRELATION CHECK ===")
    print("If there are large drifts, we should see significant peak shifts")
    
    for t in [0, 5, 10, 15, 19]:
        # Simple normalized cross-correlation
        current = img_data[t]
        
        # Normalize both images
        ref_norm = (reference - np.mean(reference)) / np.std(reference)
        cur_norm = (current - np.mean(current)) / np.std(current)
        
        # Cross-correlation using FFT
        f_ref = np.fft.fft2(ref_norm)
        f_cur = np.fft.fft2(cur_norm)
        
        # Cross-power spectrum
        cross_power = f_ref * np.conj(f_cur)
        cross_power_norm = cross_power / (np.abs(cross_power) + 1e-12)
        
        # Get correlation
        correlation = np.fft.ifft2(cross_power_norm).real
        
        # Find peak
        peak_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
        py, px = peak_idx
        
        # Handle wrap-around for shift calculation
        if py > H // 2:
            py -= H
        if px > W // 2:
            px -= W
            
        print(f"T={t}: Raw peak position at ({py}, {px}) pixels")
        
        # Also check peak sharpness
        peak_value = correlation.max()
        noise_level = np.percentile(correlation, 95)
        snr = peak_value / noise_level
        print(f"      Peak value: {peak_value:.3f}, SNR: {snr:.2f}")
        
    print("\n=== MANUAL FRAME DIFFERENCE CHECK ===")
    print("Computing simple frame differences to see motion")
    
    for t in [1, 5, 10, 15, 19]:
        diff = img_data[t] - img_data[0]
        mean_diff = np.mean(np.abs(diff))
        max_diff = np.max(np.abs(diff))
        
        print(f"T={t} vs T=0: Mean |diff|={mean_diff:.1f}, Max |diff|={max_diff:.1f}")
    
    # Check if this is just a very stable sample
    print(f"\n=== DATASET STABILITY CHECK ===")
    total_variation = 0
    for t in range(1, T):
        frame_diff = np.mean(np.abs(img_data[t] - img_data[t-1]))
        total_variation += frame_diff
        
    avg_frame_to_frame_change = total_variation / (T-1)
    print(f"Average frame-to-frame change: {avg_frame_to_frame_change:.3f}")
    
    if avg_frame_to_frame_change < 10:
        print("⚠️  This dataset appears to be very stable (low frame-to-frame variation)")
        print("   The small detected shifts might be correct for this data")
    else:
        print("✓ Dataset shows significant variation - should detect larger drifts")

if __name__ == "__main__":
    check_raw_drift()