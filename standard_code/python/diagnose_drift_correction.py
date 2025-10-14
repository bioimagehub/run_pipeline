"""
Drift Correction Diagnostic Tool

Visualizes intermediate steps to debug registration problems:
1. Original frames
2. DoG-filtered frames
3. Detected shifts
4. Registration quality metrics

MIT License
Copyright (c) 2024 BIPHUB - Bioimage Informatics Hub, University of Oslo
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation
from skimage.filters import difference_of_gaussians
from scipy.ndimage import shift as scipy_shift
import bioimage_pipeline_utils as rp
from pathlib import Path

# Configuration
TEST_FILE = r"E:\Oyvind\BIP-hub-test-data\drift\input\test_3\1_Meng_timecrop.tif"
OUTPUT_DIR = Path(__file__).parent / "diagnostics"
TIMEPOINTS_TO_TEST = [0, 5, 10, 15, 20]  # Which timepoints to visualize


def normalize_for_display(img: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    """Normalize image to [0, 1] using percentile clipping."""
    vmin = np.percentile(img, 100 - percentile)
    vmax = np.percentile(img, percentile)
    
    if vmax - vmin < 1e-10:
        return np.zeros_like(img)
    
    img_norm = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    return img_norm


def diagnose_registration(
    img: rp.BioImage,
    bandpass_low_sigma: float = 20,
    bandpass_high_sigma: float = 100,
    reference: str = "first",
    upsample_factor: int = 10
):
    """
    Diagnose drift correction by visualizing intermediate steps.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Drift Correction Diagnostic")
    print("="*70)
    print(f"Input: {TEST_FILE}")
    print(f"DoG filter: {bandpass_low_sigma}→{bandpass_high_sigma}")
    print(f"Reference: {reference}")
    print(f"Shape: T={img.shape[0]}, C={img.shape[1]}, Z={img.shape[2]}, Y={img.shape[3]}, X={img.shape[4]}")
    print()
    
    # Extract data (assume 2D for now)
    T, C, Z, Y, X = img.shape
    data = img.data[:, 0, 0, :, :]  # (T, Y, X)
    
    # Apply DoG filter
    print("Applying DoG filter...")
    filtered_data = np.zeros_like(data, dtype=np.float32)
    for t in range(T):
        filtered_data[t] = difference_of_gaussians(
            data[t].astype(np.float32),
            bandpass_low_sigma,
            bandpass_high_sigma
        )
    
    # Compute statistics
    print(f"\nOriginal data:")
    print(f"  Range: [{data.min():.1f}, {data.max():.1f}]")
    print(f"  Mean: {data.mean():.1f}, Std: {data.std():.1f}")
    
    print(f"\nFiltered data (DoG):")
    print(f"  Range: [{filtered_data.min():.1f}, {filtered_data.max():.1f}]")
    print(f"  Mean: {filtered_data.mean():.1f}, Std: {filtered_data.std():.1f}")
    
    # Check for issues
    if np.abs(filtered_data.mean()) < 1e-6:
        print("  ⚠️  WARNING: Mean is ~0 (this is expected for DoG)")
    if filtered_data.std() < 10:
        print("  ⚠️  WARNING: Very low contrast after filtering!")
    
    # Normalize filtered data to positive values for phase correlation
    print("\nNormalizing filtered data to positive range...")
    filtered_normalized = filtered_data - filtered_data.min() + 1.0
    print(f"  Normalized range: [{filtered_normalized.min():.1f}, {filtered_normalized.max():.1f}]")
    
    # Compute shifts on both filtered and normalized data
    print(f"\n{'='*70}")
    print("Computing shifts with different approaches:")
    print(f"{'='*70}")
    
    ref_idx = 0 if reference == "first" else None
    
    approaches = [
        ("Original (no filter)", data),
        ("DoG filtered (raw)", filtered_data),
        ("DoG filtered (normalized)", filtered_normalized),
    ]
    
    all_shifts = {}
    
    for approach_name, test_data in approaches:
        print(f"\n{approach_name}:")
        shifts = np.zeros((T, 2), dtype=np.float64)
        errors = np.zeros(T, dtype=np.float64)
        
        if reference == "first":
            ref_frame = test_data[0]
        
        for t in range(1, min(T, 21)):  # Only first 20 frames for speed
            if reference == "previous":
                ref_frame = test_data[t-1]
            
            try:
                shift, error, _ = phase_cross_correlation(
                    ref_frame,
                    test_data[t],
                    upsample_factor=upsample_factor
                )
                shifts[t] = shift
                errors[t] = error
            except Exception as e:
                print(f"  ⚠️  Error at t={t}: {e}")
                shifts[t] = [np.nan, np.nan]
                errors[t] = np.nan
        
        all_shifts[approach_name] = shifts
        
        # Statistics
        valid_shifts = shifts[~np.isnan(shifts).any(axis=1)]
        if len(valid_shifts) > 0:
            print(f"  Mean shift: {np.mean(valid_shifts, axis=0)}")
            print(f"  Max shift: {np.max(np.abs(valid_shifts)):.2f} pixels")
            print(f"  Mean error: {np.mean(errors[~np.isnan(errors)]):.4f}")
        else:
            print(f"  ⚠️  No valid shifts computed!")
    
    # Visualize results
    print(f"\n{'='*70}")
    print("Creating visualizations...")
    print(f"{'='*70}\n")
    
    # 1. Compare frames at different timepoints
    fig, axes = plt.subplots(3, len(TIMEPOINTS_TO_TEST), figsize=(20, 12))
    
    for col, t in enumerate(TIMEPOINTS_TO_TEST):
        if t >= T:
            continue
        
        # Original
        axes[0, col].imshow(normalize_for_display(data[t]), cmap='gray')
        axes[0, col].set_title(f'T={t}\nOriginal')
        axes[0, col].axis('off')
        
        # DoG filtered
        axes[1, col].imshow(normalize_for_display(filtered_data[t]), cmap='gray')
        axes[1, col].set_title(f'DoG filtered')
        axes[1, col].axis('off')
        
        # DoG normalized
        axes[2, col].imshow(normalize_for_display(filtered_normalized[t]), cmap='gray')
        axes[2, col].set_title(f'DoG normalized')
        axes[2, col].axis('off')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "comparison_frames.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    # 2. Plot shifts over time
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    for approach_name, shifts in all_shifts.items():
        valid = ~np.isnan(shifts).any(axis=1)
        t_valid = np.arange(T)[valid]
        
        axes[0].plot(t_valid, shifts[valid, 0], 'o-', label=f'{approach_name} (Y)', alpha=0.7)
        axes[1].plot(t_valid, shifts[valid, 1], 'o-', label=f'{approach_name} (X)', alpha=0.7)
    
    axes[0].set_ylabel('Y Shift (pixels)')
    axes[0].set_title('Detected Shifts Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_ylabel('X Shift (pixels)')
    axes[1].set_xlabel('Timepoint')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "shifts_over_time.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    # 3. Show correlation quality
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    for approach_name in all_shifts.keys():
        # Recompute for visualization
        test_data = approaches[[a[0] for a in approaches].index(approach_name)][1]
        errors = []
        
        for t in range(1, min(T, 21)):
            try:
                _, error, _ = phase_cross_correlation(
                    test_data[0], test_data[t], upsample_factor=upsample_factor
                )
                errors.append(error)
            except:
                errors.append(np.nan)
        
        ax.plot(range(1, len(errors)+1), errors, 'o-', label=approach_name, alpha=0.7)
    
    ax.set_xlabel('Timepoint')
    ax.set_ylabel('Registration Error (lower = better)')
    ax.set_title('Registration Quality Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "registration_quality.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    print(f"\n{'='*70}")
    print("Diagnostic complete!")
    print(f"{'='*70}")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nRecommendations:")
    print("  1. Check 'comparison_frames.png' - Do filtered frames still show cell structure?")
    print("  2. Check 'shifts_over_time.png' - Are shifts consistent across methods?")
    print("  3. Check 'registration_quality.png' - Which method has lowest error?")
    print("\nIf DoG filtered images look too noisy or lose structure:")
    print("  → Try increasing low_sigma (e.g., 30 or 40)")
    print("  → Try decreasing high_sigma (e.g., 80 or 50)")
    print("  → Or disable DoG filtering entirely")


def main():
    """Run the diagnostic."""
    if not Path(TEST_FILE).exists():
        print(f"✗ Error: File not found: {TEST_FILE}")
        print("Please update TEST_FILE in the script.")
        return
    
    print("Loading image...")
    img = rp.load_tczyx_image(TEST_FILE)
    
    diagnose_registration(
        img,
        bandpass_low_sigma=20,
        bandpass_high_sigma=100,
        reference="first",
        upsample_factor=10
    )


if __name__ == "__main__":
    import sys
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
