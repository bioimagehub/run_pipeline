"""
Debug script to diagnose trackpy tracking issues.
This will show:
1. Mask counts before tracking
2. Centroid positions and distances
3. TrackPy linking results
4. Mask counts after tracking
"""
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass
import trackpy as tp
import logging
import sys
import os

# Add standard_code to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'standard_code', 'python'))
from standard_code.python import bioimage_pipeline_utils as rp

# Input file - same as debug_time_fill_spatial.py
input_file = r"E:\Coen\Sarah\6849908-IMB-Coen-Sarah-Photoconv\SP20250627\output_masks\SP20250625_PC_R3_LKO8_004_segmentation.tif"

print(f"Loading image from: {input_file}")
img = rp.load_tczyx_image(input_file)
indexed_masks = img.data

print(f"\n{'='*60}")
print(f"Image shape: {indexed_masks.shape} (TCZYX)")
print(f"Data type: {indexed_masks.dtype}")
print(f"Unique labels: {np.unique(indexed_masks)}")

# Analyze mask counts BEFORE tracking
T, C, Z, Y, X = indexed_masks.shape
channel_zero_base = 0
mask_channel = indexed_masks[:, channel_zero_base]

print(f"\n{'='*60}")
print("BEFORE TRACKING - Mask Analysis")
print(f"{'='*60}")

mask_counts_before = []
max_labels_before = []
for t in range(T):
    labels_at_t = []
    for z in range(Z):
        slice_mask = mask_channel[t, z]
        unique_labels = np.unique(slice_mask)
        unique_labels = unique_labels[unique_labels != 0]
        labels_at_t.extend(unique_labels.tolist())
    
    unique_at_t = np.unique(labels_at_t) if labels_at_t else np.array([])
    mask_counts_before.append(len(unique_at_t))
    max_labels_before.append(int(np.max(unique_at_t)) if len(unique_at_t) > 0 else 0)

print(f"Found {len(mask_counts_before)} timepoints with mask counts: {mask_counts_before}")
print(f"Max labels at each timepoint: {max_labels_before}")

# Extract centroids for tracking (same as track_indexed_mask.py)
print(f"\n{'='*60}")
print("Extracting centroids for TrackPy")
print(f"{'='*60}")

records = []
for t in range(T):
    for z in range(Z):
        slice_mask = mask_channel[t, z]
        labels = np.unique(slice_mask)
        labels = labels[labels != 0]
        for label in labels:
            coords = center_of_mass(slice_mask == label)
            records.append({
                'frame': t,
                'z': z,
                'y': coords[0],
                'x': coords[1],
                'label': label,
                't_local': t * Z + z
            })

df = pd.DataFrame(records)
print(f"\nExtracted {len(df)} centroids")
print(f"\nFirst 10 centroids:")
print(df.head(10))

# Analyze centroid distances between consecutive frames
print(f"\n{'='*60}")
print("Centroid Movement Analysis")
print(f"{'='*60}")

for t in range(min(5, T-1)):  # Check first 5 frames
    current = df[df['frame'] == t]
    next_frame = df[df['frame'] == t+1]
    
    if len(current) > 0 and len(next_frame) > 0:
        for _, curr_obj in current.iterrows():
            # Calculate distance to all objects in next frame
            distances = []
            for _, next_obj in next_frame.iterrows():
                dx = next_obj['x'] - curr_obj['x']
                dy = next_obj['y'] - curr_obj['y']
                dist = np.sqrt(dx**2 + dy**2)
                distances.append(dist)
            
            min_dist = min(distances) if distances else np.nan
            print(f"T={t} label={curr_obj['label']:.0f} -> T={t+1}: min distance = {min_dist:.1f} pixels")

# Run TrackPy with different search ranges
search_ranges = [10, 50, 100, 200]

print(f"\n{'='*60}")
print("Testing TrackPy with different search ranges")
print(f"{'='*60}")

logging.getLogger('trackpy').setLevel(logging.WARNING)

for search_range in search_ranges:
    print(f"\n--- Search Range = {search_range} pixels ---")
    
    df_tracked = tp.link_df(df.copy(), search_range=search_range, pos_columns=['x', 'y'], t_column='t_local')
    
    n_particles = len(df_tracked['particle'].unique())
    print(f"TrackPy assigned {n_particles} unique particle IDs")
    
    # Show how particles are distributed across time
    particle_time_distribution = df_tracked.groupby('particle')['frame'].agg(['min', 'max', 'count'])
    print(f"\nParticle time spans:")
    print(particle_time_distribution)
    
    # Check if any particle appears in multiple timepoints
    particles_across_time = particle_time_distribution[particle_time_distribution['count'] > 1]
    print(f"\nParticles tracked across multiple timepoints: {len(particles_across_time)}")
    
    if search_range == 10:  # Save detailed results for default search range
        df_tracked_default = df_tracked.copy()

# Use default search_range=10 for final analysis
print(f"\n{'='*60}")
print("AFTER TRACKING - Detailed Results (search_range=10)")
print(f"{'='*60}")

# Reconstruct the mask with new IDs
new_indexed_masks = np.copy(indexed_masks)
new_indexed_masks[:, channel_zero_base] = 0

for row in df_tracked_default.itertuples():
    t, z = row.frame, row.z
    original_label = row.label
    slice_mask = mask_channel[t, z]
    new_indexed_masks[t, channel_zero_base, z][slice_mask == original_label] = row.particle + 1

# Analyze mask counts AFTER tracking
mask_counts_after = []
max_labels_after = []
mask_channel_after = new_indexed_masks[:, channel_zero_base]

for t in range(T):
    labels_at_t = []
    for z in range(Z):
        slice_mask = mask_channel_after[t, z]
        unique_labels = np.unique(slice_mask)
        unique_labels = unique_labels[unique_labels != 0]
        labels_at_t.extend(unique_labels.tolist())
    
    unique_at_t = np.unique(labels_at_t) if labels_at_t else np.array([])
    mask_counts_after.append(len(unique_at_t))
    max_labels_after.append(int(np.max(unique_at_t)) if len(unique_at_t) > 0 else 0)

print(f"\nFound {len(mask_counts_after)} timepoints with mask counts: {mask_counts_after}")
print(f"Max labels at each timepoint: {max_labels_after}")

# Compare before and after
print(f"\n{'='*60}")
print("COMPARISON")
print(f"{'='*60}")
print(f"{'T':>3} | {'Before':>6} | {'After':>6} | {'Change':>6}")
print("-" * 35)
for t in range(T):
    change = "✓" if mask_counts_before[t] == mask_counts_after[t] else "✗ SPLIT"
    print(f"{t:3d} | {mask_counts_before[t]:6d} | {mask_counts_after[t]:6d} | {change}")

# Show where splits occur
print(f"\n{'='*60}")
print("SPLIT ANALYSIS - Where one object becomes multiple IDs")
print(f"{'='*60}")

for t in range(T):
    if mask_counts_after[t] > mask_counts_before[t]:
        print(f"\nT={t}: {mask_counts_before[t]} object(s) split into {mask_counts_after[t]} IDs")
        
        # Show which particles appear at this timepoint
        particles_at_t = df_tracked_default[df_tracked_default['frame'] == t]['particle'].unique()
        print(f"  Particle IDs at T={t}: {sorted(particles_at_t)}")
        
        # Show centroids
        centroids_at_t = df_tracked_default[df_tracked_default['frame'] == t][['particle', 'x', 'y', 'label']]
        print(f"  Centroids:")
        for _, row in centroids_at_t.iterrows():
            print(f"    Particle {row['particle']}: (x={row['x']:.1f}, y={row['y']:.1f}) from original label {row['label']:.0f}")

print(f"\n{'='*60}")
print("DIAGNOSIS")
print(f"{'='*60}")

if max(mask_counts_after) > 1:
    print("❌ PROBLEM DETECTED: Single object is being split into multiple IDs")
    print("\nPossible causes:")
    print("1. Object centroid moves > search_range pixels between frames")
    print("2. Object temporarily disappears/reappears (causing new ID)")
    print("3. Object shape changes dramatically (centroid jumps)")
    print("4. Z-stacks are being treated as separate time points (t_local = t * Z + z)")
    print("\nRecommendations:")
    print("- Increase search_range parameter")
    print("- Check if object disappears in any frames")
    print("- Verify object motion is continuous")
    print("- Consider tracking in 3D (x, y, z) if object moves in Z")
else:
    print("✓ Tracking successful: Single object maintains single ID across time")

print("\n" + "="*60)
