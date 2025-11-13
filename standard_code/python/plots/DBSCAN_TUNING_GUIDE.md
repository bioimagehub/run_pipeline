# DBSCAN Clustering Parameter Tuning Guide

## Quick Reference for `quantify_distance_heatmap_v3.py`

### Problem: Clusters Too Large

**Symptoms:**
- Single cluster encompasses both signal and dirt
- Cluster spatial extent (Δdist, ΔT) is too large
- Multiple dirt particles merged with real signal

**Solution Priority:**
1. **Decrease `--clustering-eps`** (most effective)
   - Current: `0.20`
   - Try: `0.15`, `0.10`, `0.08`, `0.05`
   - Effect: Smaller neighborhood radius = tighter, more separated clusters

2. **Increase `--clustering-intensity-threshold`**
   - Current: `50` (50th percentile)
   - Try: `60`, `70`, `80`
   - Effect: Pre-filters more low-intensity pixels before clustering

3. **Increase `--clustering-min-samples`**
   - Current: `30`
   - Try: `40`, `50`, `100`
   - Effect: Requires denser regions to form clusters (filters elongated artifacts)

### Problem: No Clusters Found (All Noise)

**Symptoms:**
- DBSCAN returns 0 clusters
- Warning: "No clusters found - all pixels marked as noise"

**Solution Priority:**
1. **Increase `--clustering-eps`**
   - Try: `0.25`, `0.30`, `0.40`
   - Effect: Larger neighborhood allows more pixels to connect

2. **Decrease `--clustering-min-samples`**
   - Try: `20`, `15`, `10`
   - Effect: Allows smaller clusters to form

3. **Decrease `--clustering-intensity-threshold`**
   - Try: `40`, `30`, `20`
   - Effect: Includes more candidate pixels

### Problem: Signal Split Into Multiple Small Clusters

**Symptoms:**
- Real signal fragmented across 2+ clusters
- Selected cluster misses parts of the signal

**Solution:**
1. **Increase `--clustering-eps`**
   - Try: `0.25`, `0.30`
   - Effect: Bridges gaps between signal fragments

2. **Decrease `--clustering-min-samples`**
   - Try: `20`, `15`
   - Effect: Allows smaller fragments to form valid clusters

---

## Systematic Tuning Workflow

### Step 1: Baseline Run (Current Settings)
```yaml
--clustering-eps: 0.20
--clustering-min-samples: 30
--clustering-intensity-threshold: 50
--no-parallel
--force-show
```

**Check log output:**
- Number of clusters found
- Selected cluster size and spatial extent (Δdist, ΔT)
- Cluster center position
- Distance to origin

### Step 2: Adjust eps (Neighborhood Radius)

**If clusters TOO LARGE:**
```yaml
# Tighten clustering (try each sequentially)
--clustering-eps: 0.15  # Moderate tightening
--clustering-eps: 0.10  # Strong tightening
--clustering-eps: 0.08  # Very tight (for well-separated signal)
```

**If NO clusters found:**
```yaml
# Loosen clustering
--clustering-eps: 0.25
--clustering-eps: 0.30
```

### Step 3: Fine-Tune Intensity Threshold

**If still including too much dirt:**
```yaml
--clustering-intensity-threshold: 60  # Keep top 40%
--clustering-intensity-threshold: 70  # Keep top 30%
--clustering-intensity-threshold: 80  # Keep top 20% (aggressive)
```

**If losing real signal:**
```yaml
--clustering-intensity-threshold: 40  # Keep top 60%
--clustering-intensity-threshold: 30  # Keep top 70%
```

### Step 4: Adjust Minimum Cluster Size

**If dirt particles still present:**
```yaml
--clustering-min-samples: 50   # Filter small artifacts
--clustering-min-samples: 100  # Aggressive filtering
```

**If signal too fragmented:**
```yaml
--clustering-min-samples: 20
--clustering-min-samples: 15
```

---

## Recommended Starting Points by Scenario

### Clean Data (Minimal Dirt)
```yaml
--clustering-eps: 0.20
--clustering-min-samples: 30
--clustering-intensity-threshold: 50
```

### Moderate Dirt (Separated from Signal)
```yaml
--clustering-eps: 0.15          # Tighter neighborhoods
--clustering-min-samples: 40     # Filter small artifacts
--clustering-intensity-threshold: 60  # Higher threshold
```

### Heavy Dirt (Close to Signal)
```yaml
--clustering-eps: 0.10          # Very tight neighborhoods
--clustering-min-samples: 50     # Strong artifact filtering
--clustering-intensity-threshold: 70  # Aggressive pre-filtering
```

### Weak Signal (Low Intensity)
```yaml
--clustering-eps: 0.25          # Looser neighborhoods
--clustering-min-samples: 20     # Allow smaller clusters
--clustering-intensity-threshold: 40  # Lower threshold
```

---

## Understanding Log Output

### Good Clustering Result
```
Step 3: Running DBSCAN (eps=0.15, min_samples=30)
  Total candidate pixels: 1200
  Clusters found: 3
  Noise pixels: 150 (12.5%)
  Cluster IDs: [0, 1, 2]

  Cluster 0: size=800, center=(5.2, T2.1), dist_to_origin=0.085, extent=(Δdist=15.3, ΔT=5.2)
  Cluster 1: size=150, center=(45.7, T3.5), dist_to_origin=0.450, extent=(Δdist=8.1, ΔT=2.1)
  Cluster 2: size=100, center=(78.3, T1.2), dist_to_origin=0.720, extent=(Δdist=5.5, ΔT=1.5)

Step 4: Selected cluster 0 as signal
  Size: 800 pixels
  Center: distance=5.2, time=T2.1
  Distance to origin: 0.085
```

**Interpretation:**
- ✅ Cluster 0 is near origin (distance=5.2, T≈2) → Real signal
- ✅ Clusters 1 & 2 are far from origin (distance>45) → Dirt particles
- ✅ Selected cluster has reasonable spatial extent

### Bad Clustering (Too Large)
```
  Cluster 0: size=2500, center=(25.5, T5.3), dist_to_origin=0.350, extent=(Δdist=85.0, ΔT=15.0)
```

**Issues:**
- ❌ Huge spatial extent (Δdist=85.0) suggests signal + dirt merged
- ❌ Center far from origin (distance=25.5)
- **Solution:** Decrease `--clustering-eps` to 0.10 or 0.08

### Bad Clustering (No Clusters)
```
  Clusters found: 0
  Noise pixels: 1200 (100.0%)
  WARNING: No clusters found - all pixels marked as noise
```

**Issues:**
- ❌ All pixels classified as noise
- **Solution:** Increase `--clustering-eps` to 0.25 or 0.30

---

## Advanced: Understanding Normalized Feature Space

DBSCAN clusters in **normalized (distance, time, intensity)** space:

```python
# Normalization scales
distance_scale = max(distance_bins) - min(distance_bins)  # e.g., 100
time_scale = num_timepoints                               # e.g., 20
intensity_scale = max(heatmap)                            # e.g., 10000

# Normalized features per pixel
distance_norm = distance / distance_scale      # [0, 1]
time_norm = timepoint / time_scale             # [0, 1]
intensity_norm = intensity / intensity_scale   # [0, 1]
```

**eps** defines the maximum Euclidean distance in this normalized space:
```python
dist = sqrt((d1-d2)² + (t1-t2)² + (i1-i2)²)
```

**Key Insight:** 
- Small `eps` (0.10) → pixels must be close in ALL dimensions
- Large `eps` (0.30) → pixels can be far apart but still cluster

---

## Tips for Your Current Dataset

Based on your command:
```yaml
--smooth-sigma-distance: 3
--use-clustering
--subtract-t0
--clustering-eps: 0.20
--clustering-min-samples: 30
--clustering-intensity-threshold: 50
```

**If clusters are too large, try:**

1. **First attempt** (moderate tightening):
```yaml
--clustering-eps: 0.12
--clustering-intensity-threshold: 60
```

2. **If still too large** (aggressive):
```yaml
--clustering-eps: 0.08
--clustering-min-samples: 50
--clustering-intensity-threshold: 70
```

3. **If too aggressive** (no clusters), back off:
```yaml
--clustering-eps: 0.15
--clustering-min-samples: 40
--clustering-intensity-threshold: 55
```

**Monitor the log output** for:
- Cluster spatial extents (want Δdist < 30, ΔT < 10 for typical photoconversion)
- Selected cluster should have `dist_to_origin < 0.20` (near origin)
- Other clusters should have `dist_to_origin > 0.40` (far from origin = dirt)

---

## Quick Copy-Paste Test Configs

### Test 1: Tighter Clustering
```yaml
- --clustering-eps: 0.12
- --clustering-min-samples: 40
- --clustering-intensity-threshold: 60
```

### Test 2: Very Tight (Aggressive)
```yaml
- --clustering-eps: 0.08
- --clustering-min-samples: 50
- --clustering-intensity-threshold: 70
```

### Test 3: Looser (If Test 2 Too Aggressive)
```yaml
- --clustering-eps: 0.18
- --clustering-min-samples: 35
- --clustering-intensity-threshold: 55
```

---

**Remember:** Always run with `--no-parallel` and `--force-show` during tuning so you can see the clustering visualization and log output immediately!
