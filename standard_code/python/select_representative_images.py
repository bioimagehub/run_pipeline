import argparse
from tqdm import tqdm
import os
import logging
import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skimage import filters, measure

# local imports
import bioimage_pipeline_utils as rp  

# ==============================
# FEATURE EXTRACTION
# ==============================

def compute_features(img_2d: np.ndarray):
    img = img_2d.astype(np.float32)

    mean_int = np.mean(img)
    std_int = np.std(img)
    p5 = np.percentile(img, 5)
    p95 = np.percentile(img, 95)

    lap_var = np.var(filters.laplace(img))
    sobel_mean = np.mean(filters.sobel(img))

    try:
        threshold = filters.threshold_otsu(img)
        binary = img > threshold
        fg_fraction = np.sum(binary) / binary.size
        labeled = measure.label(binary)
        n_objects = labeled.max()
    except:
        fg_fraction = 0
        n_objects = 0

    return np.array([
        mean_int,
        std_int,
        p5,
        p95,
        lap_var,
        sobel_mean,
        fg_fraction,
        n_objects
    ])


def extract_timepoint_features(path, channel):
    try:
        img = rp.load_tczyx_image(path)
        img_np = img.data  # TCZYX
        t_dim = img_np.shape[0]

        results = []

        for t in range(t_dim):
            img_2d = img_np[t, channel, 0]  # Z assumed = 1
            feats = compute_features(img_2d)
            results.append((path, t, feats))

        return results

    except Exception as e:
        logging.warning(f"Failed processing {path}: {e}")
        return []


# ==============================
# MAIN LOGIC
# ==============================

def process_folder(args, use_parallel=True):

    files = rp.get_files_to_process2(args.input_search_pattern, False)
    logging.info(f"Found {len(files)} files")

    # ==============================
    # FEATURE EXTRACTION (per timepoint)
    # ==============================

    if use_parallel:
        from contextlib import contextmanager

        @contextmanager
        def _tqdm_joblib(tqdm_object):
            import joblib
            class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
                def __call__(self, *args, **kwargs):
                    tqdm_object.update(n=self.batch_size)
                    return super().__call__(*args, **kwargs)
            old_batch_callback = joblib.parallel.BatchCompletionCallBack
            joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
            try:
                yield tqdm_object
            finally:
                joblib.parallel.BatchCompletionCallBack = old_batch_callback
                tqdm_object.close()

        with tqdm(total=len(files), desc="Extracting features", unit="file") as pbar:
            with _tqdm_joblib(pbar):
                all_results = list(Parallel(n_jobs=rp.resolve_maxcores(args.maxcores, len(files)))(
                    delayed(extract_timepoint_features)(f, args.nuclear_channel)
                    for f in files
                ))
    else:
        all_results = []
        for f in tqdm(files, desc="Extracting features", unit="file"):
            all_results.append(extract_timepoint_features(f, args.nuclear_channel))

    # Flatten list
    flattened = [item for sublist in all_results for item in sublist]

    if len(flattened) == 0:
        logging.error("No valid images found.")
        return

    paths = [x[0] for x in flattened]
    timepoints = [x[1] for x in flattened]
    features = np.vstack([x[2] for x in flattened])

    # ==============================
    # STANDARDIZE + PCA
    # ==============================

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=2)
    embedding = pca.fit_transform(features_scaled)

    # ==============================
    # CLUSTERING
    # ==============================

    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42)
    labels = kmeans.fit_predict(embedding)

    selected_indices = []

    for cluster_id in range(args.n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]

        if len(cluster_indices) == 0:
            continue

        cluster_points = embedding[cluster_indices]
        center = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(cluster_points - center, axis=1)

        sorted_idx = np.argsort(distances)

        n_pick = min(args.samples_per_cluster, len(sorted_idx))
        step = max(1, len(sorted_idx) // n_pick)
        chosen = sorted_idx[::step][:n_pick]

        selected_indices.extend(cluster_indices[chosen])

    logging.info(f"Selected {len(selected_indices)} timepoints")

    # ==============================
    # SAVE SELECTED TIMEPOINTS
    # ==============================

    os.makedirs(args.output_folder, exist_ok=True)


    import h5py
    import json
    for idx in tqdm(selected_indices, desc="Saving selected timepoints", unit="image"):
        path = paths[idx]
        t = timepoints[idx]

        img = rp.load_tczyx_image(path)
        img_np = img.data  # TCZYX
        single_tp = img_np[t:t+1]  # keep T dimension
        base = os.path.basename(path)
        name, ext = os.path.splitext(base)

        for output_format in args.output_format:
            if output_format == "tif":
                output_name = f"{name}{args.output_suffix}_T{t:03d}.tif"
                output_path = os.path.join(args.output_folder, output_name)
                rp.save_tczyx_image(
                    single_tp,
                    output_path,
                    dim_order="TCZYX",
                    physical_pixel_sizes=img.physical_pixel_sizes
                )
            elif output_format == "npy":
                output_name = f"{name}{args.output_suffix}_T{t:03d}.npy"
                output_path = os.path.join(args.output_folder, output_name)
                np.save(output_path, single_tp)
            elif output_format == "ilastik-h5":
                # Export in TZYXC format (channel LAST) to match Ilastik's ImageJ plugin format
                output_name = f"{name}{args.output_suffix}_T{t:03d}.h5"
                output_path = os.path.join(args.output_folder, output_name)
                # Convert TCZYX -> TZYXC
                t_, c, z, y, x = single_tp.shape
                out_img_ilastik = np.transpose(single_tp, (0, 2, 3, 4, 1))  # TCZYX -> TZYXC
                axis_keys = ['t', 'z', 'y', 'x', 'c']
                axis_configs = [
                    {'key': 't', 'typeFlags': 8, 'resolution': 0, 'description': ''},
                    {'key': 'z', 'typeFlags': 2, 'resolution': 0, 'description': ''},
                    {'key': 'y', 'typeFlags': 2, 'resolution': 0, 'description': ''},
                    {'key': 'x', 'typeFlags': 2, 'resolution': 0, 'description': ''},
                    {'key': 'c', 'typeFlags': 1, 'resolution': 0, 'description': ''},
                ]
                with h5py.File(output_path, 'w') as f:
                    dset = f.create_dataset('data', data=out_img_ilastik, compression='gzip', compression_opts=4)
                    dset.attrs['axistags'] = json.dumps({'axes': axis_configs})
            else:
                raise ValueError(f"Unsupported output format: {output_format}")

    logging.info("Finished saving selected timepoints.")


# ==============================
# CLI ENTRY
# ==============================

def parse_output_formats(fmt_str):
    """Parse output format(s) from string, handling both space-separated and single values."""
    if isinstance(fmt_str, list):
        # Already a list from nargs
        return fmt_str
    # Split on spaces if single string with multiple formats
    formats = fmt_str.split()
    valid_formats = {"tif", "npy", "ilastik-h5"}
    for fmt in formats:
        if fmt not in valid_formats:
            raise argparse.ArgumentTypeError(f"Invalid format '{fmt}'. Choose from: tif, npy, ilastik-h5")
    return formats


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Select representative timepoints using stratified clustering."
    )


    parser.add_argument("--input-search-pattern", type=str, required=True,
                        help="Glob pattern for input images, e.g. 'folder/**/*.tif'")

    parser.add_argument("--output-folder", type=str, required=True,
                        help="Folder to save selected timepoints")

    parser.add_argument("--output-suffix", type=str, default="_representative",
                        help="Suffix appended to output filenames before the selected timepoint index")

    parser.add_argument("--nuclear-channel", type=int, default=0,
                        help="Channel index used for feature extraction")

    parser.add_argument("--n-clusters", type=int, default=10,
                        help="Number of variability clusters")

    parser.add_argument("--samples-per-cluster", type=int, default=5,
                        help="Timepoints selected per cluster")

    parser.add_argument("--output-format", type=parse_output_formats, default="tif",
                        help="Output format(s): 'tif' (OME-TIFF), 'npy' (NumPy array), or 'ilastik-h5' (HDF5 for Ilastik). Can specify multiple formats as space-separated list, e.g., 'tif ilastik-h5'")

    parser.add_argument("--no-parallel", action="store_true",
                        help="Disable parallel processing")
    parser.add_argument("--maxcores", type=int, default=None,
                        help="Maximum CPU cores to use for parallel processing (default: all available CPU cores minus 1). Ignored if --no-parallel is set.")
    parser.add_argument("--log-level", type=str, default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (default: WARNING)")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    process_folder(args, use_parallel=not args.no_parallel)