import os
import argparse
from typing import Optional, List, Dict, Any, Iterable

def _build_file_info(
    *,
    input_path: str,
    output_dir: Optional[str],
    ch: int,
    t_start: int,
    t_end: Optional[int],
) -> Any:
    """
    Construct a verified FileInfo object and select channel/time range.
    """
    from nellie.im_info.verifier import FileInfo

    fi = FileInfo(input_path, output_dir=output_dir)
    fi.find_metadata()
    fi.load_metadata()
    try:
        fi.change_selected_channel(ch)
    except Exception:
        pass
    try:
        fi.select_temporal_range(start=t_start, end=t_end)
        fi._get_output_path()
    except Exception:
        pass
    return fi

def segment_image(
    *,
    input_path: str,
    output_dir: Optional[str] = None,
    ch: int = 0,
    t_start: int = 0,
    t_end: Optional[int] = None,
    remove_edges: bool = False,
    otsu_thresh_intensity: bool = False,
    threshold: Optional[float] = None,
    cleanup_intermediates: bool = False,
) -> Dict[str, Any]:
    """
    Run Nellie segmentation on a single image using the pip-installed API.
    """
    from nellie.run import run as nellie_run

    fi = _build_file_info(
        input_path=input_path,
        output_dir=output_dir,
        ch=ch,
        t_start=t_start,
        t_end=t_end,
    )

    im_info = nellie_run(
        fi,
        remove_edges=remove_edges,
        otsu_thresh_intensity=otsu_thresh_intensity,
        threshold=threshold,
    )

    if cleanup_intermediates:
        try:
            im_info.remove_intermediates()
        except Exception:
            pass

    return {
        "output_dir": fi.output_dir,
        "user_output_base": getattr(fi, "user_output_path_no_ext", None),
        "pipeline_paths": getattr(im_info, "pipeline_paths", {}),
        "im_info": im_info,
        "file_info": fi,
    }

def segment_directory(
    *,
    directory: str,
    substring: str,
    output_dir: Optional[str],
    ch: int = 0,
    num_t: Optional[int] = None,
    remove_edges: bool = False,
    otsu_thresh_intensity: bool = False,
    threshold: Optional[float] = None,
) -> Iterable[Dict[str, Any]]:
    """
    Mirror of the CLI directory runner, but programmatic.
    """
    files = sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if substring in f and (f.endswith('.tiff') or f.endswith('.tif'))
    )
    t_end = None if num_t is None else max(0, num_t - 1)
    for idx, path in enumerate(files, 1):
        print(f"Processing file {idx} of {len(files)}, channel {ch + 1} of 1")
        try:
            yield segment_image(
                input_path=path,
                output_dir=output_dir,
                ch=ch,
                t_start=0,
                t_end=t_end,
                remove_edges=remove_edges,
                otsu_thresh_intensity=otsu_thresh_intensity,
                threshold=threshold,
            )
        except Exception as e:
            print(f"Failed to run {path}: {e}")
            continue

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Nellie segmentation using the installed pip package (no external CLI).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input-search-pattern', required=True, help='Glob pattern for input images, e.g. "folder/*.tif" or "folder/somefile*.tif". Use a single file path for one image.')
    parser.add_argument('--output_dir', help='Where to place outputs', default=None)
    parser.add_argument('--ch', type=int, default=0, help='Channel index to process')
    parser.add_argument('--t_start', type=int, default=0, help='Start time index (file mode)')
    parser.add_argument('--t_end', type=int, default=None, help='End time index inclusive (file mode)')
    parser.add_argument('--remove_edges', action='store_true', help='Enable edge removal in preprocessing')
    parser.add_argument('--otsu_thresh_intensity', action='store_true', help='Use Otsu threshold for intensity')
    parser.add_argument('--threshold', type=float, default=None, help='Manual intensity threshold')
    parser.add_argument('--cleanup_intermediates', action='store_true', help='Remove intermediate non-CSV files')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing (default: parallel enabled)')
    args = parser.parse_args(argv)

    import run_pipeline_helper_functions as rp
    from tqdm import tqdm
    image_files = rp.get_files_to_process(args.input_search_pattern, '', False)
    is_batch = len(image_files) > 1
    output_dir = args.output_dir
    if output_dir is None and is_batch:
        output_dir = os.path.dirname(args.input_search_pattern)
    if is_batch and output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def process_job(image_path):
        try:
            res = segment_image(
                input_path=image_path,
                output_dir=output_dir,
                ch=args.ch,
                t_start=args.t_start,
                t_end=args.t_end,
                remove_edges=args.remove_edges,
                otsu_thresh_intensity=args.otsu_thresh_intensity,
                threshold=args.threshold,
                cleanup_intermediates=args.cleanup_intermediates,
            )
            print(f"Output dir: {res['output_dir']}")
            print(f"User output base: {res['user_output_base']}")
            print("Pipeline outputs:")
            for k, v in sorted(res["pipeline_paths"].items()):
                print(f"  {k}: {v}")
            return (image_path, None)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return (image_path, str(e))

    if not args.no_parallel and is_batch:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_job, img) for img in image_files]
            for f in tqdm(as_completed(futures), total=len(image_files), desc='Segmenting'):
                _, err = f.result()
                if err:
                    print(f"Error: {err}")
    else:
        for img in tqdm(image_files, desc='Segmenting'):
            _, err = process_job(img)
            if err:
                print(f"Error: {err}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
