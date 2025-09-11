
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='Test script for environment and file processing.')
    parser.add_argument('--input-search-pattern', type=str, required=True, help='Glob pattern to search for input files (e.g. "./data/*.tif")')
    parser.add_argument('--no-parallel', action='store_true', help='Do not use parallel processing.')
    parser.add_argument('--name', type=str, help='write a string')
    args = parser.parse_args()

    environment_name = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')

    # Example: process files using get_files_to_process
    try:
        import run_pipeline_helper_functions as rp
        files = rp.get_files_to_process(args.input_search_pattern, '', False)
    except Exception as e:
        files = []
        print(f"Could not import or run get_files_to_process: {e}")

    def process_single_file(file):
        print(f"Processing file: {file}")

    if not args.no_parallel:
        try:
            from joblib import Parallel, delayed
            Parallel(n_jobs=-1)(delayed(process_single_file)(file) for file in files)
        except ImportError:
            print("joblib not installed, running sequentially.")
            for file in files:
                process_single_file(file)
    else:
        for file in files:
            process_single_file(file)

    print(f"Hello {args.name} you are running a python command in the {environment_name} environment")

if __name__ == "__main__":
    main()

