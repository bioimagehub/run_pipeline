#!/usr/bin/env python
import os
import sys
import yaml
import argparse
import subprocess
from tqdm import tqdm

def collapse_filename(file_path, base_folder, delimiter):
    """
    Collapse the file path into a single filename, preserving directory info.
    """
    rel_path = os.path.relpath(file_path, start=base_folder)
    collapsed = delimiter.join(rel_path.split(os.sep))
    return collapsed

def process_folder(args):
    folder_path = args.folder_path
    extension = args.extension
    search_subfolders = args.search_subfolders
    collapse_delimiter = args.collapse_delimiter

    # Gather files (search subfolders if requested)
    files_to_process = []
    if search_subfolders:
        for dirpath, _, filenames in os.walk(folder_path):
            for filename in filenames:
                if filename.endswith(extension):
                    files_to_process.append(os.path.join(dirpath, filename))
    else:
        with os.scandir(folder_path) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(extension):
                    files_to_process.append(entry.path)

    for file_path in tqdm(files_to_process, desc="Processing files", unit="file"):
        
        collapsed_filename = collapse_filename(file_path, folder_path, collapse_delimiter)
        collapsed_filename = os.path.splitext(collapsed_filename)[0] + ".tif"
        destination_folder = folder_path + "_tif"
        os.makedirs(destination_folder, exist_ok=True)
        tif_file_path = os.path.join(destination_folder, collapsed_filename)

        # Command to run the external script
        cmd = [
            "python",  # Ensure the right Python interpreter is chosen
            os.path.join(os.getcwd(), "standard_code",  "bfconvert_file.py"),
            "-i", file_path,
            "-o", tif_file_path,
        ]
        
        # Print the command for troubleshooting
        # print(f"Processing command: {cmd}")

        # Run the command
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Print the output and errors for troubleshooting
        if result.returncode == 0:
            print(f"Successfully processed: {collapsed_filename}")
            print(f"Output: {result.stdout.decode()}")  # Print standard output
        else:
            print(f"Error processing {file_path}: {result.stderr.decode()}")  # Print standard error


def main():
    parser = argparse.ArgumentParser()
    # Arguments for folder processing (default mode)
    parser.add_argument("-p", "--folder_path", type=str, help="Path to the folder to be processed")
    parser.add_argument("-e", "--extension", type=str, default=None, help="File extension to filter files in folder")
    parser.add_argument("-R", "--search_subfolders", action="store_true", help="Search for files in subfolders")
    parser.add_argument("--collapse_delimiter", type=str, default="__", help="Delimiter used to collapse file paths")
    args = parser.parse_args()

    # Ensure the user provides a valid folder path
    while args.folder_path is None or not os.path.isdir(args.folder_path):
        if args.folder_path is not None and not os.path.isdir(args.folder_path):
            print("The provided folder path does not exist.")
        args.folder_path = input("Please provide a valid path to a folder with images: ")
    
    process_folder(args)

if __name__ == "__main__":
    main()
