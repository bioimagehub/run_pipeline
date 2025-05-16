import shutil
import os
import argparse

def delete_folders(folders):
    """
    Deletes the specified folders.

    Parameters:
        folders (list): A list of folder paths to delete.
    """
    for folder in folders:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)  # Delete the folder and its contents
                print(f"Successfully deleted: {folder}")
            except Exception as e:
                print(f"Error deleting {folder}: {e}")
        else:
            print(f"Folder does not exist: {folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Delete specified folders.')
    parser.add_argument('folders', nargs='+', help='List of folder paths to delete')

    args = parser.parse_args()

    delete_folders(args.folders)
