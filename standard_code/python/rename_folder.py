import os
import shutil
import argparse

def merge_folders(source_folder, target_folder):
    """Merges the source_folder into target_folder, renaming conflicting files."""
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for item in os.listdir(source_folder):
        source_path = os.path.join(source_folder, item)
        target_path = os.path.join(target_folder, item)

        # Handling files
        if os.path.isfile(source_path):
            if os.path.exists(target_path):
                # Rename the file in the target folder to avoid conflict
                base, ext = os.path.splitext(item)
                suffix = 1
                while os.path.exists(target_path):
                    new_name = f"{base}_file_conflict_{suffix}{ext}"
                    target_path = os.path.join(target_folder, new_name)
                    suffix += 1

            shutil.move(source_path, target_path)
            print(f'Moved file: {source_path} to {target_path}')

        # Handling directories
        elif os.path.isdir(source_path):
            merge_folders(source_path, target_path)

def rename_or_move_folder(old_folder_path, new_folder_path):
    """Renames or moves the folder to the new path, merging if necessary."""
    
    if not os.path.exists(old_folder_path):
        print(f"The folder '{old_folder_path}' does not exist.")
        return

    # If the new folder already exists, merge it
    if os.path.exists(new_folder_path):
        print(f"The folder '{new_folder_path}' already exists. Merging...")
        merge_folders(old_folder_path, new_folder_path)
        shutil.rmtree(old_folder_path)  # Remove the original folder after merging
        print(f'Merged and removed original folder: {old_folder_path}')
    else:
        shutil.move(old_folder_path, new_folder_path)
        print(f'Renamed/moved folder: {old_folder_path} to {new_folder_path}')

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Rename or move a folder and merge if necessary.')
    parser.add_argument('--old_folder_path', type=str, help='The path to the folder to rename or move')
    parser.add_argument('--new_folder_path', type=str, help='The new path for the folder')

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with provided folder paths
    rename_or_move_folder(args.old_folder_path, args.new_folder_path)

if __name__ == "__main__":
    main()
