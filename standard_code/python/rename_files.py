import os
import argparse

def rename_files(target_directory, search_term, replace_term):
    """Renames files in the target directory that contain search_term to replace_term."""
    try:
        for filename in os.listdir(target_directory):
            if search_term in filename:
                old_file = os.path.join(target_directory, filename)
                new_filename = filename.replace(search_term, replace_term)
                new_file = os.path.join(target_directory, new_filename)
                os.rename(old_file, new_file)
                print(f'Renamed: {old_file} to {new_file}')
    except Exception as e:
        print(f'An error occurred: {e}')

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Rename files in the target directory by replacing a search term with a replace term.')
    parser.add_argument('--directory', type=str, help='The target directory to rename files in')
    parser.add_argument('--search_term', type=str, help='The term to search for in the file names')
    parser.add_argument('--replace_term', type=str, help='The term to replace the search term with in the file names')

    # Parse the arguments
    args = parser.parse_args()

    # Call the rename function with the provided directory, search term, and replace term
    rename_files(args.directory, args.search_term, args.replace_term)

if __name__ == "__main__":
    main()
