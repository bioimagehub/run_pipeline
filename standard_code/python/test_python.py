import argparse
import os


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Rename files in the target directory by replacing a search term with a replace term.')
    parser.add_argument('--name', type=str, help='write a string')
    
    # Parse the arguments
    args = parser.parse_args()

    environment_name = os.environ['CONDA_DEFAULT_ENV']

    # Call the rename function with the provided directory, search term, and replace term
    print(f"Hello {args.name} you are running a python command in the {environment_name} environment" )

if __name__ == "__main__":
    main()

