import shutil
import os
import argparse
import logging

logger = logging.getLogger(__name__)

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
                logger.info(f"Successfully deleted: {folder}")
            except Exception as e:
                logger.error(f"Error deleting {folder}: {e}")
        else:
            logger.warning(f"Folder does not exist: {folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Delete specified folders.')
    parser.add_argument('--folders', nargs='+', required=True, help='List of folder paths to delete')
    parser.add_argument('--log-level', type=str, default='WARNING', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level (default: WARNING)')

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    delete_folders(args.folders)
