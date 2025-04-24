from bioio import BioImage
import os


def load_bioio(path: str) -> BioImage:
    """
    Load a BioImage object from a file. The file format is determined by the file extension.
    :param args: Command line arguments containing the input file path.
    :return: A BioImage object.
    """
    # Load the image using the appropriate reader based on the file extension   
    if path.endswith(".tif"):
        import bioio_ome_tiff
        img = BioImage(path, reader=bioio_ome_tiff.Reader)
        
    elif path.endswith(".nd2"):
        import bioio_nd2
        img = BioImage(path, reader=bioio_nd2.Reader)
    
    # TODO Add more readers here if needed
    else:
        # Bioformats has this annoying printout so I prefer to use a different reader 
        import bioio_bioformats
        img = BioImage(path, reader=bioio_bioformats.Reader)
    
    return img

def get_files_to_process(folder_path: str, extension: str, search_subfolders: bool) -> list:
    """
    Get a list of files in the specified folder with the specified extension.
    """
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
    
    # replace \ wit h / in the file paths
    files_to_process = [file_path.replace("\\", "/") for file_path in files_to_process]
    return files_to_process


def collapse_filename(file_path: str, base_folder: str, delimiter: str = "__") -> str:
    """
    Collapse a full file path into a single string filename that encodes the relative
    path structure using a custom delimiter.

    Parameters:
    - file_path: The full path to the file.
    - base_folder: The base directory from which the relative path is derived.
    - delimiter: The string used to replace path separators (default: "__").

    Returns:
    - A string that represents the relative path as a flat filename, with
      directory separators replaced by the delimiter.
    """
    # Compute the path relative to the base folder
    rel_path = os.path.relpath(file_path, start=base_folder)
    
    # Replace os-specific separators with the chosen delimiter
    collapsed = delimiter.join(rel_path.split(os.sep))
    
    return collapsed

def uncollapse_filename(collapsed: str, base_folder: str, delimiter: str = "__") -> str:
    """
    Reconstruct the original file path from a collapsed filename.
    
    Parameters:
    - collapsed: The collapsed filename string.
    - base_folder: The base directory to prepend to the reconstructed path.
    - delimiter: The delimiter used in the collapse_filename function.
    
    Returns:
    - The reconstructed original file path.
    """
    parts = collapsed.split(delimiter)
    rel_path = os.path.join(*parts)

    original_path:str = os.path.join(base_folder, rel_path)
    return original_path
