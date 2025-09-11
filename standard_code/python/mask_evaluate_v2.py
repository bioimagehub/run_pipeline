import argparse
import os
from typing import List, Optional
import numpy as np
import pandas as pd
import napari
from bioio_ome_tiff.writers import OmeTiffWriter

#from concurrent.futures import ProcessPoolExecutor, as_completed

# Local imports
import run_pipeline_helper_functions as rp





import argparse
import os
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed

# Local imports
import run_pipeline_helper_functions as rp

import yaml

import numpy as np
import pandas as pd
import os



def flatten(
    image_path: str,
    suffix: str,
    mask: Optional[np.ndarray] = None
) -> Tuple[str, pd.DataFrame]:

    image = rp.load_bioio(image_path).data

    if image.ndim != 5:
        raise ValueError(f"Expected 5D image (TCZYX), got shape {image.shape} in {image_path}")

    T, C, Z, Y, X = image.shape
    basename = os.path.basename(image_path).split(".")[0].replace(suffix, "")

    coords = np.indices((T, Z, Y, X)).reshape(4, -1).T
    df = pd.DataFrame(coords, columns=["Frame", "Z", "Y", "X"])

    if mask is not None:
        mask_flat = mask.reshape(-1)
        valid_idx = np.where(mask_flat > 0)[0]
        df = df.iloc[valid_idx]

    df["basename"] = basename

    for c in range(C):
        data_flat = image[:, c].reshape(-1)
        if mask is not None:
            data_flat = data_flat[valid_idx]
        colname = os.path.splitext(suffix.lstrip("_"))[0]  # Remove leading underscore and extension

        if colname == "":
            df[f"C{c}"] = data_flat
        elif C == 1:
            df[colname] = data_flat
        else:
            df[f"{colname}_C{c}"] = data_flat

    return basename, df




def process_folder(args: argparse.Namespace) -> None:

    image_folders: List[str] = args.image_folders
    image_suffixes: List[List[str]] = args.image_suffixes
    processing_tasks = {}

    # Make a dictionary where the keys are the image basenames (without the suffix at the end) and the values are the full paths to the images
    for folder, suffixes in zip(image_folders, image_suffixes):
        for suffix in suffixes:
            pattern = os.path.join(folder, f"*{suffix}")
            image_files = rp.get_files_to_process2(pattern, False)
            for image_file in image_files:
                basename = os.path.basename(image_file)[:-len(suffix)]
                if basename not in processing_tasks:
                    processing_tasks[basename] = []
                processing_tasks[basename].append(image_file)

    tasks = list(processing_tasks.items())
    
    
    
    start_napari(tasks)


# --- Napari interactive navigation ---
def start_napari(tasks):
    """
    Launch a Napari viewer to browse images in tasks.
    Allows navigation with n/p keys and clickable file list.
    """
    from qtpy.QtWidgets import QListWidget, QVBoxLayout, QWidget
    from qtpy.QtCore import Qt
    import napari

    class TaskNavigator(QWidget):
        def __init__(self, tasks, viewer):
            super().__init__()
            self.tasks = tasks
            self.viewer = viewer
            self.idx = 0
            self.list_widget = QListWidget()
            for name, _ in tasks:
                self.list_widget.addItem(name)
            self.list_widget.currentRowChanged.connect(self.jump_to)
            layout = QVBoxLayout()
            layout.addWidget(self.list_widget)
            self.setLayout(layout)
            self.list_widget.setCurrentRow(0)

        def jump_to(self, idx):
            if 0 <= idx < len(self.tasks):
                self.idx = idx
                self.load_current()

        def load_current(self):
            self.viewer.layers.clear()
            name, files = self.tasks[self.idx]
            if not files:
                return
            
            # First file is image
            image_path = files[0]
            image = rp.load_bioio(image_path).dask_data # TCZYX             
            self.viewer.add_image(image, name=name)
            
            # All other files are masks
            for mask_path in files[1:]:
                mask = rp.load_bioio(mask_path).dask_data # TCZYX
                mask_layer_name_base = os.path.splitext(os.path.basename(mask_path))[0]
                mask_layer_name_base = mask_layer_name_base.replace(name, "")  # Remove basename from mask name

                # Set visibility: only '_segmentation' is visible, others are hidden
                visible = mask_layer_name_base == "_segmentation"

                # If mask has more than one channel, split and add each as a separate label layer
                if mask.shape[1] > 1:
                    for c in range(mask.shape[1]):
                        mask_c = mask[c]
                        mask_data = np.repeat(mask_c, image.shape[1], axis=1)
                        self.viewer.add_labels(mask_data, name=f"{mask_layer_name_base}_C{c}", visible=visible)
                else:
                    mask_data = np.repeat(mask, image.shape[1], axis=1)
                    self.viewer.add_labels(mask_data, name=mask_layer_name_base, visible=visible)

            self.viewer.status = f"{name} ({self.idx+1}/{len(self.tasks)})"
            self.list_widget.setCurrentRow(self.idx)

        def next(self):
            if self.idx < len(self.tasks) - 1:
                self.idx += 1
                self.load_current()

        def prev(self):
            if self.idx > 0:
                self.idx -= 1
                self.load_current()

    viewer = napari.Viewer()
    nav = TaskNavigator(tasks, viewer)
    dock = viewer.window.add_dock_widget(nav, area='right', name='Task List')

    @viewer.bind_key('n')
    def _next(event):
        nav.next()

    @viewer.bind_key('p')
    def _prev(event):
        nav.prev()

    nav.load_current()
    napari.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folders", type=str, nargs= "+", required=True,
                        help="Path to folders with images to flatten")
    parser.add_argument("--image-suffixes", type=rp.split_comma_separated_strstring, nargs= "+", required=True,
                        help="Suffixes of input images; CSV-style list per folder")
    args = parser.parse_args()

    process_folder(args = args)

