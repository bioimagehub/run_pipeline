# This is a troubleshooting script for drift correction in image processing.

import numpy as np

import bioimage_pipeline_utils as rp


import imagej as ij


def run_correct_3d_drift_with_imagej(input_path: str, output_path: str, max_frames: int | None = None, headless: bool = True) -> None:
    """
    Run ImageJ's "Correct 3D Drift" plugin on an input image and save the corrected stack.

    This function standardizes I/O via `bioimage_pipeline_utils`, exports a temporary OME-TIFF
    for ImageJ, executes the plugin via an IJ1 macro, and writes the result to `output_path`.

    Args:
        input_path: Path to the input image (any format supported by rp.load_tczyx_image).
        output_path: Destination OME-TIFF path for the corrected stack.
        max_frames: Optionally limit the number of timepoints for faster debugging.
        headless: Initialize ImageJ in headless mode.
    """
    import os
    import tempfile
    from pathlib import Path

    # 1) Load image using rp to ensure 5D TCZYX standardization
    img = rp.load_tczyx_image(input_path)
    data = img.data
    if max_frames is not None:
        data = data[:max_frames, ...]

    # 2) Export a temporary ImageJ-compatible TIFF for robust macro opening
    import tifffile
    tmp_dir = tempfile.mkdtemp(prefix="drift3d_")
    tmp_tif = os.path.join(tmp_dir, "input_hyperstack_imagej.tif")
    # Reorder to ImageJ hyperstack axes TZCYX with trailing S (samples) of size 1
    ij_data = np.transpose(data, (0, 2, 1, 3, 4))  # TCZYX -> TZCYX
    ij_data = np.expand_dims(ij_data, axis=-1)     # add S=1
    tifffile.imwrite(tmp_tif, ij_data, imagej=True, metadata={'axes': 'TZCYXS'})

    # Ensure output folder exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 3) Initialize ImageJ/Fiji and run macro
    # Initialize with modern API; fall back to headless flag if needed
    try:
        mode = "headless" if headless else "gui"
        ij_instance = ij.init(mode=mode)
    except Exception:
        ij_instance = ij.init(headless=headless)

    # Use forward slashes for IJ macros on Windows
    tmp_tif_macro = tmp_tif.replace("\\", "/")
    output_macro = output_path.replace("\\", "/")

    macro = f"""
        setBatchMode(true);
        open("{tmp_tif_macro}");
        // Run ImageJ plugin (defaults); if plugin needs args, adapt here
        run("Correct 3D Drift");
        // Save corrected result
        saveAs("Tiff", "{output_macro}");
        run("Close All");
    """

    try:
        ij_instance.py.run_macro(macro)
    except Exception:
        # Try running the bundled Fiji Jython script with programmatic options (no dialogs)
        script_path = os.path.join(str(Path(__file__).resolve().parents[1]), "imagej", "correct_3d_drift.py")
        with open(script_path, "r", encoding="utf-8") as fh:
            base_script = fh.read()

        tail = f"""
imp = IJ.openImage("{tmp_tif_macro}")

options = {{
  'channel': 1,
  'correct_only_xy': False,
  'multi_time_scale': False,
  'subpixel': False,
  'process': False,
  'background': 0,
  'z_min': 1,
  'z_max': imp.getNSlices(),
  'max_shifts': [50, 50, 50],
  'virtual': False,
  'only_compute': False
}}

IJ.log("Computing drift (dt=1)...")
dt = 1
shifts = compute_and_update_frame_translations_dt(imp, dt, options)
shifts = invert_shifts(shifts)
shifts = convert_shifts_to_integer(shifts)
registered_imp = register_hyperstack(imp, shifts, None, options['virtual'])

fs = FileSaver(registered_imp)
fs.saveAsTiff("{output_macro}")
"""

        full_script = base_script + "\n\n" + tail
        ij_instance.py.run_script("python", full_script, {})
    finally:
        # Best-effort cleanup of temp input file
        try:
            if os.path.exists(tmp_tif):
                os.remove(tmp_tif)
            if os.path.isdir(tmp_dir):
                os.rmdir(tmp_dir)
        except Exception:
            pass

    # 4) Verify output; if plugin missing, fallback to PyStackReg
    if not os.path.exists(output_path):
        try:
            from standard_code.python.drift_correction_utils.translation_pystackreg import register_image_xy
        except Exception:
            from drift_correction_utils.translation_pystackreg import register_image_xy
        reg_img, _ = register_image_xy(img, reference='previous', channel=0, show_progress=False)
        rp.save_tczyx_image(reg_img, output_path)


def main():

    path = r"C:\Users\oyvinode\Desktop\del_drift_correction_test\input\allLCISHGF_Acxtivator_t5s_005_2.nd2"
    outpath = r"C:\Users\oyvinode\Desktop\del_drift_correction_test\input_drift\allLCISHGF_Acxtivator_t5s_005_2_corrected.tif"
    debug_mode = True

    print("Loading input and invoking ImageJ Correct 3D Drift...")
    max_frames = 10 if debug_mode else None
    run_correct_3d_drift_with_imagej(path, outpath, max_frames=max_frames, headless=False)
    print(f"Done. Output saved to: {outpath}")





if __name__ == "__main__":
    main()



