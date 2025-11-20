# Jython script for Bio-Formats import/export via Fiji headless mode
# This script is called by biphub_multifile_open_v4.mac
# Parameters are passed via command line in format: key='value'
#
# Example usage:
# fiji --headless --run "ij_bridge_bioformats.py" "filePath='D:/input.ims', outputPath='D:/output.tif', outputFolder='D:/temp'"


#@ File (label = "Input file", style = "file") filePath
#@ File (label = "NIS-Elements AR path", style = "file", value =  "C:\\Program Files\\NIS-Elements\\nis_ar.exe") nis_path
#@ Boolean (label = "Process all files in folder with same extension") processAll


import shutil
import time
from loci.plugins import BF
from loci.plugins.in import ImporterOptions
from loci.formats import ImageReader, ImageWriter
from loci.formats.meta import IMetadata
from loci.formats import MetadataTools
from loci.common import DataTools
from ij import IJ, ImageStack, ImagePlus
from ij.process import ColorProcessor
from java.io import File
import os


def process_single_file(input_file_path, nis_exe_path):
    """Process a single file: convert to OME-TIF, import to NIS-Elements, save as ND2"""
    
    # Create output folder as subfolder with same name as input file (without extension)
    input_path = str(input_file_path)
    input_dir = os.path.dirname(input_path)
    input_basename = os.path.splitext(os.path.basename(input_path))[0]
    
    # Check if .nd2 file already exists - skip if it does
    nd2_check = os.path.splitext(input_path)[0] + ".nd2"
    if os.path.exists(nd2_check):
        print("Skipping - .nd2 already exists: " + nd2_check)
        return
    
    outputFolder = os.path.join(input_dir, input_basename)
    
    # Create output folder if it doesn't exist, or clean it if it does
    output_dir = File(outputFolder)
    if not output_dir.exists():
        output_dir.mkdirs()
    
    # Import using Bio-Formats with options (proper API way)
    options = ImporterOptions()
    options.setId(input_path)
    options.setColorMode(ImporterOptions.COLOR_MODE_COMPOSITE)
    imps = BF.openImagePlus(options)
    
    if imps and len(imps) > 0:
        print("Found " + str(len(imps)) + " series/image(s) in file")
        
        # Process each series/image
        for series_idx, imp in enumerate(imps):
            series_num = series_idx + 1
            print("\n--- Processing series " + str(series_num) + " of " + str(len(imps)) + " ---")
            
            # Get image dimensions
            width = imp.getWidth()
            height = imp.getHeight()
            nChannels = imp.getNChannels()
            nSlices = imp.getNSlices()
            nFrames = imp.getNFrames()
            stack = imp.getStack()
            
            print("Image dimensions: " + str(width) + "x" + str(height) + 
                  ", C=" + str(nChannels) + ", Z=" + str(nSlices) + ", T=" + str(nFrames))
            
            # Create series-specific output folder if multiple series
            if len(imps) > 1:
                series_outputFolder = outputFolder + "_series" + str(series_num)
                series_output_dir = File(series_outputFolder)
                if not series_output_dir.exists():
                    series_output_dir.mkdirs()
            else:
                series_outputFolder = outputFolder
                series_output_dir = output_dir
            
            # Export using Bio-Formats Exporter with write_each_z_section write_each_channel
            basename = os.path.splitext(os.path.basename(input_path))[0]
            if len(imps) > 1:
                basename = basename + "_series" + str(series_num)
            out_file = os.path.join(series_outputFolder, basename + ".ome.tif")
            
            # Use Bio-Formats Exporter to split by Z and channel
            export_params = "save=" + out_file + " write_each_z_section write_each_timepoint write_each_channel export compression=Uncompressed"
            IJ.run(imp, "Bio-Formats Exporter", export_params)
            
            imp.close()
            
            print("Export complete: " + str(nSlices * nChannels) + " files created")
            
            # Create done marker file
            done_file = File(series_outputFolder, "done.txt")
            from java.io import FileWriter
            writer = FileWriter(done_file)
            writer.write("done")
            writer.close()
            
            # Check for NIS-Elements and open first exported file
            nis_exe = File(nis_exe_path)
            
            if nis_exe.exists():
                print("Found NIS-Elements at: " + nis_exe_path)
                
                # Find first .ome.tif file in output folder
                output_files = series_output_dir.listFiles()
                first_ome_tif = None
                for f in output_files:
                    if f.getName().endswith(".ome.tif"):
                        first_ome_tif = f
                        break
                
                if first_ome_tif:
                    print("Opening in NIS-Elements: " + first_ome_tif.getAbsolutePath())
                    
                    # Create NIS-Elements macro to save as .nd2
                    # Build output path: same as input but with .nd2 extension (and series suffix if multiple)
                    if len(imps) > 1:
                        nd2_output = os.path.splitext(input_path)[0] + "_series" + str(series_num) + ".nd2"
                    else:
                        nd2_output = os.path.splitext(input_path)[0] + ".nd2"
                    # Escape backslashes for macro string
                    nd2_output_escaped = nd2_output.replace("\\", "\\\\")
                    
                    macro_path = os.path.join(series_outputFolder, "tmp_nis_macro.mac")
                    macro_file = File(macro_path)
                    from java.io import FileWriter
                    macro_writer = FileWriter(macro_file)
                    macro_content = 'ImageSaveAs("' + nd2_output_escaped + '",14,0);\n'
                    macro_content += 'CloseAllDocuments(0);\n'
                    # Add Python command to create done.txt marker file
                    # Use forward slashes and single quotes exactly as in working example
                    outputFolder_forward = series_outputFolder.replace("\\", "/")
                    macro_content += "Python_RunString(\"import os; "
                    macro_content += "done_path = os.path.join('" + outputFolder_forward + "', 'done.txt'); "
                    macro_content += "open(done_path, 'w').write('done')\");\n"
                    macro_writer.write(macro_content)
                    macro_writer.close()
                    print("Created NIS macro: " + macro_path)
                    print("Macro will save to: " + nd2_output)
                    
                    # Build command to open file in NIS-Elements and execute macro
                    import subprocess
                    cmd = [nis_exe_path, "-f", first_ome_tif.getAbsolutePath(), "-mv", macro_path]
                    subprocess.Popen(cmd)
                    print("NIS-Elements launched with file and executing macro to save as .nd2")

                    # Wait for both the .nd2 file AND the done marker from NIS-Elements
                    done_marker = os.path.join(series_outputFolder, "done.txt")
                    print("Waiting for NIS-Elements to complete...")
                    while not (os.path.exists(nd2_output) and os.path.exists(done_marker)):
                        time.sleep(1)
                    
                    time.sleep(10)
                    
                    print("NIS-Elements completed: .nd2 saved and done marker created")
                    
                    # Clean up temporary folder after launching NIS-Elements
                    try:
                        shutil.rmtree(series_outputFolder)
                        print("Cleaned up temporary folder: " + series_outputFolder)
                    except Exception as e:
                        print("Note: Could not delete temporary folder (may still be in use): " + str(e))
                else:
                    print("No .ome.tif files found in output folder")
            else:
                print("Could not find NIS-Elements at: " + nis_exe_path)
                print("You can manually select a file from the output folder and open it in NIS-Elements:")
                print("  Output folder: " + series_outputFolder)
            
            print("--- Completed series " + str(series_num) + " ---\n")
    else:
        print("ERROR: Could not open image: " + input_path)


def run():
    """Main entry point - process single file or all files in folder"""
    input_path_str = str(filePath)
    nis_path_str = str(nis_path)
    
    if processAll:
        # Get the directory and extension from the input file
        input_dir = os.path.dirname(input_path_str)
        input_ext = os.path.splitext(input_path_str)[1]
        
        print("Processing all files in folder: " + input_dir)
        print("Looking for files with extension: " + input_ext)
        
        # Get all files in the directory with the same extension
        all_files = []
        for filename in os.listdir(input_dir):
            if filename.endswith(input_ext):
                all_files.append(os.path.join(input_dir, filename))
        
        all_files.sort()
        print("Found " + str(len(all_files)) + " files to process")
        
        # Process each file
        for i, file_path in enumerate(all_files):
            print("\n=== Processing file " + str(i+1) + " of " + str(len(all_files)) + " ===")
            print("File: " + file_path)
            process_single_file(file_path, nis_path_str)
            print("=== Completed file " + str(i+1) + " ===\n")
        
        print("All files processed!")
    else:
        # Process single file
        print("Processing single file: " + input_path_str)
        process_single_file(input_path_str, nis_path_str)


# Run the main function
run()
