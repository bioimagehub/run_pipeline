# Jython script for Bio-Formats import/export via Fiji headless mode
# This script is called by biphub_multifile_open_v4.mac
# Parameters are passed via command line in format: key='value'
#
# Example usage:
# fiji --headless --run "ij_bridge_bioformats.py" "filePath='D:/input.ims', outputPath='D:/output.tif', outputFolder='D:/temp'"

#@ File filePath


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



# Create output folder as subfolder with same name as input file (without extension)
input_path = str(filePath)
input_dir = os.path.dirname(input_path)
input_basename = os.path.splitext(os.path.basename(input_path))[0]
outputFolder = os.path.join(input_dir, input_basename)


# Create output folder if it doesn't exist, or clean it if it does
output_dir = File(outputFolder)
if not output_dir.exists():
    output_dir.mkdirs()

# Import using Bio-Formats with options (proper API way)
options = ImporterOptions()
options.setId(str(filePath))
options.setColorMode(ImporterOptions.COLOR_MODE_COMPOSITE)
imps = BF.openImagePlus(options)

if imps and len(imps) > 0:
    imp = imps[0]  # Get first image
    
    # Get image dimensions
    width = imp.getWidth()
    height = imp.getHeight()
    nChannels = imp.getNChannels()
    nSlices = imp.getNSlices()
    nFrames = imp.getNFrames()
    stack = imp.getStack()
    
    print("Image dimensions: " + str(width) + "x" + str(height) + 
          ", C=" + str(nChannels) + ", Z=" + str(nSlices) + ", T=" + str(nFrames))
    
    # Export using Bio-Formats Exporter with write_each_z_section write_each_channel
    basename = os.path.splitext(os.path.basename(str(filePath)))[0]
    out_file = os.path.join(outputFolder, basename + ".ome.tif")
    
    # Use Bio-Formats Exporter to split by Z and channel
    export_params = "save=" + out_file + " write_each_z_section write_each_timepoint write_each_channel export compression=Uncompressed"
    IJ.run(imp, "Bio-Formats Exporter", export_params)
    
    imp.close()
    
    print("Export complete: " + str(nSlices * nChannels) + " files created")
    
    # Create done marker file
    done_file = File(outputFolder, "done.txt")
    from java.io import FileWriter
    writer = FileWriter(done_file)
    writer.write("done")
    writer.close()
    
    # Check for NIS-Elements and open first exported file
    nis_path = "C:\\Program Files\\NIS-Elements\\nis_ar.exe"
    nis_exe = File(nis_path)
    
    if nis_exe.exists():
        print("Found NIS-Elements at: " + nis_path)
        
        # Find first .ome.tif file in output folder
        output_files = output_dir.listFiles()
        first_ome_tif = None
        for f in output_files:
            if f.getName().endswith(".ome.tif"):
                first_ome_tif = f
                break
        
        if first_ome_tif:
            print("Opening in NIS-Elements: " + first_ome_tif.getAbsolutePath())
            
            # Create NIS-Elements macro to save as .nd2
            # Build output path: same as input but with .nd2 extension
            nd2_output = os.path.splitext(input_path)[0] + ".nd2"
            # Escape backslashes for macro string
            nd2_output_escaped = nd2_output.replace("\\", "\\\\")
            
            macro_path = os.path.join(outputFolder, "tmp_nis_macro.mac")
            macro_file = File(macro_path)
            from java.io import FileWriter
            macro_writer = FileWriter(macro_file)
            macro_content = 'ImageSaveAs("' + nd2_output_escaped + '",14,0);\n'
            macro_content += 'CloseAllDocuments(0);\n'
            macro_writer.write(macro_content)
            macro_writer.close()
            print("Created NIS macro: " + macro_path)
            print("Macro will save to: " + nd2_output)
            
            # Build command to open file in NIS-Elements and execute macro
            import subprocess
            cmd = [nis_path, "-f", first_ome_tif.getAbsolutePath(), "-mv", macro_path]
            subprocess.Popen(cmd)
            print("NIS-Elements launched with file and executing macro to save as .nd2")
        else:
            print("No .ome.tif files found in output folder")
    else:
        print("Could not find NIS-Elements at: " + nis_path)
        print("You can manually select a file from the output folder and open it in NIS-Elements:")
        print("  Output folder: " + outputFolder)
else:
    print("ERROR: Could not open image: " + str(filePath))
