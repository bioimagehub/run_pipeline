#@ File (label = "Input directory", style = "directory", value="E:/Coen/Sarah/6849908-IMB-Coen-Sarah-Photoconv_global/cellprofiler_input") input
#@ String(label="File suffix", value=".tif") suffix
#@ Integer(label="Minimum size", value=35000) minsize
#@ Integer(label="Maximum size", value=120000) maxsize
#@ Integer(label="median_xy", value=8) median_xy
#@ Integer(label="median_z", value=2) median_z
max_objects = 1 

"""
ImageJ/Fiji Python (Jython) script for simple threshold-based segmentation
Command-line compatible version for run_pipeline.exe

Usage:
  ImageJ-win64.exe --headless --console --run simple_threshold.py "input='E:/path/to/input',suffix='.tif',minsize=35000,maxsize=120000"
"""

import os
import sys

from ij import IJ, ImagePlus
from ij import IJ, WindowManager

from ij.gui import Roi

from ij.plugin import ChannelSplitter
from ij.plugin.filter import ParticleAnalyzer as PA
from ij.plugin.frame import RoiManager

from ij.measure import ResultsTable, Measurements
from ij.measure import Calibration

from ij.process import ImageProcessor

from loci.plugins import BF


 
def getRoisByParticleAnalysis(mask, minsize=0.01, maxsize=100.0, mincirc=0.1, maxcirc=1.0, pixel_units=True, exclude_edge=True):
    """
    Function to get array of rois from input mask.
    Input: 
    mask imp- 2D mask of objects (white objects on black background) 
    min and max size objects- in pixels if pixel_units=True, otherwise image calibration units
    min and max circ- between 0 and 1, 1 being perfect circle
    pixel_units- if True, temporarily removes calibration to use pixel measurements (default: True)
    exclude_edge- if True, excludes particles touching image edges (default: True)
    Returns: array of rois
    """
    
    # Save and remove calibration if pixel_units requested
    original_cal = None
    if pixel_units:
        original_cal = mask.getCalibration().copy()
        pixel_cal = Calibration()
        mask.setCalibration(pixel_cal)
    
    # Measurements to record
    measurements = Measurements.AREA + Measurements.PERIMETER 
    
    # Options for particle analyzer
    options = PA.ADD_TO_MANAGER + PA.SHOW_NONE
    if exclude_edge:
        options += PA.EXCLUDE_EDGE_PARTICLES
    
    roi_manager = RoiManager.getInstance()
    if not roi_manager:
        roi_manager = RoiManager()
        
    rtt = ResultsTable()
    
    pa = PA(options, measurements, rtt, minsize, maxsize, mincirc, maxcirc)
    pa.setHideOutputImage(True)
    pa.analyze(mask)
    rois_array = roi_manager.getRoisAsArray()
    
    # Restore original calibration if it was modified
    if original_cal is not None:
        mask.setCalibration(original_cal)

    return rois_array


# Convert java.io.File to string
input = str(input.getAbsolutePath())

# Processing parameters
#median_xy = 8
#median_z = 2



# Parse parameters (these come from script parameters above)
print("Input directory: " + str(input))
print("File suffix: " + str(suffix))
print("Min size: " + str(minsize))
print("Max size: " + str(maxsize))

# Validate input directory
if not os.path.exists(input):
    print("ERROR: Input directory does not exist: " + input)
    sys.exit(1)

# Create output directory
output = input + "_threshold"
if not os.path.exists(output):
    os.makedirs(output)

def process_file(input_dir, output_dir, filename):
    """Process a single image file"""
    output_basename = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, output_basename + "_mask.tif")
    
    # Skip if already processed
    if os.path.exists(output_path):
        print("Skipping (already exists): " + filename)
        return
    
    input_path = os.path.join(input_dir, filename)
    print("Processing: " + input_path)
    
    # Open image using Bio-Formats
    imps = BF.openImagePlus(input_path)
    
    for imp in imps:
        if imp is None:
            print("ERROR: Could not open image: " + filename)
            return
        
        imp.setTitle("input")
        
        # Get dimensions
        nFrames = imp.getNFrames()  # Number of timepoints
        nSlices = imp.getNSlices()  # Number of Z-slices
        nChannels = imp.getNChannels()
        
        print("Image dimensions - Frames: " + str(nFrames) + ", Slices: " + str(nSlices) + ", Channels: " + str(nChannels))
        
        # Duplicate for preprocessing
        preprocessed = imp.duplicate()
        preprocessed.setTitle("preprocessed")
        
        # Median filter on entire stack
        IJ.run(preprocessed, "Median 3D...", "x=" + str(median_xy) + " y=" + str(median_xy) + " z=" + str(median_z))
        
        # Create intensity-based mask
        intensity_mask = preprocessed.duplicate()
        intensity_mask.setTitle("intensity_mask")
        
        IJ.setAutoThreshold(intensity_mask, "Mean dark stack")
        IJ.run(intensity_mask, "Convert to Mask", "background=Dark black")
        IJ.run(intensity_mask, "Fill Holes", "stack")
        
        # Create output mask stack (same dimensions as input)
        mask_result = IJ.createImage("mask_result", "8-bit black", 
                                     intensity_mask.getWidth(), 
                                     intensity_mask.getHeight(), 
                                     nChannels, 
                                     nSlices, 
                                     nFrames)
        
        # Loop through each timeframe
        for t in range(1, nFrames + 1):
            print("Processing frame " + str(t) + "/" + str(nFrames))
            
            # Set current position to this timeframe
            intensity_mask.setPosition(1, 1, t)  # (channel, slice, frame)
            
            
            # Get ROIs for current frame
            roi_list = getRoisByParticleAnalysis(intensity_mask, minsize=minsize, maxsize=maxsize)
            
            
            if len(roi_list) == 0:
                print("  No objects found in frame " + str(t))
                
                roi_list_large = getRoisByParticleAnalysis(intensity_mask, minsize=minsize, maxsize=float('inf'))
                if len(roi_list_large) > 0:
                    print("  Found " + str(len(roi_list_large)) + " objects without max size limit")
                    print("Trying watershed")
                    IJ.run(intensity_mask, "Watershed", "stack")
                    roi_list = getRoisByParticleAnalysis(intensity_mask, minsize=minsize, maxsize=maxsize)
                    
            intensity_mask.show()
            print("  Found " + str(len(roi_list)) + " objects in frame " + str(t))
            


            
            # Select largest object if multiple found
            if len(roi_list) > max_objects:
                roi_areas = []
                for roi in roi_list:
                    roi_areas.append(roi.getStatistics().area)
                largest_indices = sorted(range(len(roi_areas)), key=lambda i: roi_areas[i], reverse=True)[:max_objects]
                
                # Keep only largest objects
                for i, roi in enumerate(roi_list):
                    if i in largest_indices:
                        mask_result.setPosition(1, 1, t)
                        mask_result.getProcessor().fill(roi)
            else:
                # All objects fit criteria, add them all
                for roi in roi_list:
                    mask_result.setPosition(1, 1, t)
                    mask_result.getProcessor().fill(roi)
        
        mask_result.show()
        crash
        # Save the result
        IJ.saveAsTiff(mask_result, output_path)
        print("Saved: " + output_path)
        
        # Clean up
        preprocessed.close()
        intensity_mask.close()
        mask_result.close()
        imp.close()
        
    IJ.run("Close All")


def process_folder(input_dir):
    """Scan folder for files with correct suffix"""
    files = os.listdir(input_dir)
    files.sort()
    
    for filename in files:
        full_path = os.path.join(input_dir, filename)
        
        # Process subdirectories recursively
        if os.path.isdir(full_path):
            process_folder(full_path)
        
        # Process files with matching suffix
        elif filename.endswith(suffix):
            process_file(input_dir, output, filename)

# Main execution
IJ.run("Close All")
process_folder(input)
print("Processing complete!")