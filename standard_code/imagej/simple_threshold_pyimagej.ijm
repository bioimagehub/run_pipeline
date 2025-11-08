/*
 * PyImageJ-compatible macro for threshold segmentation
 * Uses #@ parameter syntax for automatic argument injection
 * 
 * Usage with PyImageJ:
 *   python run_imagej_script.py --script simple_threshold_pyimagej.ijm --input /data --suffix .tif --minsize 35000 --maxsize 120000
 *
 * Usage with direct ImageJ CLI (fallback):
 *   ImageJ-win64.exe --headless --console -macro simple_threshold_pyimagej.ijm "input=/data,suffix=.tif,minsize=35000,maxsize=120000"
 */

// PyImageJ parameter declarations (automatic injection when using ij.py.run_macro)
#@ String(label="Input directory", description="Directory containing images to process") input
#@ String(label="File suffix", description="File extension to filter (e.g., .tif)", value=".tif") suffix
#@ Integer(label="Minimum size", description="Minimum object size in pixels", value=35000) minsize
#@ Integer(label="Maximum size", description="Maximum object size in pixels", value=120000) maxsize

// Processing parameters (hardcoded - could be exposed as parameters too)
median_xy = 8;
median_z = 2;
max_objects = 1;

// Validate required parameters
if (input == "") {
    exit("ERROR: 'input' parameter is required");
}
if (!File.exists(input)) {
    exit("ERROR: Input directory does not exist: " + input);
}

// Create output directory
output = input + "_threshold";
File.makeDirectory(output);

print("Input directory: " + input);
print("Output directory: " + output);
print("File suffix: " + suffix);
print("Min size: " + minsize);
print("Max size: " + maxsize);

setBatchMode(true);

// Get list of files
list = getFileList(input);
list = Array.sort(list);

processedCount = 0;
skippedCount = 0;

for (i = 0; i < list.length; i++) {
    if (endsWith(list[i], suffix)) {
        processFile(input, output, list[i]);
        processedCount++;
    }
}

setBatchMode(false);

print("Processing complete!");
print("Processed: " + processedCount + " files");
print("Skipped: " + skippedCount + " files (already exist)");

function processFile(input, output, file) {
    // Get file paths
    basename = substring(file, 0, lastIndexOf(file, "."));
    output_tif = output + File.separator + basename + "_mask.tif";
    output_roi = output + File.separator + basename + "_roi.zip";
    
    // Skip if already processed
    if (File.exists(output_tif)) {
        print("Skipping (already exists): " + file);
        skippedCount++;
        return;
    }
    
    print("Processing: " + file);
    
    // Open image
    open(input + File.separator + file);
    rename("input");
    run("Enhance Contrast", "saturated=0.35");
    
    // Duplicate for preprocessing
    run("Duplicate...", "title=preprocessed duplicate");
    
    // Median filter
    run("Median 3D...", "x=" + median_xy + " y=" + median_xy + " z=" + median_z);
    
    // Create intensity-based mask
    run("Duplicate...", "title=intensity_mask duplicate");
    setAutoThreshold("Mean dark stack");
    run("Convert to Mask", "background=Dark black");
    run("Fill Holes", "stack");
    
    // Get ROI Manager
    roiManager("reset");
    
    // Analyze particles
    selectWindow("intensity_mask");
    run("Analyze Particles...", "size=" + minsize + "-" + maxsize + " pixel exclude clear add stack");
    
    n_rois = roiManager("count");
    frames = nSlices;
    
    print("Found " + n_rois + " ROIs in " + frames + " frames");
    
    // Check if we have the expected number of objects
    if (n_rois > 0 && n_rois != frames * max_objects) {
        selectWindow("intensity_mask");
        getStatistics(area, mean, min, max, std, histogram);
        
        // If no signal, skip
        if (max == 0) {
            print("No signal detected, skipping");
            run("Close All");
            return;
        }
        
        // Try watershed if too many pixels
        nPixels = area;
        if (nPixels > maxsize) {
            print("Trying watershed...");
            run("Watershed", "stack");
            roiManager("reset");
            run("Analyze Particles...", "size=" + minsize + "-" + maxsize + " pixel exclude clear add stack");
            n_rois = roiManager("count");
        }
    }
    
    // Clear outside ROI
    selectWindow("intensity_mask");
    if (n_rois > 0) {
        for (j = 0; j < n_rois; j++) {
            roiManager("select", j);
            run("Clear Outside", "slice");
        }
    }
    
    // Save mask
    selectWindow("intensity_mask");
    saveAs("Tiff", output_tif);
    print("Saved mask: " + output_tif);
    
    // Save ROIs
    if (n_rois > 0) {
        roiManager("Save", output_roi);
        print("Saved ROIs: " + output_roi);
    }
    
    // Clean up
    run("Close All");
    roiManager("reset");
}
