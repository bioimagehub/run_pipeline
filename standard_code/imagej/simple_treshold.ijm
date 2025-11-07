/*
 * Macro template to process multiple images in a folder
 * Command-line compatible version for run_pipeline.exe
 * 
 * Usage:
 *   ImageJ-win64.exe --headless --console -macro simple_treshold.ijm "input=E:\path\to\input,suffix=.tif,max_objects=-1,minsize=10000,maxsize=100000"
 *   
 * For interactive debugging: Just run without arguments and it will use defaults below
 */

// Parse command-line arguments
args = getArgument();

// Default values (for debugging - modify these for quick testing)
input = "E:\\Coen\\Sarah\\6849908-IMB-Coen-Sarah-Photoconv_global\\cellprofiler_input";  // Change this to your test folder
suffix = ".tif";
max_objects = 1;
minsize = 35000;
maxsize = 120000;

median_xy = 8;
median_z = 2;

// Parse comma-separated key=value pairs (overrides defaults if provided)
if (args != "") {
    pairs = split(args, ",");
    for (i = 0; i < pairs.length; i++) {
        pair = split(pairs[i], "=");
        if (pair.length == 2) {
            key = pair[0];
            value = pair[1];
            
            if (key == "input") input = value;
            else if (key == "suffix") suffix = value;
            //else if (key == "max_objects") max_objects = parseInt(value);
            else if (key == "minsize") minsize = parseInt(value);
            else if (key == "maxsize") maxsize = parseInt(value);
        }
    }
}

// Validate required parameters
if (input == "") {
    exit("ERROR: 'input' parameter is required (set default in script or provide via arguments)");
}
if (!File.exists(input)) {
    exit("ERROR: Input directory does not exist: " + input);
}

output = input + "_threshold";

// Create output directory
File.makeDirectory(output);

// See also Process_Folder.py for a version of this code
// in the Python scripting language.
run("Close All");
processFolder(input);

// function to scan folders/subfolders/files to find files with correct suffix
function processFolder(input) {
	list = getFileList(input);
	list = Array.sort(list);
	for (i = 0; i < list.length; i++) {
		if(File.isDirectory(input + File.separator + list[i]))
			processFolder(input + File.separator + list[i]);
		if(endsWith(list[i], suffix))
			processFile(input, output, list[i]);
	}
}

function processFile(input, output, file) {
	// Do the processing here by adding your own code.
	// Leave the print statements until things work, then remove them.
	
	output_basename = File.getNameWithoutExtension(file);
	output_tif = output + File.separator + output_basename + "_mask.tif";
	output_roi = output + File.separator + output_basename + "_roi.zip";
	
	
	if(File.exists(output_tif)){
		return;	
	}
	
	
	
	setBatchMode("hide");
	
	print("Processing: " + input + File.separator + file);
	run("Bio-Formats Importer", "open=[" + input + File.separator + file +"] color_mode=Default rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT");
	rename("input");
	run("Enhance Contrast", "saturated=0.35");	
	
	// Duplicate image to create preprocessed image
	run("Duplicate...", "title=preprocessed duplicate");
	
	
	// Median filter
	run("Median 3D...", "x=" + median_xy +" y=" + median_xy + " z=" + median_z);
	
	
	////////////////////////////////////////////////////////
	// Create intensity-based mask from preprocessed image
 	//
	/////////////////////////////////////////////////////////
	
	run("Duplicate...", "title=intensity_mask duplicate");
	setAutoThreshold("Mean dark stack");
	run("Convert to Mask", "background=Dark black");
	run("Fill Holes", "stack");
	
	
	// Count objects
	selectWindow("intensity_mask");	
	n = roiManager('count');
	if (n>0) {		
		roiManager("Deselect");
		roiManager("Delete");
	}
	run("Analyze Particles...", "size=" + minsize + "-" + maxsize + " pixel exclude clear add stack");
	
	n = roiManager('count');
	
	
	
	
	// if not the correct number of objects try to resolve
	getDimensions(width, height, channels, slices, frames);

	if(n/frames != max_objects){ // One ROI per frame
		
		// is there any signal?
		
		getRawStatistics(nPixels, mean, min, max, std, histogram);
		
		// There is no hope if no signal
		if(max==0){
			return;
		}
		
		// If the nPixels is too large try watershed
		if(nPixels > maxsize){
			run("Watershed", "stack");
			run("Analyze Particles...", "size=" + minsize + "-" + maxsize + " pixel exclude clear add stack");
		}


		n = roiManager('count');
		
		// There is no hope if no ROIs after
		if(n==0){
			return;
		}		
	}

	// Clear outside ROI
	n = roiManager('count');
	for (i = 0; i < n; i++) {
	    roiManager('select', i);
	    // process roi here
	    run("Clear Outside", "slice");
	}


	// Keep ROI
	//if (n>0) {		
	//	roiManager("Deselect");
	//	roiManager("Delete");
	//}
	
		
	/**
	///////////////////////////////////////////////////////////////////////
	// Create slightly oversized edge mask and remove things outside it  //
 	// This helps split out debres that are close to the edge            //
	///////////////////////////////////////////////////////////////////////
	
	selectWindow("preprocessed");
	run("Duplicate...", "title=edge_mask duplicate");
	
	// Clear outside ROI
	// We only care about shrinking the mask if there is debris included in the threshold mask
	n = roiManager('count');
	for (i = 0; i < n; i++) {
	    roiManager('select', i);
	    // process roi here
	    run("Clear Outside", "slice");
	}
	
	if (n>0) {		
		roiManager("Deselect");
		roiManager("Delete");
	}
	run("Select None");
	run("Gaussian Blur...", "sigma=3 stack");
	
	// threshold on edge mask
	run("Find Edges", "stack");
	
	
	run("Convert to Mask", "method=Li background=Dark calculate black");
	
	// Label connected components and get largest
	run("Connected Components Labeling", "connectivity=6 type=[16 bits]");
	run("Label Size Filtering", "operation=Greater_Than size=0");

	run("Keep Largest Label");
	
	run("Fill Holes", "stack");
	
	// Clean up intermediate images
	selectWindow("edge_mask");
	close();
	selectImage("edge_mask-lbl");
	close();
	selectImage("edge_mask-lbl-sizeFilt");
	close();
	
	selectImage("edge_mask-lbl-sizeFilt-largest");
	rename("edge_mask");

	

	
	// ensure there is a mask
	selectWindow("edge_mask");	
	n = roiManager('count');
	if (n>0) {		
		roiManager("Deselect");
		roiManager("Delete");
	}
	
	selectWindow("edge_mask");
	
	run("Analyze Particles...", "size=" + minsize + "-" + maxsize + " pixel exclude clear add stack");
	
	 
	
	n = roiManager('count');
	
	
	// If edgemask found ask user for help
	if (n>0) {	
		
		// Clear outsite edge mask for spline to work
		for (i = 0; i < n; i++) {
		    roiManager('select', i);
		    // process roi here
		    run("Clear Outside", "slice");
		}
		
		
		// Delete tmp ROIs
		if (n>0) {		
			roiManager("Deselect");
			roiManager("Delete");
		}
		
		// Smooth the edge mask using morphological operations
		selectWindow("edge_mask");
		run("Options...", "iterations=3 count=1 black do=Dilate stack");
		run("Options...", "iterations=3 count=1 black do=Erode stack");

		imageCalculator("AND create stack", "edge_mask", "intensity_mask");

		rename("mask");
		selectWindow("intensity_mask");	
		close();
		

	}
 else {
		// if no edge mask found use threshold only
		selectWindow("intensity_mask");	
		rename("mask");
	}

	selectWindow("edge_mask");
	close();

		
	// Analyze final mask
	
	selectWindow("mask");
	n = roiManager('count');
	if (n>0) {		
		roiManager("Deselect");
		roiManager("Delete");
	}
	run("Analyze Particles...", "size=" + minsize + "-" + maxsize + " pixel exclude clear add stack");

	
	
	// only keep labels that have a defined roi 
	// remove on edge and size 
	n = roiManager('count');
	print(n);
	
	
	
	print(slices);
	*/
	
	getDimensions(width, height, channels, slices, frames);

	// Debugging checks (disabled in headless mode)
	if(n/frames != max_objects){ // One ROI per frame
		
	 	setBatchMode("exit and display");
	 	waitForUser("Not one mask per frame");
	 	setBatchMode("hide");
	 }
	
	// Save mask
	run("OME-TIFF...", "save=[" + output_tif + "] export compression=Uncompressed");
	
	// save ROI
	n = roiManager('count');
	if (n>0) {
		roiManager("Deselect");
		roiManager("Save",  output_roi);
		
		roiManager("Deselect");
		roiManager("Delete");
	}
	run("Close All");
	setBatchMode("exit and display");
	
	// save measurements can come here if need be
	
}

