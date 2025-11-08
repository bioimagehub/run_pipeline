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
minsize = 30000;
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
	
	fullPath = input + File.separator + file;
	print("Processing: " + fullPath);
		
	// Open
	run("Bio-Formats Importer", "open=[" + fullPath + "] autoscale color_mode=Default rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT");
	
	
	// Verify an image was opened
	if (nImages == 0) {
		print("ERROR: Could not open image: " + file);
		setBatchMode("exit and display");
		return;
	}
	
	rename("input");
	
	
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

	if(n/frames != max_objects){ // not One ROI per frame
		
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
		
		run("Erode", "stack");
		run("Erode", "stack");
	    run("Connected Components Labeling", "connectivity=6 type=[16 bits]");
	    run("Label Size Filtering", "operation=Greater_Than size=" +  minsize);

	    run("Keep Largest Label");
	    run("Erode", "stack");
	}
	
	
	run("Connected Components Labeling", "connectivity=6 type=[16 bits]");
	

	run("Re-order Hyperstack ...", "channels=[Channels (c)] slices=[Frames (t)] frames=[Slices (z)]");
	
	
	getDimensions(width, height, channels, slices, frames);

	// Debugging checks (disabled in headless mode)
	//if(n/frames != max_objects){ // One ROI per frame
		
	// 	setBatchMode("exit and display");
	 //	waitForUser("Not one mask per frame");
	 //	setBatchMode("hide");
	// }
	
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
	run("Close All");
	
	// save measurements can come here if need be
	
}

