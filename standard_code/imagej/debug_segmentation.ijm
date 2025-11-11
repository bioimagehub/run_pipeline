/*
 * Macro template to process multiple images in a folder
 */

#@ File (label = "Input directory", style = "directory") input
// String (label = "File suffix", value = ".tif") suffix

suffix = "_Probabilities.h5"
output = input + "_ij_qc"

File.makeDirectory(output);

minsize = 24000;
maxsize = 120000;
minperim = 750;
maxperim = 1500;

max_watershed_rounds = 5;

run("Close All");

// See also Process_Folder.py for a version of this code
// in the Python scripting language.

processFolder(input);

// function to scan folders/subfolders/files to find files with correct suffix
function processFolder(input) {
	list = getFileList(input);
	list = Array.sort(list);
	for (i = 0; i < list.length; i++) {
		failname = File.getNameWithoutExtension(list[i]) + "_fail.tif";
		maskname = File.getNameWithoutExtension(list[i]) + ".tif";
		if(File.isDirectory(input + File.separator + list[i]))
			processFolder(input + File.separator + list[i]);
		
		if (File.exists(output + File.separator + maskname)) {
			continue;
		}
		if (File.exists(output + File.separator + failname)) {
			continue;
		}
		if(endsWith(list[i], suffix))
			processFile(input, list[i]);
	}
}


function printArea(text){
	run("Create Selection");
	getRawStatistics(nPixels, mean, min, max, std, histogram);
	run("Select None");
	print(text + ": " + nPixels); 
	
	return nPixels
	
}


function validate_object(){
	
	run("Duplicate...", "title=tmp_measure duplicate");
	
	if (on_edge()){
		selectWindow("tmp_measure");
		close();
		return false
	}
	
	getRawStatistics(nPixels, mean, min, max, std, histogram);
	if(max==0){
		selectWindow("tmp_measure");
		close();
		return false
	}
	


	run("Set Measurements...", "area perimeter shape redirect=None decimal=0");
	run("8-bit");
	setThreshold(1, 255);
	run("Create Selection");
	
	n = roiManager('count');
	if (n>0) {
		roiManager("Deselect");
		roiManager("Delete");
	}
	
	roiManager("Add");
	

	n = roiManager('count');
	if (n>0) {
		
		roiManager("Measure");
		area = getResult("Area", 0);
		perim = getResult("Perim.", 0);
		
	}

	
	
	// close everything
	run("Clear Results");
	n = roiManager('count');
	if (n>0) {
		roiManager("Deselect");
		roiManager("Delete");
	}
	run("Select None");
	
	selectWindow("tmp_measure");
	close();
	
	print(area);
	if(area < minsize){
		print("area < minsize: " + area + " < " + minsize);
		return false
	}
	if(area > maxsize){
		print("area > maxsize: " + area + " > " + maxsize);
		return false
	}


	if(perim < minperim){
		print("perim < minperim: " + perim + " < " + minperim);
		return false
	}
	if(perim > maxperim){
		print("perim > maxperim: " + perim + " < " + maxperim);
		return false
	}	

	// success
	return true
	
}


function erode_preserve_border() {
    // Erode but preserve border pixels
    run("Duplicate...", "title=border_backup duplicate");
    
    // Get border pixels
    run("Select All");
    run("Enlarge...", "enlarge=-1");
    run("Make Inverse");
    run("Clear Outside", "stack");
    run("Select None");
    
    // Do normal erosion on original
    selectWindow("input-lbl-largest");
    run("Erode", "stack");
    
    // Restore border
    imageCalculator("Max stack", "input-lbl-largest", "border_backup");
    selectWindow("border_backup");
    close();
}

function on_edge(){
	run("Duplicate...", "duplicate");
	rename("tmp_stack");
	
	if(nSlices>1){
		run("Z Project...", "projection=[Max Intensity]");
	}

	
	
	run("Select All");
	run("Enlarge...", "enlarge=-1");
	run("Make Inverse");
	getRawStatistics(nPixels, mean, min, max, std, histogram);
	
	selectWindow("tmp_stack");
	run("Close");
	selectWindow("MAX_tmp_stack");
	run("Close");
	
	
	if(max >0){
		return true
	} else{
		return false
	}	
}

function particle_filter(min_size, max_size, exclude_edges){
		run("Analyze Particles...", "size="  + min_size + "-" + max_size + " " + exclude_edges +" clear add stack");// Dont exclude min here, to alow for ower watershedding 
		
		n = roiManager('count');
		
		// Process each slice
		for (slice = 1; slice <= nSlices; slice++) {
		    setSlice(slice);
		    
		    // Find all ROIs that belong to this slice
		    roiIndices = newArray(0);
		    for (r = 0; r < n; r++) {
		        roiManager('select', r);
		        roiSlice = getSliceNumber();  // Gets the slice this ROI is on
		        if (roiSlice == slice) {
		            roiIndices = Array.concat(roiIndices, r);
		        }
		    }
		    
		    if (roiIndices.length > 0) {
		        // Select all ROIs for this slice at once
		        roiManager('select', roiIndices);
		        
		        // Combine them into a composite selection
		        roiManager('Combine');
		        
		        // Now clear outside once for all ROIs combined
		        run("Clear Outside", "slice");
		    }
		}
		
		// Clean up ROI manager
		run("Select None");
		n = roiManager('count');
		if (n > 0) {
		    roiManager("Deselect");
		    roiManager("Delete");
		}
		
}




function processFile(input, file) {
	// Do the processing here by adding your own code.
	// Leave the print statements until things work, then remove them.
	
	setBatchMode("hide");
	

	
	imput_file = input + File.separator + file;
	imput_file = replace(imput_file, "\\", "/");
	
	print(imput_file);
	
	
	edge_file = replace(input + File.separator + file, "Label2.tif", "Label3.tif");
	

	
	output_file = output + File.separator + File.getNameWithoutExtension(file) + ".tif";
	print(output_file);
	output_fail = output + File.separator + File.getNameWithoutExtension(file) + "_fail.tif";
	

	// Load h5 file
	run("Import HDF5", "select=[" +imput_file +"] datasetname=/exported_data axisorder=tzyxc");
	
	run("Duplicate...", "duplicate channels=2");
	rename("input");
	
	close("\\Others");
	
	lower = 0.5;
	upper = 1;
	setThreshold(lower, upper);
	
	// Make binary stack from threshold
	setOption("BlackBackground", true);
	run("Convert to Mask", "method=Default background=Dark black");
	rename("mask");
	
		
	area = printArea("Input area");
	
	if(area<minsize){
		print(output_fail);
		save(output_fail);
		run("Close All");
		return;
	}	
		
	// Fill holes
	run("Fill Holes", "stack");
	area = printArea("Filled holes");
	
	// remove very small things and remove on edges
	particle_filter(200, "Infinity", "exclude");
	
	run("Connected Components Labeling", "connectivity=6 type=[16 bits]");
		
	run("Keep Largest Label");
	
	
	area = printArea("Largest object");	
	
	// Check if there is something on the edges
	selectWindow("mask-lbl-largest");
	close("\\Others");
		
	max_watershed_rounds = 5;

	
	for (i = 0; i < max_watershed_rounds; i++) {
		selectWindow("mask-lbl-largest");

		// Check if we still have edge-touching objects
	    
	    area = printArea("Largest object");	
	    
	    if(!on_edge() && area < maxsize){
	    	//waitForUser("Nothing on edges after i=" + i);	  
	    	
	        break;  // Success! Nothing on  
	    }
	    
	    
	    // Try watershed to see if things improve
		run("Watershed", "stack");
		
		particle_filter(minsize, "Infinity", "exclude");
		
		
		
		
		for (j = 0; j < i; j++) { // 0 times in first run, 1 in second and so on
			erode_preserve_border();
		}
		selectWindow("mask-lbl-largest");
		

		
		
		// Remove masks that are touching the edge
		// and remove very small things
		particle_filter(200, "Infinity", "exclude");
		
		// Re- merge over watersheded
		run("Dilate", "stack");
		run("Erode", "stack");

				
		run("Connected Components Labeling", "connectivity=6 type=[16 bits]");
		
		run("Keep Largest Label");
		
		
		for (j = 0; j < i+1; j++) { // 0 times in first run, 1 in second and so on 
			run("Dilate", "stack");
		}

	}
	
	selectWindow("mask-lbl-largest");	
	rename("mask");
	
	particle_filter(minsize, maxsize, "exclude");


	qc_status = validate_object();
	
	if(qc_status){
		
		
		save(output_file);
		print(output_file);
		run("Close All");
		setBatchMode("exit and display");
		return;
	}
		
	// Try to load the edge segmentation for this file since everyhtong failed
	// combine it with the best attempt from DAPI stain
	
	
	rename("input-failed-attempt");
	
	for (i = 1; i <= nSlices; i++) {
    	setSlice(i);
    	// do something here;
    	run("Create Selection");
		run("Convex Hull");
		changeValues(0, 255, 255);
	}


	/*
	particle_filter(0, "Infinity", "exclude");
	
	imageCalculator("Add create stack", "input","edge_input");
	rename("merged stack");
	
	grow_steps = 5;
	for (i = 0; i < grow_steps; i++) {
		run("Dilate", "stack");
	}
	run("Fill Holes", "stack");
	
	for (i = 0; i < grow_steps; i++) {
		run("Erode", "stack");
	}
	*/
		
	particle_filter(minsize, maxsize, "exclude");
	
	
	qc_status = validate_object();
	if(qc_status){
		
		save(output_file);
		run("Close All");
		setBatchMode("exit and display");
		return;
	}
	
	save(output_fail);
		

	selectWindow("input");
	run("Close");
	
	run("Close All");
	setBatchMode("exit and display");
}
