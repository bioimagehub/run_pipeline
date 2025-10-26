package main

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"
)

// FileListResult represents the result of querying files
type FileListResult struct {
	Files []string `json:"files"`
	Count int      `json:"count"`
	Error string   `json:"error,omitempty"`
}

// GetFilesFromPattern calls Python's get_files_to_process2 function
func GetFilesFromPattern(searchPattern string, searchSubfolders bool) (*FileListResult, error) {
	// Create Python script that calls get_files_to_process2
	pythonCode := fmt.Sprintf(`
import sys
import json
sys.path.insert(0, 'standard_code/python')

try:
    from bioimage_pipeline_utils import get_files_to_process2
    
    search_pattern = %q
    search_subfolders = %t
    
    files = get_files_to_process2(search_pattern, search_subfolders)
    
    result = {
        "files": files,
        "count": len(files)
    }
    print(json.dumps(result))
except Exception as e:
    result = {
        "files": [],
        "count": 0,
        "error": str(e)
    }
    print(json.dumps(result))
`, searchPattern, searchSubfolders)

	// Execute Python code
	cmd := exec.Command("python", "-c", pythonCode)
	output, err := cmd.Output()
	if err != nil {
		return &FileListResult{
			Files: []string{},
			Count: 0,
			Error: fmt.Sprintf("Failed to execute Python: %v", err),
		}, err
	}

	// Parse JSON result
	var result FileListResult
	if err := json.Unmarshal(output, &result); err != nil {
		return &FileListResult{
			Files: []string{},
			Count: 0,
			Error: fmt.Sprintf("Failed to parse result: %v", err),
		}, err
	}

	return &result, nil
}

// GetFileListPreview returns a preview of files (first 10 + count)
func GetFileListPreview(searchPattern string, searchSubfolders bool) string {
	result, err := GetFilesFromPattern(searchPattern, searchSubfolders)
	if err != nil || result.Error != "" {
		return fmt.Sprintf("Error: %s", result.Error)
	}

	if result.Count == 0 {
		return "No files found"
	}

	preview := fmt.Sprintf("Found %d files:\n", result.Count)

	// Show first 10 files
	maxShow := 10
	if len(result.Files) < maxShow {
		maxShow = len(result.Files)
	}

	for i := 0; i < maxShow; i++ {
		// Get just the filename, not full path
		parts := strings.Split(result.Files[i], "/")
		filename := parts[len(parts)-1]
		preview += fmt.Sprintf("  %d. %s\n", i+1, filename)
	}

	if result.Count > maxShow {
		preview += fmt.Sprintf("  ... and %d more", result.Count-maxShow)
	}

	return preview
}
