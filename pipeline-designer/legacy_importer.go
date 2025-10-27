package main

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/google/uuid"
)

// LegacyImportReport contains validation results from importing a legacy YAML
type LegacyImportReport struct {
	TotalSteps      int
	SuccessfulSteps int
	Warnings        []string
	Errors          []string
	UnmatchedArgs   map[string][]string // NodeID -> list of unmatched args
	MissingArgs     map[string][]string // NodeID -> list of missing required args
}

// ImportLegacyYAML processes a YAML pipeline that doesn't have _designer_metadata
// It attempts to match scripts to CLI definitions and creates proper visual nodes
func ImportLegacyYAML(yamlPipeline *YAMLPipeline, definitionsManager *CLIDefinitionsManager) (*Pipeline, *LegacyImportReport, error) {
	report := &LegacyImportReport{
		TotalSteps:    len(yamlPipeline.Run),
		Warnings:      make([]string, 0),
		Errors:        make([]string, 0),
		UnmatchedArgs: make(map[string][]string),
		MissingArgs:   make(map[string][]string),
	}

	pipeline := &Pipeline{
		Nodes:       make([]CLINode, 0),
		Connections: make([]SocketConnection, 0),
		Metadata: PipelineMetadata{
			Name: yamlPipeline.PipelineName,
		},
	}

	// Layout parameters for left-to-right node placement
	const nodeSpacing = 450.0 // Horizontal spacing between nodes
	const startX = 100.0
	const startY = 200.0

	for i, step := range yamlPipeline.Run {
		// Extract script path from commands
		scriptPath := extractScriptPath(step.Commands)
		if scriptPath == "" {
			scriptPath = step.Script // Fallback to Script field
		}

		if scriptPath == "" {
			report.Errors = append(report.Errors, fmt.Sprintf("Step %d ('%s'): No script path found in commands", i+1, step.Name))
			continue
		}

		// Extract script filename (e.g., "convert_to_tif.py" from "./standard_code/python/convert_to_tif.py")
		scriptFilename := filepath.Base(scriptPath)
		scriptName := strings.TrimSuffix(scriptFilename, filepath.Ext(scriptFilename))

		if appLogger != nil {
			appLogger.Printf("[LEGACY_IMPORT] Step %d: Looking for CLI definition matching script '%s'", i+1, scriptName)
		}

		// Find matching CLI definition
		definition := findCLIDefinitionByScript(scriptFilename, definitionsManager)
		if definition == nil {
			report.Errors = append(report.Errors, fmt.Sprintf("Step %d ('%s'): No CLI definition found for script '%s'", i+1, step.Name, scriptFilename))
			// Create a basic node without definition
			node := createBasicNodeFromStep(step, i, startX, startY, nodeSpacing)
			pipeline.Nodes = append(pipeline.Nodes, node)
			continue
		}

		if appLogger != nil {
			appLogger.Printf("[LEGACY_IMPORT] Step %d: Found CLI definition '%s' (ID: %s)", i+1, definition.Name, definition.ID)
		}

		// Create node from definition
		position := Point{X: startX + float64(i)*nodeSpacing, Y: startY}
		node := CreateNodeFromDefinition(definition, position)

		// Override node ID if present in YAML
		if step.NodeID != "" {
			node.ID = step.NodeID
		}

		// Extract arguments from YAML commands
		yamlArgs := extractArgumentsFromCommands(step.Commands)

		// Match YAML arguments to node sockets and populate values
		unmatchedArgs, missingArgs := populateSocketsFromYAMLArgs(node, yamlArgs, definition)

		if len(unmatchedArgs) > 0 {
			report.UnmatchedArgs[node.ID] = unmatchedArgs
			report.Warnings = append(report.Warnings, fmt.Sprintf("Step %d ('%s'): %d unmatched arguments from YAML: %v", i+1, step.Name, len(unmatchedArgs), unmatchedArgs))
		}

		if len(missingArgs) > 0 {
			report.MissingArgs[node.ID] = missingArgs
			report.Errors = append(report.Errors, fmt.Sprintf("Step %d ('%s'): %d REQUIRED arguments missing in YAML: %v", i+1, step.Name, len(missingArgs), missingArgs))
		}

		if len(unmatchedArgs) == 0 && len(missingArgs) == 0 {
			report.SuccessfulSteps++
		}

		pipeline.Nodes = append(pipeline.Nodes, *node)
	}

	if appLogger != nil {
		appLogger.Printf("[LEGACY_IMPORT] Import complete: %d/%d steps successful", report.SuccessfulSteps, report.TotalSteps)
		appLogger.Printf("[LEGACY_IMPORT] Warnings: %d, Errors: %d", len(report.Warnings), len(report.Errors))
	}

	return pipeline, report, nil
}

// extractScriptPath finds the .py script path from YAML commands array
// Looks for strings ending with .py (ignoring "python" interpreter command)
func extractScriptPath(commands interface{}) string {
	if appLogger != nil {
		appLogger.Printf("[EXTRACT_SCRIPT] Parsing commands: %T - %+v", commands, commands)
	}

	switch cmds := commands.(type) {
	case []interface{}:
		if appLogger != nil {
			appLogger.Printf("[EXTRACT_SCRIPT] Found array with %d elements", len(cmds))
		}
		for i, cmd := range cmds {
			if appLogger != nil {
				appLogger.Printf("[EXTRACT_SCRIPT]   Element %d: %T - %+v", i, cmd, cmd)
			}
			switch v := cmd.(type) {
			case string:
				if appLogger != nil {
					appLogger.Printf("[EXTRACT_SCRIPT]     String value: '%s'", v)
				}
				// Skip interpreter commands
				if v == "python" || v == "imagej" {
					if appLogger != nil {
						appLogger.Printf("[EXTRACT_SCRIPT]     Skipping interpreter: '%s'", v)
					}
					continue
				}
				// Check if it's a .py file path
				if strings.HasSuffix(v, ".py") {
					if appLogger != nil {
						appLogger.Printf("[EXTRACT_SCRIPT]     ✓ Found .py script: '%s'", v)
					}
					return v
				}
			}
		}
	}
	if appLogger != nil {
		appLogger.Printf("[EXTRACT_SCRIPT] No script found, returning empty string")
	}
	return ""
}

// findCLIDefinitionByScript searches all CLI definitions for a matching script filename
func findCLIDefinitionByScript(scriptFilename string, manager *CLIDefinitionsManager) *CLIDefinition {
	if appLogger != nil {
		appLogger.Printf("[FIND_DEFINITION] Searching for script: '%s'", scriptFilename)
	}

	allDefs := manager.GetAllDefinitions()

	if appLogger != nil {
		appLogger.Printf("[FIND_DEFINITION] Total definitions to search: %d", len(allDefs))
	}

	for i, def := range allDefs {
		// Extract filename from definition's script path
		defScriptFilename := filepath.Base(def.Script)

		if appLogger != nil && i < 5 { // Log first 5 for debugging
			appLogger.Printf("[FIND_DEFINITION]   Def %d: script='%s', basename='%s', match=%v", i, def.Script, defScriptFilename, defScriptFilename == scriptFilename)
		}

		if defScriptFilename == scriptFilename {
			if appLogger != nil {
				appLogger.Printf("[FIND_DEFINITION] ✓ MATCH FOUND: '%s' == '%s'", defScriptFilename, scriptFilename)
			}
			return def
		}
	}

	if appLogger != nil {
		appLogger.Printf("[FIND_DEFINITION] ✗ NO MATCH: Could not find definition for '%s'", scriptFilename)
	}
	return nil
}

// extractArgumentsFromCommands parses YAML commands array and extracts flag-value pairs
// Returns a map of flag -> value
func extractArgumentsFromCommands(commands interface{}) map[string]string {
	args := make(map[string]string)

	switch cmds := commands.(type) {
	case []interface{}:
		for _, cmd := range cmds {
			switch v := cmd.(type) {
			case map[string]interface{}:
				// Flag-value pairs stored as maps
				for flag, value := range v {
					args[flag] = fmt.Sprintf("%v", value)
				}
			}
		}
	}

	return args
}

// populateSocketsFromYAMLArgs matches YAML arguments to node sockets and sets their values
// Returns lists of unmatched args (in YAML but not in definition) and missing args (required in definition but not in YAML)
func populateSocketsFromYAMLArgs(node *CLINode, yamlArgs map[string]string, definition *CLIDefinition) ([]string, []string) {
	unmatchedArgs := make([]string, 0)
	missingArgs := make([]string, 0)

	if appLogger != nil {
		appLogger.Printf("[POPULATE_SOCKETS] Node '%s': Processing %d YAML args", node.Name, len(yamlArgs))
	}

	// Create a map of flag -> socket for quick lookup
	socketsByFlag := make(map[string]*Socket)
	for i := range node.InputSockets {
		socketsByFlag[node.InputSockets[i].ArgumentFlag] = &node.InputSockets[i]
		if appLogger != nil {
			appLogger.Printf("[POPULATE_SOCKETS]   Input socket: %s (default: '%s')",
				node.InputSockets[i].ArgumentFlag, node.InputSockets[i].DefaultValue)
		}
	}
	for i := range node.OutputSockets {
		socketsByFlag[node.OutputSockets[i].ArgumentFlag] = &node.OutputSockets[i]
		if appLogger != nil {
			appLogger.Printf("[POPULATE_SOCKETS]   Output socket: %s (default: '%s')",
				node.OutputSockets[i].ArgumentFlag, node.OutputSockets[i].DefaultValue)
		}
	}

	// Match YAML args to sockets
	for flag, value := range yamlArgs {
		socket, exists := socketsByFlag[flag]
		if !exists {
			unmatchedArgs = append(unmatchedArgs, flag)
			if appLogger != nil {
				appLogger.Printf("[POPULATE_SOCKETS]   ✗ Unmatched YAML arg: %s = '%s'", flag, value)
			}
			continue
		}

		// Populate socket value
		socket.Value = value
		if appLogger != nil {
			appLogger.Printf("[POPULATE_SOCKETS]   ✓ Matched %s = '%s'", flag, value)
		}
	}

	// Check for missing required arguments
	for _, argDef := range definition.Arguments {
		if !argDef.IsRequired {
			continue
		}

		// Check if this required arg has a value
		socket, exists := socketsByFlag[argDef.Flag]
		if !exists {
			continue // Socket wasn't created (shouldn't happen)
		}

		if socket.Value == "" && yamlArgs[argDef.Flag] == "" {
			missingArgs = append(missingArgs, argDef.Flag)
			if appLogger != nil {
				appLogger.Printf("[POPULATE_SOCKETS]   ✗ Missing required arg: %s", argDef.Flag)
			}
		}
	}

	if appLogger != nil {
		appLogger.Printf("[POPULATE_SOCKETS] Complete: %d unmatched, %d missing", len(unmatchedArgs), len(missingArgs))
	}

	return unmatchedArgs, missingArgs
}

// createBasicNodeFromStep creates a minimal node when no CLI definition is found
func createBasicNodeFromStep(step YAMLStep, index int, startX, startY, spacing float64) CLINode {
	nodeID := step.NodeID
	if nodeID == "" {
		nodeID = uuid.New().String()
	}

	position := Point{X: startX + float64(index)*spacing, Y: startY}

	node := CLINode{
		ID:            nodeID,
		Name:          step.Name,
		Environment:   step.Environment,
		Script:        step.Script,
		Position:      position,
		Size:          Size{Width: 300, Height: 150},
		InputSockets:  make([]Socket, 0),
		OutputSockets: make([]Socket, 0),
		Category:      "Uncategorized",
		TestStatus:    TestNotRun,
	}

	// Parse commands into sockets (basic fallback)
	if step.Commands != nil {
		parseCommandsToSockets(&node, step.Commands)
	}

	// Note: No longer adding __link__ sockets - visual connections are handled by frontend

	return node
}

// PrintLegacyImportReport logs a formatted report of the import process
func PrintLegacyImportReport(report *LegacyImportReport) {
	appLogger.Println("========================================")
	appLogger.Println("LEGACY YAML IMPORT REPORT")
	appLogger.Println("========================================")
	appLogger.Printf("Total steps: %d", report.TotalSteps)
	appLogger.Printf("Successfully matched: %d", report.SuccessfulSteps)
	appLogger.Printf("Warnings: %d", len(report.Warnings))
	appLogger.Printf("Errors: %d", len(report.Errors))
	appLogger.Println("")

	if len(report.Warnings) > 0 {
		appLogger.Println("⚠ WARNINGS:")
		for _, warning := range report.Warnings {
			appLogger.Printf("  - %s", warning)
		}
		appLogger.Println("")
	}

	if len(report.Errors) > 0 {
		appLogger.Println("❌ ERRORS:")
		for _, err := range report.Errors {
			appLogger.Printf("  - %s", err)
		}
		appLogger.Println("")
	}

	if len(report.UnmatchedArgs) > 0 {
		appLogger.Println("📋 UNMATCHED ARGUMENTS (in YAML but not in CLI definition):")
		for nodeID, args := range report.UnmatchedArgs {
			appLogger.Printf("  Node %s: %v", nodeID, args)
		}
		appLogger.Println("")
	}

	if len(report.MissingArgs) > 0 {
		appLogger.Println("⚠ MISSING REQUIRED ARGUMENTS (in CLI definition but not in YAML):")
		for nodeID, args := range report.MissingArgs {
			appLogger.Printf("  Node %s: %v", nodeID, args)
		}
		appLogger.Println("")
	}

	appLogger.Println("========================================")
}
