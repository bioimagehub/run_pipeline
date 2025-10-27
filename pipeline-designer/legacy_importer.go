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
	switch cmds := commands.(type) {
	case []interface{}:
		for _, cmd := range cmds {
			switch v := cmd.(type) {
			case string:
				// Skip interpreter commands
				if v == "python" || v == "imagej" {
					continue
				}
				// Check if it's a .py file path
				if strings.HasSuffix(v, ".py") {
					return v
				}
			}
		}
	}
	return ""
}

// findCLIDefinitionByScript searches all CLI definitions for a matching script filename
func findCLIDefinitionByScript(scriptFilename string, manager *CLIDefinitionsManager) *CLIDefinition {
	allDefs := manager.GetAllDefinitions()

	for _, def := range allDefs {
		// Extract filename from definition's script path
		defScriptFilename := filepath.Base(def.Script)

		if defScriptFilename == scriptFilename {
			return def
		}
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

	// Create a map of flag -> socket for quick lookup
	socketsByFlag := make(map[string]*Socket)
	for i := range node.InputSockets {
		socketsByFlag[node.InputSockets[i].ArgumentFlag] = &node.InputSockets[i]
	}
	for i := range node.OutputSockets {
		socketsByFlag[node.OutputSockets[i].ArgumentFlag] = &node.OutputSockets[i]
	}

	// Match YAML args to sockets
	for flag, value := range yamlArgs {
		socket, exists := socketsByFlag[flag]
		if !exists {
			unmatchedArgs = append(unmatchedArgs, flag)
			continue
		}

		// Populate socket value
		socket.Value = value
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
		}
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

	// Add visual link sockets
	linkIn := Socket{ID: uuid.New().String(), NodeID: node.ID, ArgumentFlag: "__link_in__", Type: TypeString, SocketSide: SocketInput, Value: "", IsRequired: false, DefaultValue: "", Description: "Visual link-only handle (ignored in YAML/command)", Validation: "", SkipEmit: true}
	linkOut := Socket{ID: uuid.New().String(), NodeID: node.ID, ArgumentFlag: "__link_out__", Type: TypeString, SocketSide: SocketOutput, Value: "", IsRequired: false, DefaultValue: "", Description: "Visual link-only handle (ignored in YAML/command)", Validation: "", SkipEmit: true}
	node.InputSockets = append(node.InputSockets, linkIn)
	node.OutputSockets = append(node.OutputSockets, linkOut)

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
