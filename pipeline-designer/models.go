package main

import (
	"time"
)

// Point represents 2D coordinates on the canvas
type Point struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
}

// Size represents node dimensions
type Size struct {
	Width  float64 `json:"width"`
	Height float64 `json:"height"`
}

// ArgumentType defines the data type of a CLI argument
type ArgumentType string

const (
	TypePath        ArgumentType = "path"
	TypeGlobPattern ArgumentType = "glob_pattern"
	TypeString      ArgumentType = "string"
	TypeInt         ArgumentType = "int"
	TypeFloat       ArgumentType = "float"
	TypeBool        ArgumentType = "bool"
	TypeFileList    ArgumentType = "file_list" // List of file paths
)

// SocketSide defines whether a socket is input or output
type SocketSide string

const (
	SocketInput  SocketSide = "input"
	SocketOutput SocketSide = "output"
)

// Socket represents a connection point on a node
type Socket struct {
	ID           string       `json:"id"`
	NodeID       string       `json:"nodeId"`
	ArgumentFlag string       `json:"argumentFlag"`
	Type         ArgumentType `json:"type"`
	SocketSide   SocketSide   `json:"socketSide"`
	Value        string       `json:"value"`
	ConnectedTo  *string      `json:"connectedTo"` // Socket ID it connects to (nil if unconnected)
	IsRequired   bool         `json:"isRequired"`
	DefaultValue string       `json:"defaultValue"`
	Description  string       `json:"description"`
	Validation   string       `json:"validation"`
	SkipEmit     bool         `json:"skipEmit,omitempty"` // If true, do not emit into YAML commands
}

// SocketConnection represents a connection between two sockets
type SocketConnection struct {
	ID           string `json:"id"`
	FromNodeID   string `json:"fromNodeId"`
	FromSocketID string `json:"fromSocketId"`
	ToNodeID     string `json:"toNodeId"`
	ToSocketID   string `json:"toSocketId"`
	IsValid      bool   `json:"isValid"`
}

// TestStatus represents the state of a node test execution
type TestStatus string

const (
	TestNotRun  TestStatus = "not_run"
	TestRunning TestStatus = "running"
	TestSuccess TestStatus = "success"
	TestFailed  TestStatus = "failed"
)

// CLINode represents a visual node on the canvas
type CLINode struct {
	ID             string     `json:"id"`
	DefinitionID   string     `json:"definitionId"`
	Name           string     `json:"name"`
	Position       Point      `json:"position"`
	Size           Size       `json:"size"`
	Environment    string     `json:"environment"`
	Executable     string     `json:"executable"`
	Script         string     `json:"script"`
	InputSockets   []Socket   `json:"inputSockets"`
	OutputSockets  []Socket   `json:"outputSockets"`
	Icon           string     `json:"icon"`
	Color          string     `json:"color"`
	IsSelected     bool       `json:"isSelected"`
	IsCollapsed    bool       `json:"isCollapsed"`
	Category       string     `json:"category"`
	TestStatus     TestStatus `json:"testStatus"`
	LastTestFile   string     `json:"lastTestFile"`
	LastTestOutput string     `json:"lastTestOutput"`
	LastTestTime   string     `json:"lastTestTime"`
	TestError      string     `json:"testError"`
}

// ArgumentDefinition defines a CLI argument in a definition template
type ArgumentDefinition struct {
	Flag         string       `json:"flag"`
	Type         ArgumentType `json:"type"`
	SocketSide   SocketSide   `json:"socketSide"`
	IsRequired   bool         `json:"isRequired"`
	DefaultValue string       `json:"defaultValue"`
	Description  string       `json:"description"`
	Validation   string       `json:"validation"`
	UserOverride bool         `json:"userOverride"`
	SkipEmit     bool         `json:"skipEmit,omitempty"` // If true, create visual socket but do not emit
}

// CLIDefinition represents a reusable CLI tool template
type CLIDefinition struct {
	ID          string               `json:"id"`
	Name        string               `json:"name"`
	Category    string               `json:"category"`
	Icon        string               `json:"icon"`
	Color       string               `json:"color"`
	Description string               `json:"description"`
	Environment string               `json:"environment"`
	Executable  string               `json:"executable"`
	Script      string               `json:"script"`
	HelpCommand string               `json:"helpCommand"`
	Arguments   []ArgumentDefinition `json:"arguments"`
	Version     string               `json:"version"`
	Author      string               `json:"author"`
	LastParsed  string               `json:"lastParsed,omitempty"` // String to avoid time parsing issues
}

// PipelineMetadata contains metadata about a pipeline
type PipelineMetadata struct {
	Name        string    `json:"name"`
	Description string    `json:"description"`
	Version     string    `json:"version"`
	Author      string    `json:"author"`
	Created     time.Time `json:"created"`
	Modified    time.Time `json:"modified"`
}

// Pipeline represents the complete visual pipeline
type Pipeline struct {
	Nodes       []CLINode          `json:"nodes"`
	Connections []SocketConnection `json:"connections"`
	Metadata    PipelineMetadata   `json:"metadata"`
}

// YAMLStep represents a step in the YAML pipeline format (for compatibility with run_pipeline.go)
// YAMLStep represents a single step in the pipeline
type YAMLStep struct {
	Name                string            `yaml:"name"`
	Type                string            `yaml:"type,omitempty"`
	Message             string            `yaml:"message,omitempty"`
	Environment         string            `yaml:"environment,omitempty"`
	Commands            interface{}       `yaml:"commands,omitempty"` // Can be []interface{} with strings and maps
	Command             string            `yaml:"command,omitempty"`
	Script              string            `yaml:"script,omitempty"`
	Args                map[string]string `yaml:"args,omitempty"`
	ExpectedOutputFiles map[string]string `yaml:"expected_output_files,omitempty"` // Output socket values
	NodeID              string            `yaml:"_node_id,omitempty"`              // Link to visual node
	LastProcessed       string            `yaml:"last_processed,omitempty"`
	CodeVersion         string            `yaml:"code_version,omitempty"`
	RunDuration         string            `yaml:"run_duration,omitempty"`
}

// VisualNodeMetadata stores visual layout information for a node
type VisualNodeMetadata struct {
	ID       string `yaml:"id"`
	Position Point  `yaml:"position"`
	Size     Size   `yaml:"size"`
}

// VisualConnectionMetadata stores visual connection information
type VisualConnectionMetadata struct {
	ID           string `yaml:"id"`
	FromNodeID   string `yaml:"from_node_id"`
	ToNodeID     string `yaml:"to_node_id"`
	FromSocketID string `yaml:"from_socket_id"`
	ToSocketID   string `yaml:"to_socket_id"`
}

// DesignerMetadata stores all visual designer information
type DesignerMetadata struct {
	Nodes       []VisualNodeMetadata       `yaml:"nodes"`
	Connections []VisualConnectionMetadata `yaml:"connections"`
}

// YAMLPipeline represents the YAML pipeline format
type YAMLPipeline struct {
	PipelineName     string            `yaml:"pipeline_name,omitempty"`
	DesignerMetadata *DesignerMetadata `yaml:"-"` // Excluded from YAML - stored in .reactflow.json instead
	Run              []YAMLStep        `yaml:"run"`
}

// PathToken represents a special path token that can be used in YAML files
type PathToken struct {
	Token        string `json:"token"`
	Description  string `json:"description"`
	ResolvedPath string `json:"resolvedPath"`
}
