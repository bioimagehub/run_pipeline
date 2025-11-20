// types.ts - Re-export Wails generated types
// This ensures consistency between frontend and backend

import * as models from '../wailsjs/go/models';

// Re-export the main namespace
export { models };

// Convenience type aliases
export type CLINode = models.main.CLINode;
export type Socket = models.main.Socket;
export type CLIDefinition = models.main.CLIDefinition;
export type ArgumentDefinition = models.main.ArgumentDefinition;
export type Pipeline = models.main.Pipeline;
export type PipelineMetadata = models.main.PipelineMetadata;
export type SocketConnection = models.main.SocketConnection;
export type Point = models.main.Point;
export type Size = models.main.Size;

// String unions for type safety (these are strings in Go, unions in TS for IDE help)
export type ArgumentType = "string" | "int" | "float" | "boolean" | "file" | "folder" | "path" | "glob_pattern" | "bool";
export type SocketSide = "input" | "output";
export type TestStatus = "not-tested" | "success" | "failure" | "running" | "not_run" | "failed";
