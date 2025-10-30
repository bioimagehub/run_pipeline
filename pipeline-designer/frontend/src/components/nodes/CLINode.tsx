import React, { useState } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { usePipelineStore } from '../../stores/pipelineStore';
import { LogFrontend, CountFilesMatchingPattern, PathExists } from '../../../wailsjs/go/main/App';

interface Socket {
  id: string;
  argumentFlag: string;
  value: string;
  type: string;
  socketSide: 'input' | 'output';
  isRequired?: boolean;
  connectedTo?: string | null;
  defaultValue?: string;
}

interface CLINodeData {
  id: string;
  name: string;
  category: string;
  icon: string;
  color: string;
  inputSockets: Socket[];
  outputSockets: Socket[];
  environment: string;
  isCollapsed?: boolean;
  width?: number; // Store node width for persistence
}

const CLINode: React.FC<NodeProps<CLINodeData>> = ({ data, selected, id }) => {
  const [showAllArgs, setShowAllArgs] = useState(false);
  const [localValues, setLocalValues] = useState<{ [socketId: string]: string }>({});
  const [fileCounts, setFileCounts] = useState<{ [socketId: string]: number }>({});
  const [pathExists, setPathExists] = useState<{ [socketId: string]: boolean }>({});
  const [nodeWidth, setNodeWidth] = useState(data.width || 300); // Initialize from data or default
  const { updateNodeSocket, deleteSelectedNode, updateNodeWidth } = usePipelineStore();
  const edges = usePipelineStore((s) => s.edges);
  const nodes = usePipelineStore((s) => s.nodes);
  const currentFilePath = usePipelineStore((s) => s.currentFilePath);
  
  // Helper function to check if types are compatible
  const areTypesCompatible = (type1: string, type2: string): boolean => {
    if (type1 === type2) return true;
    if (type1 === 'path' && (type2 === 'string' || type2 === 'glob_pattern')) return true;
    if (type2 === 'path' && (type1 === 'string' || type1 === 'glob_pattern')) return true;
    if (type1 === 'glob_pattern' && type2 === 'string') return true;
    if (type2 === 'glob_pattern' && type1 === 'string') return true;
    return false;
  };
  
  // Helper function to get socket color based on connection validity
  const getSocketColor = (socket: Socket, isOutput: boolean): string => {
    // Find connected edge
    const connectedEdge = edges.find(edge => 
      isOutput 
        ? edge.source === id && edge.sourceHandle === socket.id
        : edge.target === id && edge.targetHandle === socket.id
    );
    
    if (!connectedEdge) {
      return '#999'; // Default gray for unconnected
    }
    
    // Find the connected socket to check type compatibility
    const connectedNode = nodes.find(n => 
      n.id === (isOutput ? connectedEdge.target : connectedEdge.source)
    );
    
    if (!connectedNode) return '#4CAF50'; // Green if connected but can't find node
    
    const connectedSocketId = isOutput ? connectedEdge.targetHandle : connectedEdge.sourceHandle;
    const connectedSocket = isOutput
      ? connectedNode.data.inputSockets?.find(s => s.id === connectedSocketId)
      : connectedNode.data.outputSockets?.find(s => s.id === connectedSocketId);
    
    if (!connectedSocket) return '#4CAF50'; // Green if connected but can't find socket
    
    // Check type compatibility
    const typesMatch = areTypesCompatible(socket.type, connectedSocket.type);
    return typesMatch ? '#4CAF50' : '#f48771'; // Green for valid, red for invalid
  };
  
  // Helper function to apply transformations to values (defined early for useMemo)
  const applyTransform = (value: string, transform: string): string => {
    if (!value) return '';
    
    switch (transform) {
      case 'dirname': {
        // Extract directory name from glob pattern
        // e.g., "%YAML%/input/**/*.nd2" -> "%YAML%/input"
        // e.g., "%YAML%/data/subfolder/**/*.tif" -> "%YAML%/data/subfolder"
        let dir = value;
        
        // Find the position of glob patterns (*, ?, [)
        const globIndex = dir.search(/[\*\?\[]/);
        if (globIndex !== -1) {
          // Get everything before the glob pattern
          dir = dir.substring(0, globIndex);
          // Remove any trailing slashes or partial path segments
          dir = dir.replace(/\/*$/, ''); // Remove trailing slashes
        }
        
        return dir;
      }
      case 'basename': {
        // Extract filename without extension
        const parts = value.split('/');
        const filename = parts[parts.length - 1];
        return filename.replace(/\.[^.]*$/, '');
      }
      default:
        return value;
    }
  };
  
  // Pre-calculate all resolved input values (two-pass to handle dependencies)
  const resolvedInputValues = React.useMemo(() => {
    const values: { [socketId: string]: string } = {};
    
    if (!data.inputSockets) return values;
    
    // First pass: collect all direct values (no placeholders)
    data.inputSockets.forEach((socket) => {
      if (socket.value) {
        values[socket.id] = socket.value;
      } else if (socket.defaultValue && !socket.defaultValue.includes('<')) {
        values[socket.id] = socket.defaultValue;
      } else {
        values[socket.id] = '';
      }
    });
    
    // Second pass: resolve placeholders using first pass values
    data.inputSockets.forEach((socket) => {
      if (!socket.value && socket.defaultValue && socket.defaultValue.includes('<')) {
        let resolved = socket.defaultValue;
        
        // Resolve placeholders from other sockets
        data.inputSockets?.forEach((otherSocket) => {
          if (otherSocket.id === socket.id) return; // Skip self
          
          const flagName = otherSocket.argumentFlag.replace(/^--/, '').replace(/-/g, '_');
          const otherValue = values[otherSocket.id] || '';
          
          // Handle transformations like <input_search_pattern:dirname>
          const transformRegex = new RegExp(`<${flagName}:([^>]+)>`, 'g');
          resolved = resolved.replace(transformRegex, (match, transform) => {
            return applyTransform(otherValue, transform);
          });
          
          // Simple placeholder replacement
          const placeholder = `<${flagName}>`;
          resolved = resolved.split(placeholder).join(otherValue);
        });
        
        values[socket.id] = resolved;
      }
    });
    
    return values;
  }, [data.inputSockets]);
  
  // Pre-calculate all resolved output values (using resolved input values)
  const resolvedOutputValues = React.useMemo(() => {
    const values: { [socketId: string]: string } = {};
    
    if (!data.outputSockets) return values;
    
    // Output sockets resolve placeholders from input sockets
    data.outputSockets.forEach((socket) => {
      if (socket.defaultValue && socket.defaultValue.includes('<')) {
        let resolved = socket.defaultValue;
        
        // Resolve placeholders from input sockets using their resolved values
        data.inputSockets?.forEach((inputSocket) => {
          const flagName = inputSocket.argumentFlag.replace(/^--/, '').replace(/-/g, '_');
          const inputValue = resolvedInputValues[inputSocket.id] || '';
          
          // Handle transformations like <output_folder:dirname>
          const transformRegex = new RegExp(`<${flagName}:([^>]+)>`, 'g');
          resolved = resolved.replace(transformRegex, (match, transform) => {
            return applyTransform(inputValue, transform);
          });
          
          // Simple placeholder replacement
          const placeholder = `<${flagName}>`;
          resolved = resolved.split(placeholder).join(inputValue);
        });
        
        values[socket.id] = resolved;
      } else {
        // No placeholder, use value or default
        values[socket.id] = socket.value || socket.defaultValue || '';
      }
    });
    
    return values;
  }, [data.outputSockets, resolvedInputValues]);

  const getCategoryColor = (category: string) => {
    const colors: Record<string, string> = {
      'Segmentation': '#c586c0',
      'Image Processing': '#569cd6',
      'Measurement': '#4ec9b0',
      'Tracking': '#ce9178',
      'Visualization': '#f48771',
      'Utilities': '#858585',
    };
    return colors[category] || '#858585';
  };

  const handleInputChange = (socketId: string, value: string) => {
    console.log('Input changed:', { nodeId: id, socketId, value });
    // Update local state immediately for responsive typing
    setLocalValues(prev => ({ ...prev, [socketId]: value }));
    
    // For path-type sockets, trigger validation during typing (debounced)
    const inputSocket = data.inputSockets?.find(s => s.id === socketId);
    const outputSocket = data.outputSockets?.find(s => s.id === socketId);
    const socket = inputSocket || outputSocket;
    
    if (socket?.type === 'path' && value && value.trim() !== '') {
      // Debounce: clear previous timeout and set a new one
      if ((window as any).pathValidationTimeouts?.[socketId]) {
        clearTimeout((window as any).pathValidationTimeouts[socketId]);
      }
      if (!(window as any).pathValidationTimeouts) {
        (window as any).pathValidationTimeouts = {};
      }
      (window as any).pathValidationTimeouts[socketId] = setTimeout(() => {
        checkPathExists(socketId, value);
      }, 500); // Wait 500ms after user stops typing
    }
  };

  const handleInputCommit = (socketId: string, value: string) => {
    console.log('Input committed:', { nodeId: id, socketId, value });
    // Update the store and propagate to connected nodes
    updateNodeSocket(id, socketId, value);
    
    // If this is a glob_pattern socket, update the file count
    const inputSocket = data.inputSockets?.find(s => s.id === socketId);
    const outputSocket = data.outputSockets?.find(s => s.id === socketId);
    const socket = inputSocket || outputSocket;
    
    if (socket) {
      if (socket.type === 'glob_pattern') {
        updateFileCount(socketId, value);
      } else if (socket.type === 'path') {
        checkPathExists(socketId, value);
      }
    }
  };

  // Update file count for a glob_pattern socket
  const updateFileCount = async (socketId: string, pattern: string) => {
    if (!pattern || pattern.trim() === '') {
      setFileCounts(prev => ({ ...prev, [socketId]: 0 }));
      return;
    }

    try {
      const count = await CountFilesMatchingPattern(pattern, currentFilePath || '');
      setFileCounts(prev => ({ ...prev, [socketId]: count }));
    } catch (error) {
      console.error('Failed to count files:', error);
      setFileCounts(prev => ({ ...prev, [socketId]: 0 }));
    }
  };

  // Handle horizontal resize
  const handleResizeMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    const startX = e.clientX;
    const startWidth = nodeWidth;
    let currentWidth = startWidth;

    const handleMouseMove = (moveEvent: MouseEvent) => {
      const deltaX = moveEvent.clientX - startX;
      currentWidth = Math.max(300, Math.min(800, startWidth + deltaX)); // Min 300px, max 800px
      setNodeWidth(currentWidth);
    };

    const handleMouseUp = () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = ''; // Reset cursor
      
      // Save the final width to the store (use currentWidth, not nodeWidth)
      updateNodeWidth(id, currentWidth);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    document.body.style.cursor = 'ew-resize'; // Set cursor during drag
  };

  // Check if a path exists for a path socket
  const checkPathExists = async (socketId: string, path: string) => {
    if (!path || path.trim() === '') {
      setPathExists(prev => ({ ...prev, [socketId]: false }));
      return;
    }

    try {
      const exists = await PathExists(path);
      setPathExists(prev => ({ ...prev, [socketId]: exists }));
    } catch (error) {
      console.error('Failed to check path existence:', error);
      setPathExists(prev => ({ ...prev, [socketId]: false }));
    }
  };

  // Sync local values with props when data changes externally
  React.useEffect(() => {
    const newLocalValues: { [socketId: string]: string } = {};
    
    // Update input sockets - only sync from store for connected inputs or initial load
    data.inputSockets?.forEach(socket => {
      // Check if socket is connected (target of an edge)
      const isConnected = edges.some(
        edge => edge.target === id && edge.targetHandle === socket.id
      );
      
      // If connected, always update from store (controlled by connection)
      // If not connected, only initialize if we don't have a local value yet
      if (isConnected) {
        newLocalValues[socket.id] = socket.value || '';
      } else if (!(socket.id in localValues)) {
        newLocalValues[socket.id] = socket.value || '';
      }
    });
    
    // Update output sockets - only initialize if we don't have a local value yet
    data.outputSockets?.forEach(socket => {
      if (!(socket.id in localValues)) {
        newLocalValues[socket.id] = socket.value || '';
      }
    });
    
    if (Object.keys(newLocalValues).length > 0) {
      setLocalValues(prev => ({ ...prev, ...newLocalValues }));
    }
  }, [data.inputSockets, data.outputSockets, edges, id, localValues]);

  // Initialize file counts for glob_pattern sockets and path validation for path sockets
  React.useEffect(() => {
    data.inputSockets?.forEach(socket => {
      if (socket.type === 'glob_pattern' && socket.value) {
        updateFileCount(socket.id, socket.value);
      } else if (socket.type === 'path' && socket.value) {
        checkPathExists(socket.id, socket.value);
      }
    });
    
    // Also initialize file counts for output sockets with glob_pattern type
    data.outputSockets?.forEach(socket => {
      if (socket.type === 'glob_pattern' && socket.value) {
        updateFileCount(socket.id, socket.value);
      } else if (socket.type === 'path' && socket.value) {
        checkPathExists(socket.id, socket.value);
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data.inputSockets, data.outputSockets]);

  // Sync node width when data.width changes (e.g., when loading from file)
  React.useEffect(() => {
    if (data.width !== undefined && data.width !== nodeWidth) {
      setNodeWidth(data.width);
    }
  }, [data.width, nodeWidth]);

  const handleUpdateSocket = (socket: Socket) => {
    // Resolve placeholders in the socket's defaultValue and update the socket value
    let resolvedValue = socket.defaultValue || '';
    
    if (resolvedValue.includes('<') && resolvedValue.includes('>')) {
      // Use RESOLVED input socket values (not raw values) for placeholder resolution
      // This ensures multi-level placeholders work correctly
      data.inputSockets?.forEach((inputSocket) => {
        const flagName = inputSocket.argumentFlag.replace(/^--/, '').replace(/-/g, '_');
        
        // Use resolvedInputValues instead of inputSocket.value
        // This handles cases where input sockets themselves have placeholders
        const inputValue = resolvedInputValues[inputSocket.id] || inputSocket.value || '';
        
        // Only replace placeholders if the input value is non-empty
        if (inputValue.trim() !== '') {
          // Handle transformations like <input_search_pattern:dirname>
          const transformRegex = new RegExp(`<${flagName}:([^>]+)>`, 'g');
          resolvedValue = resolvedValue.replace(transformRegex, (match, transform) => {
            return applyTransform(inputValue, transform);
          });
          
          // Simple placeholder replacement
          const placeholder = `<${flagName}>`;
          resolvedValue = resolvedValue.split(placeholder).join(inputValue);
        } else {
          // If input is empty, remove the placeholder entirely (replace with empty string)
          const transformRegex = new RegExp(`<${flagName}:([^>]+)>`, 'g');
          resolvedValue = resolvedValue.replace(transformRegex, '');
          
          const placeholder = `<${flagName}>`;
          resolvedValue = resolvedValue.split(placeholder).join('');
        }
      });
    }
    
    // Update both local state and store
    setLocalValues(prev => ({ ...prev, [socket.id]: resolvedValue }));
    updateNodeSocket(id, socket.id, resolvedValue);
    
    const msg = `[CLINode] Update button clicked for socket: ${socket.argumentFlag} (ID: ${socket.id}, NodeId: ${id}, Resolved: ${resolvedValue})`;
    console.log(msg);
    LogFrontend(msg).catch(console.error);
  };

  const handleUpdateAll = () => {
    const msg = `[CLINode] Update All button clicked for node: ${data.name} (ID: ${id})`;
    console.log(msg);
    LogFrontend(msg).catch(console.error);
    
    let updateCount = 0;
    
    // Update all input sockets with placeholders
    if (data.inputSockets) {
      data.inputSockets.forEach((socket) => {
        if (socket.defaultValue && socket.defaultValue.includes('<') && socket.defaultValue.includes('>')) {
          handleUpdateSocket(socket);
          updateCount++;
        }
      });
    }
    
    // Update all output sockets with placeholders
    if (data.outputSockets) {
      data.outputSockets.forEach((socket) => {
        if (socket.defaultValue && socket.defaultValue.includes('<') && socket.defaultValue.includes('>')) {
          handleUpdateSocket(socket);
          updateCount++;
        }
      });
    }
    
    LogFrontend(`[CLINode] Updated ${updateCount} sockets with placeholders`).catch(console.error);
  };

  // Helper function to resolve placeholders in output socket values
  const resolveOutputPlaceholders = (outputValue: string): string => {
    if (!outputValue) return outputValue;
    
    let resolved = outputValue;
    
    // Get all input socket values
    if (data.inputSockets) {
      data.inputSockets.forEach((inputSocket) => {
        // Convert flag to placeholder format: --output-folder -> <output_folder>
        const flagName = inputSocket.argumentFlag.replace(/^--/, '').replace(/-/g, '_');
        const value = inputSocket.value || '';
        
        // Handle transformations like <input_search_pattern:dirname>
        const transformRegex = new RegExp(`<${flagName}:([^>]+)>`, 'g');
        resolved = resolved.replace(transformRegex, (match, transform) => {
          return applyTransform(value, transform);
        });
        
        // Simple placeholder replacement
        const placeholder = `<${flagName}>`;
        resolved = resolved.split(placeholder).join(value);
      });
    }
    
    return resolved;
  };

  // Helper function to get display value for input socket
  const getInputSocketDisplayValue = (socket: Socket): string => {
    // Use local value if available (for responsive typing), otherwise fall back to socket value
    if (localValues[socket.id] !== undefined) {
      return localValues[socket.id];
    }
    // Use pre-calculated resolved value
    return resolvedInputValues[socket.id] || socket.value || '';
  };

  // Helper function to get display value for output socket
  const getOutputSocketDisplayValue = (socket: Socket): string => {
    // Use local value if available (for responsive typing), otherwise fall back to socket value
    if (localValues[socket.id] !== undefined) {
      return localValues[socket.id];
    }
    return socket.value || '';
  };

  const color = getCategoryColor(data.category);
  
  // Split sockets into required and optional
  const requiredArgs = data.inputSockets?.filter(s => s.isRequired) || [];
  const optionalArgs = data.inputSockets?.filter(s => !s.isRequired) || [];
  const hasOptionalArgs = optionalArgs.length > 0;
  const shouldCollapse = requiredArgs.length > 4;
  
  // Determine which args to show
  const argsToShow = showAllArgs || !shouldCollapse 
    ? data.inputSockets || []
    : requiredArgs;

  return (
    <div
      className={`cli-node ${selected ? 'selected' : ''}`}
      style={{ borderColor: color, width: `${nodeWidth}px`, position: 'relative' }}
    >
      {/* Resize Handle */}
      <div
        className="nodrag"
        onMouseDown={handleResizeMouseDown}
        style={{
          position: 'absolute',
          right: 0,
          top: 0,
          bottom: 0,
          width: '8px',
          cursor: 'ew-resize',
          background: 'transparent',
          zIndex: 10,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
        title="Drag to resize node width"
      >
        <div style={{
          width: '2px',
          height: '30px',
          background: selected ? '#2563eb' : '#444',
          borderRadius: '1px',
          transition: 'background 0.2s',
        }} />
      </div>
      {/* Node Header */}
      <div className="node-header" style={{ backgroundColor: color }}>
        <span className="node-icon">{data.icon}</span>
        <span className="node-name">{data.name}</span>
        {selected && (
          <>
            <button
              className="nodrag"
              style={{
                marginLeft: 'auto',
                background: '#2a5a2a',
                color: '#bfb',
                border: '1px solid #3a7a3a',
                borderRadius: '3px',
                padding: '2px 8px',
                cursor: 'pointer',
                fontSize: '12px',
                marginRight: '4px',
                fontWeight: '500',
              }}
              title="Update all fields with placeholders"
              onClick={(e) => {
                e.stopPropagation();
                handleUpdateAll();
              }}
            >🔄 Update All</button>
            <button
              className="nodrag"
              style={{
                background: '#1a7f37',
                color: '#fff',
                border: 'none',
                borderRadius: '3px',
                padding: '2px 8px',
                cursor: 'pointer',
                fontSize: '12px',
                marginRight: '4px',
              }}
              title="Run This Node"
              onClick={(e) => {
                e.stopPropagation();
                // Trigger run event
                window.dispatchEvent(new CustomEvent('runNode', { detail: { nodeId: id } }));
              }}
            >▶️ Run</button>
            <button
              className="nodrag"
              style={{
                background: '#222',
                color: '#f48771',
                border: 'none',
                borderRadius: '3px',
                padding: '2px 8px',
                cursor: 'pointer',
                fontSize: '12px',
              }}
              title="Delete Node (or press Delete key)"
              onClick={(e) => {
                e.stopPropagation();
                deleteSelectedNode();
                LogFrontend(`[CLINode] Delete button clicked for node: ${id}`).catch(console.error);
              }}
            >🗑️ Delete</button>
          </>
        )}
      </div>

      {/* Node Body */}
      {!data.isCollapsed && (
        <div className="node-body" style={{ display: 'flex', gap: '12px', padding: '12px' }}>
          {/* Left Column: Input Parameters */}
          <div style={{ flex: 1, minWidth: '140px' }}>
            <div style={{ 
              fontSize: '10px', 
              fontWeight: 'bold', 
              color: '#888', 
              marginBottom: '8px',
              textTransform: 'uppercase'
            }}>
              Inputs
            </div>
            {argsToShow && argsToShow.length > 0 ? (
              <div>
                {argsToShow.map((socket) => (
                  <div key={socket.id} style={{ marginBottom: '8px', position: 'relative' }}>
                    <Handle
                      type="target"
                      position={Position.Left}
                      id={socket.id}
                      style={{
                        position: 'absolute',
                        top: '8px',
                        left: '-20px',
                        background: getSocketColor(socket, false),
                        transition: 'background 0.2s ease',
                      }}
                      title={`Type: ${socket.type}`}
                    />
                    <label style={{ 
                      fontSize: '10px', 
                      color: '#aaa', 
                      display: 'block', 
                      marginBottom: '2px',
                      textAlign: 'left'
                    }}>
                      {socket.argumentFlag}
                      {socket.isRequired && <span style={{ color: '#f48771' }}>*</span>}
                    </label>
                    {/* If this input socket is the TARGET of any edge, lock editing here and allow the source (output) to control the value */}
                    {(() => {
                      const isTargetConnected = edges.some(
                        (edge) => edge.target === id && edge.targetHandle === socket.id
                      );
                      
                      // Use local value for responsive typing
                      const displayValue = getInputSocketDisplayValue(socket);
                      
                      // Check if socket has placeholders in defaultValue
                      const hasPlaceholders = socket.defaultValue && socket.defaultValue.includes('<') && socket.defaultValue.includes('>');

                      return (
                        <div style={{ display: 'flex', gap: '2px', alignItems: 'center' }}>
                          {hasPlaceholders && (
                            <button
                              className="nodrag"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleUpdateSocket(socket);
                              }}
                              onMouseDown={(e) => e.stopPropagation()}
                              title="Update socket value from placeholders"
                              style={{
                                background: '#2a2a2a',
                                border: '1px solid #444',
                                borderRadius: '3px',
                                color: '#ddd',
                                cursor: 'pointer',
                                fontSize: '11px',
                                padding: '4px 6px',
                                flexShrink: 0,
                              }}
                            >
                              🔄
                            </button>
                          )}
                          {/* Render dropdown for bool type */}
                          {socket.type === 'bool' ? (
                            <select
                              className="nodrag nopan"
                              value={displayValue || ''}
                              onChange={(e) => {
                                if (!isTargetConnected) {
                                  const newValue = e.target.value;
                                  handleInputChange(socket.id, newValue);
                                  handleInputCommit(socket.id, newValue);
                                }
                              }}
                              onMouseDown={(e) => e.stopPropagation()}
                              onClick={(e) => e.stopPropagation()}
                              disabled={isTargetConnected}
                              style={{
                                flex: 1,
                                padding: '4px 6px',
                                fontSize: '11px',
                                background: isTargetConnected ? '#222' : '#2a2a2a',
                                border: isTargetConnected ? '1px dashed #555' : '1px solid #444',
                                borderRadius: '3px',
                                color: isTargetConnected ? '#888' : '#ddd',
                                cursor: isTargetConnected ? 'not-allowed' : 'pointer',
                              }}
                            >
                              <option value="">No</option>
                              <option value="true">Yes</option>
                            </select>
                          ) : (
                            <input
                              type="text"
                              className="nodrag nopan"
                              value={displayValue}
                              onChange={(e) => {
                                // Only allow changes when not target-connected
                                if (isTargetConnected) {
                                  return;
                                }
                                handleInputChange(socket.id, e.target.value);
                              }}
                              onBlur={(e) => {
                                // Commit changes on blur
                                if (!isTargetConnected) {
                                  handleInputCommit(socket.id, e.target.value);
                                }
                              }}
                              onKeyDown={(e) => {
                                // Commit changes on Enter or Tab
                                if ((e.key === 'Enter' || e.key === 'Tab') && !isTargetConnected) {
                                  handleInputCommit(socket.id, e.currentTarget.value);
                                }
                              }}
                              onMouseDown={(e) => e.stopPropagation()}
                              onClick={(e) => e.stopPropagation()}
                              placeholder={socket.defaultValue || `Enter ${socket.argumentFlag}...`}
                              disabled={isTargetConnected}
                              style={{
                                flex: 1,
                                padding: '4px 6px',
                                fontSize: '11px',
                                background: isTargetConnected ? '#222' : '#2a2a2a',
                                border: isTargetConnected ? '1px dashed #555' : '1px solid #444',
                                borderRadius: '3px',
                                color: isTargetConnected ? '#888' : '#ddd',
                                textAlign: 'left',
                                cursor: isTargetConnected ? 'not-allowed' : 'text',
                              }}
                            />
                          )}
                          {/* Show file count for glob_pattern type */}
                          {socket.type === 'glob_pattern' && displayValue && (
                            <span
                              style={{
                                fontSize: '10px',
                                color: fileCounts[socket.id] === 0 ? '#f48771' : '#4ec9b0',
                                fontWeight: 'bold',
                                padding: '2px 6px',
                                background: '#1a1a1a',
                                borderRadius: '3px',
                                border: '1px solid #444',
                                flexShrink: 0,
                                minWidth: '30px',
                                textAlign: 'center',
                              }}
                              title={`${fileCounts[socket.id] || 0} file(s) matching pattern`}
                            >
                              ({fileCounts[socket.id] ?? '...'})
                            </span>
                          )}
                          {/* Show green checkmark for path type if path exists */}
                          {socket.type === 'path' && displayValue && pathExists[socket.id] && (
                            <span
                              style={{
                                fontSize: '14px',
                                color: '#4ec9b0',
                                flexShrink: 0,
                                lineHeight: 1,
                              }}
                              title="Path exists"
                            >
                              ✓
                            </span>
                          )}
                        </div>
                      );
                    })()}
                  </div>
                ))}
                
                {/* Expand/Collapse Button */}
                {shouldCollapse && hasOptionalArgs && (
                  <button
                    className="nodrag"
                    onClick={(e) => {
                      e.stopPropagation();
                      setShowAllArgs(!showAllArgs);
                    }}
                    style={{
                      width: '100%',
                      padding: '4px',
                      background: '#333',
                      border: '1px solid #555',
                      borderRadius: '3px',
                      color: '#aaa',
                      fontSize: '10px',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '4px',
                      marginTop: '4px'
                    }}
                  >
                    {showAllArgs ? (
                      <>
                        <ChevronDown size={12} />
                        Hide Optional ({optionalArgs.length})
                      </>
                    ) : (
                      <>
                        <ChevronRight size={12} />
                        Show All ({optionalArgs.length} optional)
                      </>
                    )}
                  </button>
                )}
              </div>
            ) : (
              <div style={{ fontSize: '10px', color: '#666', fontStyle: 'italic' }}>No inputs</div>
            )}
          </div>

          {/* Vertical Divider */}
          <div style={{ width: '1px', background: '#444', margin: '0 4px' }}></div>

          {/* Right Column: Output Sockets */}
          <div style={{ flex: 1, minWidth: '140px' }}>
            <div style={{ 
              fontSize: '10px', 
              fontWeight: 'bold', 
              color: '#888', 
              marginBottom: '8px',
              textTransform: 'uppercase',
              textAlign: 'right'
            }}>
              Output Files
            </div>
            {data.outputSockets && data.outputSockets.length > 0 ? (
              <div>
                {data.outputSockets.map((socket) => {
                  // Use local value for responsive typing
                  const displayValue = getOutputSocketDisplayValue(socket);
                  
                  // Check if socket has placeholders in defaultValue
                  const hasPlaceholders = socket.defaultValue && socket.defaultValue.includes('<') && socket.defaultValue.includes('>');
                  
                  // Debug logging for output sockets
                  if (data.name === "Convert to tif") {
                    console.log('[CLINode Output Socket Debug]', {
                      nodeName: data.name,
                      socketFlag: socket.argumentFlag,
                      socketId: socket.id,
                      defaultValue: socket.defaultValue,
                      hasPlaceholders: hasPlaceholders,
                      value: socket.value
                    });
                  }
                  
                  return (
                    <div key={socket.id} style={{ position: 'relative', marginBottom: '8px' }}>
                      <Handle
                        type="source"
                        position={Position.Right}
                        id={socket.id}
                        style={{
                          position: 'absolute',
                          top: '8px',
                          right: '-20px',
                          background: getSocketColor(socket, true),
                          transition: 'background 0.2s ease',
                        }}
                        title={`Type: ${socket.type}`}
                      />
                      <div style={{ 
                        fontSize: '10px', 
                        color: '#aaa', 
                        textAlign: 'right',
                        marginBottom: '2px'
                      }}>
                        {socket.argumentFlag}
                      </div>
                      <div style={{ display: 'flex', gap: '2px', alignItems: 'center', justifyContent: 'flex-end' }}>
                        {hasPlaceholders && (
                          <button
                            className="nodrag"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleUpdateSocket(socket);
                            }}
                            onMouseDown={(e) => e.stopPropagation()}
                            title="Update output value from placeholders"
                            style={{
                              background: '#2a2a2a',
                              border: '1px solid #444',
                              borderRadius: '3px',
                              color: '#ddd',
                              cursor: 'pointer',
                              fontSize: '11px',
                              padding: '4px 6px',
                              flexShrink: 0,
                            }}
                          >
                            🔄
                          </button>
                        )}
                        <input
                          type="text"
                          className="nodrag nopan"
                          value={displayValue}
                          onChange={(e) => handleInputChange(socket.id, e.target.value)}
                          onBlur={(e) => handleInputCommit(socket.id, e.target.value)}
                          onKeyDown={(e) => {
                            // Commit changes on Enter or Tab
                            if (e.key === 'Enter' || e.key === 'Tab') {
                              handleInputCommit(socket.id, e.currentTarget.value);
                            }
                          }}
                          onMouseDown={(e) => e.stopPropagation()}
                          onClick={(e) => e.stopPropagation()}
                          placeholder={socket.defaultValue || 'output value'}
                          title={socket.defaultValue ? `Template: ${socket.defaultValue}` : 'Output value'}
                          style={{
                            flex: 1,
                            padding: '4px 6px',
                            fontSize: '11px',
                            background: '#2a2a2a',
                            border: '1px solid #444',
                            borderRadius: '3px',
                            color: '#ddd',
                            textAlign: 'right',
                            cursor: 'text',
                          }}
                        />
                        {/* Show file count for glob_pattern type */}
                        {socket.type === 'glob_pattern' && displayValue && (
                          <span
                            style={{
                              fontSize: '10px',
                              color: fileCounts[socket.id] === 0 ? '#f48771' : '#4ec9b0',
                              fontWeight: 'bold',
                              padding: '2px 6px',
                              background: '#1a1a1a',
                              borderRadius: '3px',
                              border: '1px solid #444',
                              flexShrink: 0,
                              minWidth: '30px',
                              textAlign: 'center',
                            }}
                            title={`${fileCounts[socket.id] || 0} file(s) matching pattern`}
                          >
                            ({fileCounts[socket.id] ?? '...'})
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div style={{ fontSize: '10px', color: '#666', fontStyle: 'italic', textAlign: 'right' }}>No outputs</div>
            )}
          </div>
        </div>
      )}

      {/* Node Footer */}
      <div className="node-footer" style={{ position: 'relative', height: '24px' }}>
        <span className="environment-badge">{data.environment}</span>
        {/* Generic anchor: lower left (input) */}
        <Handle
          type="target"
          position={Position.Bottom}
          id={`generic-in-${id}`}
          style={{
            left: 0,
            bottom: 0,
            background: '#666',
            borderRadius: '50%',
            width: '12px',
            height: '12px',
            position: 'absolute',
          }}
        />
        {/* Generic anchor: lower right (output) */}
        <Handle
          type="source"
          position={Position.Bottom}
          id={`generic-out-${id}`}
          style={{
            right: 0,
            bottom: 0,
            background: '#666',
            borderRadius: '50%',
            width: '12px',
            height: '12px',
            position: 'absolute',
          }}
        />
      </div>
    </div>
  );
};

export default CLINode;
