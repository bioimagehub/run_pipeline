import React, { useState } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { usePipelineStore } from '../../stores/pipelineStore';
import { LogFrontend } from '../../../wailsjs/go/main/App';

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
}

const CLINode: React.FC<NodeProps<CLINodeData>> = ({ data, selected, id }) => {
  const [showAllArgs, setShowAllArgs] = useState(false);
  const { updateNodeSocket } = usePipelineStore();
  const edges = usePipelineStore((s) => s.edges);
  
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
    updateNodeSocket(id, socketId, value);
  };

  const handleUpdateSocket = (socket: Socket) => {
    const msg = `[CLINode] Update button clicked for socket: ${socket.argumentFlag} (ID: ${socket.id}, NodeId: ${id}, Value: ${socket.value || resolvedInputValues[socket.id]})`;
    console.log(msg);
    LogFrontend(msg).catch(console.error);
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
    // Use pre-calculated resolved value
    return resolvedInputValues[socket.id] || socket.value || '';
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
      style={{ borderColor: color }}
    >
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
              title="Delete Node"
              onClick={(e) => {
                e.stopPropagation();
                // Delete selected node via store
                window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Delete' }));
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
                        background: socket.connectedTo ? '#4CAF50' : '#999',
                      }}
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
                      
                      // Get display value (resolved if placeholder-based default)
                      const displayValue = getInputSocketDisplayValue(socket);

                      return (
                        <div style={{ display: 'flex', gap: '2px', alignItems: 'center' }}>
                          <button
                            className="nodrag"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleUpdateSocket(socket);
                            }}
                            onMouseDown={(e) => e.stopPropagation()}
                            title="Update socket value"
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
                            onMouseDown={(e) => e.stopPropagation()}
                            onClick={(e) => e.stopPropagation()}
                            placeholder={`Enter ${socket.argumentFlag}...`}
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
              Outputs
            </div>
            {data.outputSockets && data.outputSockets.length > 0 ? (
              <div>
                {data.outputSockets.map((socket) => {
                  // Use pre-calculated resolved value
                  const displayValue = resolvedOutputValues[socket.id] || socket.value || '';
                  
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
                          background: socket.connectedTo ? '#4CAF50' : '#999',
                        }}
                      />
                      <div style={{ 
                        fontSize: '10px', 
                        color: '#aaa', 
                        textAlign: 'right',
                        marginBottom: '2px'
                      }}>
                        {socket.argumentFlag}
                      </div>
                      <input
                        type="text"
                        className="nodrag nopan"
                        value={displayValue}
                        readOnly
                        onMouseDown={(e) => e.stopPropagation()}
                        onClick={(e) => e.stopPropagation()}
                        placeholder="output value"
                        title={`Resolved pattern: ${displayValue}`}
                        style={{
                          width: '100%',
                          padding: '4px 6px',
                          fontSize: '11px',
                          background: '#2a2a2a',
                          border: '1px solid #444',
                          borderRadius: '3px',
                          color: '#ddd',
                          textAlign: 'right',
                          cursor: 'default',
                          opacity: 0.8,
                        }}
                      />
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
