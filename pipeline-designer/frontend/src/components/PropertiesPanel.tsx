import React, { useState } from 'react';
import { usePipelineStore } from '../stores/pipelineStore';
import { GetFileListPreview, LogFrontend } from '../../wailsjs/go/main/App';

const PropertiesPanel: React.FC = () => {
  // CRITICAL: Subscribe to the ENTIRE store to ensure re-renders
  const store = usePipelineStore();
  const selectedNode = store.selectedNode;
  const updateNodeSocket = store.updateNodeSocket;
  
  const [filePreview, setFilePreview] = useState<{ [key: string]: string }>({});
  const [loadingPreview, setLoadingPreview] = useState<{ [key: string]: boolean }>({});
  // Local state for socket values - MUST be declared before any conditional returns
  const [socketValues, setSocketValues] = useState<{ [key: string]: string }>({});

  // Helper function to resolve placeholders in output socket values
  const resolveOutputPlaceholders = (outputValue: string): string => {
    if (!selectedNode || !outputValue) return outputValue;
    
    let resolved = outputValue;
    
    // Get all input socket values
    if (selectedNode.data && selectedNode.data.inputSockets) {
      selectedNode.data.inputSockets.forEach((inputSocket: any) => {
        // Convert flag to placeholder format: --output-folder -> <output_folder>
        const flagName = inputSocket.argumentFlag.replace(/^--/, '').replace(/-/g, '_');
        const value = inputSocket.value || inputSocket.defaultValue || '';
        
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

  // Helper function to apply transformations to values
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

  // Log when PropertiesPanel renders
  React.useEffect(() => {
    const msg = `[PropertiesPanel] Rendered - selectedNode: ${selectedNode ? `ID=${selectedNode.id}, hasData=${!!selectedNode.data}` : 'null'}`;
    console.log(msg);
    LogFrontend(msg).catch(console.error);
  }, [selectedNode]);

  if (!selectedNode) {
    console.log('[PropertiesPanel] No node selected, showing empty state');
    LogFrontend('[PropertiesPanel] No node selected, showing empty state').catch(console.error);
    return (
      <div className="properties-panel">
        <div className="panel-header">
          <h2>Properties</h2>
        </div>
        <div className="panel-content empty">
          <p>Select a node to edit its properties</p>
        </div>
      </div>
    );
  }

  React.useEffect(() => {
    if (selectedNode && selectedNode.data && selectedNode.data.inputSockets) {
      // Initialize local state from socket values, resolving placeholders
      const initialValues: { [key: string]: string } = {};
      
      // First pass: set all direct values (no placeholders)
      selectedNode.data.inputSockets.forEach((socket: any) => {
        if (socket.value) {
          initialValues[socket.id] = socket.value;
        } else if (socket.defaultValue && !socket.defaultValue.includes('<')) {
          // Default value without placeholders
          initialValues[socket.id] = socket.defaultValue;
        } else {
          initialValues[socket.id] = '';
        }
      });
      
      // Second pass: resolve placeholders now that we have all direct values
      selectedNode.data.inputSockets.forEach((socket: any) => {
        if (!socket.value && socket.defaultValue && socket.defaultValue.includes('<')) {
          // Resolve placeholders in default value
          const resolved = resolveInputPlaceholders(
            socket.defaultValue, 
            socket.argumentFlag,
            initialValues
          );
          console.log(`[PropertiesPanel] Resolving ${socket.argumentFlag}: "${socket.defaultValue}" -> "${resolved}"`);
          initialValues[socket.id] = resolved;
        }
      });
      
      setSocketValues(initialValues);
    }
  }, [selectedNode]);

  // Helper function to resolve placeholders in input socket default values
  const resolveInputPlaceholders = (
    defaultValue: string, 
    currentFlag: string,
    currentValues: { [key: string]: string } = {}
  ): string => {
    if (!selectedNode || !defaultValue) return defaultValue;
    
    let resolved = defaultValue;
    
    // Get all input socket values (for cross-references)
    if (selectedNode.data && selectedNode.data.inputSockets) {
      selectedNode.data.inputSockets.forEach((inputSocket: any) => {
        // Skip self-reference
        if (inputSocket.argumentFlag === currentFlag) return;
        
        const flagName = inputSocket.argumentFlag.replace(/^--/, '').replace(/-/g, '_');
        
        // Try to get value from: current state > socket value > socket default
        let value = currentValues[inputSocket.id] || inputSocket.value || inputSocket.defaultValue || '';
        
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

  const handleSocketInputChange = (socketId: string, newValue: string) => {
    setSocketValues((prev) => {
      const updated = { ...prev, [socketId]: newValue };
      
      // Update dependent sockets that have placeholder-based defaults
      if (selectedNode && selectedNode.data && selectedNode.data.inputSockets) {
        selectedNode.data.inputSockets.forEach((socket: any) => {
          // Only update if this socket has no user-entered value and has a placeholder default
          if (!socket.value && socket.defaultValue && socket.defaultValue.includes('<')) {
            const resolved = resolveInputPlaceholders(
              socket.defaultValue,
              socket.argumentFlag,
              updated
            );
            updated[socket.id] = resolved;
          }
        });
      }
      
      return updated;
    });
  };

  const handleSocketInputBlur = (socketId: string) => {
    updateNodeSocket(selectedNode.id, socketId, socketValues[socketId]);
    setFilePreview(prev => ({ ...prev, [socketId]: '' }));
  };

  const handleUndo = () => {
    // Use global undo from store
    // (button can call usePipelineStore().undo)
  };

  const handleRedo = () => {
    // Use global redo from store
    // (button can call usePipelineStore().redo)
  };

  const handlePreviewFiles = async (socket: any) => {
    setLoadingPreview(prev => ({ ...prev, [socket.id]: true }));
    try {
      const pattern = socket.value || socket.defaultValue;
      const preview = await GetFileListPreview(pattern, false);
      setFilePreview(prev => ({ ...prev, [socket.id]: preview }));
    } catch (error) {
      console.error('Error loading file preview:', error);
      setFilePreview(prev => ({ ...prev, [socket.id]: `Error: ${error}` }));
    } finally {
      setLoadingPreview(prev => ({ ...prev, [socket.id]: false }));
    }
  };

  return (
    <div className="properties-panel">
      <div className="panel-header">
        <h2>Properties</h2>
      </div>

      <div className="panel-content">
        {/* Node Name */}
        <div className="property-section">
          <h3>Node: {selectedNode.data.name}</h3>
          <p className="category-badge" style={{ backgroundColor: selectedNode.data.color }}>
            {selectedNode.data.category}
          </p>
        </div>

        {/* Input Sockets */}
        {selectedNode.data.inputSockets && selectedNode.data.inputSockets.length > 0 && (
          <div className="property-section">
            <h4>Input Sockets</h4>
            {selectedNode.data.inputSockets.map((socket) => (
              <div key={socket.id} className="socket-editor">
                <label className="socket-label">
                  <span className="socket-flag">{socket.argumentFlag}</span>
                  {socket.isRequired && <span className="required-badge">*</span>}
                </label>
                <div className="socket-input-container">
                  <input
                    type="text"
                    className="socket-input"
                      value={socketValues[socket.id] || ''}
                      onChange={(e) => handleSocketInputChange(socket.id, e.target.value)}
                      onBlur={() => handleSocketInputBlur(socket.id)}
                      placeholder={socket.defaultValue}
                      disabled={!!socket.connectedTo}
                  />
                  {(socket.type === 'glob_pattern' || socket.type === 'file_list') && !socket.connectedTo && (
                    <button
                      className="preview-button"
                      onClick={() => handlePreviewFiles(socket)}
                      disabled={loadingPreview[socket.id]}
                      title="Preview matching files"
                    >
                      {loadingPreview[socket.id] ? '⏳' : '🔍'}
                    </button>
                  )}
                </div>
                {socket.connectedTo && (
                  <span className="connected-badge">⚡ Connected</span>
                )}
                {filePreview[socket.id] && (
                  <pre className="file-preview">{filePreview[socket.id]}</pre>
                )}
                {socket.description && (
                  <p className="socket-description">{socket.description}</p>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Output Sockets */}
        {selectedNode.data.outputSockets && selectedNode.data.outputSockets.length > 0 && (
          <div className="property-section">
            <h4>Output Sockets</h4>
            {selectedNode.data.outputSockets.map((socket) => {
              // Resolve placeholders for display
              const displayValue = resolveOutputPlaceholders(socket.value || socket.defaultValue || '');
              
              return (
                <div key={socket.id} className="socket-editor">
                  <label className="socket-label">
                    <span className="socket-flag">{socket.argumentFlag}</span>
                  </label>
                  <div className="socket-input-container">
                    <input
                      type="text"
                      className="socket-input"
                      value={displayValue}
                      readOnly
                      placeholder={socket.defaultValue}
                      style={{ cursor: 'default', opacity: 0.8 }}
                      title={`Resolved pattern: ${displayValue}`}
                    />
                    {(socket.type === 'file_list' || socket.type === 'glob_pattern') && !socket.connectedTo && (
                      <button
                        className="preview-button"
                        onClick={() => handlePreviewFiles(socket)}
                        disabled={loadingPreview[socket.id]}
                        title="Preview file list"
                      >
                        {loadingPreview[socket.id] ? '⏳' : '🔍'}
                      </button>
                    )}
                  </div>
                  {socket.connectedTo && (
                    <span className="connected-badge">⚡ Connected</span>
                  )}
                  {filePreview[socket.id] && (
                    <pre className="file-preview">{filePreview[socket.id]}</pre>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* Environment Info */}
        <div className="property-section">
          <h4>Execution Configuration</h4>
          <div className="info-item">
            <span className="info-label">Environment:</span>
            <span className="info-value">{selectedNode.data.environment}</span>
          </div>
          <div className="info-item">
            <span className="info-label">Script:</span>
            <span className="info-value">{selectedNode.data.script}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PropertiesPanel;
