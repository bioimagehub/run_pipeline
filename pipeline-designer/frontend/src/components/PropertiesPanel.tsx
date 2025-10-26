import React, { useState } from 'react';
import { usePipelineStore } from '../stores/pipelineStore';
import { GetFileListPreview } from '../../wailsjs/go/main/App';

const PropertiesPanel: React.FC = () => {
  const { selectedNode, updateNodeSocket } = usePipelineStore();
  const [filePreview, setFilePreview] = useState<{ [key: string]: string }>({});
  const [loadingPreview, setLoadingPreview] = useState<{ [key: string]: boolean }>({});

  if (!selectedNode) {
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

  const handleSocketValueChange = (socketId: string, newValue: string) => {
    updateNodeSocket(selectedNode.id, socketId, newValue);
    // Clear preview when value changes
    setFilePreview(prev => ({ ...prev, [socketId]: '' }));
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
          <h3>Node: {selectedNode.name}</h3>
          <p className="category-badge" style={{ backgroundColor: selectedNode.color }}>
            {selectedNode.category}
          </p>
        </div>

        {/* Input Sockets */}
        {selectedNode.inputSockets && selectedNode.inputSockets.length > 0 && (
          <div className="property-section">
            <h4>Input Sockets</h4>
            {selectedNode.inputSockets.map((socket) => (
              <div key={socket.id} className="socket-editor">
                <label className="socket-label">
                  <span className="socket-flag">{socket.argumentFlag}</span>
                  {socket.isRequired && <span className="required-badge">*</span>}
                </label>
                <div className="socket-input-container">
                  <input
                    type="text"
                    className="socket-input"
                    value={socket.value || ''}
                    onChange={(e) => handleSocketValueChange(socket.id, e.target.value)}
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
        {selectedNode.outputSockets && selectedNode.outputSockets.length > 0 && (
          <div className="property-section">
            <h4>Output Sockets</h4>
            {selectedNode.outputSockets.map((socket) => (
              <div key={socket.id} className="socket-editor">
                <label className="socket-label">
                  <span className="socket-flag">{socket.argumentFlag}</span>
                </label>
                <div className="socket-input-container">
                  <input
                    type="text"
                    className="socket-input"
                    value={socket.value || ''}
                    onChange={(e) => handleSocketValueChange(socket.id, e.target.value)}
                    placeholder={socket.defaultValue}
                    disabled={!!socket.connectedTo}
                  />
                  {socket.type === 'file_list' && !socket.connectedTo && (
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
            ))}
          </div>
        )}

        {/* Environment Info */}
        <div className="property-section">
          <h4>Execution Configuration</h4>
          <div className="info-item">
            <span className="info-label">Environment:</span>
            <span className="info-value">{selectedNode.environment}</span>
          </div>
          <div className="info-item">
            <span className="info-label">Script:</span>
            <span className="info-value">{selectedNode.script}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PropertiesPanel;
