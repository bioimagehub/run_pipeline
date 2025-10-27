import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';

interface Socket {
  id: string;
  argumentFlag: string;
  value: string;
  type: string;
  socketSide: 'input' | 'output';
  isRequired?: boolean;
  connectedTo?: string | null;
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

const CLINode: React.FC<NodeProps<CLINodeData>> = ({ data, selected }) => {
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

  const color = getCategoryColor(data.category);

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
          <button
            style={{
              marginLeft: 'auto',
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
        )}
      </div>

      {/* Node Body */}
      {!data.isCollapsed && (
        <div className="node-body" style={{ display: 'flex', gap: '12px' }}>
          {/* Left Column: Input Sockets */}
          <div style={{ flex: 1, minWidth: '140px' }}>
            <div style={{ 
              fontSize: '11px', 
              fontWeight: 'bold', 
              color: '#888', 
              marginBottom: '8px',
              textTransform: 'uppercase'
            }}>
              Inputs
            </div>
            {data.inputSockets && data.inputSockets.length > 0 ? (
              <div className="sockets-section">
                {data.inputSockets.map((socket, index) => (
                  <div key={socket.id} className="socket-row input" style={{ position: 'relative', marginBottom: '6px' }}>
                    <Handle
                      type="target"
                      position={Position.Left}
                      id={socket.id}
                      style={{
                        position: 'absolute',
                        top: '50%',
                        transform: 'translateY(-50%)',
                        left: '-8px',
                        background: socket.connectedTo ? '#4CAF50' : '#999',
                      }}
                    />
                    <div className="socket-content" style={{ fontSize: '11px' }}>
                      <span className="socket-label" style={{ fontSize: '11px', display: 'block' }}>
                        {socket.argumentFlag}
                        {socket.isRequired && <span className="required" style={{ color: '#f48771' }}>*</span>}
                      </span>
                    </div>
                  </div>
                ))}
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
              fontSize: '11px', 
              fontWeight: 'bold', 
              color: '#888', 
              marginBottom: '8px',
              textTransform: 'uppercase'
            }}>
              Outputs
            </div>
            {data.outputSockets && data.outputSockets.length > 0 ? (
              <div className="sockets-section">
                {data.outputSockets.map((socket, index) => (
                  <div key={socket.id} className="socket-row output" style={{ position: 'relative', marginBottom: '6px' }}>
                    <div className="socket-content right" style={{ fontSize: '11px', textAlign: 'right' }}>
                      <span className="socket-label" style={{ fontSize: '11px', display: 'block' }}>
                        {socket.argumentFlag}
                      </span>
                    </div>
                    <Handle
                      type="source"
                      position={Position.Right}
                      id={socket.id}
                      style={{
                        position: 'absolute',
                        top: '50%',
                        transform: 'translateY(-50%)',
                        right: '-8px',
                        background: socket.connectedTo ? '#4CAF50' : '#999',
                      }}
                    />
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ fontSize: '10px', color: '#666', fontStyle: 'italic', textAlign: 'right' }}>No outputs</div>
            )}
          </div>
        </div>
      )}

      {/* Node Footer */}
      <div className="node-footer">
        <span className="environment-badge">{data.environment}</span>
      </div>
    </div>
  );
};

export default CLINode;
