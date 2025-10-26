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
      </div>

      {/* Node Body */}
      {!data.isCollapsed && (
        <div className="node-body">
          {/* Input Sockets */}
          {data.inputSockets && data.inputSockets.length > 0 && (
            <div className="sockets-section">
              {data.inputSockets.map((socket, index) => (
                <div key={socket.id} className="socket-row input">
                  <Handle
                    type="target"
                    position={Position.Left}
                    id={socket.id}
                    style={{
                      top: `${30 + index * 40}px`,
                      background: socket.connectedTo ? '#4CAF50' : '#999',
                    }}
                  />
                  <div className="socket-content">
                    <span className="socket-label">
                      {socket.argumentFlag}
                      {socket.isRequired && <span className="required">*</span>}
                    </span>
                    {!socket.connectedTo && (
                      <input
                        type="text"
                        className="socket-value"
                        value={socket.value || ''}
                        placeholder={`Enter ${socket.type}`}
                        readOnly
                      />
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Output Sockets */}
          {data.outputSockets && data.outputSockets.length > 0 && (
            <div className="sockets-section">
              {data.outputSockets.map((socket, index) => (
                <div key={socket.id} className="socket-row output">
                  <div className="socket-content right">
                    <span className="socket-label">{socket.argumentFlag}</span>
                    {!socket.connectedTo && (
                      <input
                        type="text"
                        className="socket-value"
                        value={socket.value || ''}
                        placeholder={`Enter ${socket.type}`}
                        readOnly
                      />
                    )}
                  </div>
                  <Handle
                    type="source"
                    position={Position.Right}
                    id={socket.id}
                    style={{
                      top: `${30 + (data.inputSockets?.length || 0) * 40 + index * 40}px`,
                      background: socket.connectedTo ? '#4CAF50' : '#999',
                    }}
                  />
                </div>
              ))}
            </div>
          )}
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
