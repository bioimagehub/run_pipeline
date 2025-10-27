import React, { useState } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { usePipelineStore } from '../../stores/pipelineStore';

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

const CLINode: React.FC<NodeProps<CLINodeData>> = ({ data, selected, id }) => {
  const [showAllArgs, setShowAllArgs] = useState(false);
  const { updateNodeSocket } = usePipelineStore();
  
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
        <div className="node-body" style={{ padding: '12px' }}>
          {/* Input Parameters */}
          {argsToShow && argsToShow.length > 0 && (
            <div style={{ marginBottom: '8px' }}>
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
                    marginBottom: '2px' 
                  }}>
                    {socket.argumentFlag}
                    {socket.isRequired && <span style={{ color: '#f48771' }}>*</span>}
                  </label>
                  <input
                    type="text"
                    value={socket.value || ''}
                    onChange={(e) => {
                      e.stopPropagation();
                      updateNodeSocket(id, socket.id, e.target.value);
                    }}
                    onClick={(e) => e.stopPropagation()}
                    placeholder={`Enter ${socket.argumentFlag}...`}
                    style={{
                      width: '100%',
                      padding: '4px 6px',
                      fontSize: '11px',
                      background: '#2a2a2a',
                      border: '1px solid #444',
                      borderRadius: '3px',
                      color: '#ddd',
                    }}
                  />
                </div>
              ))}
            </div>
          )}
          
          {/* Expand/Collapse Button */}
          {shouldCollapse && hasOptionalArgs && (
            <button
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
              }}
            >
              {showAllArgs ? (
                <>
                  <ChevronDown size={12} />
                  Hide Optional Args ({optionalArgs.length})
                </>
              ) : (
                <>
                  <ChevronRight size={12} />
                  Show All Args ({optionalArgs.length} optional)
                </>
              )}
            </button>
          )}
          
          {/* Output Sockets */}
          {data.outputSockets && data.outputSockets.length > 0 && (
            <div style={{ marginTop: '12px', paddingTop: '8px', borderTop: '1px solid #444' }}>
              <div style={{ fontSize: '10px', color: '#888', marginBottom: '4px', textTransform: 'uppercase' }}>
                Outputs
              </div>
              {data.outputSockets.map((socket) => (
                <div key={socket.id} style={{ position: 'relative', marginBottom: '4px' }}>
                  <Handle
                    type="source"
                    position={Position.Right}
                    id={socket.id}
                    style={{
                      position: 'absolute',
                      top: '50%',
                      transform: 'translateY(-50%)',
                      right: '-20px',
                      background: socket.connectedTo ? '#4CAF50' : '#999',
                    }}
                  />
                  <div style={{ fontSize: '10px', color: '#aaa', textAlign: 'right' }}>
                    {socket.argumentFlag}
                  </div>
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
