import React, { useCallback } from 'react';
import ReactFlow, {
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  NodeChange,
  EdgeChange,
  applyNodeChanges,
  applyEdgeChanges,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { usePipelineStore } from '../stores/pipelineStore';
import CLINode from './nodes/CLINode';
import { LogFrontend } from '../../wailsjs/go/main/App';

// Define custom node types
const nodeTypes = {
  cliNode: CLINode,
};

const Canvas: React.FC = () => {
  const { 
    nodes: storeNodes, 
    edges: storeEdges, 
    setNodes, 
    setEdges, 
    setSelectedNode, 
    updateNodeSocket
  } = usePipelineStore();
  const [nodes, setLocalNodes, onNodesChange] = useNodesState(storeNodes);
  const [edges, setLocalEdges, onEdgesChange] = useEdgesState(storeEdges);
  const [showMiniMap, setShowMiniMap] = React.useState(false); // Minimap off by default

  // Sync store changes to local state
  React.useEffect(() => {
    setLocalNodes(storeNodes);
  }, [storeNodes]); // Sync whenever store nodes change (not just length)

  React.useEffect(() => {
    setLocalEdges(storeEdges);
  }, [storeEdges]); // Sync whenever store edges change (not just length)

  // Handle node changes (dragging, selecting, etc.)
  const handleNodesChange = useCallback(
    (changes: NodeChange[]) => {
      // Apply changes immediately to local state for smooth interaction
      onNodesChange(changes);
      
      // Only update store for certain change types
      const shouldUpdateStore = changes.some(
        change => change.type === 'position' && !(change as any).dragging
      );
      
      if (shouldUpdateStore) {
        // Update store after drag ends
        setLocalNodes((nds) => {
          // Check for overlaps and adjust positions if needed
          const adjustedNodes = nds.map((node, index) => {
            const overlapping = nds.find((other, otherIndex) => {
              if (index === otherIndex) return false;
              
              const nodeWidth = 300;
              const nodeHeight = 150;
              const padding = 20;
              
              // Check if nodes overlap
              return (
                Math.abs(node.position.x - other.position.x) < nodeWidth + padding &&
                Math.abs(node.position.y - other.position.y) < nodeHeight + padding
              );
            });
            
            // If overlapping, shift the node to the right
            if (overlapping) {
              return {
                ...node,
                position: {
                  x: overlapping.position.x + 350,
                  y: node.position.y
                }
              };
            }
            
            return node;
          });
          
          setNodes(adjustedNodes);
          return adjustedNodes;
        });
      }
    },
    [onNodesChange, setNodes, setLocalNodes]
  );

  // Handle edge changes
  const handleEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      onEdgesChange(changes);
      // Update store immediately for edge changes
      setLocalEdges((eds) => {
        setEdges(eds);
        return eds;
      });
    },
    [onEdgesChange, setEdges, setLocalEdges]
  );

  // Handle connection between nodes
  const onConnect = useCallback(
    (params: Connection) => {
      setLocalEdges((eds) => {
        const newEdges = addEdge(params, eds);
        setEdges(newEdges);
        
        // Sync value from output socket to input socket
        const sourceNode = nodes.find(n => n.id === params.source);
        if (sourceNode && params.sourceHandle) {
          const outputSocket = sourceNode.data.outputSockets?.find(s => s.id === params.sourceHandle);
          if (outputSocket && outputSocket.value && params.target && params.targetHandle) {
            // Update the target socket with the source socket's value
            setTimeout(() => {
              updateNodeSocket(params.target!, params.targetHandle!, outputSocket.value);
            }, 0);
          }
        }
        
        return newEdges;
      });
    },
    [setLocalEdges, setEdges, nodes, updateNodeSocket]
  );

  // Handle node selection
  const onNodeClick = useCallback(
    (_event: React.MouseEvent, node: any) => {
      const logMessage = `[Canvas] Node clicked: id=${node.id}, type=${node.type}, hasData=${!!node.data}, dataKeys=${node.data ? Object.keys(node.data).join(',') : 'none'}`;
      console.log(logMessage, { fullNode: node });
      LogFrontend(logMessage).catch(err => console.error('Failed to log:', err));
      setSelectedNode(node);
      console.log('[Canvas] setSelectedNode called with node ID:', node.id);
      LogFrontend(`[Canvas] setSelectedNode called with node ID: ${node.id}`).catch(err => console.error('Failed to log:', err));
    },
    [setSelectedNode]
  );

  // Handle canvas click (deselect)
  const onPaneClick = useCallback(() => {
    setSelectedNode(null);
  }, [setSelectedNode]);

  return (
    <div className="canvas-container">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={handleNodesChange}
        onEdgesChange={handleEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        nodeTypes={nodeTypes}
        fitView
        className="react-flow-dark"
        defaultEdgeOptions={{
          type: 'smoothstep',
          animated: true,
        }}
        minZoom={0.2}
        maxZoom={4}
        snapToGrid={true}
        snapGrid={[15, 15]}
        nodesDraggable={true}
        nodesConnectable={true}
        elementsSelectable={true}
      >
        <Controls 
          showInteractive={true}
          position="top-left"
        />
        {showMiniMap && <MiniMap />}
        <Background color="#555" gap={16} />
        
        {/* Minimap Toggle Button */}
        <div style={{
          position: 'absolute',
          bottom: '10px',
          right: '10px',
          zIndex: 5,
        }}>
          <button
            onClick={() => setShowMiniMap(!showMiniMap)}
            style={{
              padding: '8px 12px',
              backgroundColor: '#2d2d30',
              border: '1px solid #3e3e42',
              borderRadius: '4px',
              color: '#cccccc',
              cursor: 'pointer',
              fontSize: '12px',
            }}
            title={showMiniMap ? 'Hide Minimap' : 'Show Minimap'}
          >
            {showMiniMap ? '🗺️ Hide Map' : '🗺️ Show Map'}
          </button>
        </div>
      </ReactFlow>
    </div>
  );
};

export default Canvas;
