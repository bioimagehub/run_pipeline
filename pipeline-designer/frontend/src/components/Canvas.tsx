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
    selectedNode,
    deleteSelectedNode,
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

  // Handle keyboard shortcuts (Delete key)
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Only handle Delete key when:
      // 1. A node is selected
      // 2. User is NOT typing in an input field
      if (e.key === 'Delete' && selectedNode) {
        const target = e.target as HTMLElement;
        const isTyping = target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable;
        
        if (!isTyping) {
          e.preventDefault();
          deleteSelectedNode();
          LogFrontend(`[Canvas] Node deleted via Delete key: ${selectedNode.id}`).catch(console.error);
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedNode, deleteSelectedNode]);

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

  // Helper function to resolve placeholders in output socket values for connections
  const resolveOutputValue = (outputSocket: any, sourceNode: any): string => {
    if (!outputSocket || !sourceNode) return '';
    
    // If socket has a custom value set by user, use that
    if (outputSocket.value) {
      return outputSocket.value;
    }
    
    // Otherwise resolve the default value placeholders
    let resolved = outputSocket.defaultValue || '';
    
    if (!resolved || !sourceNode.data || !sourceNode.data.inputSockets) {
      return resolved;
    }
    
    // Resolve placeholders using source node's input socket values
    sourceNode.data.inputSockets.forEach((inputSocket: any) => {
      const flagName = inputSocket.argumentFlag.replace(/^--/, '').replace(/-/g, '_');
      const inputValue = inputSocket.value || inputSocket.defaultValue || '';
      
      // Handle transformations like <input_search_pattern:dirname>
      const transformRegex = new RegExp(`<${flagName}:([^>]+)>`, 'g');
      resolved = resolved.replace(transformRegex, (match: string, transform: string) => {
        return applyTransform(inputValue, transform);
      });
      
      // Simple placeholder replacement
      const placeholder = `<${flagName}>`;
      resolved = resolved.split(placeholder).join(inputValue);
    });
    
    return resolved;
  };

  // Helper function to apply transformations
  const applyTransform = (value: string, transform: string): string => {
    if (!value) return '';
    
    switch (transform) {
      case 'dirname': {
        let dir = value;
        const globIndex = dir.search(/[\*\?\[]/);
        if (globIndex !== -1) {
          dir = dir.substring(0, globIndex);
          dir = dir.replace(/\/*$/, '');
        }
        return dir;
      }
      case 'basename': {
        const parts = value.split('/');
        const filename = parts[parts.length - 1];
        return filename.replace(/\.[^.]*$/, '');
      }
      default:
        return value;
    }
  };

  // Helper function to check if socket types are compatible
  const areTypesCompatible = (sourceType: string, targetType: string): boolean => {
    // Exact match is always valid
    if (sourceType === targetType) return true;
    
    // Allow some flexibility: path can connect to string or glob_pattern
    if (sourceType === 'path' && (targetType === 'string' || targetType === 'glob_pattern')) return true;
    if (targetType === 'path' && (sourceType === 'string' || sourceType === 'glob_pattern')) return true;
    
    // Allow glob_pattern to connect to string (but warn visually)
    if (sourceType === 'glob_pattern' && targetType === 'string') return true;
    if (targetType === 'glob_pattern' && sourceType === 'string') return true;
    
    // Otherwise, types don't match
    return false;
  };

  // Handle connection between nodes
  const onConnect = useCallback(
    (params: Connection) => {
      setLocalEdges((eds) => {
        // Check type compatibility
        const sourceNode = nodes.find(n => n.id === params.source);
        const targetNode = nodes.find(n => n.id === params.target);
        
        let isValidConnection = true;
        let warningMessage = '';
        
        if (sourceNode && targetNode && params.sourceHandle && params.targetHandle) {
          const outputSocket = sourceNode.data.outputSockets?.find(s => s.id === params.sourceHandle);
          const inputSocket = targetNode.data.inputSockets?.find(s => s.id === params.targetHandle);
          
          if (outputSocket && inputSocket) {
            const typesMatch = areTypesCompatible(outputSocket.type, inputSocket.type);
            
            if (!typesMatch) {
              warningMessage = `Type mismatch: ${outputSocket.type} → ${inputSocket.type}`;
              console.warn(warningMessage, {
                source: { node: sourceNode.data.name, socket: outputSocket.argumentFlag, type: outputSocket.type },
                target: { node: targetNode.data.name, socket: inputSocket.argumentFlag, type: inputSocket.type }
              });
              // Don't prevent connection, just mark it as invalid
              isValidConnection = false;
            }
          }
        }
        
        // Create edge with validation metadata
        const newEdge = {
          ...params,
          id: `${params.source}-${params.sourceHandle}-${params.target}-${params.targetHandle}`,
          animated: true,
          style: isValidConnection ? { stroke: '#007acc' } : { stroke: '#f48771', strokeWidth: 2 },
          data: { isValidConnection, warningMessage }
        };
        
        const newEdges = [...eds, newEdge];
        setEdges(newEdges);
        
        // Sync value from output socket to input socket
        if (sourceNode && params.sourceHandle) {
          const outputSocket = sourceNode.data.outputSockets?.find(s => s.id === params.sourceHandle);
          if (outputSocket && params.target && params.targetHandle) {
            // Get the actual value from the output socket (not just default, but resolved value)
            const outputValue = outputSocket.value || resolveOutputValue(outputSocket, sourceNode);
            
            console.log('[Canvas] Copying value on connection:', {
              from: { node: sourceNode.data.name, socket: outputSocket.argumentFlag, value: outputValue },
              to: { node: params.target, socket: params.targetHandle }
            });
            
            // Update the target socket with the output value immediately
            setTimeout(() => {
              updateNodeSocket(params.target!, params.targetHandle!, outputValue);
              LogFrontend(`[Canvas] Connected and copied value: "${outputValue}" from ${sourceNode.data.name} to ${targetNode?.data.name}`).catch(console.error);
            }, 0);
          }
        }
        
        // Show warning if types don't match
        if (!isValidConnection && warningMessage) {
          setTimeout(() => {
            LogFrontend(`[Canvas] Connection warning: ${warningMessage}`).catch(console.error);
          }, 0);
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
