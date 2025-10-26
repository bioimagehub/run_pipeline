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

// Define custom node types
const nodeTypes = {
  cliNode: CLINode,
};

const Canvas: React.FC = () => {
  const { nodes: storeNodes, edges: storeEdges, setNodes, setEdges, setSelectedNode } = usePipelineStore();
  const [nodes, setLocalNodes, onNodesChange] = useNodesState(storeNodes);
  const [edges, setLocalEdges, onEdgesChange] = useEdgesState(storeEdges);

  // Sync store changes to local state (only when store changes externally)
  React.useEffect(() => {
    setLocalNodes(storeNodes);
  }, [storeNodes.length]); // Only sync when nodes are added/removed

  React.useEffect(() => {
    setLocalEdges(storeEdges);
  }, [storeEdges.length]); // Only sync when edges are added/removed

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
          setNodes(nds);
          return nds;
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
        return newEdges;
      });
    },
    [setLocalEdges, setEdges]
  );

  // Handle node selection
  const onNodeClick = useCallback(
    (_event: React.MouseEvent, node: any) => {
      console.log('Node clicked:', node);
      setSelectedNode(node.data);
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
      >
        <Controls />
        <MiniMap />
        <Background color="#555" gap={16} />
      </ReactFlow>
    </div>
  );
};

export default Canvas;
