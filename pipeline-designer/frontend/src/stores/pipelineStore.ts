import { create } from 'zustand';
import { Node, Edge } from 'reactflow';
import { LoadPipeline, SavePipeline, GetCLIDefinitions, CreateNodeFromDefinition, OpenFileDialog, SaveFileDialog, CreateEmptyPipeline } from '../../wailsjs/go/main/App';
import { models } from '../types';

// Use Wails generated types
type Socket = models.main.Socket;
type CLIDefinition = models.main.CLIDefinition;

interface CLINodeData {
  id: string;
  definitionId: string;
  name: string;
  category: string;
  icon: string;
  color: string;
  inputSockets: Socket[];
  outputSockets: Socket[];
  environment: string;
  script: string;
  isCollapsed?: boolean;
}

interface PipelineState {
  // Nodes and edges
  nodes: Node<CLINodeData>[];
  edges: Edge[];
  selectedNode: CLINodeData | null;
  definitions: CLIDefinition[];
  currentFilePath: string | null;
  hasUnsavedChanges: boolean;
  
  // Actions
  setNodes: (nodes: Node<CLINodeData>[]) => void;
  setEdges: (edges: Edge[]) => void;
  setSelectedNode: (node: CLINodeData | null) => void;
  loadDefinitions: () => Promise<void>;
  addNodeFromDefinition: (definitionId: string, x: number, y: number) => Promise<void>;
  updateNodeSocket: (nodeId: string, socketId: string, value: string) => void;
  loadPipeline: (filePath: string) => Promise<void>;
  savePipeline: (filePath: string) => Promise<void>;
  saveCurrentPipeline: () => Promise<void>;
  saveAsPipeline: () => Promise<void>;
  openPipeline: () => Promise<void>;
  promptForSaveLocation: () => Promise<boolean>;
  setCurrentFilePath: (path: string | null) => void;
  markAsModified: () => void;
}

export const usePipelineStore = create<PipelineState>((set, get) => ({
  nodes: [],
  edges: [],
  selectedNode: null,
  definitions: [],
  currentFilePath: null,
  hasUnsavedChanges: false,

  setNodes: (nodes) => {
    set({ nodes, hasUnsavedChanges: true });
  },
  
  setEdges: (edges) => {
    set({ edges, hasUnsavedChanges: true });
  },
  
  setSelectedNode: (node) => set({ selectedNode: node }),

  setCurrentFilePath: (path) => set({ currentFilePath: path }),

  markAsModified: () => set({ hasUnsavedChanges: true }),

  loadDefinitions: async () => {
    try {
      console.log('Loading CLI definitions...');
      const defs = await GetCLIDefinitions();
      console.log('Loaded definitions:', defs);
      set({ definitions: defs || [] });
    } catch (error) {
      console.error('Failed to load CLI definitions:', error);
    }
  },

  addNodeFromDefinition: async (definitionId, x, y) => {
    try {
      console.log('Adding node from definition:', definitionId, 'at', x, y);
      
      // Calculate position - place to the right of the last node
      const state = get();
      let finalX = x;
      let finalY = y;
      
      if (state.nodes.length > 0) {
        // Find the rightmost node
        const rightmostNode = state.nodes.reduce((max, node) => 
          node.position.x > max.position.x ? node : max
        , state.nodes[0]);
        
        // Position new node to the right with spacing
        finalX = rightmostNode.position.x + 350; // 300 width + 50 spacing
        finalY = rightmostNode.position.y;
      }
      
      const newNode = await CreateNodeFromDefinition(definitionId, finalX, finalY);
      console.log('Created node:', newNode);
      if (newNode) {
        // Convert to React Flow node format
        const flowNode: Node<CLINodeData> = {
          id: newNode.id,
          type: 'cliNode',
          position: { x: newNode.position.x, y: newNode.position.y },
          data: {
            id: newNode.id,
            definitionId: newNode.definitionId,
            name: newNode.name,
            category: newNode.category,
            icon: '🔧', // Default icon
            color: '#569cd6', // Default color
            inputSockets: newNode.inputSockets || [],
            outputSockets: newNode.outputSockets || [],
            environment: newNode.environment,
            script: newNode.script,
          },
        };
        
        console.log('Adding flow node to store:', flowNode);
        set((state) => ({ nodes: [...state.nodes, flowNode] }));
        console.log('Node added successfully');
      }
    } catch (error) {
      console.error('Failed to add node:', error);
    }
  },

  updateNodeSocket: (nodeId, socketId, value) => {
    set((state) => {
      const updatedNodes = state.nodes.map((node) => {
        if (node.id === nodeId) {
          const updatedData = { ...node.data };
          
          // Update input sockets
          if (updatedData.inputSockets) {
            updatedData.inputSockets = updatedData.inputSockets.map((socket) =>
              socket.id === socketId ? { ...socket, value } : socket
            );
          }
          
          // Update output sockets
          if (updatedData.outputSockets) {
            updatedData.outputSockets = updatedData.outputSockets.map((socket) =>
              socket.id === socketId ? { ...socket, value } : socket
            );
          }
          
          return { ...node, data: updatedData };
        }
        return node;
      });

      // Find connections where this socket is the source (output socket)
      const connectionsFromSocket = state.edges.filter(
        edge => edge.source === nodeId && edge.sourceHandle === socketId
      );

      // Propagate value to connected input sockets
      if (connectionsFromSocket.length > 0) {
        connectionsFromSocket.forEach(connection => {
          const targetNodeIndex = updatedNodes.findIndex(n => n.id === connection.target);
          if (targetNodeIndex !== -1) {
            const targetNode = updatedNodes[targetNodeIndex];
            const updatedData = { ...targetNode.data };
            
            if (updatedData.inputSockets) {
              updatedData.inputSockets = updatedData.inputSockets.map((socket) =>
                socket.id === connection.targetHandle ? { ...socket, value, connectedTo: socketId } : socket
              );
            }
            
            updatedNodes[targetNodeIndex] = { ...targetNode, data: updatedData };
          }
        });
      }

      // Also update selectedNode if it's the one being modified or connected to
      let newSelectedNode = state.selectedNode;
      const selectedNodeUpdated = updatedNodes.find(n => n.id === state.selectedNode?.id);
      if (selectedNodeUpdated) {
        newSelectedNode = selectedNodeUpdated.data;
      }

      return { 
        nodes: updatedNodes,
        selectedNode: newSelectedNode,
        hasUnsavedChanges: true
      };
    });
  },

  loadPipeline: async (filePath) => {
    try {
      const pipeline = await LoadPipeline(filePath);
      if (pipeline) {
        // Convert pipeline nodes to React Flow nodes
        const flowNodes: Node<CLINodeData>[] = pipeline.nodes.map((node: any) => ({
          id: node.id,
          type: 'cliNode',
          position: { x: node.position.x, y: node.position.y },
          data: {
            id: node.id,
            definitionId: node.definitionID,
            name: node.name,
            category: node.category,
            icon: '🔧',
            color: '#569cd6',
            inputSockets: node.inputSockets || [],
            outputSockets: node.outputSockets || [],
            environment: node.environment,
            script: node.script,
          },
        }));
        
        // Convert pipeline connections to React Flow edges
        const flowEdges: Edge[] = (pipeline.connections || []).map((conn: any) => ({
          id: conn.id,
          source: conn.fromNodeID,
          target: conn.toNodeID,
          sourceHandle: conn.fromSocketID,
          targetHandle: conn.toSocketID,
        }));
        
        set({ nodes: flowNodes, edges: flowEdges });
      }
    } catch (error) {
      console.error('Failed to load pipeline:', error);
    }
  },

  savePipeline: async (filePath) => {
    try {
      const state = get();
      
      // Import Wails model classes
      const { main } = await import('../../wailsjs/go/models');
      
      // Convert React Flow nodes back to pipeline format
      const pipelineNodes = state.nodes.map((node) => main.CLINode.createFrom({
        id: node.id,
        definitionId: node.data.definitionId,
        name: node.data.name,
        position: { x: node.position.x, y: node.position.y },
        size: { width: 300, height: 150 },
        environment: node.data.environment,
        executable: '',
        script: node.data.script,
        inputSockets: node.data.inputSockets,
        outputSockets: node.data.outputSockets,
        isSelected: false,
        isCollapsed: false,
        category: node.data.category,
        testStatus: 'not_run',
        lastTestFile: '',
        lastTestOutput: '',
        lastTestTime: '',
        testError: '',
      }));
      
      // Convert React Flow edges back to pipeline connections
      const pipelineConnections = state.edges.map((edge) => main.SocketConnection.createFrom({
        id: edge.id,
        fromNodeId: edge.source,
        toNodeId: edge.target,
        fromSocketId: edge.sourceHandle || '',
        toSocketId: edge.targetHandle || '',
        isValid: true,
      }));
      
      const metadata = main.PipelineMetadata.createFrom({
        name: 'Pipeline',
        description: '',
        version: '1.0.0',
        author: '',
        created: new Date().toISOString(),
        modified: new Date().toISOString(),
      });
      
      const pipeline = main.Pipeline.createFrom({
        nodes: pipelineNodes,
        connections: pipelineConnections,
        metadata: metadata,
      });
      
      await SavePipeline(pipeline, filePath);
      set({ hasUnsavedChanges: false, currentFilePath: filePath });
    } catch (error) {
      console.error('Failed to save pipeline:', error);
      throw error;
    }
  },

  saveCurrentPipeline: async () => {
    const state = get();
    if (state.currentFilePath) {
      await get().savePipeline(state.currentFilePath);
    } else {
      await get().saveAsPipeline();
    }
  },

  saveAsPipeline: async () => {
    try {
      const state = get();
      const defaultName = state.currentFilePath 
        ? state.currentFilePath.split('\\').pop()?.split('/').pop() || 'pipeline.yaml'
        : 'pipeline.yaml';
      
      const filePath = await SaveFileDialog(defaultName);
      if (filePath) {
        // Ensure file has .yaml extension
        const finalPath = filePath.endsWith('.yaml') || filePath.endsWith('.yml') 
          ? filePath 
          : `${filePath}.yaml`;
        
        // Create empty file first if it doesn't exist
        await CreateEmptyPipeline(finalPath);
        
        // Now save the pipeline
        await get().savePipeline(finalPath);
      }
    } catch (error) {
      console.error('Failed to save pipeline as:', error);
      throw error;
    }
  },

  openPipeline: async () => {
    try {
      const filePath = await OpenFileDialog();
      if (filePath) {
        await get().loadPipeline(filePath);
        set({ currentFilePath: filePath, hasUnsavedChanges: false });
      }
    } catch (error) {
      console.error('Failed to open pipeline:', error);
      throw error;
    }
  },

  promptForSaveLocation: async () => {
    try {
      const filePath = await SaveFileDialog('new_pipeline.yaml');
      if (filePath) {
        // Ensure file has .yaml extension
        const finalPath = filePath.endsWith('.yaml') || filePath.endsWith('.yml') 
          ? filePath 
          : `${filePath}.yaml`;
        
        // Create empty file
        await CreateEmptyPipeline(finalPath);
        
        // Set as current file path
        set({ currentFilePath: finalPath, hasUnsavedChanges: false });
        return true;
      }
      return false;
    } catch (error) {
      console.error('Failed to prompt for save location:', error);
      return false;
    }
  },
}));
