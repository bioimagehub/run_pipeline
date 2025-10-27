import React, { useEffect, useState } from 'react';
import MenuBar from './components/MenuBar';
import Canvas from './components/Canvas';
import CommandExplorer from './components/CommandExplorer';
import FileBrowser from './components/FileBrowser';
import { usePipelineStore } from './stores/pipelineStore';
import { GetStartupFilePath } from '../wailsjs/go/main/App';
import './styles/globals.css';

function App() {
  const { loadDefinitions, addNodeFromDefinition, nodes, saveCurrentPipeline, saveAsPipeline, openPipeline, currentFilePath, promptForSaveLocation, loadPipeline, setCurrentFilePath, deleteSelectedNode, runNode } = usePipelineStore();
  const [isInitialized, setIsInitialized] = useState(false);

  useEffect(() => {
    // Load CLI definitions on app start
    loadDefinitions();
    
    // Check if we need to prompt for save location
    // This happens when no file path was provided at startup
    const initializeApp = async () => {
      try {
        // Get file path from command line arguments
        const filePathFromArgs = await GetStartupFilePath();
        
        if (filePathFromArgs && filePathFromArgs.trim() !== '') {
          // Load the file provided via command line
          await loadPipeline(filePathFromArgs);
          setCurrentFilePath(filePathFromArgs);
        } else {
          // No file was provided, prompt user for save location
          const success = await promptForSaveLocation();
          if (!success) {
            // User cancelled, show message
            console.log('User cancelled save dialog - pipeline will not be saved automatically');
          }
        }
      } catch (error) {
        console.error('Error during initialization:', error);
      } finally {
        setIsInitialized(true);
      }
    };
    
    initializeApp();
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey && !e.shiftKey && e.key === 's') {
        e.preventDefault();
        saveCurrentPipeline();
      } else if (e.ctrlKey && e.shiftKey && e.key === 'S') {
        e.preventDefault();
        saveAsPipeline();
      } else if (e.ctrlKey && e.key === 'o') {
        e.preventDefault();
        openPipeline();
      } else if (e.key === 'Delete') {
        // Delete selected node
        deleteSelectedNode();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [saveCurrentPipeline, saveAsPipeline, openPipeline, deleteSelectedNode]);

  // Handle custom runNode event
  useEffect(() => {
    const handleRunNode = (e: Event) => {
      const customEvent = e as CustomEvent<{ nodeId: string }>;
      if (customEvent.detail?.nodeId) {
        runNode(customEvent.detail.nodeId);
      }
    };

    window.addEventListener('runNode', handleRunNode);
    return () => window.removeEventListener('runNode', handleRunNode);
  }, [runNode]);

  return (
    <div className="app-container">
      <MenuBar />
      <div className="app-content">
        {/* Left Panel: File Browser (20%) */}
        <aside className="left-panel">
          <FileBrowser />
        </aside>

        {/* Center Panel: Canvas (60%) */}
        <main className="center-panel">
          <Canvas />
        </main>

        {/* Right Panel: Available Nodes (20%) */}
        <aside className="right-panel">
          <CommandExplorer />
        </aside>
      </div>
    </div>
  );
}

export default App
