import React, { useEffect, useState } from 'react';
import MenuBar from './components/MenuBar';
import Canvas from './components/Canvas';
import CommandExplorer from './components/CommandExplorer';
import PropertiesPanel from './components/PropertiesPanel';
import { usePipelineStore } from './stores/pipelineStore';
import { GetStartupFilePath } from '../wailsjs/go/main/App';
import './styles/globals.css';

function App() {
  const { loadDefinitions, addNodeFromDefinition, nodes, saveCurrentPipeline, saveAsPipeline, openPipeline, currentFilePath, promptForSaveLocation, loadPipeline, setCurrentFilePath } = usePipelineStore();
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

  useEffect(() => {
    // Add File Selector node by default if no nodes exist and app is initialized
    if (isInitialized && nodes.length === 0) {
      setTimeout(() => {
        addNodeFromDefinition('file_selector', 200, 200);
      }, 500); // Wait for definitions to load
    }
  }, [nodes.length, isInitialized]);

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
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [saveCurrentPipeline, saveAsPipeline, openPipeline]);

  return (
    <div className="app-container">
      <MenuBar />
      <div className="app-content">
        {/* Left Panel: Command Explorer (20%) */}
        <aside className="left-panel">
          <CommandExplorer />
        </aside>

        {/* Center Panel: Canvas (60%) */}
        <main className="center-panel">
          <Canvas />
        </main>

        {/* Right Panel: Properties (20%) */}
        <aside className="right-panel">
          <PropertiesPanel />
        </aside>
      </div>
    </div>
  );
}

export default App
