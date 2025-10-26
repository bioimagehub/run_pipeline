import React, { useEffect } from 'react';
import Canvas from './components/Canvas';
import CommandExplorer from './components/CommandExplorer';
import PropertiesPanel from './components/PropertiesPanel';
import { usePipelineStore } from './stores/pipelineStore';
import './styles/globals.css';

function App() {
  const { loadDefinitions } = usePipelineStore();

  useEffect(() => {
    // Load CLI definitions on app start
    loadDefinitions();
  }, []);

  return (
    <div className="app-container">
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
  );
}

export default App;
