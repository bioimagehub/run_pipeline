import React from 'react';
import { usePipelineStore } from '../stores/pipelineStore';
import '../styles/MenuBar.css';

const MenuBar: React.FC = () => {
  const { saveCurrentPipeline, saveAsPipeline, openPipeline, currentFilePath, hasUnsavedChanges } = usePipelineStore();
  const [fileMenuOpen, setFileMenuOpen] = React.useState(false);

  const handleSave = async () => {
    try {
      await saveCurrentPipeline();
      setFileMenuOpen(false);
    } catch (error) {
      console.error('Save failed:', error);
      alert('Failed to save pipeline');
    }
  };

  const handleSaveAs = async () => {
    try {
      await saveAsPipeline();
      setFileMenuOpen(false);
    } catch (error) {
      console.error('Save As failed:', error);
      alert('Failed to save pipeline');
    }
  };

  const handleOpen = async () => {
    try {
      await openPipeline();
      setFileMenuOpen(false);
    } catch (error) {
      console.error('Open failed:', error);
      alert('Failed to open pipeline');
    }
  };

  const getFileName = () => {
    if (!currentFilePath) return 'Untitled';
    const parts = currentFilePath.split('\\').pop()?.split('/').pop();
    return parts || 'Untitled';
  };

  return (
    <div className="menu-bar">
      <div className="menu-item" onClick={() => setFileMenuOpen(!fileMenuOpen)}>
        File
        {fileMenuOpen && (
          <div className="menu-dropdown">
            <div className="menu-option" onClick={handleOpen}>
              Open... <span className="shortcut">Ctrl+O</span>
            </div>
            <div className="menu-separator"></div>
            <div className="menu-option" onClick={handleSave}>
              Save <span className="shortcut">Ctrl+S</span>
            </div>
            <div className="menu-option" onClick={handleSaveAs}>
              Save As... <span className="shortcut">Ctrl+Shift+S</span>
            </div>
          </div>
        )}
      </div>
      
      <div className="menu-title">
        {getFileName()}{hasUnsavedChanges && ' *'}
      </div>
    </div>
  );
};

export default MenuBar;
