import React from 'react';
import { Folder, File } from 'lucide-react';

const FileBrowser: React.FC = () => {
  return (
    <div className="file-browser">
      <div className="panel-header">
        <h2>📁 File Browser</h2>
      </div>
      <div className="panel-content">
        <div style={{ padding: '1rem', color: '#888', textAlign: 'center' }}>
          <Folder size={48} style={{ margin: '0 auto 1rem', opacity: 0.3 }} />
          <p>File browser coming soon...</p>
          <p style={{ fontSize: '0.85rem', marginTop: '0.5rem' }}>
            This will allow you to browse and select input files for your pipeline.
          </p>
        </div>
      </div>
    </div>
  );
};

export default FileBrowser;
