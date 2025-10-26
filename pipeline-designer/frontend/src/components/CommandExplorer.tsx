import React, { useState } from 'react';
import { Search } from 'lucide-react';
import { usePipelineStore } from '../stores/pipelineStore';
import { CLIDefinition } from '../types';

const CommandExplorer: React.FC = () => {
  const { definitions, addNodeFromDefinition } = usePipelineStore();
  const [searchTerm, setSearchTerm] = useState('');
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set(['Segmentation', 'Image Processing', 'Input/Output'])
  );

  // Debug logging
  React.useEffect(() => {
    console.log('CommandExplorer: definitions updated', definitions);
  }, [definitions]);

  // Group definitions by category
  const definitionsByCategory = React.useMemo(() => {
    const grouped: Record<string, CLIDefinition[]> = {};
    definitions.forEach((def) => {
      if (!grouped[def.category]) {
        grouped[def.category] = [];
      }
      grouped[def.category].push(def);
    });
    return grouped;
  }, [definitions]);

  // Filter definitions by search term
  const filteredCategories = React.useMemo(() => {
    if (!searchTerm) return definitionsByCategory;

    const filtered: Record<string, CLIDefinition[]> = {};
    Object.entries(definitionsByCategory).forEach(([category, defs]) => {
      const matchingDefs = defs.filter((def) =>
        def.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        def.description.toLowerCase().includes(searchTerm.toLowerCase())
      );
      if (matchingDefs.length > 0) {
        filtered[category] = matchingDefs;
      }
    });
    return filtered;
  }, [definitionsByCategory, searchTerm]);

  const toggleCategory = (category: string) => {
    setExpandedCategories((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(category)) {
        newSet.delete(category);
      } else {
        newSet.add(category);
      }
      return newSet;
    });
  };

  const handleAddNode = (definitionId: string) => {
    // Add node at center of canvas
    addNodeFromDefinition(definitionId, 400, 300);
  };

  return (
    <div className="command-explorer">
      <div className="explorer-header">
        <h2>Commands</h2>
      </div>

      {/* Search bar */}
      <div className="search-bar">
        <Search size={16} />
        <input
          type="text"
          placeholder="Search commands..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
      </div>

      {/* Category tree */}
      <div className="category-tree">
        {Object.entries(filteredCategories).map(([category, defs]) => (
          <div key={category} className="category-group">
            <div
              className="category-header"
              onClick={() => toggleCategory(category)}
            >
              <span className="expand-icon">
                {expandedCategories.has(category) ? '▼' : '▶'}
              </span>
              <span className="category-name">{category}</span>
              <span className="category-count">({defs.length})</span>
            </div>

            {expandedCategories.has(category) && (
              <div className="category-items">
                {defs.map((def) => (
                  <div
                    key={def.id}
                    className="command-item"
                    onClick={() => handleAddNode(def.id)}
                    title={def.description}
                  >
                    <span className="command-icon">{def.icon}</span>
                    <span className="command-name">{def.name}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default CommandExplorer;
