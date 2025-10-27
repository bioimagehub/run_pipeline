import React, { useState } from 'react';
import { Search } from 'lucide-react';
import { usePipelineStore } from '../stores/pipelineStore';
import { CLIDefinition } from '../types';

interface TreeNode {
  name: string;
  fullPath: string;
  definitions: CLIDefinition[];
  children: Map<string, TreeNode>;
}

const CommandExplorer: React.FC = () => {
  const { definitions, addNodeFromDefinition } = usePipelineStore();
  const [searchTerm, setSearchTerm] = useState('');
  const [expandedPaths, setExpandedPaths] = useState<Set<string>>(
    new Set(['Segmentation', 'Image Processing', 'Input Output'])
  );

  // Debug logging
  React.useEffect(() => {
    console.log('CommandExplorer: definitions updated', definitions);
  }, [definitions]);

  // Build hierarchical tree from categories
  const categoryTree = React.useMemo(() => {
    const root: TreeNode = {
      name: '',
      fullPath: '',
      definitions: [],
      children: new Map(),
    };

    definitions.forEach((def) => {
      const parts = def.category.split(' > ').map(p => p.trim());
      let currentNode = root;
      let currentPath = '';

      parts.forEach((part, index) => {
        currentPath = currentPath ? `${currentPath} > ${part}` : part;
        
        if (!currentNode.children.has(part)) {
          currentNode.children.set(part, {
            name: part,
            fullPath: currentPath,
            definitions: [],
            children: new Map(),
          });
        }
        
        currentNode = currentNode.children.get(part)!;
        
        // Add definition to the leaf node
        if (index === parts.length - 1) {
          currentNode.definitions.push(def);
        }
      });
    });

    return root;
  }, [definitions]);

  // Filter tree by search term
  const filterTree = (node: TreeNode, searchLower: string): TreeNode | null => {
    if (!searchLower) return node;

    const matchingDefs = node.definitions.filter((def) =>
      def.name.toLowerCase().includes(searchLower) ||
      def.description.toLowerCase().includes(searchLower)
    );

    const filteredChildren = new Map<string, TreeNode>();
    node.children.forEach((child, key) => {
      const filteredChild = filterTree(child, searchLower);
      if (filteredChild && (filteredChild.definitions.length > 0 || filteredChild.children.size > 0)) {
        filteredChildren.set(key, filteredChild);
      }
    });

    if (matchingDefs.length === 0 && filteredChildren.size === 0) {
      return null;
    }

    return {
      ...node,
      definitions: matchingDefs,
      children: filteredChildren,
    };
  };

  const filteredTree = React.useMemo(() => {
    return filterTree(categoryTree, searchTerm.toLowerCase());
  }, [categoryTree, searchTerm]);

  const togglePath = (path: string) => {
    setExpandedPaths((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(path)) {
        newSet.delete(path);
      } else {
        newSet.add(path);
      }
      return newSet;
    });
  };

  const handleAddNode = (definitionId: string) => {
    // Add node at center of canvas
    addNodeFromDefinition(definitionId, 400, 300);
  };

  // Recursive tree renderer
  const renderTreeNode = (node: TreeNode, depth: number = 0): React.ReactNode => {
    const hasChildren = node.children.size > 0;
    const hasDefs = node.definitions.length > 0;
    const isExpanded = expandedPaths.has(node.fullPath);
    const totalCount = node.definitions.length;
    const hasContent = hasChildren || hasDefs; // Has either children or definitions

    if (!node.name) {
      // Root node - just render children
      return Array.from(node.children.values()).map((child) => (
        <React.Fragment key={child.fullPath}>
          {renderTreeNode(child, 0)}
        </React.Fragment>
      ));
    }

    return (
      <div key={node.fullPath} className="category-group" style={{ marginLeft: `${depth * 12}px` }}>
        <div
          className="category-header"
          onClick={() => togglePath(node.fullPath)}
        >
          <span className="expand-icon">
            {hasContent ? (isExpanded ? '▼' : '▶') : ''}
          </span>
          <span className="category-name">{node.name}</span>
          {totalCount > 0 && <span className="category-count">({totalCount})</span>}
        </div>

        {isExpanded && (
          <>
            {/* Render definitions in this category */}
            {hasDefs && (
              <div className="category-items" style={{ marginLeft: '20px' }}>
                {node.definitions.map((def) => (
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

            {/* Render child categories */}
            {hasChildren && (
              <div className="subcategories">
                {Array.from(node.children.values()).map((child) => (
                  <React.Fragment key={child.fullPath}>
                    {renderTreeNode(child, depth + 1)}
                  </React.Fragment>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    );
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
        {filteredTree && renderTreeNode(filteredTree)}
      </div>
    </div>
  );
};

export default CommandExplorer;
