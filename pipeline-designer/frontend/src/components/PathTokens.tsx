import React, { useState, useEffect } from 'react';
import { GetPathTokens } from '../../wailsjs/go/main/App';
import { main } from '../../wailsjs/go/models';
import '../styles/PathTokens.css';

type PathToken = main.PathToken;

export const PathTokens: React.FC = () => {
  const [tokens, setTokens] = useState<PathToken[]>([]);
  const [copiedToken, setCopiedToken] = useState<string | null>(null);
  const [isCollapsed, setIsCollapsed] = useState(false);

  useEffect(() => {
    loadTokens();
  }, []);

  const loadTokens = async () => {
    try {
      const pathTokens = await GetPathTokens();
      setTokens(pathTokens || []);
    } catch (error) {
      console.error('Failed to load path tokens:', error);
    }
  };

  const copyToClipboard = async (token: string) => {
    try {
      await navigator.clipboard.writeText(token);
      setCopiedToken(token);
      setTimeout(() => setCopiedToken(null), 2000);
    } catch (error) {
      console.error('Failed to copy:', error);
    }
  };

  if (tokens.length === 0) {
    return null; // Don't show if no tokens available
  }

  return (
    <div className="path-tokens">
      <div className="path-tokens-header" onClick={() => setIsCollapsed(!isCollapsed)}>
        <h3>📍 Path Tokens {isCollapsed ? '▶' : '▼'}</h3>
        <span className="hint">Click to copy</span>
      </div>
      
      {!isCollapsed && (
        <>
          <div className="tokens-list">
            {tokens.map((token) => (
              <div
                key={token.token}
                className={`token-item ${copiedToken === token.token ? 'copied' : ''}`}
                onClick={() => copyToClipboard(token.token)}
                title={`Resolves to: ${token.resolvedPath}`}
              >
                <div className="token-badge">
                  {token.token}
                </div>
                <div className="token-info">
                  <div className="token-description">{token.description}</div>
                  <div className="token-path" title={token.resolvedPath}>
                    {token.resolvedPath}
                  </div>
                </div>
                {copiedToken === token.token && (
                  <span className="copied-indicator">✓</span>
                )}
              </div>
            ))}
          </div>
          
          <div className="tokens-help">
            <p><strong>Usage:</strong> %REPO%/data/input</p>
            <p><strong>Tip:</strong> Use these tokens in file paths for portability</p>
          </div>
        </>
      )}
    </div>
  );
};
