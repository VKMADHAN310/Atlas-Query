import React, { useState } from 'react';

interface ErrorLoggerProps {
  logs: string[];
  onClear: () => void;
}

const ErrorLogger: React.FC<ErrorLoggerProps> = ({ logs, onClear }) => {
  const [isOpen, setIsOpen] = useState(false);

  if (logs.length === 0) {
    return null; // Don't render anything if there are no logs
  }

  return (
    <div className="bg-gray-900 border-t-2 border-red-500/50">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex justify-between items-center p-2 text-left text-red-300 hover:bg-red-500/10 transition-colors"
        aria-expanded={isOpen}
        aria-controls="error-log-panel"
      >
        <div className="flex items-center space-x-2">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
          </svg>
          <span className="font-bold">Error Log</span>
          <span className="bg-red-500 text-white text-xs font-bold rounded-full px-2 py-0.5" aria-label={`${logs.length} errors`}>{logs.length}</span>
        </div>
        <div className="flex items-center space-x-4">
          <button
            onClick={(e) => {
              e.stopPropagation();
              onClear();
            }}
            className="text-gray-400 hover:text-white text-sm"
            aria-label="Clear logs"
          >
            Clear
          </button>
          <svg xmlns="http://www.w3.org/2000/svg" className={`h-5 w-5 transform transition-transform ${isOpen ? 'rotate-180' : ''}`} viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
          </svg>
        </div>
      </button>
      {isOpen && (
        <div id="error-log-panel" className="p-4 bg-gray-800/50 max-h-48 overflow-y-auto">
          <ul className="space-y-2 text-sm">
            {logs.map((log, index) => (
              <li key={index} className="font-mono text-red-300 break-words">
                {log}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default ErrorLogger;