
import React from 'react';

const LoadingSpinner: React.FC = () => {
  return (
    <div className="flex flex-col items-center justify-center space-y-2">
        <div 
          className="w-12 h-12 rounded-full animate-spin border-4 border-solid border-cyan-400 border-t-transparent"
          role="status"
          aria-live="polite"
        >
           <span className="sr-only">Loading...</span>
        </div>
        <p className="text-cyan-400">Loading Map Data...</p>
    </div>
  );
};

export default LoadingSpinner;
