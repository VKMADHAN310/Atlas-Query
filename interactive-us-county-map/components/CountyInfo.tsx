
import React, { useEffect, useState } from 'react';
import type { County } from '../types';

interface CountyInfoProps {
  county: County | null;
  onClear: () => void;
}

const CountyInfo: React.FC<CountyInfoProps> = ({ county, onClear }) => {
  // Collapsible panel state, auto-opens when a county is selected
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    if (county) setIsOpen(true);
  }, [county]);

  return (
    <div className="absolute top-4 left-2 z-20 flex items-start">
      {/* Side arrow toggle */}
      <button
        aria-label={isOpen ? 'Close county info' : 'Open county info'}
        onClick={() => setIsOpen((v) => !v)}
        className="px-2 py-2 rounded-r bg-gray-800/90 hover:bg-gray-700 border border-cyan-500/30 text-cyan-300 shadow-lg"
      >
        {isOpen ? '‹' : '›'}
      </button>

      {isOpen && (
        <div className="ml-2 bg-gray-900/80 backdrop-blur-sm p-4 rounded-lg shadow-lg border border-cyan-500/30 w-64 text-white">
          <div className="flex justify-between items-center mb-2">
            <h2 className="text-lg font-bold text-cyan-400">County Information</h2>
            {county && (
              <button onClick={onClear} className="text-gray-400 hover:text-white text-xl" aria-label="Clear selection">&times;</button>
            )}
          </div>

          {county ? (
            <div>
              <p className="text-gray-300">
                <span className="font-semibold">FIPS Code:</span> {county.id}
              </p>
              {county.name && (
                <p className="text-gray-300 mt-1">
                  <span className="font-semibold">Name:</span> {county.name}
                </p>
              )}
              <p className="text-gray-300 mt-1">
                <span className="font-semibold">State:</span> {county.state || '—'}
              </p>
            </div>
          ) : (
            <p className="text-gray-400">Click a county on the map to see its details.</p>
          )}
        </div>
      )}
    </div>
  );
};

export default CountyInfo;
