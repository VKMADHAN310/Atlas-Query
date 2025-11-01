
import React from 'react';
import type { County } from '../types';

interface CountyInfoProps {
  county: County | null;
  onClear: () => void;
}

const CountyInfo: React.FC<CountyInfoProps> = ({ county, onClear }) => {
  return (
    <div className="absolute top-4 left-4 bg-gray-900/80 backdrop-blur-sm p-4 rounded-lg shadow-lg border border-cyan-500/30 w-64 text-white transition-opacity duration-300 z-20">
      <div className="flex justify-between items-center mb-2">
         <h2 className="text-lg font-bold text-cyan-400">County Information</h2>
         {county && (
          <button onClick={onClear} className="text-gray-400 hover:text-white text-xl">&times;</button>
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
          {county.state && (
            <p className="text-gray-300 mt-1">
              <span className="font-semibold">State:</span> {county.state}
            </p>
          )}
          <p className="mt-2 text-sm text-gray-500">
            This is where additional data from a database would be displayed (e.g., name, state, population).
          </p>
        </div>
      ) : (
        <p className="text-gray-400">Click a county on the map to see its details.</p>
      )}
    </div>
  );
};

export default CountyInfo;
