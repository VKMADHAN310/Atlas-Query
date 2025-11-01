
import React, { useState, useEffect, useMemo, useRef } from 'react';
import type { County, CountyGeoJSON, UsAtlasData } from '../types';
import type { Feature, Geometry } from 'geojson';
import LoadingSpinner from './LoadingSpinner';

// Since D3 and TopoJSON are loaded from a CDN, we need to tell TypeScript about them.
declare const d3: any;
declare const topojson: any;

const US_ATLAS_URL = 'https://cdn.jsdelivr.net/npm/us-atlas@3/counties-10m.json';

interface USMapProps {
  onCountyClick: (county: County) => void;
  selectedCountyId: string | null;
  onError: (message: string) => void;
  overlayFeatures?: Feature<Geometry, any>[]; // Backend GeoJSON features to highlight
}

const USMap: React.FC<USMapProps> = ({ onCountyClick, selectedCountyId, onError, overlayFeatures = [] }) => {
  const [geoData, setGeoData] = useState<{ counties: CountyGeoJSON[], states: any } | null>(null);
  const [hoveredCountyId, setHoveredCountyId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const svgRef = useRef<SVGSVGElement>(null);
  const zoomPaneRef = useRef<SVGRectElement>(null);
  const rootGroupRef = useRef<SVGGElement>(null);
  const zoomRef = useRef<any>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true);
        const response = await fetch(US_ATLAS_URL);
        if (!response.ok) {
          throw new Error(`Failed to fetch map data: ${response.statusText} (Status: ${response.status})`);
        }
        const usAtlasData: UsAtlasData = await response.json();
        
        const counties = topojson.feature(usAtlasData, usAtlasData.objects.counties).features as CountyGeoJSON[];
        const states = topojson.mesh(usAtlasData, usAtlasData.objects.states, (a: any, b: any) => a !== b);
        
        setGeoData({ counties, states });
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred';
        setError(errorMessage);
        onError(`USMap: ${errorMessage}`);
      } finally {
        setIsLoading(false);
      }
    };
    fetchData();
  }, [onError]);

  const projection = useMemo(() => d3.geoAlbersUsa().scale(1300).translate([487.5, 305]), []);
  const pathGenerator = useMemo(() => d3.geoPath().projection(projection), [projection]);

  // Setup zoom/pan behavior once
  useEffect(() => {
    if (!svgRef.current || !rootGroupRef.current || !geoData) return;
    const svg = d3.select(svgRef.current);
    const g = d3.select(rootGroupRef.current);
    
    const zoom = d3.zoom()
      .scaleExtent([0.5, 20]) // Increased max zoom from 8 to 20 for better detail
      .translateExtent([[-200, -200], [1175, 810]]) // Expanded bounds for panning
      .filter((event: any) => {
        // Allow all interactions except right-click context menu
        return event.type !== 'contextmenu' && event.button !== 2;
      })
      .on('start', () => {
        svg.style('cursor', 'grabbing');
      })
      .on('zoom', (event: any) => {
        g.attr('transform', event.transform);
      })
      .on('end', () => {
        svg.style('cursor', 'grab');
      });
    
    zoomRef.current = zoom;
    svg.call(zoom as any);

    return () => {
      try { svg.on('.zoom', null); } catch {}
    };
  }, [geoData]); // Re-run when geoData is loaded

  if (isLoading) {
    return <div className="flex items-center justify-center h-full"><LoadingSpinner /></div>;
  }

  if (error) {
    return <div className="flex items-center justify-center h-full text-red-500">{error}</div>;
  }

  return (
    <div className="relative w-full h-full overflow-hidden">
      {/* Reset zoom control */}
      <button
        className="absolute bottom-2 left-2 z-20 px-2 py-1 text-xs rounded bg-gray-800/80 hover:bg-gray-700 border border-gray-700 text-gray-100"
        onClick={() => {
          if (!svgRef.current || !zoomRef.current) return;
          const svg = d3.select(svgRef.current);
          svg.transition().duration(250).call(zoomRef.current.transform, d3.zoomIdentity);
        }}
      >
        Reset view
      </button>
      <svg
        ref={svgRef}
        width="100%"
        height="100%"
        viewBox="0 0 975 610"
        preserveAspectRatio="xMidYMid meet"
        style={{ touchAction: 'none', cursor: 'grab' }}
      >
        {/* Transparent pane to capture zoom/pan across empty areas */}
        <rect ref={zoomPaneRef} width="100%" height="100%" fill="transparent" pointerEvents="all" />
        <g ref={rootGroupRef}>
          <g className="counties">
            {geoData?.counties.map((feature) => {
              const isSelected = feature.id === selectedCountyId;
              const isHovered = feature.id === hoveredCountyId;
              
              let fillClass = 'fill-gray-700/50 stroke-gray-600/50';
              if (isHovered) {
                fillClass = 'fill-cyan-400/80 stroke-cyan-200';
              }
              if (isSelected) {
                fillClass = 'fill-amber-400 stroke-amber-200';
              }

              return (
                <path
                  key={feature.id}
                  d={pathGenerator(feature) ?? ''}
                  className={`county-path transition-all duration-150 ease-in-out ${fillClass} cursor-pointer`}
                  onMouseEnter={() => setHoveredCountyId(feature.id)}
                  onMouseLeave={() => setHoveredCountyId(null)}
                  onClick={() => onCountyClick({ id: feature.id })}
                  style={{ strokeWidth: 0.5 }}
                />
              );
            })}
          </g>
          {/* Overlay features from backend queries */}
          {overlayFeatures.length > 0 && (
            <g className="overlays">
              {overlayFeatures.map((feat, idx) => (
                <path
                  key={idx}
                  d={pathGenerator(feat) ?? ''}
                  className="fill-amber-400/40 stroke-amber-300"
                  style={{ strokeWidth: 1.5 }}
                  pointerEvents="none"
                />
              ))}
            </g>
          )}
          <path
            d={pathGenerator(geoData?.states) ?? ''}
            className="states"
            fill="none"
            stroke="#FFFFFF"
            strokeLinejoin="round"
            style={{ strokeWidth: 0.5 }}
          />
        </g>
      </svg>
    </div>
  );
};

export default USMap;
