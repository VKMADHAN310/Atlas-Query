import React, { useState, useCallback } from 'react';
import USMap from './components/USMap';
import CountyInfo from './components/CountyInfo';
import ErrorLogger from './components/ErrorLogger';
import QueryBar from './components/QueryBar';
import { getCountyByGeoid, type BackendAnswer } from './api';
import type { County } from './types';
import type { Feature, Geometry } from 'geojson';
import HistoryPanel from './components/HistoryPanel.tsx';

type ProviderType = 'ollama' | 'hf' | 'db';

interface HistoryEntry {
  id: string; // timestamp-based id
  query: string;
  provider: ProviderType;
  model: string;
  timestamp: number;
  answer: BackendAnswer;
  overlayFeatures: Feature<Geometry, any>[];
}

const App: React.FC = () => {
  const [selectedCounty, setSelectedCounty] = useState<County | null>(null);
  const [errorLogs, setErrorLogs] = useState<string[]>([]);
  const [overlayFeatures, setOverlayFeatures] = useState<Feature<Geometry, any>[]>([]);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [selectedHistoryId, setSelectedHistoryId] = useState<string | null>(null);

  const rowsToFeatures = (rows: Array<Record<string, any>>): Feature<Geometry, any>[] => {
    const feats: Feature<Geometry, any>[] = [];
    for (const r of rows) {
      const gj = r.geojson;
      if (!gj) continue;
      let geom: any = gj;
      if (typeof gj === 'string') {
        try { geom = JSON.parse(gj); } catch { continue; }
      }
      if (!geom || typeof geom !== 'object' || !geom.type) continue;
      feats.push({ type: 'Feature', geometry: geom as Geometry, properties: r });
    }
    return feats;
  };

  const handleCountyClick = async (county: County) => {
    setSelectedCounty(county);
    // Ask backend to visualize this county by GEOID
    try {
      const q = `visualize county by geoid ${county.id}`;
      const ans = await getCountyByGeoid(county.id);
      if (!ans.ok) {
        handleError(ans.error || 'Backend returned ok=false');
        return;
      }
      const feats = rowsToFeatures(ans.rows_preview);
      setOverlayFeatures(feats);

      // Enrich selected county with name/state from the first result row when available
      const first = ans.rows_preview?.[0] || {} as any;
      const enriched: County = {
        id: county.id,
        name: (first.namelsad || first.name || undefined) as string | undefined,
        state: (first.stateaabrv || first.stateabbrv || first.stateabbrev || first.state || undefined) as string | undefined,
      };
      setSelectedCounty(enriched);

      const entry: HistoryEntry = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        query: q,
        provider: 'db',
        model: 'direct',
        timestamp: Date.now(),
        answer: ans,
        overlayFeatures: feats,
      };
      setHistory(prev => [entry, ...prev].slice(0, 100));
      setSelectedHistoryId(entry.id);
    } catch (e) {
      handleError(e instanceof Error ? e.message : String(e));
    }
  };

  const handleClearSelection = () => {
    setSelectedCounty(null);
  }

  const handleError = useCallback((message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setErrorLogs(prev => [`[${timestamp}] ${message}`, ...prev]);
  }, []);

  const clearErrorLogs = () => {
    setErrorLogs([]);
  };

  return (
    <div className="h-screen bg-gray-900 text-white flex flex-col font-sans">
      <header className="p-4 text-center text-xl font-bold text-cyan-400 border-b border-cyan-500/30 bg-gray-900 shadow-lg tracking-wider z-10">
        <h1>Interactive US County Map</h1>
      </header>
      <QueryBar
        onResult={(q: string, ans: BackendAnswer, meta) => {
          if (!ans.ok) {
            handleError(ans.error || 'Backend returned ok=false');
            return;
          }
          // Clear any previously selected county when a new result arrives
          setSelectedCounty(null);
          const feats = rowsToFeatures(ans.rows_preview);
          setOverlayFeatures(feats);

          const entry: HistoryEntry = {
            id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
            query: q,
            provider: meta.provider,
            model: meta.model,
            timestamp: Date.now(),
            answer: ans,
            overlayFeatures: feats,
          };
          setHistory(prev => [entry, ...prev].slice(0, 100));
          setSelectedHistoryId(entry.id);
        }}
        onError={handleError}
      />
      
      {/* Main content area */}
      <div className="flex-grow flex flex-col md:flex-row overflow-hidden">
        {/* Left Column / Top Row: Map */}
        <main className="flex-grow md:w-2/3 h-1/2 md:h-full relative">
          <USMap 
            onCountyClick={handleCountyClick} 
            selectedCountyId={selectedCounty?.id ?? null}
            onError={handleError}
            overlayFeatures={overlayFeatures}
          />
          <CountyInfo county={selectedCounty} onClear={handleClearSelection} />
        </main>

        {/* Right Column / Bottom Row: History & Errors */}
        <aside className="md:w-1/3 flex flex-col bg-gray-800 border-t-2 md:border-t-0 md:border-l-2 border-cyan-500/30 h-1/2 md:h-full">
          <HistoryPanel
            items={history}
            selectedId={selectedHistoryId}
            onSelect={(id) => {
              setSelectedHistoryId(id);
              // Clear any previously selected county when switching history views
              setSelectedCounty(null);
              const item = history.find(h => h.id === id);
              if (item) {
                setOverlayFeatures(item.overlayFeatures);
              }
            }}
            onClear={() => {
              setHistory([]);
              setSelectedHistoryId(null);
              // Also clear selection & overlays when history is cleared
              setSelectedCounty(null);
              setOverlayFeatures([]);
            }}
          />
          <ErrorLogger logs={errorLogs} onClear={clearErrorLogs} />
        </aside>
      </div>
    </div>
  );
};

export default App;