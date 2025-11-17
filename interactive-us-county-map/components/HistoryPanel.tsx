import React from 'react';
import type { BackendAnswer } from '../api';

type Provider = 'ollama' | 'hf' | 'db';

interface HistoryPanelProps {
  items: Array<{
    id: string;
    query: string;
    provider: Provider;
    model: string;
    timestamp: number;
    answer: BackendAnswer;
  }>;
  selectedId: string | null;
  onSelect: (id: string) => void;
  onClear: () => void;
}

const HistoryPanel: React.FC<HistoryPanelProps> = ({ items, selectedId, onSelect, onClear }) => {
  return (
    <div className="flex flex-col max-h-[50vh] md:max-h-full overflow-hidden">
      <div className="flex items-center justify-between p-3 border-b border-cyan-500/30">
        <div className="font-semibold text-cyan-400">History</div>
        <button
          className="text-xs px-2 py-1 rounded bg-gray-700 hover:bg-gray-600"
          onClick={onClear}
          disabled={items.length === 0}
        >
          Clear
        </button>
      </div>
      <div className="flex-1 overflow-y-auto">
        {items.length === 0 ? (
          <div className="p-3 text-sm text-gray-400">No history yet. Run a query to see it here.</div>
        ) : (
          <ul className="divide-y divide-gray-700">
            {items.map(item => {
              const isSelected = selectedId === item.id;
              return (
                <li key={item.id}>
                  <button
                    className={`w-full text-left p-3 hover:bg-gray-700/50 ${isSelected ? 'bg-gray-700/70' : ''}`}
                    onClick={() => onSelect(item.id)}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2 min-w-0">
                        <span className={`text-cyan-300 transition-transform transform ${isSelected ? 'rotate-90' : ''}`}>▶</span>
                        <div className="truncate text-sm text-gray-200" title={item.query}>{item.query}</div>
                      </div>
                      <div className="text-xs text-gray-400 whitespace-nowrap">{new Date(item.timestamp).toLocaleTimeString()}</div>
                    </div>
                    <div className="mt-1 flex items-center gap-2 text-xs text-gray-400">
                      <span className="px-1.5 py-0.5 rounded bg-gray-700 text-gray-300">{item.provider === 'ollama' ? 'Local' : item.provider === 'hf' ? 'API' : 'Direct'}</span>
                      <span className="px-1.5 py-0.5 rounded bg-gray-700 text-gray-300 truncate" title={item.model}>{item.model}</span>
                      <span>• {item.answer.rows_total} rows</span>
                      <span>• LM {item.answer.lm_ms} ms</span>
                      <span>• DB {item.answer.db_ms} ms</span>
                    </div>

                    {isSelected && (
                      <div className="mt-2 p-2 border border-cyan-500/20 rounded bg-gray-800/60">
                        {item.answer.sql && (
                          <>
                            <div className="font-semibold text-cyan-400 mb-1">SQL</div>
                            <pre className="whitespace-pre-wrap break-words text-[11px] text-cyan-300 font-mono">{item.answer.sql}</pre>
                          </>
                        )}
                        <div className="mt-2 text-xs text-gray-400">LM: {item.answer.lm_ms} ms • DB: {item.answer.db_ms} ms</div>
                      </div>
                    )}
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
};

export default HistoryPanel;
