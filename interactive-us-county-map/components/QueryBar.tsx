import React, { useMemo, useRef, useState } from 'react';
import { postAnswer, type BackendAnswer } from '../api';

interface QueryBarProps {
  onResult: (query: string, result: BackendAnswer, meta: { provider: 'ollama' | 'hf'; model: string }) => void;
  onError: (message: string) => void;
}

const QueryBar: React.FC<QueryBarProps> = ({ onResult, onError }) => {
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [loadingModel, setLoadingModel] = useState(false);
  const [provider, setProvider] = useState<'ollama' | 'hf'>('ollama'); // UI label: Local
  const [model, setModel] = useState<string>('llama3.1:8b'); // UI label: Model
  const prevLocalModelRef = useRef<string>('llama3.1:8b');

  const OLLAMA_URL = (import.meta as any).env?.VITE_OLLAMA_URL || 'http://localhost:11434';

  const localModels = useMemo(() => [
    { label: 'Phi-3', value: 'phi3:mini' },
    { label: 'Llama 3.1', value: 'llama3.1:8b' },
  ], []);

  const apiModels = useMemo(() => [
    { label: 'Llama 70B', value: 'hf:meta-llama/Llama-3.3-70B-Instruct:cerebras' },
  ], []);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isLoading) return;
    setIsLoading(true);
    try {
      const ans = await postAnswer(query.trim(), { provider, model });
      if (!ans.ok) {
        onError(ans.error || 'Backend returned ok=false');
      }
      onResult(query.trim(), ans, { provider, model });
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsLoading(false);
    }
  };

  const handleProviderChange = (val: 'ollama' | 'hf') => {
    setProvider(val);
    if (val === 'ollama') {
      // Default to Llama 3.1 when switching to Local
      setModel('llama3.1:8b');
    } else {
      // Default to Llama 70B when switching to API
      setModel('hf:meta-llama/Llama-3.3-70B-Instruct:cerebras');
    }
  };

  const tryStopOllamaModel = async (m: string) => {
    try {
      await fetch(`${OLLAMA_URL}/api/stop`, { method: 'POST', body: JSON.stringify({ name: m }), headers: { 'Content-Type': 'application/json' } });
    } catch {
      /* ignore (likely CORS) */
    }
  };

  const tryWarmOllamaModel = async (m: string) => {
    try {
      // Empty prompt warm-up to load weights; keep it short
      await fetch(`${OLLAMA_URL}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: m, prompt: '', stream: false, options: { num_predict: 1, temperature: 0 } }),
      });
    } catch {
      /* ignore (likely CORS) */
    }
  };

  const handleModelChange = async (newModel: string) => {
    // If switching among Local models, show loading and try to unload/warm
    const isLocal = provider === 'ollama';
    const prevLocal = prevLocalModelRef.current;
    setModel(newModel);
    if (isLocal && newModel !== prevLocal) {
      setLoadingModel(true);
      try {
        await tryStopOllamaModel(prevLocal);
        await tryWarmOllamaModel(newModel);
      } finally {
        prevLocalModelRef.current = newModel;
        setLoadingModel(false);
      }
    }
  };

  const modelOptions = provider === 'ollama' ? localModels : apiModels;

  return (
    <form onSubmit={submit} className="p-3 bg-gray-800 border-b border-cyan-500/30 flex flex-wrap gap-3 items-center">
      <div className="flex-1 min-w-[250px]">
        <input
          className="w-full p-2 rounded bg-gray-700 text-white placeholder-gray-400 focus:outline-none"
          placeholder="Ask a question, e.g. visualize Madison County"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={isLoading || loadingModel}
        />
      </div>
      <div className="flex items-center gap-2">
        <div className="flex flex-col items-start">
          <select
            className="p-2 rounded bg-gray-700 text-white"
            value={provider}
            onChange={(e) => handleProviderChange(e.target.value as 'ollama' | 'hf')}
            disabled={isLoading || loadingModel}
            aria-label="Source"
          >
            <option value="ollama">Local</option>
            <option value="hf">API</option>
          </select>
          <span className="text-xs text-gray-400 mt-1">Source</span>
        </div>
        <div className="flex flex-col items-start">
          <select
            className="p-2 rounded bg-gray-700 text-white"
            value={model}
            onChange={(e) => handleModelChange(e.target.value)}
            disabled={isLoading || loadingModel}
            aria-label="Model"
          >
            {modelOptions.map(m => (
              <option key={m.value} value={m.value}>{m.label}</option>
            ))}
          </select>
          <span className="text-xs text-gray-400 mt-1">Model</span>
        </div>
        <button
          type="submit"
          className="px-4 py-2 rounded bg-cyan-600 hover:bg-cyan-500 disabled:bg-gray-600"
          disabled={isLoading || !query.trim() || loadingModel}
        >
          {isLoading ? 'Sending…' : 'Send'}
        </button>
      </div>
      {loadingModel && (
        <div className="w-full text-sm text-amber-300">Loading new model…</div>
      )}
    </form>
  );
};

export default QueryBar;
