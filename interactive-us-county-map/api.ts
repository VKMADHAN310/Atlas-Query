// Simple API client for the NL→SQL backend
// Uses Vite env var VITE_API_BASE_URL or defaults to http://localhost:8000

export interface BackendAnswer {
  ok: boolean;
  error?: string | null;
  rows_preview: Array<Record<string, any>>;
  rows_total: number;
  sql?: string | null;
  lm_ms: number;
  db_ms: number;
  scope_rejected: boolean;
}

const API_BASE = (import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8000';

export async function postAnswer(query: string, opts?: { provider?: 'ollama' | 'hf'; model?: string }): Promise<BackendAnswer> {
  const body = {
    query,
    ...(opts?.provider ? { provider: opts.provider } : {}),
    ...(opts?.model ? { model: opts.model } : {}),
  };

  const res = await fetch(`${API_BASE}/answer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Backend error ${res.status}: ${text}`);
  }
  const data = (await res.json()) as BackendAnswer;
  return data;
}

export async function getCountyByGeoid(geoid: string): Promise<BackendAnswer> {
  const res = await fetch(`${API_BASE}/county/${encodeURIComponent(geoid)}`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Backend error ${res.status}: ${text}`);
  }
  return (await res.json()) as BackendAnswer;
}
