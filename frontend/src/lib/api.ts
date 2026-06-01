// Typed API client. Keeps URL building + JSON parsing in one place.

const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export type Market = "us" | "india";
export type ConfigId = "naive" | "agentic";

export interface Chunk {
  id: number;
  text: string;
  company: string;
  ticker?: string;
  year: string;
  page: string | number;
  market?: string;
  source_url?: string;
  citation: string;
  sub_query?: string;
}

export interface QueryMetadata {
  model?: string;
  latency?: number;
  input_tokens?: number;
  output_tokens?: number;
  sub_queries?: string[];
  grading_score?: number | null;
  avg_grade?: number | null;
  rewrite_iterations?: number;
  critic_iterations?: number;
  needs_retry?: boolean | null;
  low_confidence?: boolean | null;
  citations?: string[];
  agentic?: QueryMetadata | null;
}

export interface QueryRequest {
  question: string;
  config: ConfigId;
  market: Market;
  company_filter?: string[] | null;
  top_k?: number;
}

export interface ConfigInfo {
  id: string;
  label: string;
  model: string;
  description: string;
}

export interface HealthResponse {
  status: string;
  collections: string[];
  configs: string[];
}

export interface HistoryItem {
  id: number;
  question: string;
  config: string;
  market: string;
  answer: string;
  latency: number;
  created_at: string;
}

// --------------------------------------------------------------------------- //
// JSON endpoints
// --------------------------------------------------------------------------- //

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

export const api = {
  health: () => getJson<HealthResponse>("/api/health"),
  configs: () => getJson<{ configs: ConfigInfo[] }>("/api/configs"),
  history: (limit = 50) => getJson<{ items: HistoryItem[] }>(`/api/history?limit=${limit}`),
};

// --------------------------------------------------------------------------- //
// SSE query stream
// --------------------------------------------------------------------------- //

export type SSEEvent =
  | { type: "status"; stage: string; label: string }
  | { type: "sources"; chunks: Chunk[]; metadata: QueryMetadata }
  | { type: "chunk"; content: string }
  | { type: "metrics"; latency: number; model?: string; input_tokens?: number; output_tokens?: number; agentic?: QueryMetadata | null }
  | { type: "error"; message: string }
  | { type: "done" };

export interface StreamHandlers {
  onEvent: (event: SSEEvent) => void;
  signal?: AbortSignal;
}

/**
 * Stream a POST /api/query response as Server-Sent Events.
 * We do the parsing ourselves because the standard `EventSource` only supports
 * GET requests and we need to POST a JSON body.
 */
export async function streamQuery(req: QueryRequest, handlers: StreamHandlers): Promise<void> {
  const res = await fetch(`${BASE}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(req),
    signal: handlers.signal,
  });

  if (!res.ok || !res.body) {
    throw new Error(`Stream failed: ${res.status} ${res.statusText}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE messages are separated by a blank line.
    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const rawEvent = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const dataLines = rawEvent
        .split("\n")
        .filter(line => line.startsWith("data:"))
        .map(line => line.slice(5).trimStart());
      if (dataLines.length === 0) continue;
      try {
        const event = JSON.parse(dataLines.join("\n")) as SSEEvent;
        handlers.onEvent(event);
      } catch (e) {
        console.error("Failed to parse SSE event", e, dataLines);
      }
    }
  }
}
