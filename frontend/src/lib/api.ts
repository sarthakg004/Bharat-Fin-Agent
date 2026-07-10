// Typed API client. Single source of truth for backend shapes + URLs.

// Same-origin by default: an empty/unset VITE_API_URL means we hit "/api/..."
// on whatever host serves the page. That works both in production (the Docker
// Space serves API + SPA on one origin) and in dev (Vite proxies /api → :8000,
// see vite.config.ts). Only set VITE_API_URL to point at a different host.
const BASE = import.meta.env.VITE_API_URL || "";

export interface Chunk {
  id: number;
  text: string;
  company: string;
  ticker?: string;
  year: string;
  page: string | number;
  source_url?: string;
  citation: string;
  sub_query?: string;
  kind?: "text" | "web" | "table" | "market" | "xbrl" | "calc" | "edgar" | "uploaded";
  /** Set when kind === "uploaded" — the originating file. */
  filename?: string;
}

export interface QueryMetadata {
  model?: string;
  latency?: number;
  input_tokens?: number;
  output_tokens?: number;
  sub_queries?: string[];
  query_routes?: string[];
  grading_score?: number | null;
  avg_grade?: number | null;
  rewrite_iterations?: number;
  critic_iterations?: number;
  needs_retry?: boolean | null;
  low_confidence?: boolean | null;
  refused?: boolean;
  // Confidence framework (#8/9).
  confidence?: number | null;
  confidence_band?: "answer" | "warn" | "refuse" | null;
  answer_status?: string | null;
  confidence_scores?: {
    retrieval?: number | null;
    verification?: number | null;
    citation?: number | null;
    critic?: number | null;
  } | null;
  /** Low-confidence draft the gate withheld; revealed on demand. Empty otherwise. */
  numeric_verification_score?: number | null;
  unverified_count?: number;
  web_hits?: number;
  table_computations?: number;
  market_calls?: number;
  citations?: string[];
  agentic?: QueryMetadata | null;
}

// --------------------------------------------------------------------------- //
// Charts
// --------------------------------------------------------------------------- //

export interface Candle {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface VolumeBar {
  time: number;
  value: number;
  color?: string;
}

export interface ChartSpec {
  type: "candlestick";
  symbol: string;
  period: string;
  interval: string;
  candles: Candle[];
  volume?: VolumeBar[];
}


// --------------------------------------------------------------------------- //
// Query (SSE)
// --------------------------------------------------------------------------- //

export interface ProviderConfig {
  provider: "groq" | "gemini" | "openai" | "anthropic";
  synth_model?: string;
  api_key?: string;
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

export interface QueryRequest {
  question: string;
  top_k?: number;
  provider_config?: ProviderConfig;
  /** Per-session conversation memory (the server is stateless). */
  chat_history?: ChatTurn[];
  /** Ids from POST /api/upload — the documents to answer over. */
  upload_ids?: string[];
}

// --------------------------------------------------------------------------- //
// Deep Research
// --------------------------------------------------------------------------- //

export interface ResearchRequest {
  question: string;
  provider_config?: ProviderConfig;
  chat_history?: ChatTurn[];
  max_agents?: number;
}

/** One planned research task (a specialist agent, or the final thesis step). */
export interface ResearchPlanTask {
  id: string;
  label: string;
}

export interface UploadResponse {
  upload_id: string;
  filename: string;
  pages: number;
  tables: number;
  chunks: number;
}

export interface HealthResponse {
  status: string;
  collections: string[];
}

export type SSEEvent =
  | { type: "chat"; chat_id: number }
  | { type: "status"; stage: string; label: string; index?: number; total?: number }
  | { type: "step_done"; stage: string; detail?: string | null }
  | { type: "research_plan"; company?: string; ticker?: string; objective?: string; tasks: ResearchPlanTask[] }
  | { type: "agent_start"; id: string }
  | { type: "agent_done"; id: string; status: "done" | "failed"; detail?: string; summary?: string; confidence?: number | null }
  | { type: "sources"; chunks: Chunk[]; metadata: QueryMetadata }
  | { type: "chart"; chart: ChartSpec }
  | { type: "chunk"; content: string }
  | { type: "metrics"; latency: number; model?: string; input_tokens?: number; output_tokens?: number; agentic?: QueryMetadata | null }
  | { type: "error"; message: string; code?: string }
  | { type: "done" };

export interface StreamHandlers {
  onEvent: (event: SSEEvent) => void;
  signal?: AbortSignal;
}

// --------------------------------------------------------------------------- //
// JSON / SSE
// --------------------------------------------------------------------------- //

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

export const api = {
  // The server is stateless: only health + the SSE query endpoint remain.
  // Chat threads live entirely client-side (see threadStore).
  health: () => getJson<HealthResponse>("/api/health"),
};

/** Upload a document; the returned id is passed as `upload_ids` on queries.
 * Uploads are ephemeral server-side (kept ~1 hour). */
export async function uploadFile(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/api/upload`, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.json().then((j) => j.detail).catch(() => null);
    throw new Error(detail || `Upload failed: ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<UploadResponse>;
}

/**
 * Stream a POST /api/query response as Server-Sent Events.
 * We do the parsing manually because `EventSource` is GET-only.
 */
export function streamQuery(req: QueryRequest, handlers: StreamHandlers): Promise<void> {
  return streamSSE("/api/query", req, handlers);
}

/** Stream a Deep Research run (POST /api/research) — same SSE framing. */
export function streamResearch(req: ResearchRequest, handlers: StreamHandlers): Promise<void> {
  return streamSSE("/api/research", req, handlers);
}

async function streamSSE(path: string, req: unknown, handlers: StreamHandlers): Promise<void> {
  const res = await fetch(`${BASE}${path}`, {
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
