// Typed API client: the backend's shapes and URLs in one place.

// Empty VITE_API_URL means same origin ("/api/..."). Set it to call another host.
const BASE = import.meta.env.VITE_API_URL || "";

export type Provider = "groq" | "gemini" | "openai" | "anthropic";

/** One evidence card. `[N]` in the answer cites the card with `id === N - 1`. */
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
  kind?: "text" | "web" | "market" | "xbrl" | "calc" | "edgar";
}

export interface QueryMetadata {
  model?: string;
  latency?: number;
  input_tokens?: number;
  output_tokens?: number;
  sub_queries?: string[];
  query_routes?: string[];
  /** Share of the answer's claims the fact-check could support; null = not checked. */
  support_score?: number | null;
  unsupported_claims?: string[];
  recoveries?: number;
  refused?: boolean;
  /** Steps that were skipped or degraded, in plain words. */
  notices?: string[];
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
// Config and query
// --------------------------------------------------------------------------- //

/** GET /api/config: which model does which job, and what the picker may offer. */
export interface ServerConfig {
  roles: Record<"planner" | "extractor" | "writer" | "critic", { provider: Provider; model: string }>;
  writer_models: Record<Provider, string[]>;
  /** Providers the server has keys for. The others need the user's own key. */
  server_keys: Provider[];
}

/** The user's writer choice. Omitted entirely when the default is used. */
export interface WriterConfig {
  provider?: Provider;
  model?: string;
  api_key?: string;
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

export interface QueryRequest {
  question: string;
  writer?: WriterConfig;
  /** Client thread id. Not stored server-side; it only groups the traces. */
  session_id?: string;
  /** The server is stateless, so recent turns are sent with each question. */
  chat_history?: ChatTurn[];
}

export type ErrorCode =
  | "rate_limit" | "busy"                       // transient: retried automatically
  | "quota" | "too_large" | "auth" | "not_found" | "error";

export type SSEEvent =
  | { type: "status"; stage: string; label: string; index?: number; total?: number }
  | { type: "step_done"; stage: string; detail?: string | null }
  | { type: "sources"; chunks: Chunk[]; metadata: QueryMetadata }
  | { type: "chart"; chart: ChartSpec }
  | { type: "chunk"; content: string }
  | ({ type: "metrics" } & QueryMetadata)
  | { type: "error"; message: string; code: ErrorCode; retryable: boolean; retry_after?: number }
  | { type: "done" };

export interface StreamHandlers {
  onEvent: (event: SSEEvent) => void;
  signal?: AbortSignal;
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

export const api = {
  health: () => getJson<{ status: string }>("/api/health"),
  config: () => getJson<ServerConfig>("/api/config"),
};

/** The stream closed before the server sent `done`: the connection dropped. */
export class ConnectionLost extends Error {
  constructor() {
    super("The connection dropped before the answer finished.");
    this.name = "ConnectionLost";
  }
}

/**
 * POST /api/query and read the Server-Sent Events. Parsed by hand because
 * `EventSource` only does GET. Throws `ConnectionLost` if the stream ends
 * without a `done` event.
 */
export async function streamQuery(req: QueryRequest, handlers: StreamHandlers): Promise<void> {
  const res = await fetch(`${BASE}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(req),
    signal: handlers.signal,
  });
  if (!res.ok || !res.body) {
    const detail = await res.json().then((j) => j.detail).catch(() => null);
    throw new Error(typeof detail === "string" ? detail : `Request failed: ${res.status} ${res.statusText}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finished = false;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const data = buffer.slice(0, idx).split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart());
      buffer = buffer.slice(idx + 2);
      if (data.length === 0) continue;
      try {
        const event = JSON.parse(data.join("\n")) as SSEEvent;
        if (event.type === "done") finished = true;
        handlers.onEvent(event);
      } catch (e) {
        console.error("Failed to parse SSE event", e, data);
      }
    }
  }
  if (!finished) throw new ConnectionLost();
}
