import { useCallback, useRef } from "react";
import { streamQuery, streamResearch } from "@/lib/api";
import type { QueryRequest, ResearchRequest, SSEEvent } from "@/lib/api";

/**
 * Tiny wrapper that abstracts the POST-and-stream pattern away from the components.
 * Returns an `abort()` function so callers can cancel mid-stream (e.g. user clicks
 * Clear or starts a new query).
 */
export function useSSE() {
  const controllerRef = useRef<AbortController | null>(null);

  const send = useCallback(async (
    req: QueryRequest | ResearchRequest,
    onEvent: (e: SSEEvent) => void,
    mode: "chat" | "research" = "chat",
  ) => {
    controllerRef.current?.abort();
    const ctrl = new AbortController();
    controllerRef.current = ctrl;
    const stream = mode === "research" ? streamResearch : streamQuery;
    try {
      await stream(req as QueryRequest & ResearchRequest, { onEvent, signal: ctrl.signal });
    } catch (err: unknown) {
      if ((err as Error).name === "AbortError") return;
      throw err;
    }
  }, []);

  const abort = useCallback(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
  }, []);

  return { send, abort };
}
