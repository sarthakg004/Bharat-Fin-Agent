import { useCallback, useRef } from "react";
import { streamQuery } from "@/lib/api";
import type { QueryRequest, SSEEvent } from "@/lib/api";

/**
 * Tiny wrapper that abstracts the POST-and-stream pattern away from the components.
 * Returns an `abort()` function so callers can cancel mid-stream (e.g. user clicks
 * Clear or starts a new query).
 */
export function useSSE() {
  const controllerRef = useRef<AbortController | null>(null);

  const send = useCallback(async (req: QueryRequest, onEvent: (e: SSEEvent) => void) => {
    controllerRef.current?.abort();
    const ctrl = new AbortController();
    controllerRef.current = ctrl;
    try {
      await streamQuery(req, { onEvent, signal: ctrl.signal });
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
