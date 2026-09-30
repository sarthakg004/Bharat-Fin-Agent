import { useCallback, useRef } from "react";
import { streamQuery } from "@/lib/api";
import type { QueryRequest, SSEEvent } from "@/lib/api";

/** POST a question and stream the events. A new `send` aborts the previous one. */
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

  return { send };
}
