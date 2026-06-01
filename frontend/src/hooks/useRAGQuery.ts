import { useCallback } from "react";
import toast from "react-hot-toast";

import type { ConfigId, Market, SSEEvent } from "@/lib/api";
import { useChatStore } from "@/store/chatStore";
import { useSSE } from "./useSSE";

/**
 * High-level "submit a question" hook — wires the SSE stream into the chat store.
 * The component just calls `ask(question)`; everything else is incremental updates
 * on the global store.
 */
export function useRAGQuery(market: Market, config: ConfigId, companyFilter: string[]) {
  const { send, abort } = useSSE();
  const {
    appendMessage,
    patchMessage,
    appendChunkToMessage,
    setStreaming,
    startNewChat,
  } = useChatStore();

  const ask = useCallback(
    async (question: string) => {
      if (!question.trim()) return;

      // Submitting a fresh question detaches us from any history row that
      // happened to be loaded — the new turn becomes its own chat.
      const wasInHistory = useChatStore.getState().currentHistoryId != null;
      if (wasInHistory) startNewChat();

      appendMessage({ role: "user", content: question, market, config });
      const assistantId = appendMessage({
        role: "assistant",
        content: "",
        market,
        config,
        streaming: true,
      });
      setStreaming(assistantId);

      try {
        await send(
          {
            question,
            config,
            market,
            company_filter: companyFilter.length ? companyFilter : null,
            top_k: 5,
          },
          (e: SSEEvent) => {
            switch (e.type) {
              case "status":
                patchMessage(assistantId, {
                  status: { stage: e.stage, label: e.label },
                });
                break;
              case "sources":
                patchMessage(assistantId, {
                  chunks: e.chunks,
                  metadata: { ...(e.metadata || {}) },
                });
                break;
              case "chunk":
                appendChunkToMessage(assistantId, e.content);
                break;
              case "metrics":
                patchMessage(assistantId, {
                  metadata: {
                    model: e.model,
                    latency: e.latency,
                    input_tokens: e.input_tokens,
                    output_tokens: e.output_tokens,
                    agentic: e.agentic ?? null,
                  },
                });
                break;
              case "error":
                patchMessage(assistantId, { error: e.message, streaming: false });
                setStreaming(null);
                toast.error(e.message);
                break;
              case "done":
                patchMessage(assistantId, { streaming: false, status: undefined });
                setStreaming(null);
                break;
            }
          },
        );
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        patchMessage(assistantId, { error: msg, streaming: false });
        setStreaming(null);
        toast.error(msg);
      }
    },
    [send, market, config, companyFilter, appendMessage, patchMessage, appendChunkToMessage, setStreaming],
  );

  return { ask, abort };
}
