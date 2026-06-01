import { useCallback } from "react";
import toast from "react-hot-toast";

import type { Market, SSEEvent } from "@/lib/api";
import { useChatStore } from "@/store/chatStore";
import { useThreadStore } from "@/store/threadStore";
import { useSSE } from "./useSSE";

/**
 * "Submit a question" hook — owns the SSE pipeline + state updates.
 *
 * - When no chat is active yet, the backend auto-creates one and emits a
 *   `chat` event with the new id; we adopt it and refresh the sidebar.
 * - Each new question detaches from any previously loaded message — the
 *   chat_id is what tracks identity, not the in-memory message ids.
 */
export function useRAGQuery(market: Market) {
  const { send } = useSSE();
  const {
    appendMessage,
    patchMessage,
    appendChunkToMessage,
    appendChartToMessage,
    setStreaming,
    setActiveChatId,
  } = useChatStore();

  const ask = useCallback(
    async (question: string) => {
      if (!question.trim()) return;

      const activeChatId = useChatStore.getState().activeChatId;
      const threads = useThreadStore.getState();

      appendMessage({ role: "user", content: question, market });
      const assistantId = appendMessage({
        role: "assistant",
        content: "",
        market,
        streaming: true,
      });
      setStreaming(assistantId);

      try {
        await send(
          { question, market, chat_id: activeChatId, top_k: 5 },
          (e: SSEEvent) => {
            switch (e.type) {
              case "chat":
                // Backend may auto-create the chat; adopt the new id and
                // refresh the sidebar so the new thread appears.
                if (e.chat_id && useChatStore.getState().activeChatId !== e.chat_id) {
                  setActiveChatId(e.chat_id);
                  threads.setActive(e.chat_id);
                  threads.refresh();
                }
                break;
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
              case "chart":
                appendChartToMessage(assistantId, e.chart);
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
                // Refresh sidebar to pick up updated_at + message_count.
                threads.refresh();
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
    [send, market, appendMessage, patchMessage, appendChunkToMessage,
     appendChartToMessage, setStreaming, setActiveChatId],
  );

  return { ask };
}
