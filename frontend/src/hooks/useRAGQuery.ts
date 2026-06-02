import { useCallback } from "react";
import toast from "react-hot-toast";

import type { ChatTurn, Market, SSEEvent } from "@/lib/api";
import { useChatStore } from "@/store/chatStore";
import { useThreadStore } from "@/store/threadStore";
import { currentProviderConfig } from "@/store/settingsStore";
import { useSSE } from "./useSSE";

const MEMORY_TURNS = 6;   // how many prior turns to send as conversation memory

/**
 * "Submit a question" hook — owns the SSE pipeline + state updates.
 *
 * The server is stateless: we send the recent conversation as `chat_history`
 * and, when the answer completes, persist the thread's messages client-side
 * (threadStore → sessionStorage). `ask` adds a new turn; `regenerate` re-runs
 * the last user question (the Retry button).
 */
export function useRAGQuery(market: Market) {
  const { send } = useSSE();
  const {
    appendMessage, patchMessage, appendChunkToMessage,
    appendChartToMessage, setStreaming,
  } = useChatStore();

  // Build chat_history from the messages BEFORE index `upto`.
  const historyBefore = useCallback((upto: number): ChatTurn[] => {
    return useChatStore.getState().messages
      .slice(0, upto)
      .filter((m) => m.content && !m.error)
      .map((m) => ({ role: m.role, content: m.content }))
      .slice(-MEMORY_TURNS);
  }, []);

  const runStream = useCallback(
    async (question: string, assistantId: string, history: ChatTurn[]) => {
      setStreaming(assistantId);
      try {
        await send(
          { question, market, top_k: 5, chat_history: history,
            provider_config: currentProviderConfig() },
          (e: SSEEvent) => {
            switch (e.type) {
              case "status":
                patchMessage(assistantId, { status: { stage: e.stage, label: e.label } });
                break;
              case "sources":
                patchMessage(assistantId, { chunks: e.chunks, metadata: { ...(e.metadata || {}) } });
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
                    model: e.model, latency: e.latency,
                    input_tokens: e.input_tokens, output_tokens: e.output_tokens,
                    agentic: e.agentic ?? null,
                  },
                });
                break;
              case "error":
                if (e.code === "rate_limit") {
                  // Daily limit hit on all keys — show it as a calm message in
                  // the thread, not a red error.
                  patchMessage(assistantId, {
                    content: e.message, streaming: false, status: undefined,
                  });
                  toast("Daily usage limit reached", { icon: "⏳" });
                } else {
                  patchMessage(assistantId, { error: e.message, streaming: false });
                  toast.error(e.message);
                }
                setStreaming(null);
                break;
              case "done":
                patchMessage(assistantId, { streaming: false, status: undefined });
                setStreaming(null);
                useThreadStore.getState().saveActive();   // persist to sessionStorage
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
    [send, market, patchMessage, appendChunkToMessage, appendChartToMessage, setStreaming],
  );

  const ask = useCallback(
    async (question: string) => {
      if (!question.trim()) return;

      // Ensure a thread exists to own this conversation.
      const threads = useThreadStore.getState();
      if (!threads.activeId) threads.createChat("New chat", market);

      const history = historyBefore(useChatStore.getState().messages.length);
      appendMessage({ role: "user", content: question, market });
      const assistantId = appendMessage({ role: "assistant", content: "", market, streaming: true });
      useThreadStore.getState().saveActive();   // capture the user turn immediately

      await runStream(question, assistantId, history);
    },
    [market, appendMessage, historyBefore, runStream],
  );

  const regenerate = useCallback(
    async () => {
      const msgs = useChatStore.getState().messages;
      const lastUserOffset = [...msgs].reverse().findIndex((m) => m.role === "user");
      if (lastUserOffset === -1) return;
      const userIdx = msgs.length - 1 - lastUserOffset;
      const question = msgs[userIdx].content;

      const history = historyBefore(userIdx);     // memory excludes the question itself
      useChatStore.getState().dropLastAssistant(); // remove the stale answer
      const assistantId = appendMessage({ role: "assistant", content: "", market, streaming: true });

      await runStream(question, assistantId, history);
    },
    [market, appendMessage, historyBefore, runStream],
  );

  return { ask, regenerate };
}
