import { useCallback } from "react";
import toast from "react-hot-toast";

import { ConnectionLost } from "@/lib/api";
import type { ChatTurn, SSEEvent } from "@/lib/api";
import { useChatStore } from "@/store/chatStore";
import { useThreadStore } from "@/store/threadStore";
import { writerPayload } from "@/store/settingsStore";
import { useServerConfig } from "./useServerConfig";
import { useSSE } from "./useSSE";

const MEMORY_TURNS = 6;        // prior turns sent as conversation memory
const MAX_AUTO_RETRIES = 2;    // automatic retries of one question, then it is manual

/**
 * Ask a question and stream the answer into the chat store.
 *
 * `ask` adds a new turn. `regenerate` re-runs the last question: the Retry
 * button calls it, and so does the countdown after a transient failure.
 */
export function useRAGQuery() {
  const { send } = useSSE();
  const config = useServerConfig();
  const {
    appendMessage, patchMessage, appendChunkToMessage,
    appendStepToMessage, markStepDone, appendChartToMessage, setStreaming,
  } = useChatStore();

  const historyBefore = useCallback((upto: number): ChatTurn[] => {
    return useChatStore.getState().messages
      .slice(0, upto)
      .filter((m) => m.content && !m.error && !m.notice)
      .map((m) => ({ role: m.role, content: m.content }))
      .slice(-MEMORY_TURNS);
  }, []);

  const runStream = useCallback(
    async (question: string, assistantId: string, history: ChatTurn[], attempt: number) => {
      setStreaming(assistantId);
      const startedAt = Date.now();
      let firstChunk = true;

      // Show a failure in plain words. A transient one (`retryInSec` given)
      // gets a countdown and is retried automatically a limited number of times.
      const fail = (notice: string, retryInSec?: number) => {
        patchMessage(assistantId, {
          notice, streaming: false, status: undefined,
          retryAt: retryInSec != null ? Date.now() + retryInSec * 1000 : undefined,
          autoRetry: retryInSec != null && attempt < MAX_AUTO_RETRIES,
        });
        setStreaming(null);
      };

      try {
        await send(
          {
            question,
            // The client's thread id; the server only uses it to group traces.
            session_id: useThreadStore.getState().activeId ?? undefined,
            chat_history: history,
            writer: writerPayload(config),
          },
          (e: SSEEvent) => {
            switch (e.type) {
              case "status":
                appendStepToMessage(assistantId, {
                  stage: e.stage, label: e.label, index: e.index, total: e.total,
                });
                break;
              case "step_done":
                markStepDone(assistantId, e.stage, e.detail);
                break;
              case "sources":
                patchMessage(assistantId, { chunks: e.chunks, metadata: e.metadata });
                break;
              case "chart":
                appendChartToMessage(assistantId, e.chart);
                break;
              case "chunk":
                if (firstChunk) {
                  firstChunk = false;       // the answer starts: freeze the timer
                  patchMessage(assistantId, { thoughtMs: Date.now() - startedAt, status: undefined });
                }
                appendChunkToMessage(assistantId, e.content);
                break;
              case "metrics": {
                const { type: _type, ...meta } = e;
                patchMessage(assistantId, { metadata: meta });
                break;
              }
              case "error":
                fail(e.message, e.retryable ? (e.retry_after ?? 5) : undefined);
                break;
              case "done":
                patchMessage(assistantId, { streaming: false, status: undefined });
                setStreaming(null);
                useThreadStore.getState().saveActive();
                break;
            }
          },
        );
      } catch (err) {
        // `fetch` throws TypeError when the server cannot be reached at all.
        if (err instanceof ConnectionLost || err instanceof TypeError) {
          fail("The connection to the server dropped before the answer finished.", 3);
        } else {
          const msg = err instanceof Error ? err.message : String(err);
          patchMessage(assistantId, { error: msg, streaming: false, status: undefined });
          setStreaming(null);
          toast.error(msg);
        }
      }
    },
    [send, config, patchMessage, appendChunkToMessage, appendStepToMessage, markStepDone,
     appendChartToMessage, setStreaming],
  );

  const ask = useCallback(
    async (question: string) => {
      if (!question.trim()) return;
      const threads = useThreadStore.getState();
      if (!threads.activeId) threads.createChat("New chat");

      const history = historyBefore(useChatStore.getState().messages.length);
      appendMessage({ role: "user", content: question });
      const assistantId = appendMessage({
        role: "assistant", content: "", streaming: true, startedAt: Date.now(), attempt: 0,
      });
      useThreadStore.getState().saveActive();
      await runStream(question, assistantId, history, 0);
    },
    [appendMessage, historyBefore, runStream],
  );

  /** Re-run the last question. `auto` = triggered by the countdown, not the user. */
  const regenerate = useCallback(
    async (auto = false) => {
      const msgs = useChatStore.getState().messages;
      const lastUserOffset = [...msgs].reverse().findIndex((m) => m.role === "user");
      if (lastUserOffset === -1) return;
      const userIdx = msgs.length - 1 - lastUserOffset;
      const previous = msgs[msgs.length - 1];
      const attempt = auto ? (previous?.attempt ?? 0) + 1 : 0;

      const history = historyBefore(userIdx);
      useChatStore.getState().dropLastAssistant();
      const assistantId = appendMessage({
        role: "assistant", content: "", streaming: true, startedAt: Date.now(), attempt,
      });
      await runStream(msgs[userIdx].content, assistantId, history, attempt);
    },
    [appendMessage, historyBefore, runStream],
  );

  return { ask, regenerate };
}
