import { useCallback } from "react";
import toast from "react-hot-toast";

import type { ChatTurn, SSEEvent } from "@/lib/api";
import { useChatStore } from "@/store/chatStore";
import { useThreadStore } from "@/store/threadStore";
import { currentProviderConfig, useSettingsStore } from "@/store/settingsStore";
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
export function useRAGQuery() {
  const { send } = useSSE();
  const {
    appendMessage, patchMessage, appendChunkToMessage,
    appendStepToMessage, markStepDone, appendChartToMessage, setStreaming,
    setResearchPlan, patchResearchTask,
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
    async (question: string, assistantId: string, history: ChatTurn[],
           mode: "chat" | "research" = "chat") => {
      setStreaming(assistantId);
      const startedAt = Date.now();
      let firstChunk = true;
      const uploadIds = useChatStore.getState().uploads.map((u) => u.id);
      // `ask` creates the thread before streaming, so this is set by now. It's
      // the client's own thread id — the server doesn't store it, it just tags
      // the traces so follow-up turns group into one Langfuse session.
      const sessionId = useThreadStore.getState().activeId ?? undefined;
      const request = mode === "research"
        ? { question, session_id: sessionId, chat_history: history,
            provider_config: currentProviderConfig() }
        : { question, session_id: sessionId, chat_history: history,
            provider_config: currentProviderConfig(),
            upload_ids: uploadIds.length ? uploadIds : undefined };
      try {
        await send(
          request,
          (e: SSEEvent) => {
            switch (e.type) {
              case "research_plan":
                setResearchPlan(assistantId, {
                  company: e.company, objective: e.objective,
                  tasks: e.tasks.map((t) => ({ ...t, status: "pending" as const })),
                });
                break;
              case "agent_start":
                patchResearchTask(assistantId, e.id, { status: "running" });
                break;
              case "agent_done":
                patchResearchTask(assistantId, e.id, {
                  status: e.status, detail: e.detail,
                  summary: e.summary, confidence: e.confidence,
                });
                break;
              case "status":
                appendStepToMessage(assistantId, {
                  stage: e.stage, label: e.label, index: e.index, total: e.total,
                });
                break;
              case "step_done":
                markStepDone(assistantId, e.stage, e.detail);
                break;
              case "sources":
                patchMessage(assistantId, { chunks: e.chunks, metadata: { ...(e.metadata || {}) } });
                break;
              case "chart":
                appendChartToMessage(assistantId, e.chart);
                break;
              case "chunk":
                if (firstChunk) {
                  // Answer is starting — freeze the "thought for Ns" timer and
                  // clear the live spinner.
                  firstChunk = false;
                  patchMessage(assistantId, {
                    thoughtMs: Date.now() - startedAt, status: undefined,
                  });
                }
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
                if (e.code === "upload_expired") {
                  // Server-side uploads expired (TTL/restart) — drop the stale
                  // chips so the user can re-attach cleanly.
                  useChatStore.setState({ uploads: [] });
                  useThreadStore.getState().saveActive();
                }
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
          mode,
        );
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        patchMessage(assistantId, { error: msg, streaming: false });
        setStreaming(null);
        toast.error(msg);
      }
    },
    [send, patchMessage, appendChunkToMessage, appendStepToMessage, markStepDone,
     appendChartToMessage, setStreaming, setResearchPlan, patchResearchTask],
  );

  const ask = useCallback(
    async (question: string) => {
      if (!question.trim()) return;

      // Ensure a thread exists to own this conversation.
      const threads = useThreadStore.getState();
      if (!threads.activeId) threads.createChat("New chat");

      // The mode toggle (Chat / Deep Research) decides which pipeline runs.
      const mode = useSettingsStore.getState().mode;

      const history = historyBefore(useChatStore.getState().messages.length);
      appendMessage({ role: "user", content: question });
      const assistantId = appendMessage({ role: "assistant", content: "", streaming: true, startedAt: Date.now() });
      useThreadStore.getState().saveActive();   // capture the user turn immediately

      await runStream(question, assistantId, history, mode);
    },
    [appendMessage, historyBefore, runStream],
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
      const assistantId = appendMessage({ role: "assistant", content: "", streaming: true, startedAt: Date.now() });

      await runStream(question, assistantId, history, useSettingsStore.getState().mode);
    },
    [appendMessage, historyBefore, runStream],
  );

  return { ask, regenerate };
}
