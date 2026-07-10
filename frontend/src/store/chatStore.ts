import { create } from "zustand";
import type { ChartSpec, Chunk, QueryMetadata } from "@/lib/api";

export type MessageRole = "user" | "assistant";

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  status?: { stage: string; label: string; index?: number; total?: number };
  /** Accumulated "thinking" trace — one entry per graph node as it runs.
   * `done` flips when the node finishes; `detail` is its short outcome line
   * ("12 passages", "2 exact figures from SEC XBRL"). */
  steps?: {
    stage: string; label: string; index?: number; total?: number;
    done?: boolean; detail?: string | null;
  }[];
  /** Furthest pipeline position reached (0..total) — drives the progress bar. */
  progressIndex?: number;
  progressTotal?: number;
  /** Epoch ms the assistant turn started — drives the live elapsed timer/ETA. */
  startedAt?: number;
  /** Wall-clock ms the agent spent thinking before the answer streamed. */
  thoughtMs?: number;
  chunks?: Chunk[];
  charts?: ChartSpec[];
  /** Present on Deep Research messages — drives the research timeline. */
  research?: ResearchState;
  metadata?: QueryMetadata;
  ragas?: RagasScores;
  error?: string;
  streaming?: boolean;
  createdAt: number;
}

/** One research task's live status in the Deep Research timeline. */
export interface ResearchTask {
  id: string;
  label: string;
  status: "pending" | "running" | "done" | "failed";
  detail?: string;
  /** First few hundred chars of the specialist's finding — the expandable
   * reasoning summary in the timeline. */
  summary?: string;
  confidence?: number | null;
}

/** Deep Research state attached to an assistant message. */
export interface ResearchState {
  company?: string;
  objective?: string;
  tasks: ResearchTask[];
}

export interface RagasScores {
  faithfulness?: number;
  answer_relevancy?: number;
  context_precision?: number;
  context_recall?: number;
}

/** A document attached to the conversation via POST /api/upload. Ephemeral:
 * the server keeps it ~1 hour; it clears with the chat. */
export interface UploadedDoc {
  id: string;
  name: string;
  pages: number;
  chunks: number;
}

interface ChatState {
  messages: ChatMessage[];
  highlightedChunkId: number | null;
  streamingId: string | null;
  /** Client thread id the current `messages` belong to (null = fresh chat). */
  activeChatId: string | null;
  /** Documents attached to the active conversation. */
  uploads: UploadedDoc[];

  addUpload: (doc: UploadedDoc) => void;
  removeUpload: (id: string) => void;

  appendMessage: (m: Omit<ChatMessage, "id" | "createdAt">) => string;
  patchMessage: (id: string, patch: Partial<ChatMessage>) => void;
  appendChunkToMessage: (id: string, text: string) => void;
  /** Append a thinking step (skips consecutive duplicate labels). */
  appendStepToMessage: (
    id: string,
    step: { stage: string; label: string; index?: number; total?: number },
  ) => void;
  /** Mark the latest unfinished occurrence of a stage as done (+ outcome). */
  markStepDone: (id: string, stage: string, detail?: string | null) => void;
  appendChartToMessage: (id: string, chart: ChartSpec) => void;
  /** Attach the research plan (Deep Research mode). */
  setResearchPlan: (id: string, research: ResearchState) => void;
  /** Update one research task's status/detail as agent events stream in. */
  patchResearchTask: (id: string, taskId: string, patch: Partial<ResearchTask>) => void;
  clear: () => void;
  setHighlight: (id: number | null) => void;
  setStreaming: (id: string | null) => void;
  startNewChat: () => void;
  setActiveChatId: (id: string | null) => void;
  /** Replace the active conversation with a thread's stored messages (+ its
   * attached documents, so uploads survive thread switches and reloads). */
  loadMessages: (chatId: string, messages: ChatMessage[], uploads?: UploadedDoc[]) => void;
  /** Drop the trailing assistant message (used by Retry). */
  dropLastAssistant: () => void;
}

function uid(): string {
  return Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  highlightedChunkId: null,
  streamingId: null,
  activeChatId: null,
  uploads: [],

  addUpload: (doc) => set((s) => ({ uploads: [...s.uploads, doc] })),
  removeUpload: (id) =>
    set((s) => ({ uploads: s.uploads.filter((u) => u.id !== id) })),

  appendMessage: (m) => {
    const id = uid();
    set((s) => ({ messages: [...s.messages, { ...m, id, createdAt: Date.now() }] }));
    return id;
  },
  patchMessage: (id, patch) =>
    set((s) => ({
      messages: s.messages.map((msg) => (msg.id === id ? { ...msg, ...patch } : msg)),
    })),
  appendChunkToMessage: (id, text) =>
    set((s) => ({
      messages: s.messages.map((msg) =>
        msg.id === id ? { ...msg, content: (msg.content || "") + text } : msg,
      ),
    })),
  appendStepToMessage: (id, step) =>
    set((s) => ({
      messages: s.messages.map((msg) => {
        if (msg.id !== id) return msg;
        const steps = msg.steps || [];
        // Progress advances MONOTONICALLY — a corrective loop revisiting an
        // earlier node must not make the bar jump backwards.
        const progressIndex = Math.max(msg.progressIndex ?? 0, step.index ?? 0);
        const progressTotal = step.total ?? msg.progressTotal ?? 0;
        // Skip a consecutive duplicate label (e.g. retrieve→grade looping) — the
        // trace list shows distinct activity, but `status`/progress still update.
        if (steps.length && steps[steps.length - 1].label === step.label) {
          return { ...msg, status: step, progressIndex, progressTotal };
        }
        return {
          ...msg, steps: [...steps, step], status: step,
          progressIndex, progressTotal,
        };
      }),
    })),
  markStepDone: (id, stage, detail) =>
    set((s) => ({
      messages: s.messages.map((msg) => {
        if (msg.id !== id || !msg.steps?.length) return msg;
        const steps = [...msg.steps];
        for (let i = steps.length - 1; i >= 0; i--) {
          if (steps[i].stage === stage && !steps[i].done) {
            steps[i] = { ...steps[i], done: true, detail: detail ?? steps[i].detail };
            return { ...msg, steps };
          }
        }
        return msg;
      }),
    })),
  appendChartToMessage: (id, chart) =>
    set((s) => ({
      messages: s.messages.map((msg) =>
        msg.id === id ? { ...msg, charts: [...(msg.charts || []), chart] } : msg,
      ),
    })),
  setResearchPlan: (id, research) =>
    set((s) => ({
      messages: s.messages.map((msg) => (msg.id === id ? { ...msg, research } : msg)),
    })),
  patchResearchTask: (id, taskId, patch) =>
    set((s) => ({
      messages: s.messages.map((msg) => {
        if (msg.id !== id || !msg.research) return msg;
        return {
          ...msg,
          research: {
            ...msg.research,
            tasks: msg.research.tasks.map((t) =>
              t.id === taskId ? { ...t, ...patch } : t),
          },
        };
      }),
    })),
  clear: () => set({
    messages: [], highlightedChunkId: null, streamingId: null, activeChatId: null,
    uploads: [],
  }),
  setHighlight: (id) => set({ highlightedChunkId: id }),
  setStreaming: (id) => set({ streamingId: id }),
  startNewChat: () => set({
    messages: [], highlightedChunkId: null, streamingId: null, activeChatId: null,
    uploads: [],
  }),
  setActiveChatId: (id) => set({ activeChatId: id }),

  loadMessages: (chatId, messages, uploads) => set({
    messages: [...messages],
    highlightedChunkId: null,
    streamingId: null,
    activeChatId: chatId,
    uploads: uploads ? [...uploads] : [],
  }),

  dropLastAssistant: () => set((s) => {
    const idx = [...s.messages].reverse().findIndex((m) => m.role === "assistant");
    if (idx === -1) return {};
    const realIdx = s.messages.length - 1 - idx;
    return { messages: s.messages.slice(0, realIdx) };
  }),
}));

/** Convenience selector — last assistant message. */
export function selectLastAssistant(state: ChatState): ChatMessage | undefined {
  for (let i = state.messages.length - 1; i >= 0; i--) {
    if (state.messages[i].role === "assistant") return state.messages[i];
  }
  return undefined;
}
