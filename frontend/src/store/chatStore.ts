import { create } from "zustand";
import type { ChartSpec, Chunk, Market, PersistedMessage, QueryMetadata } from "@/lib/api";

export type MessageRole = "user" | "assistant";

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  status?: { stage: string; label: string };
  chunks?: Chunk[];
  charts?: ChartSpec[];
  metadata?: QueryMetadata;
  market?: Market;
  ragas?: RagasScores;
  error?: string;
  streaming?: boolean;
  createdAt: number;
}

export interface RagasScores {
  faithfulness?: number;
  answer_relevancy?: number;
  context_precision?: number;
  context_recall?: number;
}

interface ChatState {
  messages: ChatMessage[];
  highlightedChunkId: number | null;
  streamingId: string | null;
  /** Backend chat thread the current `messages` belong to (null = fresh chat). */
  activeChatId: number | null;

  appendMessage: (m: Omit<ChatMessage, "id" | "createdAt">) => string;
  patchMessage: (id: string, patch: Partial<ChatMessage>) => void;
  appendChunkToMessage: (id: string, text: string) => void;
  appendChartToMessage: (id: string, chart: ChartSpec) => void;
  clear: () => void;
  setHighlight: (id: number | null) => void;
  setStreaming: (id: string | null) => void;
  startNewChat: () => void;
  setActiveChatId: (id: number | null) => void;
  loadMessagesFor: (chatId: number, persisted: PersistedMessage[]) => void;
}

function uid(): string {
  return Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  highlightedChunkId: null,
  streamingId: null,
  activeChatId: null,

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
  appendChartToMessage: (id, chart) =>
    set((s) => ({
      messages: s.messages.map((msg) =>
        msg.id === id ? { ...msg, charts: [...(msg.charts || []), chart] } : msg,
      ),
    })),
  clear: () => set({
    messages: [], highlightedChunkId: null, streamingId: null, activeChatId: null,
  }),
  setHighlight: (id) => set({ highlightedChunkId: id }),
  setStreaming: (id) => set({ streamingId: id }),
  startNewChat: () => set({
    messages: [], highlightedChunkId: null, streamingId: null, activeChatId: null,
  }),
  setActiveChatId: (id) => set({ activeChatId: id }),

  loadMessagesFor: (chatId, persisted) => {
    const now = Date.now();
    const messages: ChatMessage[] = persisted.map((m, i) => ({
      id: uid(),
      role: m.role,
      content: m.content,
      chunks: m.chunks,
      charts: m.charts,
      metadata: m.metadata,
      streaming: false,
      createdAt: now + i,
    }));
    set({
      messages,
      highlightedChunkId: null,
      streamingId: null,
      activeChatId: chatId,
    });
  },
}));

/** Convenience selector — last assistant message. */
export function selectLastAssistant(state: ChatState): ChatMessage | undefined {
  for (let i = state.messages.length - 1; i >= 0; i--) {
    if (state.messages[i].role === "assistant") return state.messages[i];
  }
  return undefined;
}
