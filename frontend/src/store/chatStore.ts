import { create } from "zustand";
import type { Chunk, ConfigId, HistoryItemFull, Market, QueryMetadata } from "@/lib/api";

export type MessageRole = "user" | "assistant";

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  status?: { stage: string; label: string };
  chunks?: Chunk[];
  metadata?: QueryMetadata;
  config?: ConfigId;
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
  /** ID of the history row currently loaded in the chat (null = fresh chat). */
  currentHistoryId: number | null;

  appendMessage: (m: Omit<ChatMessage, "id" | "createdAt">) => string;
  patchMessage: (id: string, patch: Partial<ChatMessage>) => void;
  appendChunkToMessage: (id: string, text: string) => void;
  clear: () => void;
  setHighlight: (id: number | null) => void;
  setStreaming: (id: string | null) => void;
  loadHistoryItem: (item: HistoryItemFull) => void;
  startNewChat: () => void;
}

function uid(): string {
  return Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  highlightedChunkId: null,
  streamingId: null,
  currentHistoryId: null,

  appendMessage: (m) => {
    const id = uid();
    set((s) => ({
      messages: [...s.messages, { ...m, id, createdAt: Date.now() }],
    }));
    return id;
  },
  startNewChat: () =>
    set({
      messages: [],
      highlightedChunkId: null,
      streamingId: null,
      currentHistoryId: null,
    }),
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
  clear: () =>
    set({
      messages: [],
      highlightedChunkId: null,
      streamingId: null,
      currentHistoryId: null,
    }),
  setHighlight: (id) => set({ highlightedChunkId: id }),
  setStreaming: (id) => set({ streamingId: id }),
  loadHistoryItem: (item) => {
    // Replace the entire view with the historical question + stored answer.
    // Streaming is force-cleared so any in-flight query is visually abandoned;
    // the underlying SSE call is aborted by the hook layer (App.tsx).
    const now = Date.now();
    set({
      highlightedChunkId: null,
      streamingId: null,
      currentHistoryId: item.id,
      messages: [
        {
          id: uid(),
          role: "user",
          content: item.question,
          market: item.market as Market,
          config: item.config as ConfigId,
          createdAt: now,
        },
        {
          id: uid(),
          role: "assistant",
          content: item.answer,
          chunks: item.chunks,
          metadata: item.metadata,
          market: item.market as Market,
          config: item.config as ConfigId,
          streaming: false,
          createdAt: now + 1,
        },
      ],
    });
  },
}));

/**
 * Convenience selector — last assistant message (the one citation cards
 * scroll into view for, the one CitationsPanel reflects).
 */
export function selectLastAssistant(state: ChatState): ChatMessage | undefined {
  for (let i = state.messages.length - 1; i >= 0; i--) {
    if (state.messages[i].role === "assistant") return state.messages[i];
  }
  return undefined;
}
