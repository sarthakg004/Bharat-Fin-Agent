import { create } from "zustand";
import type { Chunk, ConfigId, Market, QueryMetadata } from "@/lib/api";

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

  appendMessage: (m: Omit<ChatMessage, "id" | "createdAt">) => string;
  patchMessage: (id: string, patch: Partial<ChatMessage>) => void;
  appendChunkToMessage: (id: string, text: string) => void;
  clear: () => void;
  setHighlight: (id: number | null) => void;
  setStreaming: (id: string | null) => void;
}

function uid(): string {
  return Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  highlightedChunkId: null,
  streamingId: null,

  appendMessage: (m) => {
    const id = uid();
    set((s) => ({
      messages: [...s.messages, { ...m, id, createdAt: Date.now() }],
    }));
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
  clear: () => set({ messages: [], highlightedChunkId: null, streamingId: null }),
  setHighlight: (id) => set({ highlightedChunkId: id }),
  setStreaming: (id) => set({ streamingId: id }),
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
