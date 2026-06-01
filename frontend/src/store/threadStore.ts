/**
 * Thread store — owns the list of chat threads and the active id.
 *
 * The chat-message state (`chatStore`) is the active thread's contents. When
 * the user switches threads we fetch its messages from the backend and pour
 * them into chatStore via `loadMessagesInto()`.
 */

import { create } from "zustand";
import { api, type ChatSummary, type Market } from "@/lib/api";
import { useChatStore } from "./chatStore";

interface ThreadState {
  threads: ChatSummary[];
  activeId: number | null;
  loading: boolean;
  error: string | null;

  refresh: () => Promise<void>;
  createChat: (title?: string, market?: Market) => Promise<ChatSummary>;
  switchTo: (id: number) => Promise<void>;
  renameChat: (id: number, title: string) => Promise<void>;
  deleteChat: (id: number) => Promise<void>;
  setActive: (id: number | null) => void;
  /** Insert / update one summary locally (used after queries that auto-create). */
  upsertSummary: (chat: ChatSummary) => void;
}

export const useThreadStore = create<ThreadState>((set, get) => ({
  threads: [],
  activeId: null,
  loading: false,
  error: null,

  refresh: async () => {
    set({ loading: true, error: null });
    try {
      const { chats } = await api.listChats();
      set({ threads: chats, loading: false });
    } catch (e) {
      set({ error: (e as Error).message, loading: false });
    }
  },

  createChat: async (title = "New chat", market = "us") => {
    const chat = await api.createChat(title, market);
    set((s) => ({ threads: [chat, ...s.threads], activeId: chat.id }));
    useChatStore.getState().startNewChat();
    useChatStore.getState().setActiveChatId(chat.id);
    return chat;
  },

  switchTo: async (id) => {
    set({ activeId: id });
    try {
      const { messages } = await api.getChat(id);
      useChatStore.getState().loadMessagesFor(id, messages);
    } catch (e) {
      // Likely 404 — refresh the list to drop the missing thread and reset.
      set({ error: (e as Error).message });
      await get().refresh();
      useChatStore.getState().startNewChat();
      set({ activeId: null });
    }
  },

  renameChat: async (id, title) => {
    const updated = await api.renameChat(id, title);
    set((s) => ({
      threads: s.threads.map((t) => (t.id === id ? { ...t, ...updated } : t)),
    }));
  },

  deleteChat: async (id) => {
    await api.deleteChat(id);
    set((s) => {
      const threads = s.threads.filter((t) => t.id !== id);
      // If we deleted the active chat, drop active.
      const activeId = s.activeId === id ? null : s.activeId;
      return { threads, activeId };
    });
    if (useChatStore.getState().activeChatId === id) {
      useChatStore.getState().startNewChat();
    }
  },

  setActive: (id) => set({ activeId: id }),

  upsertSummary: (chat) => set((s) => {
    const without = s.threads.filter((t) => t.id !== chat.id);
    return { threads: [chat, ...without] };
  }),
}));
