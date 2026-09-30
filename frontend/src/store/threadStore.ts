/**
 * Thread store — client-side, per-session chat threads.
 *
 * The app is stateless on the server: threads + their messages live entirely in
 * the browser, persisted to `sessionStorage` so they survive a reload but are
 * cleared when the tab closes ("history kept as long as you're on the site").
 *
 * `chatStore` holds the *active* thread's live/streaming messages for rendering;
 * `saveActive()` writes them back into the owning thread here.
 */

import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

import { useChatStore, type ChatMessage } from "./chatStore";

export interface Thread {
  id: string;
  title: string;
  messages: ChatMessage[];
  created_at: number;
  updated_at: number;
}

function uid(): string {
  return Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
}

function deriveTitle(current: string, messages: ChatMessage[]): string {
  if (current && current !== "New chat") return current;
  const firstUser = messages.find((m) => m.role === "user");
  if (!firstUser) return current;
  const t = firstUser.content.trim().slice(0, 48);
  return t + (firstUser.content.trim().length > 48 ? "…" : "");
}

interface ThreadState {
  threads: Thread[];
  activeId: string | null;

  createChat: (title?: string) => Thread;
  switchTo: (id: string) => void;
  renameChat: (id: string, title: string) => void;
  deleteChat: (id: string) => void;
  setActive: (id: string | null) => void;
  /** Persist the active thread's current chatStore messages back into the list. */
  saveActive: () => void;
}

export const useThreadStore = create<ThreadState>()(
  persist(
    (set, get) => ({
      threads: [],
      activeId: null,

      createChat: (title = "New chat") => {
        const thread: Thread = {
          id: uid(), title, messages: [],
          created_at: Date.now(), updated_at: Date.now(),
        };
        set((s) => ({ threads: [thread, ...s.threads], activeId: thread.id }));
        useChatStore.getState().startNewChat();
        useChatStore.getState().setActiveChatId(thread.id);
        return thread;
      },

      switchTo: (id) => {
        const t = get().threads.find((x) => x.id === id);
        if (!t) return;
        set({ activeId: id });
        useChatStore.getState().loadMessages(id, t.messages);
      },

      renameChat: (id, title) =>
        set((s) => ({
          threads: s.threads.map((t) =>
            t.id === id ? { ...t, title, updated_at: Date.now() } : t),
        })),

      deleteChat: (id) =>
        set((s) => {
          const threads = s.threads.filter((t) => t.id !== id);
          let activeId = s.activeId;
          if (s.activeId === id) {
            const next = threads[0];
            activeId = next?.id ?? null;
            if (next) useChatStore.getState().loadMessages(next.id, next.messages);
            else useChatStore.getState().startNewChat();
          }
          return { threads, activeId };
        }),

      setActive: (id) => set({ activeId: id }),

      saveActive: () => {
        const { activeChatId, messages } = useChatStore.getState();
        if (!activeChatId) return;
        set((s) => ({
          threads: s.threads.map((t) =>
            t.id === activeChatId
              ? { ...t, messages, updated_at: Date.now(), title: deriveTitle(t.title, messages) }
              : t),
        }));
      },
    }),
    {
      name: "finagent.threads",
      // sessionStorage → cleared on tab close (per-session memory).
      storage: createJSONStorage(() => sessionStorage),
    },
  ),
);
