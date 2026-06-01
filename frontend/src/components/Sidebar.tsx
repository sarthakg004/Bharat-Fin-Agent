import { useRef } from "react";
import { motion } from "framer-motion";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Clock, Info, Trash2 } from "lucide-react";
import toast from "react-hot-toast";

import { api } from "@/lib/api";
import { useConfigStore } from "@/store/configStore";
import { useChatStore } from "@/store/chatStore";
import { CompanyChips } from "./CompanyChips";
import { cls, timeAgo } from "@/lib/utils";

const CONFIG_DESCRIPTIONS: Record<string, string> = {
  naive: "Embed → retrieve → stuff → generate.",
  agentic: "Plan → hybrid retrieve → grade → rewrite → synthesize → critic.",
};

interface Props {
  // Reserved for future "rerun this question as a fresh query" affordance;
  // history clicks currently load the stored chat instead.
  onAsk?: (q: string) => void;
}

export function Sidebar({ onAsk: _onAsk }: Props) {
  const { config, setConfig } = useConfigStore();
  const history = useQuery({
    queryKey: ["history"],
    queryFn: () => api.history(10),
    refetchInterval: 30_000,
    refetchOnWindowFocus: false,
  });

  return (
    <motion.aside
      // Width is controlled by the wrapper in App.tsx so it can be resized;
      // on mobile we let the bottom-sheet container drive layout instead.
      className="flex h-full w-full shrink-0 flex-col border-r border-border-default bg-bg-base"
      initial={false}
    >
      {/* Mode selector */}
      <section className="border-b border-border-subtle p-4">
        <SectionLabel>Mode</SectionLabel>
        <div className="mt-2 flex flex-col gap-1">
          {(["naive", "agentic"] as const).map((id) => {
            const active = config === id;
            return (
              <button
                key={id}
                onClick={() => setConfig(id)}
                className={cls(
                  "group flex items-center justify-between border px-3 py-2 text-left transition-colors",
                  active
                    ? "border-accent bg-accent-dim text-accent"
                    : "border-border-subtle text-text-secondary hover:border-border hover:bg-bg-hover hover:text-text-primary",
                )}
              >
                <span className="flex items-center gap-2 font-ui text-[13px]">
                  <span
                    className={cls(
                      "inline-block h-[7px] w-[7px] rounded-full",
                      active ? "bg-accent" : "bg-border-strong",
                    )}
                  />
                  {id === "naive" ? "Naive RAG" : "Agentic RAG"}
                </span>
                <span
                  className="opacity-0 transition-opacity group-hover:opacity-100"
                  title={CONFIG_DESCRIPTIONS[id]}
                >
                  <Info size={12} className="text-text-muted" />
                </span>
              </button>
            );
          })}
        </div>
        <p className="mt-2 font-mono text-[10px] leading-relaxed text-text-muted">
          {CONFIG_DESCRIPTIONS[config]}
        </p>
      </section>

      {/* Company filters */}
      <section className="border-b border-border-subtle p-4">
        <CompanyChips />
      </section>

      {/* History */}
      <HistorySection
        items={history.data?.items ?? []}
        isLoading={history.isLoading}
      />

      <ClearConversationButton />
    </motion.aside>
  );
}

function HistorySection({
  items,
  isLoading,
}: {
  items: { id: number; question: string; config: string; market: string; created_at: string }[];
  isLoading: boolean;
}) {
  const queryClient = useQueryClient();
  const loadHistoryItem = useChatStore((s) => s.loadHistoryItem);
  const startNewChat = useChatStore((s) => s.startNewChat);
  // If the user rapidly clicks several history items, only the most recent
  // request should win. We ignore any in-flight load whose token has been
  // superseded by a newer click.
  const loadTokenRef = useRef(0);

  async function handleLoad(id: number) {
    const token = ++loadTokenRef.current;
    try {
      const item = await api.historyItem(id);
      if (loadTokenRef.current !== token) return;
      loadHistoryItem(item);
    } catch (e) {
      if (loadTokenRef.current !== token) return;
      toast.error(`Couldn't load: ${(e as Error).message}`);
    }
  }

  async function handleDelete(id: number, e: React.MouseEvent) {
    e.stopPropagation();
    // If we're currently viewing this item, deleting it should also clear
    // the chat — otherwise the user is left looking at content that no
    // longer exists in history.
    const viewingThis = useChatStore.getState().currentHistoryId === id;
    try {
      await api.deleteHistoryItem(id);
      if (viewingThis) startNewChat();
      toast.success("Deleted.");
      queryClient.invalidateQueries({ queryKey: ["history"] });
    } catch (err) {
      toast.error(`Delete failed: ${(err as Error).message}`);
    }
  }

  async function handleClearAll() {
    if (items.length === 0) return;
    if (!window.confirm(`Delete all ${items.length} history items?`)) return;
    try {
      const res = await api.clearHistory();
      // Clearing all history also wipes whatever historical chat we're on.
      if (useChatStore.getState().currentHistoryId != null) startNewChat();
      toast.success(`Cleared ${res.deleted} item(s).`);
      queryClient.invalidateQueries({ queryKey: ["history"] });
    } catch (err) {
      toast.error(`Clear failed: ${(err as Error).message}`);
    }
  }

  return (
    <section className="flex flex-1 flex-col overflow-hidden">
      <div className="flex items-center justify-between border-b border-border-subtle px-4 py-2">
        <SectionLabel>
          <span className="inline-flex items-center gap-1.5">
            <Clock size={11} /> History
          </span>
        </SectionLabel>
        {items.length > 0 && (
          <button
            onClick={handleClearAll}
            className="font-mono text-[10px] uppercase tracking-wider text-text-muted hover:text-err"
            title="Delete every history item"
          >
            clear all
          </button>
        )}
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        <div className="flex flex-col gap-px">
          {isLoading && (
            <>
              <div className="h-[28px] skeleton" />
              <div className="h-[28px] skeleton" />
            </>
          )}
          {!isLoading && items.length === 0 && (
            <span className="font-mono text-[10px] text-text-muted">
              No queries yet.
            </span>
          )}
          {items.map((h) => (
            <div
              key={h.id}
              role="button"
              tabIndex={0}
              onClick={() => handleLoad(h.id)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  handleLoad(h.id);
                }
              }}
              className="group flex cursor-pointer flex-col gap-px border-b border-border-subtle px-2 py-2 text-left transition-colors hover:bg-bg-hover"
              title={`Load: ${h.question}`}
            >
              <div className="flex items-start gap-2">
                <span className="line-clamp-1 flex-1 font-ui text-[12px] text-text-primary">
                  {h.question}
                </span>
                <button
                  onClick={(e) => handleDelete(h.id, e)}
                  className="shrink-0 text-text-muted opacity-0 transition-opacity hover:text-err group-hover:opacity-100"
                  title="Delete this history item"
                  aria-label="Delete"
                >
                  <Trash2 size={11} />
                </button>
              </div>
              <span className="flex items-center justify-between font-mono text-[9px] uppercase tracking-wider text-text-muted">
                <span>{h.config} · {h.market}</span>
                <span>{timeAgo(h.created_at)}</span>
              </span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function ClearConversationButton() {
  const clear = useChatStore((s) => s.clear);
  return (
    <button
      onClick={clear}
      className="border-t border-border-subtle px-4 py-3 text-left font-mono text-[11px] uppercase tracking-wider text-text-secondary transition-colors hover:bg-bg-hover hover:text-text-primary"
    >
      Clear conversation
    </button>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-text-secondary">
      {children}
    </span>
  );
}
