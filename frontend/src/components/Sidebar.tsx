import { motion } from "framer-motion";
import { useQuery } from "@tanstack/react-query";
import { Clock, Info } from "lucide-react";

import { api } from "@/lib/api";
import { useConfigStore } from "@/store/configStore";
import { useChatStore } from "@/store/chatStore";
import { CompanyChips } from "./CompanyChips";
import { cls, timeAgo } from "@/lib/utils";

const CONFIG_DESCRIPTIONS: Record<string, string> = {
  naive: "Embed → retrieve → stuff → generate.",
  agentic: "Plan → hybrid retrieve → grade → rewrite → synthesize → critic.",
};

export function Sidebar({ onAsk }: { onAsk: (q: string) => void }) {
  const { config, setConfig } = useConfigStore();
  const history = useQuery({
    queryKey: ["history"],
    queryFn: () => api.history(10),
    refetchInterval: 30_000,
    refetchOnWindowFocus: false,
  });

  return (
    <motion.aside
      className="flex h-full w-[240px] shrink-0 flex-col border-r border-border-default bg-bg-base"
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
      <section className="flex-1 overflow-y-auto p-4">
        <SectionLabel>
          <span className="inline-flex items-center gap-1.5">
            <Clock size={11} /> History
          </span>
        </SectionLabel>
        <div className="mt-2 flex flex-col gap-px">
          {history.isLoading && (
            <>
              <div className="h-[28px] skeleton" />
              <div className="h-[28px] skeleton" />
            </>
          )}
          {history.data?.items.length === 0 && (
            <span className="font-mono text-[10px] text-text-muted">
              No queries yet.
            </span>
          )}
          {history.data?.items.map((h) => (
            <button
              key={h.id}
              onClick={() => onAsk(h.question)}
              className="group flex flex-col gap-px border-b border-border-subtle px-2 py-2 text-left transition-colors hover:bg-bg-hover"
              title={h.question}
            >
              <span className="line-clamp-1 font-ui text-[12px] text-text-primary">
                {h.question}
              </span>
              <span className="flex items-center justify-between font-mono text-[9px] uppercase tracking-wider text-text-muted">
                <span>{h.config} · {h.market}</span>
                <span>{timeAgo(h.created_at)}</span>
              </span>
            </button>
          ))}
        </div>
      </section>

      <ClearButton />
    </motion.aside>
  );
}

function ClearButton() {
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
