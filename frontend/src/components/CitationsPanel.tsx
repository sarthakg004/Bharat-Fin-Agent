import { useMemo } from "react";
import { motion } from "framer-motion";
import { PanelRightClose } from "lucide-react";

import type { Chunk } from "@/lib/api";
import { selectLastAssistant, useChatStore } from "@/store/chatStore";
import { useConfigStore } from "@/store/configStore";
import { ChunkCard } from "./ChunkCard";

/** Evidence grouped by source, most authoritative first. Each card keeps its
 * own [N], which is the number the answer cites. */
const SECTIONS: { kinds: (Chunk["kind"] | undefined)[]; label: string; color: string }[] = [
  { kinds: ["xbrl", "calc"],   label: "SEC XBRL figures", color: "var(--info)" },
  { kinds: ["text", undefined], label: "Filings",         color: "var(--text-secondary)" },
  { kinds: ["edgar"],          label: "EDGAR search",    color: "var(--chart-3)" },
  { kinds: ["web"],            label: "Web search",      color: "var(--warning)" },
  { kinds: ["market"],         label: "Market data",     color: "var(--chart-2)" },
];

export function CitationsPanel() {
  const last = useChatStore((s) => selectLastAssistant(s));
  const toggleCitations = useConfigStore((s) => s.toggleCitations);

  const chunks = last?.chunks ?? [];

  const sections = useMemo(
    () =>
      SECTIONS.map((s) => ({
        ...s,
        chunks: chunks.filter((c) => s.kinds.includes(c.kind)),
      })).filter((s) => s.chunks.length > 0),
    [chunks],
  );

  const empty = chunks.length === 0;

  const meta = last?.metadata;

  return (
    <motion.aside
      // Width is controlled by the wrapper in App.tsx so it can be resized.
      className="flex h-full w-full shrink-0 flex-col border-l border-border-default bg-bg-base"
      initial={false}
    >
      <header className="flex items-center justify-between border-b border-border-subtle px-4 py-3">
        <span className="font-mono text-[11px] uppercase tracking-wider text-text-secondary">
          Sources{chunks.length > 0 && (
            <span className="text-text-primary"> · {chunks.length} chunks</span>
          )}
        </span>
        <button
          onClick={toggleCitations}
          className="text-text-muted transition-colors hover:text-text-primary"
          aria-label="Close panel"
        >
          <PanelRightClose size={14} />
        </button>
      </header>

      <div className="flex-1 overflow-y-auto p-4">
        {empty ? (
          <EmptySources />
        ) : (
          <div className="flex flex-col gap-5">
            {sections.map((s) => (
              <section key={s.label}>
                <SectionHeader color={s.color} label={s.label} count={s.chunks.length} />
                <div className="flex flex-col gap-3">
                  {s.chunks.map((c, i) => (
                    <ChunkCard key={c.id} chunk={c} index={i} accent={s.color} />
                  ))}
                </div>
              </section>
            ))}
          </div>
        )}
      </div>

      {/* What the run did */}
      {!empty && meta && (
        <div className="border-t border-border-subtle px-4 py-3">
          <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-text-secondary">
            Run trace
          </span>
          <div className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 font-mono text-[10px] text-text-muted">
            {meta.sub_queries?.length ? (
              <>
                <span className="text-text-secondary">Sub-queries</span>
                <span className="text-text-primary">{meta.sub_queries.length}</span>
              </>
            ) : null}
            {meta.recoveries != null && (
              <>
                <span className="text-text-secondary">Recovery passes</span>
                <span className="text-text-primary">{meta.recoveries}</span>
              </>
            )}
            {typeof meta.support_score === "number" && (
              <>
                <span className="text-text-secondary">Claims supported</span>
                <span className="text-text-primary">{Math.round(meta.support_score * 100)}%</span>
              </>
            )}
          </div>
        </div>
      )}
    </motion.aside>
  );
}

function SectionHeader({ color, label, count }: { color: string; label: string; count: number }) {
  return (
    <div className="mb-2 flex items-center gap-2">
      <span
        className="inline-block h-[7px] w-[7px] shrink-0"
        style={{ background: color }}
        aria-hidden
      />
      <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-text-secondary">
        {label}
      </span>
      <span className="font-mono text-[10px] text-text-muted">{count}</span>
      <span className="h-px flex-1 bg-border-subtle" aria-hidden />
    </div>
  );
}

function EmptySources() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 text-center">
      <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-text-muted">
        no sources yet
      </span>
      <span className="font-ui text-[12px] text-text-muted">
        The evidence behind each answer appears here.
      </span>
    </div>
  );
}
