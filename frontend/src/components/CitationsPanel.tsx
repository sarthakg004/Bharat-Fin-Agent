import { useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { PanelRightClose, FileText, X } from "lucide-react";

import type { Chunk } from "@/lib/api";
import { selectLastAssistant, useChatStore } from "@/store/chatStore";
import { useThreadStore } from "@/store/threadStore";
import { useConfigStore } from "@/store/configStore";
import { ChunkCard } from "./ChunkCard";

/** Provenance lanes, ordered by authority. Each source kind renders as its own
 * titled section so uploaded documents, filings, EDGAR and web hits never blur
 * together. Colors reuse the app's semantic tokens: green = your material,
 * blue = deterministic figures, neutral = corpus filings, amber = external. */
const SECTIONS: { kinds: (Chunk["kind"] | undefined)[]; label: string; color: string }[] = [
  { kinds: ["uploaded"],       label: "Your documents",  color: "var(--accent)" },
  { kinds: ["xbrl", "calc"],   label: "SEC XBRL figures", color: "var(--info)" },
  { kinds: ["text", undefined], label: "Filings",         color: "var(--text-secondary)" },
  { kinds: ["table"],          label: "Filing tables",   color: "var(--text-secondary)" },
  { kinds: ["edgar"],          label: "EDGAR search",    color: "var(--chart-3)" },
  { kinds: ["web"],            label: "Web search",      color: "var(--warning)" },
  { kinds: ["market"],         label: "Market data",     color: "var(--chart-2)" },
];

export function CitationsPanel() {
  const last = useChatStore((s) => selectLastAssistant(s));
  const uploads = useChatStore((s) => s.uploads);
  const removeUpload = useChatStore((s) => s.removeUpload);
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

  const agenticMeta = useMemo(() => last?.metadata?.agentic ?? last?.metadata, [last]);

  function detach(id: string) {
    removeUpload(id);
    useThreadStore.getState().saveActive();
  }

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
        {/* Attached documents — live for the whole session, removable anytime. */}
        {uploads.length > 0 && (
          <section className="mb-5">
            <SectionHeader color="var(--accent)" label="Attached documents" count={uploads.length} />
            <div className="flex flex-col gap-1.5">
              <AnimatePresence initial={false}>
                {uploads.map((u) => (
                  <motion.div
                    key={u.id}
                    initial={{ opacity: 0, y: 4 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, x: 12 }}
                    className="flex items-center gap-2 border border-border-subtle bg-bg-surface px-3 py-2"
                    style={{ borderLeft: "2px solid var(--accent)" }}
                  >
                    <FileText size={13} className="shrink-0 text-accent" />
                    <div className="min-w-0 flex-1">
                      <div className="truncate font-ui text-[12.5px] text-text-primary" title={u.name}>
                        {u.name}
                      </div>
                      <div className="font-mono text-[10px] text-text-muted">
                        {u.pages} pages · {u.chunks} chunks · this session
                      </div>
                    </div>
                    <button
                      onClick={() => detach(u.id)}
                      className="shrink-0 text-text-muted transition-colors hover:text-err"
                      title={`Remove ${u.name} from this chat`}
                      aria-label={`Remove ${u.name}`}
                    >
                      <X size={13} />
                    </button>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          </section>
        )}

        {empty ? (
          <EmptySources hasUploads={uploads.length > 0} />
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

      {/* Agentic metadata strip */}
      {!empty && agenticMeta && (
        <div className="border-t border-border-subtle px-4 py-3">
          <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-text-secondary">
            Run trace
          </span>
          <div className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 font-mono text-[10px] text-text-muted">
            {agenticMeta.sub_queries?.length ? (
              <>
                <span className="text-text-secondary">Sub-queries</span>
                <span className="text-text-primary">{agenticMeta.sub_queries.length}</span>
              </>
            ) : null}
            {agenticMeta.avg_grade != null && (
              <>
                <span className="text-text-secondary">Grader avg</span>
                <span className="text-text-primary">{agenticMeta.avg_grade}</span>
              </>
            )}
            {agenticMeta.rewrite_iterations != null && (
              <>
                <span className="text-text-secondary">Rewrites</span>
                <span className="text-text-primary">{agenticMeta.rewrite_iterations}</span>
              </>
            )}
            {agenticMeta.critic_iterations != null && (
              <>
                <span className="text-text-secondary">Critic loops</span>
                <span className="text-text-primary">{agenticMeta.critic_iterations}</span>
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

function EmptySources({ hasUploads }: { hasUploads: boolean }) {
  return (
    <div className={hasUploads
      ? "flex flex-col items-center gap-2 pt-8 text-center"
      : "flex h-full flex-col items-center justify-center gap-2 text-center"}
    >
      <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-text-muted">
        no sources yet
      </span>
      <span className="font-ui text-[12px] text-text-muted">
        {hasUploads
          ? "Ask a question — passages from your documents and other sources will appear here."
          : "Retrieved chunks will appear here after your first query."}
      </span>
    </div>
  );
}
