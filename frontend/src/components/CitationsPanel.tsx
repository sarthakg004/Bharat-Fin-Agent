import { useMemo } from "react";
import { motion } from "framer-motion";
import { PanelRightClose } from "lucide-react";

import { selectLastAssistant, useChatStore } from "@/store/chatStore";
import { useConfigStore } from "@/store/configStore";
import { ChunkCard } from "./ChunkCard";
import { MetricsBadge } from "./MetricsBadge";

export function CitationsPanel() {
  const last = useChatStore((s) => selectLastAssistant(s));
  const toggleCitations = useConfigStore((s) => s.toggleCitations);

  const chunks = last?.chunks ?? [];
  const ragas = last?.ragas;

  const empty = chunks.length === 0;

  const agenticMeta = useMemo(() => last?.metadata?.agentic ?? last?.metadata, [last]);

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
          <div className="flex flex-col gap-3">
            {chunks.map((c, i) => (
              <ChunkCard key={c.id} chunk={c} index={i} />
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

      {/* RAGAS row */}
      <div className="flex flex-wrap gap-1 border-t border-border-subtle px-4 py-3">
        <MetricsBadge label="Faithfulness" score={ragas?.faithfulness} />
        <MetricsBadge label="Relevancy" score={ragas?.answer_relevancy} />
        <MetricsBadge label="Precision" score={ragas?.context_precision} />
        <MetricsBadge label="Recall" score={ragas?.context_recall} />
      </div>
    </motion.aside>
  );
}

function EmptySources() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 text-center">
      <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-text-muted">
        no sources yet
      </span>
      <span className="font-ui text-[12px] text-text-muted">
        Retrieved chunks will appear here after your first query.
      </span>
    </div>
  );
}
