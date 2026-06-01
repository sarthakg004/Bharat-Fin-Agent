import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { X, Send, Loader2 } from "lucide-react";

import { streamQuery, type SSEEvent, type Chunk, type QueryMetadata, type ConfigId } from "@/lib/api";
import { useConfigStore } from "@/store/configStore";
import { parseAnswerCitations } from "@/lib/utils";

interface Props {
  open: boolean;
  onClose: () => void;
}

interface RaceState {
  answer: string;
  chunks: Chunk[];
  metadata?: QueryMetadata;
  status?: string;
  done: boolean;
  faster?: boolean;
}

const INITIAL: RaceState = { answer: "", chunks: [], done: false };

export function CompareModal({ open, onClose }: Props) {
  const [question, setQuestion] = useState("");
  const [running, setRunning] = useState(false);
  const [naive, setNaive] = useState<RaceState>(INITIAL);
  const [agentic, setAgentic] = useState<RaceState>(INITIAL);
  const { market } = useConfigStore();
  const naiveT0 = useRef<number>(0);
  const agenticT0 = useRef<number>(0);

  useEffect(() => {
    if (!open) {
      setQuestion("");
      setNaive(INITIAL);
      setAgentic(INITIAL);
      setRunning(false);
    }
  }, [open]);

  function dispatch(target: "naive" | "agentic") {
    return (e: SSEEvent) => {
      const setter = target === "naive" ? setNaive : setAgentic;
      setter((prev) => {
        if (e.type === "status") return { ...prev, status: e.label };
        if (e.type === "sources") return { ...prev, chunks: e.chunks, metadata: e.metadata };
        if (e.type === "chunk") return { ...prev, answer: prev.answer + e.content };
        if (e.type === "metrics") return { ...prev, metadata: { ...(prev.metadata || {}), latency: e.latency } };
        if (e.type === "done") {
          const finishedFirst = target === "naive" ? !agentic.done : !naive.done;
          return { ...prev, done: true, faster: finishedFirst };
        }
        return prev;
      });
    };
  }

  async function race() {
    if (!question.trim()) return;
    setRunning(true);
    setNaive(INITIAL);
    setAgentic(INITIAL);
    naiveT0.current = performance.now();
    agenticT0.current = performance.now();

    const opts = { market, company_filter: null, top_k: 5 };
    const calls: Promise<unknown>[] = [
      streamQuery({ question, config: "naive" as ConfigId, ...opts }, { onEvent: dispatch("naive") }),
      streamQuery({ question, config: "agentic" as ConfigId, ...opts }, { onEvent: dispatch("agentic") }),
    ];

    try { await Promise.allSettled(calls); } finally { setRunning(false); }
  }

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-50 flex items-stretch bg-bg-base/95"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          <motion.div
            className="m-auto flex h-[90vh] w-[min(1400px,95vw)] flex-col border border-border-default bg-bg-base"
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.95, opacity: 0 }}
            transition={{ duration: 0.18 }}
          >
            {/* Header */}
            <div className="flex shrink-0 items-center justify-between border-b border-border-default px-5 py-3">
              <span className="font-display text-[18px] text-text-primary">
                Compare modes
              </span>
              <button
                onClick={onClose}
                className="text-text-muted hover:text-text-primary"
                aria-label="Close"
              >
                <X size={16} />
              </button>
            </div>

            {/* Input */}
            <div className="flex items-center gap-2 border-b border-border-subtle px-5 py-3">
              <input
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) race(); }}
                placeholder="Ask one question; Naive and Agentic answer in parallel."
                className="flex-1 border border-border bg-bg-surface px-3 py-2 font-ui text-[13px] text-text-primary placeholder:text-text-muted focus:outline-none"
              />
              <button
                onClick={race}
                disabled={!question.trim() || running}
                className="flex items-center gap-1.5 border border-accent bg-accent px-3 py-2 font-ui text-[12px] text-bg-base hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-50"
              >
                {running ? <Loader2 size={12} className="animate-spin" /> : <Send size={12} />}
                Race
              </button>
            </div>

            {/* Race columns */}
            <div className="grid flex-1 grid-cols-1 md:grid-cols-2">
              <Column title="Naive RAG" state={naive} />
              <div className="border-t border-border-subtle md:border-t-0 md:border-l">
                <Column title="Agentic RAG" state={agentic} />
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

function Column({ title, state }: { title: string; state: RaceState }) {
  const segments = parseAnswerCitations(state.answer);
  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex items-center justify-between border-b border-border-subtle px-4 py-2">
        <span className="font-mono text-[11px] uppercase tracking-wider text-text-secondary">
          {title}
        </span>
        {state.done && state.faster && (
          <span className="border border-accent px-2 py-[2px] font-mono text-[10px] text-accent">
            faster
          </span>
        )}
        {state.metadata?.latency != null && (
          <span className="font-mono text-[10px] text-text-muted">
            {state.metadata.latency.toFixed(2)}s
          </span>
        )}
      </div>
      <div className="flex-1 overflow-y-auto px-4 py-3">
        {state.status && !state.done && (
          <div className="mb-2 font-mono text-[10px] uppercase tracking-wider text-text-secondary">
            {state.status}
          </div>
        )}
        <p className="whitespace-pre-wrap font-ui text-[13px] leading-relaxed text-text-primary">
          {segments.map((s, i) =>
            s.kind === "text" ? <span key={i}>{s.text}</span> : <span key={i} className="citation-mark">[{s.ids.join(",")}]</span>,
          )}
          {!state.done && state.answer && <span className="streaming-cursor">▌</span>}
        </p>
      </div>
      <div className="border-t border-border-subtle px-4 py-2 font-mono text-[10px] text-text-muted">
        {state.chunks.length > 0 && <span>{state.chunks.length} chunks · </span>}
        {state.metadata?.model && <span>{state.metadata.model}</span>}
      </div>
    </div>
  );
}

