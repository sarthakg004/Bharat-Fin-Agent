import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Check, ChevronRight, ExternalLink, RotateCcw } from "lucide-react";

import type { ChatMessage } from "@/store/chatStore";
import { MarkdownAnswer } from "@/lib/markdown";
import { ChartView } from "@/components/ChartView";

interface BubbleProps {
  msg: ChatMessage;
  /** Set on the last assistant message (when not streaming) to show Retry. */
  onRetry?: () => void;
}

export function MessageBubble({ msg, onRetry }: BubbleProps) {
  if (msg.role === "user") return <UserBubble msg={msg} />;
  return <AssistantBubble msg={msg} onRetry={onRetry} />;
}

function UserBubble({ msg }: { msg: ChatMessage }) {
  return (
    <motion.div
      initial={{ y: 8, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.2 }}
      className="flex justify-end"
    >
      <div className="max-w-[80%] border border-border-default bg-bg-elevated px-4 py-3 font-ui text-[14px] text-text-primary">
        {msg.content}
      </div>
    </motion.div>
  );
}

function AssistantBubble({ msg, onRetry }: BubbleProps) {
  const showStatus = msg.streaming && (!msg.content || msg.content.length < 6);

  return (
    <motion.div
      initial={{ y: 8, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.2 }}
      className="flex flex-col gap-2"
    >
      <ThinkingTrace msg={msg} />

      {msg.error ? (
        <div className="border border-err bg-err-dim px-4 py-3 font-mono text-[12px] text-err">
          {msg.error}
        </div>
      ) : showStatus && !msg.content && !(msg.steps?.length) ? (
        <span className="font-mono text-[12px] text-text-muted">…</span>
      ) : showStatus && !msg.content ? null : (
        // The MarkdownAnswer handles headings / bullets / tables / inline
        // citation chips (`[N]` and `[N, M]`) end-to-end. The streaming
        // cursor is appended outside so the markdown parser doesn't try to
        // interpret it.
        <div className="relative">
          <MarkdownAnswer text={msg.content} />
          {msg.streaming && (
            <span className="streaming-cursor inline-block align-baseline">▌</span>
          )}
        </div>
      )}

      {/* Inline charts produced by the market-data tool lane. They arrive on
          a separate SSE channel and attach to the in-progress assistant
          message, so they appear immediately when the data lands. */}
      {msg.charts?.map((chart, i) => (
        <ChartView key={`${chart.symbol}-${i}`} spec={chart} />
      ))}

      {!msg.streaming && msg.metadata && <MetadataFooter msg={msg} />}

      {/* Retry — re-runs the last question (shown on the latest answer). */}
      {!msg.streaming && onRetry && (
        <button
          onClick={onRetry}
          className="mt-1 inline-flex w-fit items-center gap-1.5 border border-border-subtle px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-text-secondary transition-colors hover:border-accent hover:text-accent"
          title="Regenerate this answer"
        >
          <RotateCcw size={11} />
          Retry
        </button>
      )}
    </motion.div>
  );
}

/**
 * ChatGPT-style "thinking" trace. While the agent works we show each step as it
 * happens (current one spinning, finished ones checked). Once the answer starts
 * streaming we collapse the trace into a compact "Thought for Ns" row that the
 * user can expand to review what the agent did.
 */
function ThinkingTrace({ msg }: { msg: ChatMessage }) {
  const [open, setOpen] = useState(false);
  const steps = msg.steps || [];
  const thinking = msg.streaming && !msg.content;

  if (steps.length === 0 && !msg.status) return null;

  // Live view — the agent is still working, no answer text yet.
  if (thinking) {
    const current = msg.status || steps[steps.length - 1];
    return (
      <div className="flex flex-col gap-1.5">
        <AnimatePresence initial={false}>
          {steps.map((s, i) => {
            const isLast = i === steps.length - 1;
            return (
              <motion.div
                key={`${s.stage}-${i}`}
                initial={{ opacity: 0, x: -4 }}
                animate={{ opacity: isLast ? 1 : 0.5, x: 0 }}
                transition={{ duration: 0.2 }}
                className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-wider text-text-secondary"
              >
                {isLast ? (
                  <span className="inline-block h-[10px] w-[10px] animate-spin rounded-full border border-text-muted border-t-accent" />
                ) : (
                  <Check size={11} className="text-accent" />
                )}
                <span>{isLast ? current?.label ?? s.label : s.label}</span>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    );
  }

  // Collapsed view — answer is present; offer the trace on demand.
  if (steps.length === 0) return null;
  const secs = msg.thoughtMs ? Math.max(1, Math.round(msg.thoughtMs / 1000)) : null;
  return (
    <div className="flex flex-col gap-1">
      <button
        onClick={() => setOpen((v) => !v)}
        className="inline-flex w-fit items-center gap-1 font-mono text-[10px] uppercase tracking-wider text-text-muted transition-colors hover:text-text-secondary"
      >
        <ChevronRight
          size={11}
          className={"transition-transform " + (open ? "rotate-90" : "")}
        />
        {secs ? `Thought for ${secs}s` : "Thinking trace"}
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden border-l border-border-subtle pl-3"
          >
            {steps.map((s, i) => (
              <div
                key={`${s.stage}-${i}`}
                className="flex items-center gap-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-text-muted"
              >
                <Check size={10} className="shrink-0 text-accent/70" />
                <span>{s.label}</span>
              </div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function MetadataFooter({ msg }: { msg: ChatMessage }) {
  const m = msg.metadata!;
  const chunks = msg.chunks?.length ?? 0;
  const traceUrl =
    "https://smith.langchain.com/o/_/projects/" +
    encodeURIComponent(import.meta.env.VITE_LANGCHAIN_PROJECT || "finagent");

  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[10px] uppercase tracking-wider text-text-muted">
      {m.model && <span>{m.model}</span>}
      {m.latency != null && <span>· {m.latency.toFixed(2)}s</span>}
      {chunks > 0 && <span>· {chunks} chunks used</span>}
      {m.input_tokens != null && (
        <span>· {m.input_tokens}↓ / {m.output_tokens ?? 0}↑ tok</span>
      )}
      <a
        href={traceUrl}
        target="_blank"
        rel="noreferrer"
        className="inline-flex items-center gap-1 text-text-muted hover:text-accent"
      >
        <ExternalLink size={9} /> LangSmith trace
      </a>
      {m.low_confidence && <span className="text-warning">· low confidence</span>}
    </div>
  );
}
