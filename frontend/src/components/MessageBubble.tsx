import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Check, ChevronRight, Clock, Info, RotateCcw } from "lucide-react";

import type { QueryMetadata } from "@/lib/api";
import type { ChatMessage } from "@/store/chatStore";
import { MarkdownAnswer } from "@/lib/markdown";
import { ChartView } from "@/components/ChartView";

interface BubbleProps {
  msg: ChatMessage;
  /** Set on the last assistant message (when not streaming) to show Retry.
   *  `auto` = called by the countdown after a transient failure. */
  onRetry?: (auto?: boolean) => void;
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
      ) : msg.notice ? (
        // A failure explained in plain words: not a crash, so not styled as one.
        <div className="flex max-w-[80%] items-start gap-2 border border-warning/50 bg-warning-dim px-4 py-3 font-ui text-[13px] leading-relaxed text-text-primary">
          <Info size={14} className="mt-0.5 shrink-0 text-warning" />
          <span>{msg.notice}</span>
        </div>
      ) : showStatus && !msg.content ? (
        // A skeleton until the first step arrives; then the trace shows progress.
        !msg.steps?.length ? <AnswerSkeleton /> : null
      ) : (
        <div className="relative">
          <MarkdownAnswer text={msg.content} />
          {msg.streaming && (
            <span className="streaming-cursor inline-block align-baseline">▌</span>
          )}
        </div>
      )}

      {msg.retryAt && <RetryCountdown at={msg.retryAt} auto={!!msg.autoRetry} onRetry={onRetry} />}

      {/* Steps the agent had to skip, e.g. "Fact-check was skipped: ...". */}
      {!msg.streaming && (msg.metadata?.notices ?? []).map((n) => (
        <div key={n} className="flex items-start gap-1.5 font-mono text-[10.5px] text-warning">
          <Info size={11} className="mt-px shrink-0" />
          <span>{n}</span>
        </div>
      ))}

      {/* Inline charts produced by the market-data tool lane. They arrive on
          a separate SSE channel and attach to the in-progress assistant
          message, so they appear immediately when the data lands. */}
      {msg.charts?.map((chart, i) => (
        <ChartView key={`${chart.symbol}-${i}`} spec={chart} />
      ))}

      {!msg.streaming && msg.metadata && <MetadataFooter msg={msg} />}

      {/* Retry: re-runs the last question (shown on the latest answer). */}
      {!msg.streaming && onRetry && (
        <button
          onClick={() => onRetry()}
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

/** Counts down to `at`, the moment a transient failure can be retried. With
 *  `auto`, it retries by itself when the countdown ends; otherwise it tells the
 *  user the Retry button is ready. */
function RetryCountdown({ at, auto, onRetry }: {
  at: number; auto: boolean; onRetry?: (auto?: boolean) => void;
}) {
  const [left, setLeft] = useState(() => at - Date.now());
  const fired = useRef(false);

  useEffect(() => {
    setLeft(at - Date.now());
    const id = setInterval(() => setLeft(at - Date.now()), 500);
    return () => clearInterval(id);
  }, [at]);

  const ready = left <= 0;
  useEffect(() => {
    if (ready && auto && onRetry && !fired.current) {
      fired.current = true;                 // once: a re-render must not retry again
      onRetry(true);
    }
  }, [ready, auto, onRetry]);

  return (
    <div
      className={`flex w-fit items-center gap-2 border px-3 py-1.5 font-mono text-[11px] ${
        ready ? "border-accent/40 text-accent" : "border-border-subtle text-text-secondary"
      }`}
      aria-live="polite"
    >
      <Clock size={12} className={ready ? "" : "animate-pulse"} />
      {ready ? (
        <span>{auto ? "Retrying…" : "You can retry now."}</span>
      ) : (
        <span>
          {auto ? "Retrying in " : "Try again in "}
          <span className="tabular-nums text-text-primary">{formatLeft(left)}</span>
        </span>
      )}
    </div>
  );
}

/** Milliseconds → "42s" / "3m 07s" / "2h 05m". Ceil, so it never shows 0 early. */
function formatLeft(ms: number): string {
  const s = Math.ceil(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, "0")}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, "0")}m`;
}

/**
 * Placeholder for the incoming answer while the agent is still working —
 * pulsing skeleton lines where the text will appear, so the loading state
 * looks like an answer being prepared rather than a stalled chat.
 */
function AnswerSkeleton() {
  return (
    <div className="flex w-full max-w-[80%] animate-pulse flex-col gap-2 pt-1" aria-hidden>
      <div className="h-3 w-[88%] rounded-sm bg-border-subtle" />
      <div className="h-3 w-[72%] rounded-sm bg-border-subtle" />
      <div className="h-3 w-[55%] rounded-sm bg-border-subtle" />
    </div>
  );
}

/**
 * The "thinking" trace. While the agent works, one line shows the current step.
 * Once the answer arrives it collapses to "Thought for Ns", expandable to the
 * full list of steps.
 */
/** Re-render every `ms` while `active` so elapsed time / ETA tick live. */
function useNow(active: boolean, ms = 500): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => setNow(Date.now()), ms);
    return () => clearInterval(id);
  }, [active, ms]);
  return now;
}

function ThinkingTrace({ msg }: { msg: ChatMessage }) {
  const [open, setOpen] = useState(false);
  const steps = msg.steps || [];
  const thinking = msg.streaming && !msg.content;
  const now = useNow(!!thinking);

  if (steps.length === 0 && !msg.status) return null;

  // Live view — the agent is still working, no answer text yet. ONE line:
  // a spinner + the current step (cross-faded as it changes) + the previous
  // step's outcome, over a thin progress bar. The full step-by-step trace is
  // available afterwards via the "Thought for Ns" expander — while working,
  // a growing activity feed was just noise pushing the answer down.
  if (thinking) {
    const startedAt = msg.startedAt ?? msg.createdAt;
    const elapsedS = Math.max(0, (now - startedAt) / 1000);

    // Monotonic progress from the furthest pipeline stage reached. (We show
    // elapsed time + a progress bar, but NOT an ETA — the agent loops and skips
    // stages unpredictably, so any "time left" estimate would be misleading.)
    const total = msg.progressTotal ?? 0;
    const reached = (msg.progressIndex ?? 0) + 1;
    const progress = total > 0 ? Math.min(0.99, reached / total) : 0;

    const current = steps[steps.length - 1];
    const lastDone = [...steps].reverse().find((s) => s.done && s.detail);

    return (
      <div className="flex w-full max-w-[80%] flex-col gap-2">
        <div className="flex items-center gap-2 font-mono text-[11px]">
          <span className="inline-block h-[9px] w-[9px] shrink-0 animate-spin rounded-full border border-text-muted border-t-accent" />
          <AnimatePresence mode="wait" initial={false}>
            <motion.span
              key={current?.label ?? "start"}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.15 }}
              className="truncate text-text-primary"
            >
              {(current?.label ?? "Working…").replace(/…$/, "")}
              {lastDone?.detail && (
                <span className="text-text-muted"> · {lastDone.detail}</span>
              )}
            </motion.span>
          </AnimatePresence>
          <span className="ml-auto shrink-0 font-mono text-[10px] uppercase tracking-wider text-text-muted">
            {elapsedS.toFixed(0)}s
          </span>
        </div>

        {/* Progress bar — fills as the agent advances through the pipeline. */}
        {total > 0 && (
          <div className="h-[3px] w-full overflow-hidden rounded-full bg-border-subtle">
            <motion.div
              className="h-full bg-accent"
              initial={false}
              animate={{ width: `${Math.round(progress * 100)}%` }}
              transition={{ duration: 0.4, ease: "easeOut" }}
            />
          </div>
        )}
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
                className="flex items-baseline gap-2 py-0.5 font-mono text-[10px] text-text-muted"
              >
                <Check size={10} className="shrink-0 translate-y-[1px] text-accent/70" />
                <span className="uppercase tracking-wider">{s.label.replace(/…$/, "")}</span>
                {s.detail && <span className="text-text-secondary">· {s.detail}</span>}
              </div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function MetadataFooter({ msg }: { msg: ChatMessage }) {
  const m: QueryMetadata = msg.metadata ?? {};
  const chunks = msg.chunks?.length ?? 0;
  const checked = typeof m.support_score === "number";

  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[10px] uppercase tracking-wider text-text-muted">
      {m.model && <span>{m.model}</span>}
      {m.latency != null && <span>· {m.latency.toFixed(1)}s</span>}
      {chunks > 0 && <span>· {chunks} sources</span>}
      {m.input_tokens != null && (
        <span>· {m.input_tokens}↓ / {m.output_tokens ?? 0}↑ tok</span>
      )}
      {!m.refused && (
        <span title="Share of the answer's claims the fact-check found in the evidence">
          · {checked ? `fact-check ${Math.round((m.support_score as number) * 100)}%` : "not fact-checked"}
        </span>
      )}
    </div>
  );
}
