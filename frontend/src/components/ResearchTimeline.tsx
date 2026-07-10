import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Check, ChevronRight, Download, FlaskConical, X } from "lucide-react";

import type { ChatMessage, ResearchTask } from "@/store/chatStore";

/**
 * Deep Research timeline — the live agent-activity panel shown on research
 * messages. While the run is in flight every specialist row updates in place
 * (pending ○ → running spinner → done ✓ / failed ✕, with its outcome line);
 * afterwards the same list collapses behind a "Research trail" expander, and
 * each completed row can expand to its finding summary.
 */
export function ResearchTimeline({ msg }: { msg: ChatMessage }) {
  const research = msg.research;
  const [open, setOpen] = useState(false);
  if (!research?.tasks.length) return null;

  const live = !!msg.streaming && !msg.content;
  const done = research.tasks.filter((t) => t.status === "done").length;

  if (live) {
    return (
      <div className="flex w-full max-w-[80%] flex-col gap-1 border border-border-subtle bg-bg-surface px-3 py-2.5">
        <div className="mb-1 flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.18em] text-text-secondary">
          <FlaskConical size={11} className="text-accent" />
          Deep research
          {research.company && <span className="text-text-primary">· {research.company}</span>}
          <span className="ml-auto normal-case tracking-normal text-text-muted">
            {done}/{research.tasks.length}
          </span>
        </div>
        {research.tasks.map((t) => <TaskRow key={t.id} task={t} />)}
      </div>
    );
  }

  // Completed view — collapsible trail + per-agent summaries.
  return (
    <div className="flex flex-col gap-1">
      <button
        onClick={() => setOpen((v) => !v)}
        className="inline-flex w-fit items-center gap-1 font-mono text-[10px] uppercase tracking-wider text-text-muted transition-colors hover:text-text-secondary"
      >
        <ChevronRight size={11} className={"transition-transform " + (open ? "rotate-90" : "")} />
        Research trail · {done}/{research.tasks.length} agents
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
            {research.tasks.map((t) => <TaskRow key={t.id} task={t} expandable />)}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function TaskRow({ task, expandable }: { task: ResearchTask; expandable?: boolean }) {
  const [open, setOpen] = useState(false);
  const canExpand = !!expandable && !!task.summary;

  return (
    <div className="flex flex-col">
      <button
        type="button"
        onClick={canExpand ? () => setOpen((v) => !v) : undefined}
        className={
          "flex items-baseline gap-2 py-0.5 text-left font-mono text-[11px] " +
          (canExpand ? "cursor-pointer hover:text-text-primary " : "cursor-default ") +
          (task.status === "pending" ? "text-text-muted/60" : "text-text-secondary")
        }
      >
        <StatusIcon status={task.status} />
        <span className={task.status === "running" ? "text-text-primary" : ""}>
          {task.label}
        </span>
        {task.detail && (
          <span className="truncate text-[10px] text-text-muted">· {task.detail}</span>
        )}
        {canExpand && (
          <ChevronRight
            size={10}
            className={"ml-auto shrink-0 transition-transform " + (open ? "rotate-90" : "")}
          />
        )}
      </button>
      {canExpand && open && (
        <p className="mb-1 ml-5 border-l border-border-subtle pl-2 font-ui text-[11.5px] leading-snug text-text-muted">
          {task.summary}…
        </p>
      )}
    </div>
  );
}

function StatusIcon({ status }: { status: ResearchTask["status"] }) {
  if (status === "running") {
    return (
      <span className="inline-block h-[9px] w-[9px] shrink-0 translate-y-[1px] animate-spin rounded-full border border-text-muted border-t-accent" />
    );
  }
  if (status === "done") {
    return <Check size={11} className="shrink-0 translate-y-[1.5px] text-accent" />;
  }
  if (status === "failed") {
    return <X size={11} className="shrink-0 translate-y-[1.5px] text-err" />;
  }
  return (
    <span className="inline-block h-[7px] w-[7px] shrink-0 translate-y-[1px] rounded-full border border-border-strong" />
  );
}

/** Download the finished report as a markdown file. */
export function ExportReportButton({ msg }: { msg: ChatMessage }) {
  if (!msg.research || !msg.content || msg.streaming) return null;

  function exportMd() {
    const name = (msg.research?.company || "research").toLowerCase().replace(/\W+/g, "-");
    const blob = new Blob([msg.content], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${name}-investment-report.md`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <button
      onClick={exportMd}
      className="mt-1 inline-flex w-fit items-center gap-1.5 border border-border-subtle px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-text-secondary transition-colors hover:border-accent hover:text-accent"
      title="Download the report as markdown"
    >
      <Download size={11} />
      Export report
    </button>
  );
}
