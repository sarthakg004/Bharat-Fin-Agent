import { motion } from "framer-motion";
import { ExternalLink } from "lucide-react";

import type { ChatMessage } from "@/store/chatStore";
import { useChatStore } from "@/store/chatStore";
import { cls, parseAnswerCitations } from "@/lib/utils";

export function MessageBubble({ msg }: { msg: ChatMessage }) {
  if (msg.role === "user") return <UserBubble msg={msg} />;
  return <AssistantBubble msg={msg} />;
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

function AssistantBubble({ msg }: { msg: ChatMessage }) {
  const setHighlight = useChatStore((s) => s.setHighlight);
  const segments = parseAnswerCitations(msg.content || "");
  const showStatus = msg.streaming && (!msg.content || msg.content.length < 6);

  return (
    <motion.div
      initial={{ y: 8, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.2 }}
      className="flex flex-col gap-2"
    >
      {msg.status && (
        <ProgressRow stage={msg.status.stage} label={msg.status.label} />
      )}

      {msg.error ? (
        <div className="border border-err bg-err-dim px-4 py-3 font-mono text-[12px] text-err">
          {msg.error}
        </div>
      ) : (
        <p className="font-ui text-[14px] leading-relaxed text-text-primary">
          {showStatus && !msg.content ? (
            <span className="font-mono text-[12px] text-text-muted">…</span>
          ) : (
            segments.map((seg, i) =>
              seg.kind === "text" ? (
                <span key={i}>{seg.text}</span>
              ) : (
                <CitationMark
                  key={i}
                  ids={seg.ids}
                  onClick={(id) => setHighlight(id)}
                />
              ),
            )
          )}
          {msg.streaming && <span className="streaming-cursor">▌</span>}
        </p>
      )}

      {!msg.streaming && msg.metadata && <MetadataFooter msg={msg} />}
    </motion.div>
  );
}

function CitationMark({
  ids,
  onClick,
}: {
  ids: number[];
  onClick: (id: number) => void;
}) {
  return (
    <span>
      {ids.map((id, i) => (
        <span key={id}>
          {i > 0 && ","}
          <a
            className="citation-mark"
            onClick={(e) => {
              e.preventDefault();
              onClick(id);
              const el = document.getElementById(`chunk-${id}`);
              el?.scrollIntoView({ behavior: "smooth", block: "center" });
            }}
            href={`#chunk-${id}`}
          >
            [{id}]
          </a>
        </span>
      ))}
    </span>
  );
}

function ProgressRow({ label }: { stage: string; label: string }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.15 }}
      className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-wider text-text-secondary"
    >
      <span className="inline-block h-[10px] w-[10px] animate-spin rounded-full border border-text-muted border-t-accent" />
      <span>{label}</span>
    </motion.div>
  );
}

function MetadataFooter({ msg }: { msg: ChatMessage }) {
  const m = msg.metadata!;
  const chunks = (msg.chunks?.length ?? 0);
  const traceUrl =
    "https://smith.langchain.com/o/_/projects/" +
    encodeURIComponent(import.meta.env.VITE_LANGCHAIN_PROJECT || "finagent");

  return (
    <div
      className={cls(
        "flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[10px] uppercase tracking-wider text-text-muted",
      )}
    >
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
      {m.low_confidence && (
        <span className="text-warning">· low confidence</span>
      )}
    </div>
  );
}
