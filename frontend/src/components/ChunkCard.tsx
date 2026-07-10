import { useEffect, useRef, useState } from "react";
import { motion, useAnimation } from "framer-motion";
import { ExternalLink, ChevronDown } from "lucide-react";

import type { Chunk } from "@/lib/api";
import { useChatStore } from "@/store/chatStore";
import { cls } from "@/lib/utils";

interface Props {
  chunk: Chunk;
  index: number;
  /** Lane color from the owning Sources section (left rail tint). */
  accent?: string;
}

/** True for placeholder values the backend uses when a field doesn't apply. */
const blank = (v: string | number | undefined) =>
  v == null || v === "" || v === "?" || v === "—";

export function ChunkCard({ chunk, index, accent }: Props) {
  const highlighted = useChatStore((s) => s.highlightedChunkId === chunk.id);
  const ref = useRef<HTMLDivElement>(null);
  const ctrl = useAnimation();
  const [expanded, setExpanded] = useState(false);

  // Pulse + scroll into view when this chunk is highlighted from the chat.
  useEffect(() => {
    if (!highlighted) return;
    ref.current?.scrollIntoView({ behavior: "smooth", block: "center" });
    ctrl.start({
      boxShadow: [
        "0 0 0 0px var(--accent)",
        "0 0 0 2px var(--accent)",
        "0 0 0 0px var(--accent)",
      ],
      transition: { duration: 0.6 },
    });
  }, [highlighted, ctrl]);

  return (
    <motion.div
      ref={ref}
      id={`chunk-${chunk.id}`}
      initial={{ y: 6, opacity: 0 }}
      animate={ctrl}
      whileInView={{ y: 0, opacity: 1 }}
      viewport={{ once: true }}
      transition={{ delay: index * 0.04 }}
      className={cls(
        "border-y border-r bg-bg-surface border-y-border-subtle border-r-border-subtle",
        !accent && !highlighted && "border-l border-l-border-subtle",
      )}
      style={
        highlighted
          ? { borderLeft: "2px solid var(--accent)" }
          : accent
            ? { borderLeft: `2px solid ${accent}` }
            : undefined
      }
    >
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border-subtle px-3 py-2">
        <div className="flex min-w-0 items-center gap-2 font-mono text-[11px] text-text-secondary">
          <span className="shrink-0 text-accent">[{chunk.id + 1}]</span>
          <span className="truncate text-text-primary">
            {chunk.kind === "uploaded"
              ? (chunk.filename || chunk.company || "uploaded document")
              : (chunk.company || chunk.ticker || "?")}
          </span>
          {!blank(chunk.year) && (
            <>
              <span className="text-text-muted">·</span>
              <span className="shrink-0">FY {chunk.year}</span>
            </>
          )}
          {!blank(chunk.page) && (
            <>
              <span className="text-text-muted">·</span>
              <span className="shrink-0">p. {chunk.page}</span>
            </>
          )}
        </div>
        {chunk.source_url && (
          <a
            href={chunk.source_url}
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1 font-mono text-[10px] text-text-muted hover:text-accent"
          >
            open <ExternalLink size={9} />
          </a>
        )}
      </div>

      {/* Breadcrumb (sub_query for agentic, or generic) */}
      {chunk.sub_query && (
        <div className="border-b border-border-subtle bg-bg-base px-3 py-1 font-mono text-[10px] text-text-muted">
          {chunk.sub_query}
        </div>
      )}

      {/* Body */}
      <div className="px-3 py-3">
        <p
          className={cls(
            "font-ui text-[12.5px] leading-relaxed text-text-primary",
            !expanded && "line-clamp-3",
          )}
        >
          {chunk.text}
        </p>
        {chunk.text.length > 200 && (
          <button
            onClick={() => setExpanded((v) => !v)}
            className="mt-2 inline-flex items-center gap-1 font-mono text-[10px] uppercase tracking-wider text-text-muted hover:text-text-primary"
          >
            {expanded ? "Show less" : "Show more"}{" "}
            <ChevronDown
              size={11}
              className={cls("transition-transform", expanded && "rotate-180")}
            />
          </button>
        )}
      </div>
    </motion.div>
  );
}
