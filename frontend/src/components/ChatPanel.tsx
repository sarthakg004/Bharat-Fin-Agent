import { useEffect, useRef } from "react";
import { motion } from "framer-motion";

import { useChatStore } from "@/store/chatStore";
import { useRAGQuery } from "@/hooks/useRAGQuery";
import { useThreadStore } from "@/store/threadStore";
import { MessageBubble } from "./MessageBubble";
import { InputBar } from "./InputBar";
import { cls } from "@/lib/utils";
import type { BackendStatus } from "@/hooks/useBackendStatus";

// One example per tool lane, all answerable by the live US corpus + tools
// (XBRL fact, derived-metric calculator, EDGAR cross-document, market data).
// Non-corpus companies (e.g. Indian banks) fall through to web search — fine
// to ask, but not what we showcase on the empty state.
const EXAMPLE_QUERIES = [
  "What was Apple's R&D spend in FY2023?",
  "What was Coca-Cola's operating margin in FY2022?",
  "Which companies disclosed a material weakness in their 10-K?",
  "How has Tesla performed as a stock this year?",
];

interface Props {
  backendStatus: BackendStatus;
}

export function ChatPanel({ backendStatus }: Props) {
  const messages = useChatStore((s) => s.messages);
  const streamingId = useChatStore((s) => s.streamingId);
  const createChat = useThreadStore((s) => s.createChat);
  const { ask, regenerate } = useRAGQuery();
  const backendOnline = backendStatus === "online";

  const lastAssistantId = (() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "assistant") return messages[i].id;
    }
    return null;
  })();

  const scrollerRef = useRef<HTMLDivElement>(null);

  // Pin to bottom while streaming.
  useEffect(() => {
    scrollerRef.current?.scrollTo({
      top: scrollerRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages.length, streamingId, messages.map((m) => m.content.length).join(",")]);

  const empty = messages.length === 0;

  return (
    <section className="relative flex h-full min-w-0 flex-1 flex-col bg-bg-base">
      {empty ? (
        <EmptyState onAsk={ask} backendStatus={backendStatus} />
      ) : (
        <div
          ref={scrollerRef}
          className="flex-1 overflow-y-auto px-6 py-6 md:px-10"
        >
          <div className="mx-auto flex max-w-[760px] flex-col gap-6">
            {messages.map((m) => (
              <MessageBubble
                key={m.id}
                msg={m}
                onRetry={m.id === lastAssistantId && !streamingId ? regenerate : undefined}
              />
            ))}
          </div>
        </div>
      )}

      <div className="sticky bottom-0 border-t border-border-subtle bg-bg-base px-6 py-3 md:px-10">
        <div className="mx-auto max-w-[760px]">
          <InputBar
            onSend={ask}
            onClear={() => createChat("New chat")}
            streaming={!!streamingId}
            disabled={!backendOnline}
          />
        </div>
      </div>
    </section>
  );
}

function EmptyState({
  onAsk,
  backendStatus,
}: {
  onAsk: (q: string) => void;
  backendStatus: BackendStatus;
}) {
  const backendOnline = backendStatus === "online";
  return (
    <div className="flex flex-1 items-center justify-center px-6 py-10">
      <motion.div
        initial={{ y: 12, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.45 }}
        className="flex max-w-[680px] flex-col items-center text-center"
      >
        <span className="inline-flex h-[48px] items-center border border-accent px-3 font-mono text-[20px] font-medium text-accent">
          [FIN]
        </span>

        <p className="mt-6 font-display text-[28px] italic leading-tight text-text-primary">
          Ask anything about the financials.
        </p>
        <p className="mt-2 font-mono text-[11px] uppercase tracking-[0.18em] text-text-secondary">
          SEC 10-K · Indian annual reports · multi-agent RAG
        </p>

        {backendStatus === "warming" && (
          <div className="mt-6 flex items-center gap-3 border border-warning bg-warning-dim px-4 py-3 text-left">
            <span className="inline-block h-4 w-4 shrink-0 animate-spin rounded-full border-2 border-warning border-t-transparent" />
            <div>
              <p className="font-mono text-[11px] uppercase tracking-wider text-warning">
                Waking up the agent…
              </p>
              <p className="mt-1 font-ui text-[12px] text-text-secondary">
                The server scales to zero to save cost, so the first request can take
                up to a minute to cold-start. This page will connect automatically.
              </p>
            </div>
          </div>
        )}
        {backendStatus === "down" && (
          <div className="mt-6 border border-err bg-err-dim px-4 py-3 text-left">
            <p className="font-mono text-[11px] uppercase tracking-wider text-err">
              Backend unreachable
            </p>
            <p className="mt-1 font-ui text-[12px] text-text-secondary">
              Couldn't reach the API after a while. It may be redeploying — this page
              keeps retrying, so leave it open.
            </p>
          </div>
        )}

        <div className="mt-8 grid w-full grid-cols-1 gap-2 sm:grid-cols-2">
          {EXAMPLE_QUERIES.map((q) => (
            <motion.button
              key={q}
              whileHover={{ y: -2, boxShadow: "0 4px 12px #00D08430" }}
              transition={{ duration: 0.18 }}
              onClick={() => onAsk(q)}
              className={cls(
                "border border-border bg-bg-surface px-3 py-2 text-left font-ui text-[12px] text-text-primary transition-colors hover:border-accent hover:text-accent",
                !backendOnline && "pointer-events-none opacity-50",
              )}
            >
              {q}
            </motion.button>
          ))}
        </div>
      </motion.div>
    </div>
  );
}
