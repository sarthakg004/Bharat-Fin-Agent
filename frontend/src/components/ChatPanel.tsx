import { useEffect, useRef } from "react";
import { motion } from "framer-motion";

import { useChatStore } from "@/store/chatStore";
import { useConfigStore } from "@/store/configStore";
import { useRAGQuery } from "@/hooks/useRAGQuery";
import { useThreadStore } from "@/store/threadStore";
import { MessageBubble } from "./MessageBubble";
import { InputBar } from "./InputBar";
import { cls } from "@/lib/utils";

const EXAMPLE_QUERIES = [
  "What was Apple's R&D spend in FY2023?",
  "Compare TCS and Infosys revenue growth 2022–2024",
  "Which segment dragged 3M's organic growth in FY2022?",
  "HDFC Bank net interest margin FY24",
];

interface Props {
  backendOnline: boolean;
}

export function ChatPanel({ backendOnline }: Props) {
  const messages = useChatStore((s) => s.messages);
  const startNewChat = useChatStore((s) => s.startNewChat);
  const streamingId = useChatStore((s) => s.streamingId);
  const market = useConfigStore((s) => s.market);
  const createChat = useThreadStore((s) => s.createChat);
  const { ask } = useRAGQuery(market);

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
        <EmptyState onAsk={ask} backendOnline={backendOnline} />
      ) : (
        <div
          ref={scrollerRef}
          className="flex-1 overflow-y-auto px-6 py-6 md:px-10"
        >
          <div className="mx-auto flex max-w-[760px] flex-col gap-6">
            {messages.map((m) => (
              <MessageBubble key={m.id} msg={m} />
            ))}
          </div>
        </div>
      )}

      <div className="sticky bottom-0 border-t border-border-subtle bg-bg-base px-6 py-3 md:px-10">
        <div className="mx-auto max-w-[760px]">
          <InputBar
            onSend={ask}
            onClear={() => { startNewChat(); createChat("New chat", market); }}
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
  backendOnline,
}: {
  onAsk: (q: string) => void;
  backendOnline: boolean;
}) {
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

        {!backendOnline && (
          <div className="mt-6 border border-err bg-err-dim px-4 py-3 text-left">
            <p className="font-mono text-[11px] uppercase tracking-wider text-err">
              Backend offline
            </p>
            <p className="mt-1 font-ui text-[12px] text-text-secondary">
              Start the FastAPI server with:
            </p>
            <pre className="mt-1 overflow-x-auto bg-bg-elevated px-2 py-1 font-mono text-[11px] text-text-primary">
              uvicorn backend.main:app --reload --port 8000
            </pre>
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
