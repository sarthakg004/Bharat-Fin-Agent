import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Search } from "lucide-react";
import toast from "react-hot-toast";

import { useChatStore, selectLastAssistant } from "@/store/chatStore";
import { useConfigStore } from "@/store/configStore";
import { cls } from "@/lib/utils";

interface Props {
  open: boolean;
  onClose: () => void;
  onCompare: () => void;
  onShortcuts: () => void;
}

interface CommandItem {
  id: string;
  label: string;
  hint?: string;
  run: () => void;
}

export function CommandPalette({ open, onClose, onCompare, onShortcuts }: Props) {
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const { setMarket, setConfig } = useConfigStore();
  const clear = useChatStore((s) => s.clear);

  useEffect(() => {
    if (open) {
      setQuery("");
      setActive(0);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [open]);

  const commands = useMemo<CommandItem[]>(() => [
    { id: "us",     label: "Switch to US Market",   hint: "🇺🇸", run: () => { setMarket("us"); } },
    { id: "in",     label: "Switch to India Market", hint: "🇮🇳", run: () => { setMarket("india"); } },
    { id: "naive",  label: "Switch mode → Naive RAG", run: () => setConfig("naive") },
    { id: "agentic",label: "Switch mode → Agentic RAG", run: () => setConfig("agentic") },
    { id: "clear",  label: "Clear conversation", run: clear },
    { id: "compare",label: "Compare modes", hint: "C", run: onCompare },
    { id: "copy",   label: "Copy last answer", run: () => {
      const last = selectLastAssistant(useChatStore.getState());
      if (!last?.content) return toast.error("No assistant message yet.");
      navigator.clipboard?.writeText(last.content);
      toast.success("Copied.");
    } },
    { id: "trace",  label: "Open LangSmith trace", run: () => {
      window.open("https://smith.langchain.com/", "_blank");
    } },
    { id: "shortcuts", label: "Keyboard shortcuts", hint: "?", run: onShortcuts },
  ], [setMarket, setConfig, clear, onCompare, onShortcuts]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return commands;
    return commands.filter((c) => c.label.toLowerCase().includes(q));
  }, [query, commands]);

  function run(item: CommandItem) {
    item.run();
    onClose();
  }

  function onKey(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((a) => Math.min(filtered.length - 1, a + 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((a) => Math.max(0, a - 1));
    } else if (e.key === "Enter" && filtered[active]) {
      e.preventDefault();
      run(filtered[active]);
    } else if (e.key === "Escape") {
      e.preventDefault();
      onClose();
    }
  }

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-50 flex items-start justify-center pt-[10vh]"
          style={{ backdropFilter: "blur(8px)", WebkitBackdropFilter: "blur(8px)" }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
        >
          <motion.div
            onClick={(e) => e.stopPropagation()}
            className="w-[min(560px,90vw)] border border-border-default bg-bg-surface"
            initial={{ y: -8, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: -8, opacity: 0 }}
            transition={{ duration: 0.15 }}
          >
            <div className="flex items-center gap-2 border-b border-border-subtle px-3 py-3">
              <Search size={14} className="text-text-muted" />
              <input
                ref={inputRef}
                value={query}
                onChange={(e) => { setQuery(e.target.value); setActive(0); }}
                onKeyDown={onKey}
                placeholder="Type a command..."
                className="flex-1 bg-transparent font-ui text-[14px] text-text-primary placeholder:text-text-muted focus:outline-none"
              />
              <kbd className="border border-border-subtle px-1.5 py-[1px] font-mono text-[10px] text-text-muted">
                ESC
              </kbd>
            </div>

            <ul className="max-h-[50vh] overflow-y-auto py-1">
              {filtered.length === 0 && (
                <li className="px-3 py-4 text-center font-mono text-[11px] text-text-muted">
                  No matches.
                </li>
              )}
              {filtered.map((item, idx) => (
                <li
                  key={item.id}
                  onMouseEnter={() => setActive(idx)}
                  onClick={() => run(item)}
                  className={cls(
                    "flex cursor-pointer items-center justify-between px-3 py-2 font-ui text-[13px]",
                    idx === active ? "bg-bg-hover text-text-primary" : "text-text-secondary",
                  )}
                >
                  <span>{item.label}</span>
                  {item.hint && (
                    <kbd className="border border-border-subtle px-1.5 py-[1px] font-mono text-[10px] text-text-muted">
                      {item.hint}
                    </kbd>
                  )}
                </li>
              ))}
            </ul>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
