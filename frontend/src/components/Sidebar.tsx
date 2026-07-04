import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Plus, Trash2, Pencil, MessageSquare, Check, X } from "lucide-react";
import toast from "react-hot-toast";

import { useThreadStore } from "@/store/threadStore";
import { cls, timeAgo } from "@/lib/utils";

export function Sidebar() {
  const { threads, activeId, createChat, switchTo, renameChat, deleteChat } =
    useThreadStore();

  return (
    <motion.aside
      className="flex h-full w-full shrink-0 flex-col border-r border-border-default bg-bg-base"
      initial={false}
    >
      {/* New chat — primary action, top of sidebar */}
      <div className="border-b border-border-subtle p-3">
        <button
          onClick={() => createChat("New chat")}
          className="group flex w-full items-center justify-between border border-accent bg-accent px-3 py-2 font-ui text-[13px] text-bg-base transition-colors hover:bg-accent-hover"
        >
          <span className="inline-flex items-center gap-2">
            <Plus size={14} />
            New chat
          </span>
          <span className="hidden font-mono text-[10px] uppercase tracking-wider opacity-60 group-hover:opacity-100 md:inline">
            ⌘N
          </span>
        </button>
      </div>

      {/* Thread list */}
      <nav className="flex flex-1 flex-col overflow-y-auto px-2 py-2">
        {threads.length === 0 && (
          <div className="flex flex-1 flex-col items-center justify-center gap-2 text-center">
            <MessageSquare size={22} className="text-text-muted" />
            <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-text-muted">
              No chats yet
            </span>
            <span className="px-6 font-ui text-[12px] text-text-secondary">
              Click "New chat" to start asking the agent.
            </span>
          </div>
        )}

        {threads.map((t) => {
          const last = t.messages[t.messages.length - 1];
          return (
            <ThreadRow
              key={t.id}
              title={t.title}
              updated_at={t.updated_at}
              messageCount={t.messages.length}
              preview={last?.content?.slice(0, 80) || undefined}
              active={activeId === t.id}
              onSelect={() => switchTo(t.id)}
              onRename={(title) => renameChat(t.id, title)}
              onDelete={() => { deleteChat(t.id); toast.success("Deleted."); }}
            />
          );
        })}
      </nav>

      <div className="border-t border-border-subtle px-3 py-2 font-mono text-[10px] uppercase tracking-wider text-text-muted">
        {threads.length} chat{threads.length === 1 ? "" : "s"} · this session
      </div>
    </motion.aside>
  );
}

// --------------------------------------------------------------------------- //

interface ThreadRowProps {
  title: string;
  updated_at: number;
  messageCount: number;
  preview?: string;
  active: boolean;
  onSelect: () => void;
  onRename: (title: string) => void;
  onDelete: () => void;
}

function ThreadRow({ title, updated_at, messageCount, preview,
                    active, onSelect, onRename, onDelete }: ThreadRowProps) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(title);
  const inputRef = useRef<HTMLInputElement>(null);
  useEffect(() => { if (editing) setTimeout(() => inputRef.current?.select(), 30); }, [editing]);

  function commit() {
    const next = draft.trim();
    if (!next || next === title) { setEditing(false); return; }
    onRename(next);
    setEditing(false);
  }

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => !editing && onSelect()}
      onKeyDown={(e) => {
        if (editing) return;
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(); }
      }}
      className={cls(
        "group mx-1 mb-px flex cursor-pointer flex-col gap-1 border-l-2 px-2 py-2 transition-colors",
        active
          ? "border-l-accent bg-bg-hover"
          : "border-l-transparent hover:bg-bg-hover hover:border-l-border-strong",
      )}
    >
      <div className="flex items-center gap-2">
        {editing ? (
          <>
            <input
              ref={inputRef}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onClick={(e) => e.stopPropagation()}
              onKeyDown={(e) => {
                e.stopPropagation();
                if (e.key === "Enter") commit();
                if (e.key === "Escape") { setEditing(false); setDraft(title); }
              }}
              className="flex-1 border border-border bg-bg-elevated px-1.5 py-0.5 font-ui text-[12px] text-text-primary focus:outline-none"
            />
            <button
              onClick={(e) => { e.stopPropagation(); commit(); }}
              className="text-text-secondary hover:text-accent"
              aria-label="Save"
            >
              <Check size={11} />
            </button>
            <button
              onClick={(e) => { e.stopPropagation(); setEditing(false); setDraft(title); }}
              className="text-text-secondary hover:text-err"
              aria-label="Cancel"
            >
              <X size={11} />
            </button>
          </>
        ) : (
          <>
            <span className={cls(
              "line-clamp-1 flex-1 font-ui text-[12.5px]",
              active ? "text-text-primary" : "text-text-secondary group-hover:text-text-primary",
            )}>
              {title}
            </span>
            <button
              onClick={(e) => { e.stopPropagation(); setDraft(title); setEditing(true); }}
              className="shrink-0 text-text-muted opacity-0 transition-opacity hover:text-text-primary group-hover:opacity-100"
              title="Rename"
              aria-label="Rename"
            >
              <Pencil size={11} />
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                if (window.confirm(`Delete chat "${title}"?`)) onDelete();
              }}
              className="shrink-0 text-text-muted opacity-0 transition-opacity hover:text-err group-hover:opacity-100"
              title="Delete"
              aria-label="Delete"
            >
              <Trash2 size={11} />
            </button>
          </>
        )}
      </div>
      <div className="flex items-center justify-between font-mono text-[9.5px] uppercase tracking-wider text-text-muted">
        <span>{messageCount} msg{messageCount === 1 ? "" : "s"}</span>
        <span>{timeAgo(updated_at)}</span>
      </div>
      {preview && !editing && (
        <span className="line-clamp-1 font-ui text-[11px] text-text-muted">
          {preview}
        </span>
      )}
    </div>
  );
}
