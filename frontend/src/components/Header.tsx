import { useEffect, useRef, useState } from "react";
import { Command, Pencil, Check, Settings as SettingsIcon } from "lucide-react";

import { MarketToggle } from "./MarketToggle";
import { StatusBadge } from "./StatusBadge";
import { useThreadStore } from "@/store/threadStore";
import { useChatStore } from "@/store/chatStore";
import { useSettingsStore, PROVIDER_LABELS } from "@/store/settingsStore";
import { cls, modKey } from "@/lib/utils";

interface HeaderProps {
  onCommandPalette: () => void;
  onSettings: () => void;
}

export function Header({ onCommandPalette, onSettings }: HeaderProps) {
  const activeChatId = useChatStore((s) => s.activeChatId);
  const threads = useThreadStore((s) => s.threads);
  const renameChat = useThreadStore((s) => s.renameChat);
  const activeChat = threads.find((t) => t.id === activeChatId);

  const provider = useSettingsStore((s) => s.provider);
  const model = useSettingsStore((s) => s.modelByProvider[s.provider]);
  const hasUserKey = useSettingsStore((s) => !!s.keys[s.provider]);

  return (
    <header className="flex h-[56px] shrink-0 items-center gap-3 border-b border-border-default bg-bg-base px-4">
      {/* Brand mark — smaller now that the chat title is the headline */}
      <div className="flex items-center gap-2">
        <span className="inline-flex h-[26px] items-center border border-accent px-[6px] font-mono text-[11px] font-medium text-accent">
          [FIN]
        </span>
        <span className="hidden font-display text-[15px] leading-none tracking-tight text-text-primary sm:inline">
          BhāratFinAgent
        </span>
      </div>

      <span className="hidden h-5 w-px bg-border-default md:inline" />

      {/* Active chat title */}
      <ActiveChatTitle
        title={activeChat?.title}
        onRename={async (t) => {
          // Drop the short-circuit return so the arrow stays Promise<void>.
          if (activeChatId != null) await renameChat(activeChatId, t);
        }}
      />

      {/* Right cluster */}
      <div className="ml-auto flex items-center gap-3">
        <MarketToggle />

        {/* Provider + model pill — clicks open the Settings modal */}
        <button
          onClick={onSettings}
          title="Switch provider, model, or API key"
          className="hidden items-center gap-1.5 border border-border-default px-2 py-1 font-mono text-[11px] text-text-secondary transition-colors hover:border-accent hover:bg-bg-hover hover:text-text-primary md:flex"
        >
          <SettingsIcon size={11} />
          <span className="text-text-primary">{PROVIDER_LABELS[provider]}</span>
          <span className="text-text-muted">·</span>
          <span className="truncate max-w-[140px]">{model}</span>
          {hasUserKey && (
            <span
              className="inline-block h-1.5 w-1.5 rounded-full bg-info"
              title="Using your API key"
            />
          )}
        </button>

        <StatusBadge />
        <button
          onClick={onCommandPalette}
          className="hidden items-center gap-1.5 border border-border-default px-2 py-1 font-mono text-[11px] text-text-secondary transition-colors hover:bg-bg-hover hover:text-text-primary md:flex"
          aria-label="Open command palette"
        >
          <Command size={11} />
          <span>{modKey()}K</span>
        </button>
      </div>
    </header>
  );
}

interface TitleProps {
  title?: string;
  onRename: (next: string) => Promise<void>;
}

function ActiveChatTitle({ title, onRename }: TitleProps) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(title || "");
  const ref = useRef<HTMLInputElement>(null);

  useEffect(() => { setDraft(title || ""); }, [title]);
  useEffect(() => { if (editing) setTimeout(() => ref.current?.select(), 30); }, [editing]);

  async function commit() {
    const next = draft.trim();
    if (!next || next === title) { setEditing(false); return; }
    try { await onRename(next); } finally { setEditing(false); }
  }

  if (!title) {
    return (
      <span className="truncate font-display text-[15px] italic text-text-muted">
        new chat
      </span>
    );
  }

  return (
    <div className="flex min-w-0 flex-1 items-center gap-2">
      {editing ? (
        <>
          <input
            ref={ref}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") commit();
              if (e.key === "Escape") { setEditing(false); setDraft(title); }
            }}
            className="min-w-0 flex-1 border border-border bg-bg-elevated px-2 py-[3px] font-display text-[15px] text-text-primary focus:outline-none"
          />
          <button onClick={commit} className="text-accent hover:text-accent-hover" aria-label="Save title">
            <Check size={14} />
          </button>
        </>
      ) : (
        <button
          onClick={() => setEditing(true)}
          className={cls(
            "group inline-flex min-w-0 items-center gap-2 truncate text-left",
          )}
          title="Click to rename"
        >
          <span className="truncate font-display text-[15px] text-text-primary">
            {title}
          </span>
          <Pencil
            size={11}
            className="shrink-0 text-text-muted opacity-0 transition-opacity group-hover:opacity-100"
          />
        </button>
      )}
    </div>
  );
}
