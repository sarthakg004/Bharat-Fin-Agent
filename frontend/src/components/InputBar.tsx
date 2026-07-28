import { useRef, useState } from "react";
import { motion } from "framer-motion";
import { ArrowUp, X, Loader2, ChevronDown, FileText, FlaskConical, ListTree, MessageSquare, Paperclip, PenLine } from "lucide-react";
import toast from "react-hot-toast";

import { uploadFile } from "@/lib/api";
import { useChatStore } from "@/store/chatStore";
import { useThreadStore } from "@/store/threadStore";
import {
  PROVIDER_LABELS, PROVIDER_MODELS, type Provider, useSettingsStore,
} from "@/store/settingsStore";
import { cls, modKey } from "@/lib/utils";

interface Props {
  onSend: (q: string) => void;
  onClear: () => void;
  streaming: boolean;
  disabled?: boolean;
}

// The question bar is the primary control on the page, so it gets real estate:
// ~3 lines visible before it scrolls, on a 24px rhythm.
const MAX_ROWS = 8;
const LINE_HEIGHT = 24;
const MIN_HEIGHT = LINE_HEIGHT * 3;

export function InputBar({ onSend, onClear, streaming, disabled }: Props) {
  const [value, setValue] = useState("");
  const mode = useSettingsStore((s) => s.mode);
  const ref = useRef<HTMLTextAreaElement | null>(null);

  function send() {
    const q = value.trim();
    if (!q || streaming || disabled) return;
    onSend(q);
    setValue("");
    autoSize();
  }

  function autoSize() {
    const el = ref.current;
    if (!el) return;
    el.style.height = "0px";
    const next = Math.min(Math.max(el.scrollHeight, MIN_HEIGHT), LINE_HEIGHT * MAX_ROWS);
    el.style.height = next + "px";
  }

  function onKey(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Escape") {
      setValue("");
      autoSize();
      return;
    }
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      send();
    }
  }

  return (
    <div
      className={cls(
        "flex flex-col border border-border-default bg-bg-surface shadow-[0_-1px_24px_rgba(0,0,0,0.35)]",
        "transition-colors focus-within:border-border-strong",
        disabled && "opacity-60",
      )}
    >
      <UploadChips />

      {/* Question row */}
      <div className="flex items-end gap-3 px-4 pb-3 pt-3.5">
        <textarea
          ref={ref}
          value={value}
          rows={3}
          onChange={(e) => {
            setValue(e.target.value);
            autoSize();
          }}
          onKeyDown={onKey}
          placeholder={
            mode === "research"
              ? `Name a company to research — e.g. "Should I invest in Nvidia?"   ${modKey()}↵ to send`
              : `Ask a financial question...   ${modKey()}↵ to send`
          }
          style={{ height: MIN_HEIGHT }}
          className="flex-1 resize-none bg-transparent font-ui text-[15px] leading-[24px] text-text-primary placeholder:text-text-muted focus:outline-none"
          disabled={disabled}
        />
        <div className="flex shrink-0 items-end gap-1">
          <UploadButton disabled={disabled || streaming} />
          <button
            type="button"
            onClick={onClear}
            className="flex h-[38px] w-[38px] items-center justify-center border border-border-subtle text-text-muted transition-colors hover:bg-bg-hover hover:text-text-primary"
            title="New conversation"
            aria-label="New conversation"
          >
            <X size={14} />
          </button>
          <motion.button
            type="button"
            onClick={send}
            whileTap={{ scale: 0.95 }}
            className={cls(
              "flex h-[38px] w-[38px] items-center justify-center border transition-colors",
              value.trim() && !streaming
                ? "border-accent bg-accent text-bg-base hover:bg-accent-hover"
                : "border-border-subtle bg-bg-elevated text-text-muted",
            )}
            aria-label="Send"
            disabled={!value.trim() || streaming || disabled}
          >
            {streaming ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <ArrowUp size={15} />
            )}
          </motion.button>
        </div>
      </div>

      {/* Mode + models — pick the pipeline and BOTH models right here. */}
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1.5 border-t border-border-subtle bg-bg-base/40 px-4 py-2">
        <ModeToggle disabled={disabled || streaming} />
        <span className="h-4 w-px bg-border-subtle" />
        <ModelSelect />
      </div>
    </div>
  );
}

/** Chat ↔ Deep Research. Chat answers one question; Deep Research runs the
 *  specialist-agent workflow and produces a full cited investment report. */
function ModeToggle({ disabled }: { disabled?: boolean }) {
  const mode = useSettingsStore((s) => s.mode);
  const setMode = useSettingsStore((s) => s.setMode);

  const base =
    "inline-flex items-center gap-1.5 px-2 py-1 font-mono text-[10px] uppercase tracking-wider transition-colors";
  return (
    <div className="flex items-center border border-border-subtle" role="radiogroup" aria-label="Mode">
      <button
        type="button"
        role="radio"
        aria-checked={mode === "chat"}
        disabled={disabled}
        onClick={() => setMode("chat")}
        className={cls(base, mode === "chat"
          ? "bg-bg-elevated text-text-primary"
          : "text-text-muted hover:text-text-secondary")}
        title="Answer one question over the filings + tools"
      >
        <MessageSquare size={11} />
        Chat
      </button>
      <button
        type="button"
        role="radio"
        aria-checked={mode === "research"}
        disabled={disabled}
        onClick={() => setMode("research")}
        className={cls(base, mode === "research"
          ? "bg-accent/10 text-accent"
          : "text-text-muted hover:text-text-secondary")}
        title="Multi-agent research: specialist agents + a full cited investment report (takes a few minutes)"
      >
        <FlaskConical size={11} />
        Deep Research
      </button>
    </div>
  );
}

/** Attach PDF/DOCX files. Each is parsed server-side into ephemeral chunks and
 *  every question in this chat is answered over them (+ the corpus). */
function UploadButton({ disabled }: { disabled?: boolean }) {
  const addUpload = useChatStore((s) => s.addUpload);
  const [busy, setBusy] = useState(false);
  const fileRef = useRef<HTMLInputElement | null>(null);

  async function onPick(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    e.target.value = "";                    // allow re-picking the same file
    if (!files.length) return;
    // A thread must exist to own the uploads (they persist per-thread).
    const threads = useThreadStore.getState();
    if (!threads.activeId) threads.createChat("New chat");
    setBusy(true);
    // Sequential — the server parses on a single worker anyway.
    for (const file of files) {
      try {
        const res = await uploadFile(file);
        addUpload({ id: res.upload_id, name: res.filename,
                    pages: res.pages, chunks: res.chunks });
        toast.success(`${res.filename} attached (${res.pages} pages)`);
      } catch (err) {
        toast.error(err instanceof Error ? err.message : `${file.name}: upload failed`);
      }
    }
    useThreadStore.getState().saveActive();  // uploads survive reload/switch
    setBusy(false);
  }

  return (
    <>
      <input
        ref={fileRef}
        type="file"
        accept=".pdf,.docx"
        multiple
        className="hidden"
        onChange={onPick}
      />
      <button
        type="button"
        onClick={() => fileRef.current?.click()}
        disabled={disabled || busy}
        className="flex h-[38px] w-[38px] items-center justify-center border border-border-subtle text-text-muted transition-colors hover:bg-bg-hover hover:text-text-primary disabled:opacity-50"
        title="Attach PDF or DOCX files (analysed for this chat only)"
        aria-label="Attach documents"
      >
        {busy ? <Loader2 size={14} className="animate-spin" /> : <Paperclip size={14} />}
      </button>
    </>
  );
}

/** Chips for the documents attached to this chat. */
function UploadChips() {
  const uploads = useChatStore((s) => s.uploads);
  const removeUpload = useChatStore((s) => s.removeUpload);
  if (!uploads.length) return null;

  function remove(id: string) {
    removeUpload(id);
    useThreadStore.getState().saveActive();
  }

  return (
    <div className="flex flex-wrap items-center gap-1.5 border-b border-border-subtle px-3 py-1.5">
      {uploads.map((u) => (
        <span
          key={u.id}
          className="inline-flex items-center gap-1.5 border border-border-subtle bg-bg-elevated px-2 py-1 font-mono text-[10px] text-text-secondary"
          title={`${u.chunks} chunks · answered alongside the corpus`}
        >
          <FileText size={11} className="shrink-0 text-accent" />
          <span className="max-w-[220px] truncate">{u.name}</span>
          <span className="text-text-muted">· {u.pages}p</span>
          <button
            onClick={() => remove(u.id)}
            className="text-text-muted transition-colors hover:text-err"
            aria-label={`Remove ${u.name}`}
          >
            <X size={11} />
          </button>
        </span>
      ))}
    </div>
  );
}

/** The two model pickers.
 *
 *  PLAN and ANSWER are separate models because they are separate jobs: the
 *  planner emits structured retrieval keys, the synthesizer writes prose, and
 *  the best model for one is not always the best for the other. Both default to
 *  the provider's first listed model, so leaving them alone reproduces the
 *  previous single-model behaviour exactly.
 *
 *  The PROVIDER is shared and is chosen by the ANSWER picker's optgroup — one
 *  request carries one provider and one API key, so offering a second provider
 *  here would be a lie. Switching provider re-points both. */
function ModelSelect() {
  const provider = useSettingsStore((s) => s.provider);
  const model = useSettingsStore((s) => s.modelByProvider[s.provider]);
  const planner = useSettingsStore((s) => s.plannerByProvider[s.provider]);
  const keys = useSettingsStore((s) => s.keys);
  const setProvider = useSettingsStore((s) => s.setProvider);
  const setModel = useSettingsStore((s) => s.setModel);
  const setPlannerModel = useSettingsStore((s) => s.setPlannerModel);
  const setKey = useSettingsStore((s) => s.setKey);

  const [draft, setDraft] = useState("");

  function pickAnswer(value: string) {
    const [p, m] = value.split("::") as [Provider, string];
    setProvider(p);
    setModel(p, m);
    setDraft("");
  }

  function saveKey(e: React.FormEvent) {
    e.preventDefault();
    if (draft.trim()) { setKey(provider, draft.trim()); setDraft(""); }
  }

  // Groq uses the server key; the others need the user's own key.
  const needsKey = provider !== "groq" && !keys[provider];
  const hasUserKey = provider !== "groq" && !!keys[provider];

  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-1.5">
      {/* Planner — constrained to the provider the answer model selected. */}
      <Picker
        icon={<ListTree size={11} />}
        label="Plan"
        title="PLANNER — turns your question into the search queries that hit the filings, and picks the lane (filings / SEC XBRL / market data / web). Precision over prose."
        value={planner}
        onChange={(m) => setPlannerModel(provider, m)}
      >
        {PROVIDER_MODELS[provider].map((m) => (
          <option key={m} value={m}>{m}</option>
        ))}
      </Picker>

      {/* Answer — also owns the provider choice. */}
      <Picker
        icon={<PenLine size={11} />}
        label="Answer"
        title="ANSWER — reads the retrieved evidence, writes the cited answer and fact-checks its own draft. Also selects the provider for both models."
        value={`${provider}::${model}`}
        onChange={pickAnswer}
      >
        {(Object.keys(PROVIDER_MODELS) as Provider[]).map((p) => (
          <optgroup key={p} label={PROVIDER_LABELS[p]}>
            {PROVIDER_MODELS[p].map((m) => (
              <option key={`${p}::${m}`} value={`${p}::${m}`}>{m}</option>
            ))}
          </optgroup>
        ))}
      </Picker>

      {/* Non-Groq models need the user's own key — collect it right here. */}
      {needsKey && (
        <form onSubmit={saveKey} className="flex items-center gap-1">
          <input
            type="password"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder={`${PROVIDER_LABELS[provider]} API key`}
            className="w-[180px] border border-warning/60 bg-bg-elevated px-2 py-1.5 font-mono text-[11px] text-text-primary placeholder:text-text-muted focus:border-accent focus:outline-none"
            aria-label={`${PROVIDER_LABELS[provider]} API key`}
          />
          <button
            type="submit"
            disabled={!draft.trim()}
            className="border border-accent bg-accent px-2 py-1.5 font-mono text-[10px] uppercase tracking-wider text-bg-base transition-colors hover:bg-accent-hover disabled:opacity-40"
          >
            Use
          </button>
        </form>
      )}

      {hasUserKey && (
        <button
          onClick={() => setKey(provider, "")}
          className="inline-flex items-center gap-1 font-mono text-[10px] uppercase tracking-wider text-accent hover:text-text-primary"
          title="Your key is stored in this browser — click to clear"
        >
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-accent" /> key set
        </button>
      )}
    </div>
  );
}

/** A labelled <select>: the label sits INSIDE the control so the two pickers
 *  read as "Plan: <model>" / "Answer: <model>" rather than two bare dropdowns
 *  whose meaning you have to remember. */
function Picker({ icon, label, title, value, onChange, children }: {
  icon: React.ReactNode;
  label: string;
  title: string;
  value: string;
  onChange: (v: string) => void;
  children: React.ReactNode;
}) {
  return (
    <div
      className="group relative flex items-center border border-border-subtle bg-bg-elevated transition-colors focus-within:border-accent hover:border-border-default"
      title={title}
    >
      <span className="pointer-events-none flex items-center gap-1 border-r border-border-subtle py-1.5 pl-2 pr-2 font-mono text-[10px] uppercase tracking-wider text-text-muted">
        {icon}
        {label}
      </span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="cursor-pointer appearance-none bg-transparent py-1.5 pl-2 pr-6 font-mono text-[11px] text-text-secondary transition-colors group-hover:text-text-primary focus:outline-none"
        aria-label={title}
      >
        {children}
      </select>
      <ChevronDown
        size={12}
        className="pointer-events-none absolute right-1.5 top-1/2 -translate-y-1/2 text-text-muted"
      />
    </div>
  );
}
