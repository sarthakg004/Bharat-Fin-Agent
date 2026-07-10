import { useRef, useState } from "react";
import { motion } from "framer-motion";
import { ArrowUp, X, Loader2, ChevronDown, Cpu, FileText, Paperclip } from "lucide-react";
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

const MAX_ROWS = 5;
const LINE_HEIGHT = 22;

export function InputBar({ onSend, onClear, streaming, disabled }: Props) {
  const [value, setValue] = useState("");
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
    const next = Math.min(el.scrollHeight, LINE_HEIGHT * MAX_ROWS);
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
        "flex flex-col border border-border-default bg-bg-surface",
        disabled && "opacity-60",
      )}
    >
      <UploadChips />

      {/* Question row */}
      <div className="flex items-end gap-2 px-3 py-2">
        <textarea
          ref={ref}
          value={value}
          rows={1}
          onChange={(e) => {
            setValue(e.target.value);
            autoSize();
          }}
          onKeyDown={onKey}
          placeholder={`Ask a financial question...   ${modKey()}↵ to send`}
          className="flex-1 resize-none bg-transparent font-ui text-[14px] leading-[22px] text-text-primary placeholder:text-text-muted focus:outline-none"
          disabled={disabled}
        />
        <div className="flex shrink-0 items-end gap-1">
          <UploadButton disabled={disabled || streaming} />
          <button
            type="button"
            onClick={onClear}
            className="flex h-[34px] w-[34px] items-center justify-center border border-border-subtle text-text-muted transition-colors hover:bg-bg-hover hover:text-text-primary"
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
              "flex h-[34px] w-[34px] items-center justify-center border transition-colors",
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
              <ArrowUp size={14} />
            )}
          </motion.button>
        </div>
      </div>

      {/* Model row — pick the model right here (provider is implied by the model). */}
      <div className="flex items-center gap-2 border-t border-border-subtle px-3 py-1.5">
        <Cpu size={12} className="shrink-0 text-text-muted" />
        <ModelSelect />
      </div>
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
        className="flex h-[34px] w-[34px] items-center justify-center border border-border-subtle text-text-muted transition-colors hover:bg-bg-hover hover:text-text-primary disabled:opacity-50"
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

/** One combined model picker. Choosing a model also selects its provider; the
 *  provider is shown only as the optgroup label, never as a separate control. */
function ModelSelect() {
  const provider = useSettingsStore((s) => s.provider);
  const model = useSettingsStore((s) => s.modelByProvider[s.provider]);
  const keys = useSettingsStore((s) => s.keys);
  const setProvider = useSettingsStore((s) => s.setProvider);
  const setModel = useSettingsStore((s) => s.setModel);
  const setKey = useSettingsStore((s) => s.setKey);

  const [draft, setDraft] = useState("");

  function pick(value: string) {
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
    <div className="flex flex-wrap items-center gap-1.5">
      <div className="relative">
        <select
          value={`${provider}::${model}`}
          onChange={(e) => pick(e.target.value)}
          className="cursor-pointer appearance-none border border-border-subtle bg-bg-elevated py-1 pl-2 pr-6 font-mono text-[11px] text-text-secondary transition-colors hover:text-text-primary focus:outline-none"
          aria-label="Model"
        >
          {(Object.keys(PROVIDER_MODELS) as Provider[]).map((p) => (
            <optgroup key={p} label={PROVIDER_LABELS[p]}>
              {PROVIDER_MODELS[p].map((m) => (
                <option key={`${p}::${m}`} value={`${p}::${m}`}>{m}</option>
              ))}
            </optgroup>
          ))}
        </select>
        <ChevronDown
          size={12}
          className="pointer-events-none absolute right-1.5 top-1/2 -translate-y-1/2 text-text-muted"
        />
      </div>

      {/* Non-Groq models need the user's own key — collect it right here. */}
      {needsKey && (
        <form onSubmit={saveKey} className="flex items-center gap-1">
          <input
            type="password"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder={`${PROVIDER_LABELS[provider]} API key`}
            className="w-[180px] border border-warning/60 bg-bg-elevated px-2 py-1 font-mono text-[11px] text-text-primary placeholder:text-text-muted focus:border-accent focus:outline-none"
            aria-label={`${PROVIDER_LABELS[provider]} API key`}
          />
          <button
            type="submit"
            disabled={!draft.trim()}
            className="border border-accent bg-accent px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-bg-base transition-colors hover:bg-accent-hover disabled:opacity-40"
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
