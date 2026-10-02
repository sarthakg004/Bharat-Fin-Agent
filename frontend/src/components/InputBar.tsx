import { useRef, useState } from "react";
import { motion } from "framer-motion";
import { ArrowUp, ChevronDown, Loader2, PenLine } from "lucide-react";

import type { Provider, ServerConfig } from "@/lib/api";
import { useServerConfig } from "@/hooks/useServerConfig";
import { PROVIDER_LABELS, resolveWriter, useSettingsStore } from "@/store/settingsStore";
import { cls, modKey } from "@/lib/utils";

interface Props {
  onSend: (q: string) => void;
  streaming: boolean;
  disabled?: boolean;
}

// About three lines visible before the box scrolls.
const MAX_ROWS = 8;
const LINE_HEIGHT = 24;
const MIN_HEIGHT = LINE_HEIGHT * 3;

export function InputBar({ onSend, streaming, disabled }: Props) {
  const [value, setValue] = useState("");
  const ref = useRef<HTMLTextAreaElement | null>(null);
  const config = useServerConfig();
  const writer = resolveWriter(config, useSettingsStore());
  // A writer the server has no key for cannot run until the user supplies one.
  const blocked = streaming || disabled || writer.needsKey;

  function send() {
    const q = value.trim();
    if (!q || blocked) return;
    onSend(q);
    setValue("");
    autoSize();
  }

  function autoSize() {
    const el = ref.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = Math.min(Math.max(el.scrollHeight, MIN_HEIGHT), LINE_HEIGHT * MAX_ROWS) + "px";
  }

  function onKey(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Escape") {
      setValue("");
      autoSize();
    } else if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
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
          placeholder={`Ask a financial question...   ${modKey()}↵ to send`}
          style={{ height: MIN_HEIGHT }}
          className="flex-1 resize-none bg-transparent font-ui text-[15px] leading-[24px] text-text-primary placeholder:text-text-muted focus:outline-none"
          disabled={disabled}
        />
        <motion.button
          type="button"
          onClick={send}
          whileTap={{ scale: 0.95 }}
          className={cls(
            "flex h-[38px] w-[38px] shrink-0 items-center justify-center border transition-colors",
            value.trim() && !blocked
              ? "border-accent bg-accent text-bg-base hover:bg-accent-hover"
              : "border-border-subtle bg-bg-elevated text-text-muted",
          )}
          aria-label="Send"
          title={writer.needsKey ? "Add an API key for the selected writer model first" : "Send"}
          disabled={!value.trim() || blocked}
        >
          {streaming ? <Loader2 size={14} className="animate-spin" /> : <ArrowUp size={15} />}
        </motion.button>
      </div>

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 border-t border-border-subtle bg-bg-base/40 px-4 py-2">
        {config ? <ModelBar config={config} /> : (
          <span className="font-mono text-[10px] uppercase tracking-wider text-text-muted">
            loading models…
          </span>
        )}
      </div>
    </div>
  );
}

/** Which model does which job. The planner and the fact-checker are fixed and
 *  shown as text; the writer is the one the user can change. */
function ModelBar({ config }: { config: ServerConfig }) {
  const settings = useSettingsStore();
  const writer = resolveWriter(config, settings);
  const [draft, setDraft] = useState("");
  const providers = Object.keys(config.writer_models) as Provider[];
  const short = (model: string) => model.split("/").pop();

  function pick(value: string) {
    const [p, m] = value.split("::") as [Provider, string];
    const isDefault = p === config.roles.writer.provider && m === config.roles.writer.model;
    settings.setWriter(isDefault ? null : p, isDefault ? null : m);
    setDraft("");
  }

  function saveKey(e: React.FormEvent) {
    e.preventDefault();
    if (draft.trim()) {
      settings.setKey(writer.provider, draft.trim());
      setDraft("");
    }
  }

  return (
    <>
      <span
        className="cursor-help font-mono text-[10px] uppercase tracking-wider text-text-muted"
        title="The planner splits your question and picks the sources. Set by the server."
      >
        Plan <span className="normal-case text-text-secondary">{short(config.roles.planner.model)}</span>
      </span>

      <div
        className="group relative flex items-center border border-border-subtle bg-bg-elevated transition-colors focus-within:border-accent hover:border-border-default"
        title="The writer reads the evidence and writes the cited answer. This is the one model you can change."
      >
        <span className="pointer-events-none flex items-center gap-1 border-r border-border-subtle px-2 py-1.5 font-mono text-[10px] uppercase tracking-wider text-text-muted">
          <PenLine size={11} />
          Write
        </span>
        <select
          value={`${writer.provider}::${writer.model}`}
          onChange={(e) => pick(e.target.value)}
          className="cursor-pointer appearance-none bg-transparent py-1.5 pl-2 pr-6 font-mono text-[11px] text-text-secondary transition-colors group-hover:text-text-primary focus:outline-none"
          aria-label="Writer model"
        >
          {providers.map((p) => (
            <optgroup
              key={p}
              label={PROVIDER_LABELS[p] + (config.server_keys.includes(p) ? "" : " (needs your API key)")}
            >
              {config.writer_models[p].map((m) => (
                <option key={`${p}::${m}`} value={`${p}::${m}`}>
                  {m}{p === config.roles.writer.provider && m === config.roles.writer.model ? "  (default)" : ""}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
        <ChevronDown size={12} className="pointer-events-none absolute right-1.5 top-1/2 -translate-y-1/2 text-text-muted" />
      </div>

      {/* Not a control: the fact-checker is set by the server, and is never the
          writer's model (the server swaps to the default writer's model then). */}
      <span
        className="cursor-help font-mono text-[10px] uppercase tracking-wider text-text-muted"
        title="Fact-checks every claim in the draft against the evidence. Set by the server, and always a different model from the writer."
      >
        Fact-check{" "}
        <span className="normal-case text-text-secondary">
          {short(
            writer.provider === config.roles.critic.provider && writer.model === config.roles.critic.model
              ? config.roles.writer.model
              : config.roles.critic.model,
          )}
        </span>
      </span>

      {/* The server has no key for this provider: ask for the user's own. */}
      {writer.needsKey && (
        <form onSubmit={saveKey} className="flex items-center gap-1">
          <input
            type="password"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder={`${PROVIDER_LABELS[writer.provider]} API key to use this model`}
            className="w-[250px] border border-warning/60 bg-bg-elevated px-2 py-1.5 font-mono text-[11px] text-text-primary placeholder:text-text-muted focus:border-accent focus:outline-none"
            aria-label={`${PROVIDER_LABELS[writer.provider]} API key`}
            autoComplete="off"
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

      {writer.key && (
        <button
          onClick={() => settings.setKey(writer.provider, "")}
          className="inline-flex items-center gap-1 font-mono text-[10px] uppercase tracking-wider text-accent hover:text-text-primary"
          title="Your key is stored only in this browser. Click to remove it."
        >
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-accent" /> your key · remove
        </button>
      )}
    </>
  );
}
