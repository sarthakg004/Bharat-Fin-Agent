import { useRef, useState } from "react";
import { motion } from "framer-motion";
import { ArrowUp, X, Loader2, ChevronDown, Cpu } from "lucide-react";
import toast from "react-hot-toast";

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

/** One combined model picker. Choosing a model also selects its provider; the
 *  provider is shown only as the optgroup label, never as a separate control. */
function ModelSelect() {
  const provider = useSettingsStore((s) => s.provider);
  const model = useSettingsStore((s) => s.modelByProvider[s.provider]);
  const keys = useSettingsStore((s) => s.keys);
  const setProvider = useSettingsStore((s) => s.setProvider);
  const setModel = useSettingsStore((s) => s.setModel);

  function pick(value: string) {
    const [p, m] = value.split("::") as [Provider, string];
    setProvider(p);
    setModel(p, m);
    if (p !== "groq" && !keys[p]) {
      toast(`Add your ${PROVIDER_LABELS[p]} API key (gear, top-right)`, { icon: "🔑" });
    }
  }

  const needsKey = provider !== "groq" && !keys[provider];

  return (
    <div className="relative flex items-center gap-1.5">
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
      {needsKey && (
        <span className="font-mono text-[10px] uppercase tracking-wider text-warning">
          key needed
        </span>
      )}
    </div>
  );
}
