import { useRef, useState } from "react";
import { motion } from "framer-motion";
import { ArrowUp, X, Loader2 } from "lucide-react";

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
        "flex items-end gap-2 border border-border-default bg-bg-surface px-3 py-2",
        disabled && "opacity-60",
      )}
    >
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
          title="Clear conversation"
          aria-label="Clear conversation"
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
            <motion.span
              animate={{ rotate: streaming ? 45 : 0 }}
              transition={{ type: "spring", stiffness: 400, damping: 25 }}
            >
              <ArrowUp size={14} />
            </motion.span>
          )}
        </motion.button>
      </div>
    </div>
  );
}
