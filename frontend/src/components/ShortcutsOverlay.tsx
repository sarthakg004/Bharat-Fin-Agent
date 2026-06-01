import { AnimatePresence, motion } from "framer-motion";
import { X } from "lucide-react";
import { modKey } from "@/lib/utils";

const SHORTCUTS: { combo: string; label: string }[] = [
  { combo: "↵",     label: "Send message" },
  { combo: "N",     label: "New chat" },
  { combo: "K",     label: "Command palette" },
  { combo: "L",     label: "Clear current view" },
  { combo: "[",     label: "Toggle sidebar" },
  { combo: "]",     label: "Toggle citations panel" },
];

export function ShortcutsOverlay({ open, onClose }: { open: boolean; onClose: () => void }) {
  const mod = modKey();
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center bg-bg-base/80"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
        >
          <motion.div
            onClick={(e) => e.stopPropagation()}
            initial={{ scale: 0.96, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.96, opacity: 0 }}
            transition={{ duration: 0.18 }}
            className="w-[min(420px,90vw)] border border-border-default bg-bg-surface"
          >
            <div className="flex items-center justify-between border-b border-border-subtle px-4 py-3">
              <span className="font-display text-[18px] text-text-primary">Keyboard shortcuts</span>
              <button onClick={onClose} className="text-text-muted hover:text-text-primary">
                <X size={14} />
              </button>
            </div>
            <ul className="flex flex-col">
              {SHORTCUTS.map((s) => (
                <li key={s.combo} className="flex items-center justify-between border-b border-border-subtle px-4 py-2 last:border-b-0">
                  <span className="font-ui text-[13px] text-text-secondary">{s.label}</span>
                  <kbd className="border border-border-default bg-bg-elevated px-2 py-[2px] font-mono text-[11px] text-text-primary">
                    {mod}{s.combo}
                  </kbd>
                </li>
              ))}
              <li className="flex items-center justify-between px-4 py-2">
                <span className="font-ui text-[13px] text-text-secondary">Close overlays</span>
                <kbd className="border border-border-default bg-bg-elevated px-2 py-[2px] font-mono text-[11px] text-text-primary">
                  Esc
                </kbd>
              </li>
              <li className="flex items-center justify-between border-t border-border-subtle px-4 py-2">
                <span className="font-ui text-[13px] text-text-secondary">This panel</span>
                <kbd className="border border-border-default bg-bg-elevated px-2 py-[2px] font-mono text-[11px] text-text-primary">
                  ?
                </kbd>
              </li>
            </ul>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
