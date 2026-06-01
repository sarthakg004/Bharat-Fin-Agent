import { useEffect } from "react";
import { isMac } from "@/lib/utils";

export type ShortcutHandler = (e: KeyboardEvent) => void;
export interface ShortcutMap {
  [combo: string]: ShortcutHandler;
}

/**
 * Combo syntax: `mod+enter`, `mod+shift+c`, `mod+k`, `escape`, `?`.
 * `mod` resolves to ⌘ on macOS and Ctrl elsewhere.
 *
 * Shortcuts only fire when the user is NOT typing in an input or textarea,
 * with the exception of `mod+...` chords (sending a message must work from
 * the textarea).
 */
export function useKeyboardShortcuts(map: ShortcutMap): void {
  useEffect(() => {
    function comboFor(e: KeyboardEvent): string | null {
      const mod = isMac() ? e.metaKey : e.ctrlKey;
      const parts: string[] = [];
      if (mod) parts.push("mod");
      if (e.shiftKey) parts.push("shift");
      if (e.altKey) parts.push("alt");
      const key = e.key.toLowerCase();
      if (["control", "meta", "shift", "alt"].includes(key)) return null;
      parts.push(key === " " ? "space" : key === "enter" ? "enter" : key);
      return parts.join("+");
    }

    function onKey(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement | null)?.tagName?.toLowerCase();
      const inField = tag === "input" || tag === "textarea" || (e.target as HTMLElement | null)?.isContentEditable;
      const combo = comboFor(e);
      if (!combo) return;
      const handler = map[combo];
      if (!handler) return;

      const isChord = combo.startsWith("mod+");
      if (inField && !isChord && combo !== "escape") return;

      e.preventDefault();
      handler(e);
    }

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [map]);
}
