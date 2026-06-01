import { PanelLeftOpen, PanelRightOpen } from "lucide-react";
import { modKey } from "@/lib/utils";

interface Props {
  side: "left" | "right";
  onClick: () => void;
}

/**
 * Thin floating handle that appears at the edge of the chat panel when the
 * corresponding side panel is collapsed. Lets the user re-open without
 * needing the keyboard shortcut.
 */
export function PanelOpenButton({ side, onClick }: Props) {
  const Icon = side === "left" ? PanelLeftOpen : PanelRightOpen;
  const label = side === "left" ? "Open sidebar" : "Open citations panel";
  const shortcut = side === "left" ? `${modKey()}[` : `${modKey()}]`;
  return (
    <button
      onClick={onClick}
      title={`${label}  ·  ${shortcut}`}
      aria-label={label}
      className={
        "hidden md:flex h-full w-[18px] shrink-0 items-center justify-center " +
        "border-y border-border-subtle bg-bg-base text-text-secondary " +
        "transition-colors hover:bg-bg-hover hover:text-accent " +
        (side === "left" ? "border-r" : "border-l")
      }
    >
      <Icon size={13} />
    </button>
  );
}
