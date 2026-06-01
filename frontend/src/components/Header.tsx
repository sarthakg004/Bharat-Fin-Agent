import { Command } from "lucide-react";
import { MarketToggle } from "./MarketToggle";
import { StatusBadge } from "./StatusBadge";
import { useConfigStore } from "@/store/configStore";
import { modKey } from "@/lib/utils";

interface HeaderProps {
  onCommandPalette: () => void;
}

export function Header({ onCommandPalette }: HeaderProps) {
  const config = useConfigStore((s) => s.config);
  const configLabel = config === "naive" ? "Naive RAG" : "Agentic RAG";

  return (
    <header className="flex h-[56px] shrink-0 items-center border-b border-border-default bg-bg-base px-4">
      {/* Logo */}
      <div className="flex items-center gap-3">
        <span className="inline-flex h-[28px] items-center border border-accent px-[6px] font-mono text-[12px] font-medium text-accent">
          [FIN]
        </span>
        <span className="font-display text-[18px] leading-none tracking-tight text-text-primary">
          BhāratFinAgent
        </span>
        <span className="ml-1 hidden font-mono text-[10px] uppercase tracking-[0.2em] text-text-muted md:inline">
          financial research terminal
        </span>
      </div>

      {/* Market toggle — middle */}
      <div className="mx-auto">
        <MarketToggle />
      </div>

      {/* Right cluster */}
      <div className="flex items-center gap-4">
        <div className="hidden items-center gap-2 font-mono text-[11px] uppercase tracking-wider text-text-secondary md:flex">
          <span className="text-text-muted">cfg</span>
          <span className="text-text-primary">{configLabel}</span>
        </div>

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
