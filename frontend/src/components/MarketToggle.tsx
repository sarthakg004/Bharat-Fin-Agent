import { motion } from "framer-motion";
import { useConfigStore } from "@/store/configStore";
import type { Market } from "@/lib/api";
import { cls } from "@/lib/utils";

const OPTIONS: { value: Market; label: string }[] = [
  { value: "us", label: "🇺🇸  US" },
  { value: "india", label: "🇮🇳  India" },
];

export function MarketToggle() {
  const { market, setMarket } = useConfigStore();

  return (
    <div
      role="tablist"
      aria-label="Market"
      className="relative inline-flex items-center border border-border-default p-[2px]"
    >
      {OPTIONS.map((opt) => {
        const active = market === opt.value;
        return (
          <button
            key={opt.value}
            role="tab"
            aria-selected={active}
            onClick={() => setMarket(opt.value)}
            className={cls(
              "relative z-10 px-3 py-[6px] text-[12px] font-mono uppercase tracking-wider transition-colors",
              active ? "text-bg-base" : "text-text-secondary hover:text-text-primary",
            )}
          >
            {active && (
              <motion.span
                layoutId="marketPill"
                className="absolute inset-0 bg-accent"
                transition={{ type: "spring", stiffness: 350, damping: 30 }}
              />
            )}
            <span className="relative">{opt.label}</span>
          </button>
        );
      })}
    </div>
  );
}
