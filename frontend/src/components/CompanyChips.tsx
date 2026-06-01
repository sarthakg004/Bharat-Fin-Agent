import { useConfigStore } from "@/store/configStore";
import { cls } from "@/lib/utils";

const TICKERS: Record<"us" | "india", string[]> = {
  us: ["AAPL", "MSFT", "NVDA", "JPM", "WMT", "AMZN", "GOOGL", "TSLA", "BA", "DIS"],
  india: ["TCS", "INFY", "RELIANCE", "HDFC", "ICICI", "WIPRO", "ITC", "SBI", "MARUTI", "NTPC"],
};

export function CompanyChips() {
  const { market, companyFilter, toggleCompany, clearCompanies } = useConfigStore();
  const tickers = TICKERS[market];

  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-text-secondary">
          Filter
        </span>
        {companyFilter.length > 0 && (
          <button
            onClick={clearCompanies}
            className="font-mono text-[10px] uppercase tracking-wider text-text-muted hover:text-text-secondary"
          >
            clear
          </button>
        )}
      </div>
      <div className="flex flex-wrap gap-1">
        {tickers.map((t) => {
          const active = companyFilter.includes(t);
          return (
            <button
              key={t}
              onClick={() => toggleCompany(t)}
              className={cls(
                "border px-2 py-1 font-mono text-[10px] transition-colors",
                active
                  ? "border-accent bg-accent-dim text-accent"
                  : "border-border bg-bg-elevated text-text-secondary hover:bg-bg-hover hover:text-text-primary",
              )}
            >
              {t}
            </button>
          );
        })}
      </div>
    </div>
  );
}
