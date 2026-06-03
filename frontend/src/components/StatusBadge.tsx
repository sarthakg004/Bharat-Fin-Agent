import { useBackendStatus } from "@/hooks/useBackendStatus";

export function StatusBadge() {
  const { status } = useBackendStatus();

  const dot =
    status === "online" ? "bg-accent animate-pulseDot"
    : status === "warming" ? "bg-warning animate-pulse"
    : "bg-err";
  const text =
    status === "online" ? "text-text-primary"
    : status === "warming" ? "text-warning"
    : "text-err";
  const label =
    status === "online" ? "Connected"
    : status === "warming" ? "Waking up…"
    : "Offline";

  return (
    <div className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-wider text-text-secondary">
      <span aria-hidden className={"inline-block h-2 w-2 rounded-full " + dot} />
      <span className={text}>{label}</span>
    </div>
  );
}
