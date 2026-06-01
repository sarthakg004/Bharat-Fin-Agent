import { cls, scoreTier } from "@/lib/utils";

const COLOURS: Record<ReturnType<typeof scoreTier>, string> = {
  good: "border-accent text-accent",
  okay: "border-warning text-warning",
  bad: "border-err text-err",
  none: "border-border-subtle text-text-muted",
};

interface Props {
  label: string;
  score: number | null | undefined;
}

export function MetricsBadge({ label, score }: Props) {
  const tier = scoreTier(score);
  const display = score == null ? "—" : score.toFixed(2);
  return (
    <span
      className={cls(
        "inline-flex items-center gap-1.5 border px-2 py-1 font-mono text-[10px] uppercase tracking-wider",
        COLOURS[tier],
      )}
      title={`${label}: ${display}`}
    >
      <span className="text-text-secondary">{label}</span>
      <span>{display}</span>
    </span>
  );
}
