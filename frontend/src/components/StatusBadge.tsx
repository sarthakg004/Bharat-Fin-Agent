import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

export function StatusBadge() {
  const q = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 30_000,
    refetchOnWindowFocus: false,
    retry: 1,
  });
  const online = q.isSuccess;

  return (
    <div className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-wider text-text-secondary">
      <span
        aria-hidden
        className={
          "inline-block h-2 w-2 rounded-full " +
          (online ? "bg-accent animate-pulseDot" : "bg-err")
        }
      />
      <span className={online ? "text-text-primary" : "text-err"}>
        {online ? "Connected" : "Offline"}
      </span>
    </div>
  );
}
