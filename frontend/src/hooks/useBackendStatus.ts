import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { api } from "@/lib/api";

export type BackendStatus = "warming" | "online" | "down";

/**
 * Polls /api/health and turns it into a friendly status:
 *   - "warming"  → not up yet, but we're still within the cold-start window
 *   - "online"   → healthy
 *   - "down"     → still unreachable after a generous wait (~90s)
 *
 * While not online we poll every 3s (so the UI flips to "online" quickly once
 * the container finishes its cold start); once online we drop to every 30s.
 */
export function useBackendStatus(): { status: BackendStatus; elapsedSec: number } {
  const q = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: (query) => (query.state.status === "success" ? 30_000 : 3_000),
    refetchOnWindowFocus: false,
    retry: false,
  });

  const [elapsedSec, setElapsedSec] = useState(0);
  useEffect(() => {
    if (q.isSuccess) {
      setElapsedSec(0);
      return;
    }
    const t0 = Date.now();
    const id = setInterval(() => setElapsedSec(Math.floor((Date.now() - t0) / 1000)), 1000);
    return () => clearInterval(id);
  }, [q.isSuccess]);

  const status: BackendStatus = q.isSuccess
    ? "online"
    : elapsedSec < 90
      ? "warming"
      : "down";

  return { status, elapsedSec };
}
