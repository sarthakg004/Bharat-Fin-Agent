import { useQuery } from "@tanstack/react-query";

import { api } from "@/lib/api";

/** The server's model table (GET /api/config). Fetched once and cached. */
export function useServerConfig() {
  return useQuery({ queryKey: ["config"], queryFn: api.config, staleTime: Infinity, retry: 3 }).data;
}
