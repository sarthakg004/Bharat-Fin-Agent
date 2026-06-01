import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { ConfigId, Market } from "@/lib/api";

interface ConfigState {
  market: Market;
  config: ConfigId;
  companyFilter: string[];
  sidebarOpen: boolean;
  citationsOpen: boolean;

  setMarket: (m: Market) => void;
  setConfig: (c: ConfigId) => void;
  toggleCompany: (c: string) => void;
  clearCompanies: () => void;
  toggleSidebar: () => void;
  toggleCitations: () => void;
}

// We persist ONLY the active config (per the spec). Everything else is per-session.
export const useConfigStore = create<ConfigState>()(
  persist(
    (set) => ({
      market: "us",
      config: "agentic",
      companyFilter: [],
      sidebarOpen: true,
      citationsOpen: true,

      setMarket: (market) => set({ market, companyFilter: [] }),
      setConfig: (config) => set({ config }),
      toggleCompany: (c) =>
        set((s) => ({
          companyFilter: s.companyFilter.includes(c)
            ? s.companyFilter.filter((x) => x !== c)
            : [...s.companyFilter, c],
        })),
      clearCompanies: () => set({ companyFilter: [] }),
      toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
      toggleCitations: () => set((s) => ({ citationsOpen: !s.citationsOpen })),
    }),
    {
      name: "finagent.config",
      partialize: (s) => ({ config: s.config }),
    },
  ),
);
