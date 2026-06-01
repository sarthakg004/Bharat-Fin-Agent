import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { ConfigId, Market } from "@/lib/api";

// Width clamps for the side panels (in px). Sidebar can't go too narrow
// (history items become unreadable) or too wide (chat panel loses room).
export const SIDEBAR_MIN = 200;
export const SIDEBAR_MAX = 420;
export const SIDEBAR_DEFAULT = 240;

export const CITATIONS_MIN = 280;
export const CITATIONS_MAX = 560;
export const CITATIONS_DEFAULT = 340;

function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, n));
}

interface ConfigState {
  market: Market;
  config: ConfigId;
  companyFilter: string[];
  sidebarOpen: boolean;
  citationsOpen: boolean;
  sidebarWidth: number;
  citationsWidth: number;

  setMarket: (m: Market) => void;
  setConfig: (c: ConfigId) => void;
  toggleCompany: (c: string) => void;
  clearCompanies: () => void;
  toggleSidebar: () => void;
  toggleCitations: () => void;
  setSidebarWidth: (px: number) => void;
  setCitationsWidth: (px: number) => void;
}

// We persist the active config AND the resized panel widths — users who drag
// the sidebar to fit their screen don't want to redo it every page load.
// Companies / market / open-state stay session-only.
export const useConfigStore = create<ConfigState>()(
  persist(
    (set) => ({
      market: "us",
      config: "agentic",
      companyFilter: [],
      sidebarOpen: true,
      citationsOpen: true,
      sidebarWidth: SIDEBAR_DEFAULT,
      citationsWidth: CITATIONS_DEFAULT,

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
      setSidebarWidth: (px) => set({ sidebarWidth: clamp(px, SIDEBAR_MIN, SIDEBAR_MAX) }),
      setCitationsWidth: (px) =>
        set({ citationsWidth: clamp(px, CITATIONS_MIN, CITATIONS_MAX) }),
    }),
    {
      name: "finagent.config",
      partialize: (s) => ({
        config: s.config,
        sidebarWidth: s.sidebarWidth,
        citationsWidth: s.citationsWidth,
      }),
    },
  ),
);
