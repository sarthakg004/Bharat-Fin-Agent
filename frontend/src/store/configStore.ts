import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { Market } from "@/lib/api";

// Width clamps for the side panels (in px).
export const SIDEBAR_MIN = 200;
export const SIDEBAR_MAX = 420;
export const SIDEBAR_DEFAULT = 260;

export const CITATIONS_MIN = 280;
export const CITATIONS_MAX = 560;
export const CITATIONS_DEFAULT = 360;

function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, n));
}

interface ConfigState {
  market: Market;
  sidebarOpen: boolean;
  citationsOpen: boolean;
  sidebarWidth: number;
  citationsWidth: number;

  setMarket: (m: Market) => void;
  toggleSidebar: () => void;
  toggleCitations: () => void;
  setSidebarWidth: (px: number) => void;
  setCitationsWidth: (px: number) => void;
}

// Persist only the resize widths + market preference.
// Mode (always "agentic" now) and company filters are gone.
export const useConfigStore = create<ConfigState>()(
  persist(
    (set) => ({
      market: "us",
      sidebarOpen: true,
      citationsOpen: true,
      sidebarWidth: SIDEBAR_DEFAULT,
      citationsWidth: CITATIONS_DEFAULT,

      setMarket: (market) => set({ market }),
      toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
      toggleCitations: () => set((s) => ({ citationsOpen: !s.citationsOpen })),
      setSidebarWidth: (px) => set({ sidebarWidth: clamp(px, SIDEBAR_MIN, SIDEBAR_MAX) }),
      setCitationsWidth: (px) =>
        set({ citationsWidth: clamp(px, CITATIONS_MIN, CITATIONS_MAX) }),
    }),
    {
      name: "finagent.config",
      partialize: (s) => ({
        market: s.market,
        sidebarWidth: s.sidebarWidth,
        citationsWidth: s.citationsWidth,
      }),
    },
  ),
);
