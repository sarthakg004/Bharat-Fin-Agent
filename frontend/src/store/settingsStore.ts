/**
 * Settings store — provider choice, the two model picks (planner and answer),
 * and per-provider API keys.
 *
 * Keys live in `localStorage` (browser only) and are forwarded to the
 * backend in each query's `provider_config` field. The backend never
 * persists user-supplied keys.
 */

import { create } from "zustand";
import { persist } from "zustand/middleware";

export type Provider = "groq" | "gemini" | "openai" | "anthropic";

/** Models per provider that we surface in the UI. Override list is editable
 *  in this single place so adding a new option is one entry.                */
export const PROVIDER_MODELS: Record<Provider, string[]> = {
  // Verified against GET /openai/v1/models (Groq) and ListModels (Gemini).
  groq: [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
    "qwen/qwen3-32b",
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b-instant",
  ],
  gemini: [
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-flash-latest",
  ],
  openai: [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "o4-mini",
  ],
  anthropic: [
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "claude-opus-4-1",
  ],
};

export const PROVIDER_LABELS: Record<Provider, string> = {
  groq: "Groq",
  gemini: "Google Gemini",
  openai: "OpenAI",
  anthropic: "Anthropic",
};

export type QueryMode = "chat" | "research";

interface SettingsState {
  /** Execution mode: normal chat, or Deep Research (multi-agent report). */
  mode: QueryMode;
  /** Which provider every query is routed through (server default is "groq"). */
  provider: Provider;
  /** Which model WRITES THE ANSWER under that provider (synth + critic). */
  modelByProvider: Record<Provider, string>;
  /**
   * Which model DECOMPOSES the question under that provider.
   *
   * Separate from the answer model on purpose: the planner emits structured
   * retrieval keys and the synthesizer writes prose, and the best model for one
   * is not always the best for the other. Both default to the provider's first
   * listed model (gpt-oss-120b on Groq), so out-of-the-box behaviour is
   * unchanged — this exists so the two can be moved independently.
   */
  plannerByProvider: Record<Provider, string>;
  /**
   * User-supplied API keys, kept ONLY in localStorage.
   * Groq is empty by default because the Space ships with a server-side key —
   * users can override it but they don't need to.
   */
  keys: Record<Provider, string>;

  setMode: (m: QueryMode) => void;
  setProvider: (p: Provider) => void;
  setModel: (p: Provider, model: string) => void;
  setPlannerModel: (p: Provider, model: string) => void;
  setKey: (p: Provider, key: string) => void;
  clearKey: (p: Provider) => void;
}

const DEFAULT_MODELS: Record<Provider, string> = {
  groq: PROVIDER_MODELS.groq[0],
  gemini: PROVIDER_MODELS.gemini[0],
  openai: PROVIDER_MODELS.openai[0],
  anthropic: PROVIDER_MODELS.anthropic[0],
};

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      mode: "chat",
      provider: "groq",
      modelByProvider: { ...DEFAULT_MODELS },
      plannerByProvider: { ...DEFAULT_MODELS },
      keys: { groq: "", gemini: "", openai: "", anthropic: "" },

      setMode: (mode) => set({ mode }),
      setProvider: (provider) => set({ provider }),
      setModel: (p, model) =>
        set((s) => ({ modelByProvider: { ...s.modelByProvider, [p]: model } })),
      setPlannerModel: (p, model) =>
        set((s) => ({ plannerByProvider: { ...s.plannerByProvider, [p]: model } })),
      setKey: (p, key) =>
        set((s) => ({ keys: { ...s.keys, [p]: key } })),
      clearKey: (p) =>
        set((s) => ({ keys: { ...s.keys, [p]: "" } })),
    }),
    {
      name: "finagent.settings",
      // A model removed from PROVIDER_MODELS (e.g. Groq retiring kimi-k2)
      // can linger in localStorage and 404 every query — clamp any persisted
      // choice that's no longer offered back to the provider default.
      merge: (persisted, current) => {
        const s = { ...current, ...(persisted as Partial<SettingsState>) };
        // `plannerByProvider` did not exist in earlier builds, so a persisted
        // blob can be missing it entirely — fall back to the defaults, then
        // clamp both maps the same way.
        const clamp = (m: Record<Provider, string> | undefined) => {
          const out = { ...DEFAULT_MODELS, ...(m ?? {}) };
          for (const p of Object.keys(PROVIDER_MODELS) as Provider[]) {
            if (!PROVIDER_MODELS[p].includes(out[p])) out[p] = DEFAULT_MODELS[p];
          }
          return out;
        };
        return {
          ...s,
          modelByProvider: clamp(s.modelByProvider),
          plannerByProvider: clamp(s.plannerByProvider),
        };
      },
    },
  ),
);

/** Convenience: the current `provider_config` payload to send with a query. */
export function currentProviderConfig(): {
  provider: Provider;
  synth_model: string;
  planner_model: string;
  api_key?: string;
} {
  const s = useSettingsStore.getState();
  const provider = s.provider;
  const synth_model = s.modelByProvider[provider];
  const planner_model = s.plannerByProvider[provider];
  const api_key = s.keys[provider] || undefined;   // omit if empty
  return { provider, synth_model, planner_model, ...(api_key ? { api_key } : {}) };
}
