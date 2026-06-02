/**
 * Settings store — provider choice, model, and per-provider API keys.
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
  groq: [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "moonshotai/kimi-k2-instruct",
    "qwen/qwen3-32b",
  ],
  gemini: [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
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

interface SettingsState {
  /** Which provider every query is routed through (server default is "groq"). */
  provider: Provider;
  /** Which synth model to use under that provider. */
  modelByProvider: Record<Provider, string>;
  /**
   * User-supplied API keys, kept ONLY in localStorage.
   * Groq is empty by default because the Space ships with a server-side key —
   * users can override it but they don't need to.
   */
  keys: Record<Provider, string>;

  setProvider: (p: Provider) => void;
  setModel: (p: Provider, model: string) => void;
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
      provider: "groq",
      modelByProvider: { ...DEFAULT_MODELS },
      keys: { groq: "", gemini: "", openai: "", anthropic: "" },

      setProvider: (provider) => set({ provider }),
      setModel: (p, model) =>
        set((s) => ({ modelByProvider: { ...s.modelByProvider, [p]: model } })),
      setKey: (p, key) =>
        set((s) => ({ keys: { ...s.keys, [p]: key } })),
      clearKey: (p) =>
        set((s) => ({ keys: { ...s.keys, [p]: "" } })),
    }),
    { name: "finagent.settings" },
  ),
);

/** Convenience: the current `provider_config` payload to send with a query. */
export function currentProviderConfig(): {
  provider: Provider;
  synth_model: string;
  api_key?: string;
} {
  const s = useSettingsStore.getState();
  const provider = s.provider;
  const synth_model = s.modelByProvider[provider];
  const api_key = s.keys[provider] || undefined;   // omit if empty
  return { provider, synth_model, ...(api_key ? { api_key } : {}) };
}
