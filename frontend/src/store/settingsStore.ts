/**
 * The user's writer choice and API keys.
 *
 * Only the WRITER model can be changed; the planner and the fact-checker are
 * fixed on the server. Model names come from GET /api/config, so nothing here
 * hardcodes one. Keys stay in this browser (localStorage) and are sent with
 * each question; the server never stores them.
 */

import { create } from "zustand";
import { persist } from "zustand/middleware";

import type { Provider, ServerConfig, WriterConfig } from "@/lib/api";

export const PROVIDER_LABELS: Record<Provider, string> = {
  groq: "Groq",
  gemini: "Google Gemini",
  openai: "OpenAI",
  anthropic: "Anthropic",
};

interface SettingsState {
  /** null = the server's default writer. */
  provider: Provider | null;
  model: string | null;
  keys: Record<Provider, string>;
  setWriter: (provider: Provider | null, model: string | null) => void;
  setKey: (p: Provider, key: string) => void;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      provider: null,
      model: null,
      keys: { groq: "", gemini: "", openai: "", anthropic: "" },
      setWriter: (provider, model) => set({ provider, model }),
      setKey: (p, key) => set((s) => ({ keys: { ...s.keys, [p]: key } })),
    }),
    // v3: earlier versions stored a planner model and retired model names.
    { name: "finagent.settings.v3" },
  ),
);

/** The writer in effect, given the server's config. A saved model the server
 *  no longer offers falls back to the default. */
export function resolveWriter(config: ServerConfig | undefined, s: SettingsState) {
  const fallback = config?.roles.writer;
  const offered = s.provider && config?.writer_models[s.provider]?.includes(s.model ?? "");
  const provider = (offered ? s.provider : fallback?.provider) ?? "gemini";
  const model = (offered ? s.model : fallback?.model) ?? "";
  const isDefault = !offered;
  const key = s.keys[provider] || "";
  const needsKey = !!config && !config.server_keys.includes(provider) && !key;
  return { provider, model, isDefault, key, needsKey };
}

/** The `writer` field to send with a question; undefined for the default. */
export function writerPayload(config: ServerConfig | undefined): WriterConfig | undefined {
  const w = resolveWriter(config, useSettingsStore.getState());
  if (w.isDefault && !w.key) return undefined;
  return { provider: w.provider, model: w.model, ...(w.key ? { api_key: w.key } : {}) };
}
