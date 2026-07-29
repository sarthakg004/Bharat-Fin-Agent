import { AnimatePresence, motion } from "framer-motion";
import { Check, Eye, EyeOff, Key, ListTree, PenLine, X } from "lucide-react";
import { useState } from "react";

import {
  PROVIDER_LABELS,
  PROVIDER_MODELS,
  type Provider,
  useSettingsStore,
} from "@/store/settingsStore";
import { cls } from "@/lib/utils";

const PROVIDERS: Provider[] = ["groq", "openai", "anthropic", "gemini"];

interface Props {
  open: boolean;
  onClose: () => void;
}

export function SettingsModal({ open, onClose }: Props) {
  const {
    provider, modelByProvider, plannerByProvider, keys,
    setProvider, setModel, setPlannerModel, setKey,
  } = useSettingsStore();

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center bg-bg-base/85"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
        >
          <motion.div
            onClick={(e) => e.stopPropagation()}
            initial={{ scale: 0.96, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.96, opacity: 0 }}
            transition={{ duration: 0.16 }}
            className="w-[min(640px,92vw)] max-h-[88vh] overflow-y-auto border border-border-default bg-bg-surface"
          >
            <div className="flex items-center justify-between border-b border-border-default px-5 py-3">
              <div>
                <h2 className="font-display text-[18px] text-text-primary">Provider & model</h2>
                <p className="mt-0.5 font-mono text-[10px] uppercase tracking-wider text-text-muted">
                  keys stay in your browser — never persisted on the server
                </p>
              </div>
              <button onClick={onClose} className="text-text-muted hover:text-text-primary" aria-label="Close">
                <X size={16} />
              </button>
            </div>

            <div className="space-y-5 px-5 py-4">
              {/* Active provider — radio strip */}
              <section>
                <SectionLabel>Active provider</SectionLabel>
                <div className="mt-2 grid grid-cols-2 gap-1.5 md:grid-cols-4">
                  {PROVIDERS.map((p) => (
                    <button
                      key={p}
                      onClick={() => setProvider(p)}
                      className={cls(
                        "border px-3 py-2 text-left font-ui text-[12.5px] transition-colors",
                        p === provider
                          ? "border-accent bg-accent-dim text-accent"
                          : "border-border-subtle text-text-secondary hover:border-border hover:bg-bg-hover hover:text-text-primary",
                      )}
                    >
                      {PROVIDER_LABELS[p]}
                    </button>
                  ))}
                </div>
              </section>

              {/* What the two models actually do. Without this the pair reads
                  as a redundant setting; it is not — they are different jobs. */}
              <section className="border border-border-subtle bg-bg-elevated px-3 py-2.5">
                <SectionLabel>Two models, two jobs</SectionLabel>
                <dl className="mt-2 space-y-1.5 font-ui text-[12px] leading-relaxed text-text-secondary">
                  <div className="flex gap-2">
                    <dt className="inline-flex shrink-0 items-center gap-1 font-mono text-[10px] uppercase tracking-wider text-accent">
                      <ListTree size={11} /> Planner
                    </dt>
                    <dd>
                      Turns your question into the search queries that hit the
                      filings, and decides which lane answers it — filings, SEC
                      XBRL, live market data or the web. Rewards precision over
                      prose.
                    </dd>
                  </div>
                  <div className="flex gap-2">
                    <dt className="inline-flex shrink-0 items-center gap-1 font-mono text-[10px] uppercase tracking-wider text-accent">
                      <PenLine size={11} /> Answer
                    </dt>
                    <dd>
                      Reads the retrieved evidence and writes the cited answer,
                      then fact-checks its own draft. Rewards long-form reasoning
                      and careful writing.
                    </dd>
                  </div>
                </dl>
                <p className="mt-2 border-t border-border-subtle pt-2 font-mono text-[10.5px] leading-relaxed text-text-muted">
                  Both default to <code>gpt-oss-120b</code>. Leave them alone and
                  nothing changes — the split exists so a cheaper or stricter
                  model can take one job without touching the other. The rows
                  below set each provider's pair; the planner can also stay on a
                  <em> different</em> provider from the answer (a free planner
                  beside a paid answer model), which is chosen in the composer's
                  Plan dropdown.
                </p>
              </section>

              {/* Per-provider rows */}
              <section className="space-y-4">
                {PROVIDERS.map((p) => (
                  <ProviderRow
                    key={p}
                    provider={p}
                    active={p === provider}
                    model={modelByProvider[p]}
                    setModel={(m) => setModel(p, m)}
                    planner={plannerByProvider[p]}
                    setPlanner={(m) => setPlannerModel(p, m)}
                    apiKey={keys[p]}
                    setApiKey={(k) => setKey(p, k)}
                  />
                ))}
              </section>

              {/* Footer note */}
              <p className="border-t border-border-subtle pt-3 font-mono text-[10.5px] leading-relaxed text-text-muted">
                <strong className="text-text-secondary">Groq:</strong> if you leave the key empty the Space
                uses its built-in <code>GROQ_API_KEY</code> from HF Secrets, so the default Groq models
                always work without configuration.&nbsp;
                <strong className="text-text-secondary">Others:</strong> paste a key to switch provider —
                both models move together, since one request carries one provider. Keys are stored in
                <code>localStorage</code> on your device and posted only when you send a query.
              </p>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

interface RowProps {
  provider: Provider;
  active: boolean;
  model: string;
  setModel: (m: string) => void;
  planner: string;
  setPlanner: (m: string) => void;
  apiKey: string;
  setApiKey: (k: string) => void;
}

function ProviderRow({ provider, active, model, setModel, planner, setPlanner,
                      apiKey, setApiKey }: RowProps) {
  const [reveal, setReveal] = useState(false);
  const hasKey = !!apiKey;

  return (
    <div className={cls(
      "border px-3 py-2.5",
      active ? "border-accent" : "border-border-subtle",
    )}>
      <div className="flex items-center gap-2">
        <span className={cls(
          "inline-block h-[7px] w-[7px] rounded-full",
          active ? "bg-accent" : (hasKey ? "bg-info" : "bg-border-strong"),
        )} />
        <span className="font-ui text-[13px] text-text-primary">
          {PROVIDER_LABELS[provider]}
        </span>
        {active && (
          <span className="ml-auto inline-flex items-center gap-1 font-mono text-[9.5px] uppercase tracking-wider text-accent">
            <Check size={10} /> active
          </span>
        )}
      </div>

      <div className="mt-2 grid grid-cols-1 gap-2 md:grid-cols-2">
        {/* Two models, two jobs: the planner emits structured retrieval keys,
            the synthesizer writes prose. Both default to the same model, so a
            user who never touches this sees no change. */}
        <label className="block">
          <span className="block font-mono text-[10px] uppercase tracking-wider text-text-muted">
            Planner model
            <span className="ml-1 normal-case text-text-muted">(decomposes the question)</span>
          </span>
          <select
            value={planner}
            onChange={(e) => setPlanner(e.target.value)}
            className="mt-1 w-full border border-border-default bg-bg-elevated px-2 py-1.5 font-mono text-[12px] text-text-primary focus:outline-none focus:border-accent"
          >
            {PROVIDER_MODELS[provider].map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </label>

        <label className="block">
          <span className="block font-mono text-[10px] uppercase tracking-wider text-text-muted">
            Answer model
            <span className="ml-1 normal-case text-text-muted">(writes the answer)</span>
          </span>
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="mt-1 w-full border border-border-default bg-bg-elevated px-2 py-1.5 font-mono text-[12px] text-text-primary focus:outline-none focus:border-accent"
          >
            {PROVIDER_MODELS[provider].map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </label>

        <label className="block">
          <span className="flex items-center justify-between font-mono text-[10px] uppercase tracking-wider text-text-muted">
            <span className="inline-flex items-center gap-1">
              <Key size={10} /> API key
              {provider === "groq" && (
                <span className="ml-1 normal-case text-text-muted">(optional — Space has one)</span>
              )}
            </span>
            <button
              type="button"
              onClick={() => setReveal((v) => !v)}
              className="text-text-muted hover:text-text-secondary"
              aria-label="Toggle key visibility"
            >
              {reveal ? <EyeOff size={11} /> : <Eye size={11} />}
            </button>
          </span>
          <input
            type={reveal ? "text" : "password"}
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder={provider === "groq" ? "(uses Space secret)" : "sk-…"}
            className="mt-1 w-full border border-border-default bg-bg-elevated px-2 py-1.5 font-mono text-[12px] text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent"
            autoComplete="off"
            spellCheck={false}
          />
        </label>
      </div>
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-text-secondary">
      {children}
    </span>
  );
}
