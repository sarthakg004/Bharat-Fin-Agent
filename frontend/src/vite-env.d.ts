/// <reference types="vite/client" />

// Project-specific env vars used in source. Adding them to the ImportMetaEnv
// interface makes `import.meta.env.VITE_*` typed instead of `any`.
interface ImportMetaEnv {
  readonly VITE_API_URL?: string;
  readonly VITE_LANGCHAIN_PROJECT?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
