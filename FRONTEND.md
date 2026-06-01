# FinAgent — frontend & API

Bloomberg-Terminal-meets-Linear UI on top of the existing Python RAG code.

```
┌─ Header ──────────────────────────────────────────────────────────┐
│  [FIN] BhāratFinAgent  │  🇺🇸 US / 🇮🇳 India  │  cfg  status  ⌘K │
├──────────────┬────────────────────────────────┬──────────────────┤
│  Sidebar     │           Chat                 │   Citations      │
│  - Mode      │  - Empty state with examples   │   - Chunk cards  │
│  - Filters   │  - Streaming answer + cites    │   - RAGAS badges │
│  - History   │  - Input bar at the bottom     │   - Run trace    │
└──────────────┴────────────────────────────────┴──────────────────┘
```

The chat panel uses Server-Sent Events to stream tokens. Citation markers
`[1]`, `[2]` inside the answer are clickable — they scroll the matching chunk
card into view in the right-hand panel and pulse it once. `⌘K` opens a Linear-
style command palette; `⌘⇧C` opens a race-comparison modal that runs Naive
and Agentic side-by-side.

## Run it

### 1. Backend (FastAPI, port 8000)

The backend wraps `src/agents/naive_rag.py` (`NaiveRAG`) and
`src/graph/corrective_rag.py` (`AgenticRAGv2`). Keep them loaded as singletons.

```bash
# Same conda env as the existing pipeline
conda activate finagent
pip install fastapi uvicorn   # if not present already
uvicorn backend.main:app --reload --port 8000
```

Endpoints:

| method | path           | what it does                                   |
|--------|----------------|------------------------------------------------|
| GET    | /api/health    | up-status + available collections + configs    |
| GET    | /api/configs   | the two modes the UI offers                    |
| GET    | /api/history   | last 50 questions from `data/finagent.db`      |
| POST   | /api/query     | SSE stream: `status`, `sources`, `chunk`, `metrics`, `done` |

The SSE stream pseudo-streams the final answer (Groq returns one chunk; we
split it on words for the UI). Retrieval and metadata are emitted as separate
events so the right-panel + metadata footer can light up before the text
arrives.

### 2. Frontend (Vite, port 5173)

```bash
cd frontend
npm install                          # one-time
npm run dev                          # http://localhost:5173
```

The Vite dev server proxies `/api/*` to `http://localhost:8000` (see
`vite.config.ts`), so the URLs in `src/lib/api.ts` work the same in dev and
prod. For HF Spaces / non-localhost deploys, set `VITE_API_URL` in
`frontend/.env`.

## Keyboard shortcuts

| key             | action                       |
|-----------------|------------------------------|
| `⌘↵` / `Ctrl↵`  | send message                 |
| `⌘K`            | command palette              |
| `⌘⇧C`           | compare modes                |
| `⌘L`            | clear conversation           |
| `⌘[`            | toggle sidebar               |
| `⌘]`            | toggle citations panel       |
| `?`             | shortcuts overlay            |
| `Esc`           | close any overlay / input    |

## Design rules baked in

- No gradients, no glass-morphism, no purple, no rounded-everything, no Inter.
- Border radius capped at `4px`.
- Three fonts loaded from Google Fonts: **DM Serif Display** (display),
  **Geist** (UI), **IBM Plex Mono** (numbers, citations, code).
- Pure-SVG grain texture on `body::before` at 3.5% opacity.
- `prefers-reduced-motion` is honoured both via CSS and via Framer Motion's
  `useReducedMotion()` hook — entrance + panel-slide animations skip entirely.

## File map

```
backend/
  main.py            FastAPI app + SSE encoder
  rag_service.py     Naive + AgenticRAGv2 singletons; output normalisation
  models.py          Pydantic request/response schemas
  history.py         tiny SQLite log
frontend/
  vite.config.ts     Vite + /api proxy
  tailwind.config.ts custom tokens (no `DEFAULT`-only — keys are `subtle`,
                     `default`, `strong` so `border-border-default` works)
  src/
    main.tsx         QueryClient root
    App.tsx          three-panel layout, entrance stagger, mobile sheets
    styles/globals.css  CSS variables + grain + skeleton + scrollbar
    lib/
      api.ts         typed API client + custom SSE parser (POST + stream)
      utils.ts       cls, timeAgo, parseAnswerCitations, scoreTier
    store/
      chatStore.ts   Zustand: messages, streamingId, highlightedChunkId
      configStore.ts Zustand: market, config, companyFilter, panel visibility
    hooks/
      useSSE.ts                wraps streamQuery + AbortController
      useKeyboardShortcuts.ts  global shortcut table (mod resolves per-OS)
      useRAGQuery.ts           SSE event dispatcher into chatStore
    components/
      Header / MarketToggle / StatusBadge
      Sidebar / CompanyChips
      ChatPanel / MessageBubble / InputBar
      CitationsPanel / ChunkCard / MetricsBadge
      CompareModal / CommandPalette / ShortcutsOverlay
```

## Notes

- The agentic mode uses `AgenticRAGv2` with `BAAI/bge-reranker-base` so the
  backend starts fast (~280 MB load). Swap to `bge-reranker-large` in
  `backend/rag_service.py` once you don't mind a 1.3 GB download.
- Rate-limit / TPD failures route through `RotatingChatModel`, so multi-key
  rotation works inside both modes without any extra wiring here.
- LangSmith traces link out from each answer's metadata footer; set
  `VITE_LANGCHAIN_PROJECT` in `frontend/.env` to override the project name.
- The Compare modal races the two configs in parallel (`Promise.allSettled`
  over two streamQuery calls) — both write into a column-local React state
  so neither can stall the other.
