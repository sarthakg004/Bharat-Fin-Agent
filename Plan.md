## India angle — datasets and why this is the better move

There are several solid public datasets for Indian financial QA, and combining them gives you everything FinanceBench offers, *plus* an angle that differentiates you in the Indian job market.

**Datasets that work for an India version (all on HuggingFace, free):**

- **`sweatSmile/FinanceQA`** — a curated dataset of ~4,000 financial Q&A entries extracted from Indian company annual reports, balance sheets, and financial statements, with structured fields for queries, answers, and contextual excerpts. Includes companies like ICICI Bank. This is your closest direct analog to FinanceBench.
- **`adityarane/financial-qa-dataset`** — ~475 Q&A-context triplets with sample notebooks for basic RAG with evaluation already included.
- **`bharatgenai/BhashaBench-Finance`** — 19,433 rigorously validated questions across 25+ Indian financial government and institutional exams, covering 30+ financial domains, in both English and Hindi. Use this as a second evaluation set to show robustness, and the Hindi capability is a unique angle for Indian fintech roles.
- **`kdave/Indian_Financial_News`** — Indian financial news articles, perfect to plug into your web-search / current-events agent.

**For the underlying corpus (the actual annual reports you'll do RAG over):**

You don't need a pre-built dataset of PDFs because you can pull them directly from NSE. Two public Python libraries handle this cleanly: nselib is a Python library to fetch publicly available data from NSE India covering capital market data, FII statistics, AMFI reports, and more, and `bsedata` does the same for BSE. The NSE corporate filings page (`nseindia.com/companies-listing/corporate-filings-annual-reports`) hosts annual report PDFs directly. SEBI annual reports are also publicly downloadable.

A clean approach: pick **15-20 Nifty 50 companies across sectors** (Reliance, TCS, Infosys, HDFC Bank, ICICI Bank, ITC, Bharti Airtel, Hindustan Unilever, L&T, Maruti, etc.), download their last 3 annual reports each. That gives you ~45-60 PDFs — comparable corpus size to FinanceBench's 40 companies, but Indian.

**Why India is actually a stronger choice for *you* specifically:**

1. **Recruiter recognition.** A recruiter at Razorpay, Zerodha, Groww, Cred, Jupiter, INDmoney, HDFC, ICICI's AI team, Sarvam AI, Krutrim, or any of the big GCC AI teams in Mumbai/Bengaluru (Goldman Sachs, JPMC, Morgan Stanley, Citi) will *immediately* understand what "agent over Reliance's annual report" means. SEC 10-Ks of US companies feel abstract to them by comparison.
2. **Differentiation.** Thousands of Indian candidates are building US-stock projects to mimic Western tutorials. An India-native finance agent is genuinely rare and signals you understand the local market.
3. **Bonus angle: bilingual.** If you wire BhashaBench-Finance in as an additional eval and demo a few Hindi queries (*"पिछले साल रिलायंस का राजस्व क्या था?"*), you've now got a multilingual agentic RAG project, which is on-trend with Sarvam AI / BharatGen / IndicLLM hiring waves.
4. **Indian accounting standards.** Indian annual reports use Ind-AS, which differs subtly from US-GAAP (different line item names, different segment disclosures). Handling this in your prompts/tools is a real-world engineering challenge that's worth talking about in interviews.

**My recommendation:** Build the India version. Use `sweatSmile/FinanceQA` as your primary eval set and the NSE corporate filings as your underlying corpus. Use BhashaBench-Finance as a secondary eval. Call the project something like **"BhāratFinAgent"** or **"NiftyAgent"** — a name that sounds Indian-confident on a resume.

One thing to note: the FinanceQA dataset doesn't come with the same level of paper-published baseline as FinanceBench (where GPT-4-Turbo's 81% failure rate is a known number). You'll have to *generate* your own baseline by running naive RAG yourself before turning on the agentic features. That's actually fine — the *comparison* table is the deliverable, not the absolute number.

Now to the detailed plan.

---

## The detailed six-week plan

This assumes ~10-15 hrs/week. If you're full-time on it, compress to 3 weeks. I've structured each week as "goal → tasks → deliverable at end of week" so you always know where you should be.

### Week 1 — Project setup, data acquisition, and naive RAG baseline

**Goal:** A working naive RAG pipeline running on Indian annual reports with first eval numbers checked into the repo.

**Day 1 — Environment & repo skeleton**
1. Create a new GitHub repo: `bharat-fin-agent` (or whatever name). Make it public.
2. Set up Python 3.11 in a fresh `venv` or `uv` environment.
3. Initial `requirements.txt`: `langchain langgraph langsmith langchain-openai langchain-community chromadb sentence-transformers rank-bm25 unstructured[pdf] pypdf pandas pydantic streamlit ragas datasets python-dotenv`.
4. Folder structure:
   ```
   bharat-fin-agent/
   ├── data/
   │   ├── pdfs/              # downloaded annual reports
   │   └── eval/              # FinanceQA + BhashaBench cached
   ├── src/
   │   ├── ingestion/         # PDF parsing, chunking, embedding
   │   ├── agents/            # individual LangGraph nodes
   │   ├── graph/             # LangGraph wiring
   │   ├── eval/              # RAGAS scripts
   │   └── ui/                # Streamlit app
   ├── notebooks/             # exploratory work
   ├── results/               # eval CSVs, comparison tables
   ├── .env.example
   └── README.md
   ```
5. Sign up for: OpenAI or Anthropic API key, LangSmith free tier (essential — sign up at smith.langchain.com), and optionally Tavily for web search (later).
6. Initial commit, push to GitHub. From day 1, work in feature branches and merge via PRs — this looks professional.

**Day 2 — Pick your companies and download annual reports**
1. Pick 15 Nifty 50 companies across 5 sectors (so you get diverse content): IT (TCS, Infosys, Wipro), Banking (HDFC Bank, ICICI Bank, SBI), Energy (Reliance, ONGC, NTPC), FMCG (HUL, ITC, Nestle India), Auto (Maruti, M&M, Tata Motors).
2. For each company, download the last 3 annual reports (FY22, FY23, FY24) from NSE corporate filings or the company's IR page. ~45 PDFs total, will land at 2-5GB. Store under `data/pdfs/{COMPANY}/{YEAR}.pdf`.
3. Build a simple `data/metadata.csv` with columns: company, year, sector, pdf_path, num_pages.

**Day 3 — Load and cache the eval datasets**
1. `from datasets import load_dataset; ds = load_dataset("sweatSmile/FinanceQA")`. Inspect the schema, save it to `data/eval/financeqa.parquet`.
2. Same for `adityarane/financial-qa-dataset` and `bharatgenai/BhashaBench-Finance`.
3. Filter FinanceQA to questions about companies whose annual reports you actually downloaded. This is your held-out eval set — aim for at least 100 questions.

**Day 4 — PDF parsing and text chunking**
1. Use `unstructured.io` with `strategy="hi_res"` to extract both text and tables from PDFs. Initially just store text; tables come in Week 4.
2. Write `src/ingestion/parse.py` that takes a PDF and returns a list of (text, metadata) chunks where metadata is `{company, year, page_num, section}`.
3. Use LangChain's `RecursiveCharacterTextSplitter` with `chunk_size=1000`, `chunk_overlap=200`.
4. Run on all 45 PDFs, cache intermediate results to `data/chunks/` so you don't reparse every time.

**Day 5 — Embeddings and vector store**
1. Pick embeddings: `BAAI/bge-large-en-v1.5` (free, runs locally on CPU but slow) or OpenAI `text-embedding-3-small` (cheap, ~$1 for the whole corpus).
2. Embed all chunks, store in a persistent Chroma collection at `data/chroma/`.
3. Sanity test: write a tiny script that takes a query string and returns top-5 chunks. Try *"What was Reliance Jio's ARPU in FY23?"* and verify it returns Reliance Jio chunks.

**Day 6 — Naive RAG pipeline**
1. Build the simplest possible pipeline in `src/agents/naive_rag.py`: query → embed → top-5 retrieve → stuff into prompt → LLM → answer.
2. Use GPT-4o-mini or Claude Haiku for cost. Save outputs as JSON with fields: question, retrieved_chunks, answer, latency, cost.

**Day 7 — First evaluation run**
1. Set up RAGAS in `src/eval/run_ragas.py`. Compute faithfulness, answer_relevancy, context_precision, context_recall on your 100-question subset.
2. Use Claude Sonnet 4.6 or GPT-4o as the RAGAS judge LLM (not the same model that generated the answers — avoids self-bias).
3. Write the results to `results/week1_naive_baseline.csv`.
4. Update README with a "Baseline" section showing the numbers. This is your starting point — everything from here is improvement.

**End of week 1 deliverable:** README has a baseline table like:

| Configuration | Faithfulness | Answer Relevancy | Context Precision | Accuracy |
|---|---|---|---|---|
| Naive RAG | 0.62 | 0.71 | 0.55 | 0.48 |

---

### Week 2 — LangGraph orchestration and LangSmith observability

**Goal:** The naive pipeline is now wrapped in LangGraph with full tracing. You can see every node execution in LangSmith.

**Day 1 — Define your AgentState**
1. In `src/graph/state.py`, define a Pydantic `AgentState` TypedDict with fields: `question, sub_queries, retrieved_chunks, table_results, web_results, draft_answer, final_answer, citations, grading_score, iteration_count, errors`.
2. This is the shared state every node reads and writes.

**Day 2 — Wrap retrieval + generation as LangGraph nodes**
1. Convert your retriever into a `retrieve_node(state) -> state` function.
2. Same for `generate_node`.
3. Wire them with `StateGraph` and a single straight edge. This is functionally identical to your naive RAG but in LangGraph form.

**Day 3 — Add LangSmith tracing**
1. Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY=...` in `.env`.
2. Run a few queries and confirm traces appear in LangSmith UI. Take a screenshot — you'll use it in the README.

**Day 4 — Add the planner node**
1. `planner_node(state)` takes the question, prompts an LLM to decompose it into 1-3 sub-queries.
2. Use structured output (Pydantic model) so you get a clean list back, not free-form text.
3. Test on multi-hop questions like *"Compare TCS and Infosys revenue growth in FY23."* — verify it splits into two sub-queries.

**Day 5 — Add the synthesizer node**
1. Replace the simple "stuff into prompt" generator with a proper synthesizer that takes all retrieved chunks, all sub-query answers, and produces a final answer with inline citations like `[Reliance AR FY23, p. 102]`.
2. Use a stronger model here — GPT-4o or Claude Sonnet — quality matters.

**Day 6 — Add a basic hallucination critic**
1. `critic_node(state)` extracts factual claims from the draft answer and asks an LLM "is each claim supported by the retrieved context? yes/no."
2. If any claim fails, set `state.needs_retry = True`. For now don't actually loop yet — just log the failures.

**Day 7 — Re-run eval**
1. Run the agentic graph on the same 100-question eval set.
2. Add row to `results/comparison.csv`. You should see modest gains from the planner alone, especially on multi-hop questions.
3. Commit a fresh README update with the v1 architecture diagram and the updated comparison table.

**End of week 2 deliverable:** LangGraph version running with planner + synthesizer + critic, LangSmith traces visible, comparison row added.

---

### Week 3 — Hybrid retrieval and the Corrective-RAG loop

**Goal:** Implement self-grading retrieval with query rewriting. This is where you start seeing real lifts.

**Day 1 — Add BM25 alongside dense retrieval**
1. Install `rank_bm25`. Build a BM25 index over the same chunks.
2. Write a hybrid retriever that runs both, takes top-10 from each, and unions the results.
3. Add a cross-encoder reranker (`BAAI/bge-reranker-large`) that re-scores the union and returns top-5.

**Day 2 — Wire hybrid retrieval into the graph**
1. Replace your dense-only retriever node with the hybrid version.
2. Quick eval — you should see context precision climb noticeably.

**Day 3 — Build the relevance grader node**
1. `grader_node(state)` takes each retrieved chunk and asks an LLM "does this chunk contain information relevant to the sub-query? yes/no, plus a 1-5 relevance score."
2. Use a small fast model — Haiku or GPT-4o-mini.
3. Store grades in state.

**Day 4 — Build the query rewriter node**
1. `rewrite_node(state)` triggers when the grader's average score is below a threshold (try 3.0 to start).
2. Prompts an LLM to reformulate the question, optionally adding domain terms (e.g. user asks "how much did they earn" → rewriter expands to "net profit attributable to shareholders revenue").
3. Maintain an `iteration_count` in state to prevent infinite loops — cap at 3 retries.

**Day 5 — Wire conditional edges**
1. In LangGraph, after `grader_node`, add a conditional edge:
   - If `avg_score >= threshold` → go to `synthesizer_node`
   - If `avg_score < threshold AND iteration_count < 3` → go to `rewrite_node` → back to `retrieve_node`
   - If `iteration_count >= 3` → go to `synthesizer_node` with a "low confidence" flag.

**Day 6 — Wire the critic loop properly**
1. After `synthesizer_node`, `critic_node` runs. If it flags hallucination, loop back to `retrieve_node` with a hint to find more grounding evidence. Cap at 2 critic retries.

**Day 7 — Eval and error analysis**
1. Re-run RAGAS. Faithfulness should climb significantly (this is the biggest lift you'll see).
2. Open the failures in LangSmith. Categorize them: "missing from corpus", "retrieved wrong company", "table not extracted yet" (this is your Week 4 motivation), "math error".
3. Commit `results/comparison.csv` updated with the v2 agentic numbers.

**End of week 3 deliverable:** Corrective-RAG loop is live. Comparison table shows 3 rows: naive, agentic v1, agentic v2 (with grader). You can name the failure categories that motivate Week 4.

---

### Week 4 — The table agent (the highest-value week)

**Goal:** Numeric questions that require reading financial tables now work. This is where your project starts massively outperforming a vanilla RAG.

**Day 1 — Re-ingest with table extraction**
1. Re-run `unstructured.io` with `infer_table_structure=True`. This returns `Table` elements with HTML representations.
2. For each table, store: company, year, page_num, table_title (extracted via LLM call on the surrounding context), HTML, and a converted Pandas DataFrame.
3. Save to `data/tables/{company}_{year}_{page}.parquet` plus a master `data/tables/index.json`.

**Day 2 — Embed table titles for retrieval**
1. Create a second Chroma collection (`tables`) where each "document" is a table's title + first row + last row (so embeddings capture what the table is about).
2. Test: query "Reliance balance sheet 2023" → should return the actual balance sheet table.

**Day 3 — Build the table agent node**
1. `table_agent_node(state)` does: retrieve top-3 relevant tables → load them as DataFrames → prompt an LLM to write a small Python/pandas snippet that answers the question from those DataFrames.
2. Execute the snippet in a sandboxed REPL (use `langchain_experimental.tools.PythonREPLTool` with `globals` restricted to the DataFrames only — never let it execute arbitrary user code in production, but for a portfolio project this is fine with a clear disclaimer).
3. Return the computed answer + reference to which table it came from.

**Day 4 — Update the router**
1. Modify the planner or add a new `router_node` that, for each sub-query, classifies it as: `narrative` (→ dense+BM25 retriever), `numeric` (→ table agent), or `external` (→ web search, coming Week 5).
2. Use few-shot examples in the router prompt. Numeric markers: "what was", "how much", "ratio", "margin", "growth %", specific years, currency amounts.

**Day 5 — Wire table agent into the graph**
1. Add the conditional edge from router to table_agent_node when query type is numeric.
2. Synthesizer now needs to merge text-retrieval results with table-agent computed results.

**Day 6 — Test on the hardest questions**
1. Specifically pick questions from FinanceQA that involve ratios, multi-year comparisons, segment breakdowns. These are where table extraction matters.
2. Debug failures in LangSmith — most will be parsing errors (tables that span multiple pages, merged cells).

**Day 7 — Eval**
1. Re-run RAGAS + accuracy on the full eval set.
2. You should see a big jump on numeric questions. Add row to comparison table.

**End of week 4 deliverable:** Table agent working, comparison table has 4 rows, and you can demo a question like *"What was HDFC Bank's net interest margin in FY23?"* with the agent computing it from the actual table.

---

### Week 5 — Web search agent, critic polish, and bilingual support

**Goal:** Handle out-of-corpus questions and tighten quality. Add the bilingual angle.

**Day 1 — Tavily web search integration**
1. Sign up for Tavily free tier. `pip install tavily-python`.
2. `web_search_node(state)` runs a search and returns top-3 results with snippets.
3. Wire as router option for "external" queries like "what was Reliance's latest quarterly result" (post-cutoff of your downloaded annual reports).

**Day 2 — Add the `Indian_Financial_News` dataset as a local web fallback**
1. Index the news dataset into a third Chroma collection. When Tavily isn't available, search this instead.
2. This is realistic — many enterprises restrict outbound web calls, so showing both options is good engineering.

**Day 3 — Hindi/bilingual support**
1. Add a language-detection step at the start of the graph (use `langdetect` or just an LLM call).
2. If Hindi, translate the query to English for retrieval (since your corpus is English), retrieve in English, then translate the final answer back to Hindi.
3. Test with a few BhashaBench questions. This is a 1-day effort that adds a huge wow-factor to the demo.

**Day 4 — Critic polish**
1. Improve critic prompts based on Week 3 failure analysis. Add a "numeric verification" mode that specifically checks if numbers in the answer appear in retrieved tables.
2. Add a "refusal" path — if after all retries the agent can't ground its answer, it should explicitly say *"I don't have enough information to answer this from the available filings."* instead of hallucinating.

**Day 5 — Final error analysis**
1. Run full eval. Open the top 20 failures. Fix what's cheap to fix (prompt tweaks, threshold tuning).
2. Document the remaining failure modes in `LIMITATIONS.md`. Recruiters appreciate honest project limitations more than over-claiming.

**Day 6 — Run BhashaBench-Finance eval**
1. Run a sample (200 questions) from BhashaBench-Finance as a *second*, independent eval set.
2. This shows your system generalizes beyond the dataset you developed against.

**Day 7 — Big comparison table**
1. Finalize `results/comparison.csv` with all configurations across both eval sets, including latency and cost per query.
2. Generate a nice visualization (matplotlib bar chart, saved as PNG) for the README.

**End of week 5 deliverable:** Full agentic system with web search, bilingual support, refusal handling. Two independent eval sets show consistent improvement.

---

### Week 6 — Streamlit UI, documentation, deployment, and a blog post

**Goal:** Make it shareable, demo-able, and recruiter-discoverable.

**Day 1 — Streamlit UI**
1. `src/ui/app.py` — chat interface with question input, streaming answer output, citations shown as clickable expanders showing the source chunk and page number.
2. Sidebar: dropdown to pick which configuration to run (naive vs full agentic) — this lets recruiters *see* the difference live.
3. Show LangSmith trace link for each query so curious viewers can dig in.

**Day 2 — Architecture diagram and README**
1. Use [excalidraw.com](https://excalidraw.com) or Mermaid to draw the full architecture (similar to the one I sketched above but with your specific node names).
2. Write README sections: Problem, Architecture, Dataset, Setup, Results (the comparison table is the hero), Limitations, Future Work, Acknowledgments.
3. README first impression matters most — put the results table and architecture diagram in the first scroll.

**Day 3 — Loom video walkthrough**
1. Record 3-5 minutes: 30 seconds problem framing, 1 minute architecture explanation pointing at the diagram, 2 minutes live demo of 3-4 representative queries (one narrative, one numeric, one comparative, one Hindi), 30 seconds results table.
2. Embed the Loom in the README.

**Day 4 — Deploy**
1. Deploy the Streamlit app to **HuggingFace Spaces** (free, Indian-recruiter-friendly because HF profiles are easy to browse) or **Streamlit Community Cloud**.
2. The corpus PDFs are too big — use a small subset (say 5 companies × 2 years = 10 PDFs) for the public demo, with a note that the full eval used the larger corpus.
3. Pin the API keys via environment variables in HF Spaces secrets.

**Day 5 — Optional blog post on Medium / LinkedIn**
1. Title: something like *"Building a multi-agent RAG system over Indian annual reports — how Corrective-RAG cut our hallucination rate by X%"*.
2. Structure: problem, dataset, naive baseline numbers, each agentic improvement with the before/after delta, final comparison, limitations.
3. Link to the GitHub and the live demo.
4. This is your single highest-leverage activity for visibility. Even one decent blog post that gets shared on LinkedIn can get you a recruiter ping.

**Day 6 — Polish and resume update**
1. Update your resume with the project bullets (template I gave in the previous message).
2. Make sure the GitHub README, the Loom video, the live demo, and your LinkedIn project entry all link to each other.
3. Pin the repo to your GitHub profile.

**Day 7 — Buffer day**
1. Always leave a buffer. Use it to fix whatever you didn't get to, polish the Loom, or start applying.

**End of week 6 deliverable:** Polished public project with live demo, GitHub repo, video, optional blog post, updated resume.

---

## A few practical tips

- **Don't get stuck in week 1 perfecting the chunking.** Ship a mediocre baseline fast, then improve. Bad ingestion is fixable later; not having a baseline blocks everything.
- **Commit eval numbers after every major change.** Your `results/comparison.csv` should grow by one row per significant feature. This is the project's spine.
- **Use LangSmith liberally.** Every time you debug a failed query, screenshot the trace. These screenshots are gold for the README and for interview discussions ("here's a trace showing how the system recovered from a poor retrieval...").
- **Stay within the free tier as long as possible.** OpenAI/Anthropic credits + Groq free tier + LangSmith hobby tier + Tavily free tier + HuggingFace Spaces free tier = ~₹0-500 total project cost.
- **Pick a specific finance vertical if you want to go even deeper.** E.g., focus only on Indian banks (HDFC, ICICI, SBI, Axis, Kotak, Bandhan, Federal). Then you can claim "specialized banking-sector RAG" which is even more recruiter-targeted for Mumbai-based banking AI roles.
- **Update your LinkedIn headline** to include "Building Agentic RAG systems" or similar — recruiters search for these keywords now.
