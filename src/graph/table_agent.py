"""
table_agent.py  ·  src/graph/table_agent.py

Answer a numeric question by retrieving relevant tables, asking an LLM to write
a tiny pandas snippet, and executing it in a sandbox.

Pipeline:
    question
        ↓ similarity-search top-K in the `tables` Chroma collection
        ↓ load each hit's parquet into a pandas DataFrame
        ↓ LLM writes a snippet that assigns the answer to `result`
        ↓ execute under a restricted globals dict (no imports, no I/O)
        ↓ return result + the code + which tables were used

Safety
------
`exec` is run with a curated `__builtins__` (no `open`, `import`, `eval`, etc.)
and a globals dict containing only `pd` and the loaded DataFrames. This is
adequate for a portfolio project — **do not** ship arbitrary user code through
this without a real sandbox (e.g. WASM, gVisor, subprocess with seccomp).

Usage as a library
------------------
    from src.graph.table_agent import TableAgent

    ta = TableAgent(chroma_dir="data/chroma", collection_name="tables")
    out = ta.answer("What was HDFC Bank's net interest margin in FY23?")
    print(out["answer"])
    print(out["code"])
    print(out["tables_used"])
"""

from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path
from typing import Optional, Union

from dotenv import load_dotenv

from src.llm import build_llm, resolve_api_key

load_dotenv()


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

CODE_SYSTEM = """\
You are a senior financial data analyst. Given one or more pandas DataFrames
extracted from a company's financial filing, write a short, defensive Python
snippet that computes the answer.

Rules:
- Use ONLY the supplied DataFrames and `pd` (pandas).
- Do NOT import any modules.
- Assign the final answer to a variable named `result`.
- Coerce numeric columns with `pd.to_numeric(..., errors="coerce")` when they
  may contain stray characters (₹, %, commas).
- Be tolerant of column-name variation (use case-insensitive substring matches).
- Keep the snippet under ~25 lines.
"""

CODE_PROMPT = """\
Question: {question}

Available DataFrames (already loaded — do not redefine):
{tables}

Return a JSON object with `code` (a string of Python code) and `explanation`
(one sentence). Remember: assign the answer to `result`.
"""


# Curated builtins so `exec` can't open files / import / spawn processes.
_SAFE_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in (
        "True", "False", "None",
        "abs", "all", "any", "bool", "dict", "enumerate", "float",
        "int", "len", "list", "max", "min", "print", "range", "round",
        "set", "sorted", "str", "sum", "tuple", "zip",
    )
}


class TableAgent:
    """Retrieval-augmented table QA via LLM-generated pandas snippets."""

    DEFAULT_CODE_MODELS = {
        "groq": "llama-3.3-70b-versatile",
        "gemini": "gemini-2.5-flash",
        "openai": "gpt-4o",
        "anthropic": "claude-sonnet-4-6",
    }

    def __init__(
        self,
        chroma_dir: Union[str, Path] = "data/chroma",
        collection_name: str = "tables",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        provider: str = "groq",
        code_model: Optional[str] = None,
        top_k: int = 3,
        api_key: Optional[str] = None,
        max_rows_preview: int = 5,
    ):
        self.chroma_dir = str(chroma_dir)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.provider = provider.lower()
        self.code_model = code_model or self.DEFAULT_CODE_MODELS[self.provider]
        self.top_k = top_k
        self.max_rows_preview = max_rows_preview

        # Lazy resources; validate the key up front.
        resolve_api_key(self.provider, api_key)
        self.api_key = api_key

        self._retriever = None
        self._llm = None

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def answer(self, question: str) -> dict:
        """Retrieve relevant tables and compute the answer with pandas.

        Returns a dict with:
            answer            str           the printable answer (or error)
            code              str           the snippet the LLM produced
            explanation       str           one-line description of the snippet
            tables_used       list[dict]    metadata for each loaded table
            stdout            str           anything the snippet printed
            error             Optional[str] exec-time error message (if any)
        """
        docs = self.retrieve_tables(question)
        if not docs:
            return {"answer": "No matching tables retrieved.",
                    "code": "", "explanation": "", "tables_used": [],
                    "stdout": "", "error": "no_tables"}

        dfs, descs = self._load_dataframes(docs)
        if not dfs:
            return {"answer": "Found candidate tables but none could be loaded as DataFrames.",
                    "code": "", "explanation": "", "tables_used": [],
                    "stdout": "", "error": "no_dataframes"}

        code, explanation, gen_error = self._generate_code(question, descs)
        if gen_error:
            return {"answer": f"Code generation failed: {gen_error}",
                    "code": code, "explanation": explanation,
                    "tables_used": descs, "stdout": "", "error": gen_error}

        result, stdout_text, exec_error = self._safe_exec(code, dfs)
        return {
            "answer": self._format_result(result) if exec_error is None else f"(exec error) {exec_error}",
            "code": code,
            "explanation": explanation,
            "tables_used": descs,
            "stdout": stdout_text,
            "error": exec_error,
        }

    def retrieve_tables(self, query: str, k: Optional[int] = None):
        return self._get_retriever().similarity_search(query, k=(k or self.top_k))

    # ------------------------------------------------------------------ #
    # DataFrame loading
    # ------------------------------------------------------------------ #

    def _load_dataframes(self, docs):
        import pandas as pd

        dfs: dict = {}
        descs: list[dict] = []
        for i, d in enumerate(docs):
            meta = d.metadata or {}
            path = meta.get("parquet_path")
            if not path:
                continue
            try:
                df = (pd.read_csv(path) if path.endswith(".csv")
                      else pd.read_parquet(path))
            except Exception as e:
                descs.append({
                    "name": f"df{i}", "title": meta.get("title", ""),
                    "company": meta.get("company", ""), "year": meta.get("year", ""),
                    "page": meta.get("page", ""),
                    "shape": None, "columns": [], "preview": f"(failed to load: {e})",
                    "parquet_path": path,
                })
                continue

            name = f"df{i}"
            dfs[name] = df
            descs.append({
                "name": name,
                "title": meta.get("title", ""),
                "company": meta.get("company", ""),
                "year": meta.get("year", ""),
                "page": meta.get("page", ""),
                "shape": list(df.shape),
                "columns": [str(c) for c in df.columns],
                "preview": df.head(self.max_rows_preview).to_string(),
                "parquet_path": path,
            })
        return dfs, descs

    # ------------------------------------------------------------------ #
    # Code generation
    # ------------------------------------------------------------------ #

    def _generate_code(self, question: str, descs: list[dict]) -> tuple[str, str, Optional[str]]:
        """Ask the LLM for a snippet via structured output. Returns (code, explanation, err)."""
        from langchain_core.messages import HumanMessage, SystemMessage

        from src.graph.state import PandasCode

        llm = self._get_llm()
        tables_block = "\n\n".join(self._describe(d) for d in descs)
        prompt = CODE_PROMPT.format(question=question, tables=tables_block)
        try:
            out: PandasCode = llm.with_structured_output(PandasCode).invoke(
                [SystemMessage(content=CODE_SYSTEM), HumanMessage(content=prompt)]
            )
            code = self._strip_code_fences(out.code)
            return code, out.explanation, None
        except Exception as e:
            # Fall back to plain text + regex extraction so we degrade gracefully.
            try:
                resp = llm.invoke(
                    [SystemMessage(content=CODE_SYSTEM), HumanMessage(content=prompt)]
                )
                code = self._strip_code_fences(resp.content)
                return code, "", f"structured-output fallback ({type(e).__name__})"
            except Exception as e2:
                return "", "", f"{type(e).__name__}: {e}; fallback: {type(e2).__name__}: {e2}"

    @staticmethod
    def _describe(d: dict) -> str:
        head = (
            f"# {d['name']}: {d.get('title','(no title)')} "
            f"— {d.get('company','?')} {d.get('year','?')} p{d.get('page','?')}\n"
            f"# shape={d.get('shape')} columns={d.get('columns', [])[:12]}"
        )
        return f"{head}\n# preview:\n# " + d.get("preview", "").replace("\n", "\n# ")

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        # ```python\n...\n``` or ```\n...\n```
        m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        return textwrap.dedent(m.group(1)) if m else text.strip()

    # ------------------------------------------------------------------ #
    # Sandboxed exec
    # ------------------------------------------------------------------ #

    @staticmethod
    def _safe_exec(code: str, dfs: dict) -> tuple[object, str, Optional[str]]:
        """Run `code` with a curated builtins dict + pd + the DataFrames.

        Captures stdout, returns (`result` value, stdout, error message).
        """
        import io
        from contextlib import redirect_stdout

        import pandas as pd

        scope = {"__builtins__": _SAFE_BUILTINS, "pd": pd, **dfs}
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                exec(code, scope)
            return scope.get("result"), buf.getvalue(), None
        except Exception as e:
            return None, buf.getvalue(), f"{type(e).__name__}: {e}"

    @staticmethod
    def _format_result(result) -> str:
        import pandas as pd

        if result is None:
            return "(snippet ran but did not assign `result`)"
        if isinstance(result, (pd.DataFrame, pd.Series)):
            return result.to_string()
        return str(result)

    # ------------------------------------------------------------------ #
    # Resources
    # ------------------------------------------------------------------ #

    def _get_retriever(self):
        if self._retriever is None:
            from langchain_chroma import Chroma
            from langchain_huggingface import HuggingFaceEmbeddings

            emb = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                encode_kwargs={"normalize_embeddings": True},
            )
            self._retriever = Chroma(
                collection_name=self.collection_name,
                embedding_function=emb,
                persist_directory=self.chroma_dir,
            )
        return self._retriever

    def _get_llm(self):
        if self._llm is None:
            self._llm = build_llm(
                self.provider, self.code_model, self.api_key, temperature=0.0
            )
        return self._llm


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description="Answer a numeric question via the table agent.")
    p.add_argument("--question", required=True)
    p.add_argument("--chroma-dir", default="data/chroma")
    p.add_argument("--collection", default="tables")
    p.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--provider", choices=["groq", "gemini", "openai", "anthropic"],
                   default="groq")
    p.add_argument("--code-model", default=None)
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    ta = TableAgent(
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        provider=args.provider,
        code_model=args.code_model,
        top_k=args.top_k,
    )
    out = ta.answer(args.question)
    print("=" * 60)
    print(f"Question: {args.question}")
    print(f"\nAnswer:\n{out['answer']}")
    print(f"\nExplanation: {out['explanation']}")
    print(f"\nCode:\n{out['code']}")
    if out.get("stdout"):
        print(f"\n--- stdout ---\n{out['stdout']}")
    if out.get("error"):
        print(f"\nError: {out['error']}")
    print("\nTables used:")
    for t in out["tables_used"]:
        print(f"  - {t.get('title')} ({t.get('company')} {t.get('year')} p{t.get('page')}) "
              f"shape={t.get('shape')}")


if __name__ == "__main__":
    main()
