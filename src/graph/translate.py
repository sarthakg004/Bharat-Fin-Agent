"""
translate.py  ·  src/graph/translate.py

Language detection + LLM translation for the bilingual flow.

Pipeline:
    detect(question)            → "en" | "hi" | other ISO code
    translate(text, target=...)  → translated text (or pass-through if already)

We use `langdetect` for cheap detection (no LLM call) and the existing chat
LLM for translation. The translator deliberately asks for "no commentary"
output via structured-output, so we don't need to strip prefatory phrases like
"Here is the translation:".

Why an LLM and not a dedicated MT model? The corpus is English; we only need
high-quality translation in/out of Hindi for a few user-facing strings. Using
the agent's existing LLM avoids a new dependency.
"""

from __future__ import annotations

from typing import Optional

from src.graph.state import Translation


# Hindi gets first-class support; everything else falls back to "en".
_ISO_TO_NAME = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "kn": "Kannada",
}


def detect_language(text: str) -> str:
    """Detect ISO 639-1 language code. Returns 'en' on failure.

    Uses `langdetect` (deterministic seed) so the same input always yields the
    same answer. Short strings are unreliable; we treat <8 chars as 'en'.
    """
    text = (text or "").strip()
    if len(text) < 8:
        return "en"
    try:
        from langdetect import detect, DetectorFactory

        DetectorFactory.seed = 0  # deterministic
        return detect(text) or "en"
    except Exception:
        return "en"


def language_name(code: str) -> str:
    return _ISO_TO_NAME.get(code, code or "English")


TRANSLATE_PROMPT = """\
Translate the text below into {target}. Preserve numbers, units (₹, crore, %),
ticker symbols, and proper nouns verbatim. Return only the translation, no
preface, no commentary.

Text:
\"\"\"{text}\"\"\"
"""


def translate_text(
    text: str,
    target_code: str,
    llm,
    source_code: Optional[str] = None,
) -> str:
    """Translate `text` into the target language using `llm`.

    `llm` must support `.with_structured_output(Translation)`. If translation
    fails for any reason the original text is returned (the caller can decide
    to flag this in errors); we never want a translation hop to break the run.
    """
    if not text:
        return text
    if source_code and source_code == target_code:
        return text
    target_name = language_name(target_code)
    prompt = TRANSLATE_PROMPT.format(target=target_name, text=text)
    # First try structured output (gives us a clean string with no preface).
    try:
        out: Translation = llm.with_structured_output(Translation).invoke(prompt)
        result = (out.text or "").strip()
        if result:
            return result
    except Exception:
        pass
    # Fallback: plain invoke. Some models won't fit the structured-output
    # schema reliably; in that case a raw call still gives a usable translation
    # (we strip a leading "Translation:"-style preface defensively).
    try:
        resp = llm.invoke(prompt)
        raw = getattr(resp, "content", None) or str(resp)
        raw = raw.strip().strip('"').strip("'").strip()
        for prefix in ("Translation:", "translation:", "Here is the translation:"):
            if raw.lower().startswith(prefix.lower()):
                raw = raw[len(prefix):].strip()
        return raw or text
    except Exception:
        return text
