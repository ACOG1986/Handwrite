"""Handwriting recognition with a local vision model, personalised with the user's corrections."""

import os
import re

from . import gpu, learning, storage

BACKEND = os.environ.get("HANDAPP_OCR", "local")  # "mock" for testing the UI without a model
AUTO_FIX_MIN_COUNT = int(os.environ.get("HANDAPP_AUTOFIX_MIN", "2"))

# Also used verbatim as the training prompt in train.py, so the fine-tuned model sees what it learned on.
BASE_PROMPT = """Transcribe this photographed handwritten notebook page. The text is mostly Lithuanian.

Rules:
- Write exactly what is written. Do not translate, summarise or correct anything.
- Keep Lithuanian letters: ą č ę ė į š ų ū ž.
- Keep the reading order. Join lines that only wrap at the edge of the page; start a new paragraph where the writer did, with a blank line between paragraphs.
- Use "# " for titles, "- " for bullet points, "1. " for numbered items, **bold** for emphasised words, __underline__ for underlined words.
- Write formulas as plain text, e.g. x^2 + 3x = 0.
- If a word is unclear, write your best guess inside double braces: {{word}}.
- Output only the transcription."""


class OCRError(Exception):
    pass


def build_prompt() -> str:
    subs = learning.top_substitutions(25)
    if not subs:
        return BASE_PROMPT
    hints = "\n".join(f"- {s['wrong']} → {s['right']}" for s in subs)
    return f"{BASE_PROMPT}\n\nThis writer's words that were misread before (wrong → correct):\n{hints}"


def _split_uncertain(raw: str) -> tuple[str, list[str]]:
    uncertain = re.findall(r"\{\{(.+?)\}\}", raw)
    clean = re.sub(r"\{\{(.+?)\}\}", r"\1", raw)
    return clean.strip(), list(dict.fromkeys(uncertain))


def apply_learned_fixes(text: str) -> tuple[str, list[str]]:
    """Replace words the user has corrected the same way at least AUTO_FIX_MIN_COUNT times."""
    fixes = {s["wrong"]: s["right"] for s in learning.top_substitutions(10_000) if s["count"] >= AUTO_FIX_MIN_COUNT}
    if not fixes:
        return text, []
    applied = []

    def swap(m):
        word = m.group(0)
        if word in fixes:
            applied.append(fixes[word])
            return fixes[word]
        return word

    return re.sub(r"\w+", swap, text), applied


def recognize(page_id: str) -> tuple[str, list[str]]:
    """Returns (text, words to double-check)."""
    if BACKEND == "mock":
        raw = "# Lietuvių kalba\n\nŠiandien mokemės apie {{dalyvius}} ir padalyvius.\n\n- ąžuolas\n- čiužinys\n- **ėjimas**"
    else:
        from . import local_model

        failure = None
        try:
            raw = local_model.generate(storage.page_image_path(page_id), build_prompt())
        except Exception as e:  # model download, out of memory, missing CUDA...
            failure = f"Локальная модель не смогла распознать страницу: {type(e).__name__}: {e}"
            if "OutOfMemory" in type(e).__name__:
                failure += " Не хватило памяти видеокарты: уменьшите HANDAPP_MAX_PIXELS в .env или закройте другие программы."
        if failure:
            gpu.free_cache()  # outside the except block, so the failed run's tensors are already released
            raise OCRError(failure)
    if not raw.strip():
        raise OCRError("Модель вернула пустой ответ.")
    text, uncertain = _split_uncertain(raw)
    text, auto_fixed = apply_learned_fixes(text)
    return text, list(dict.fromkeys(uncertain + auto_fixed))
