"""Second pass: a text model (Gemma 3) re-reads the recognized page and fixes words that don't fit the context.

The reader often confuses similar letters (e/ė/ę, a/ą, s/š, u/ų/ū) or misreads a word. Gemma sees the whole
text, the words the reader was unsure of and the ones the dictionary doesn't know, and proposes fixes. Only
small, look-alike substitutions into real words are kept, and every changed word is shown to the user.

On a 12 GB card the reader and Gemma take turns on the GPU, so pages are proofread in the background once
the reader has been idle for a few seconds: a batch of photos is read first, then proofread together.
"""

import difflib
import os
import re
import threading
import time
import unicodedata

from . import gpu, learning, spell, storage

BACKEND = os.environ.get("HANDAPP_OCR", "local")
IDLE_SECONDS = float(os.environ.get("HANDAPP_FIX_IDLE", "8"))
MAX_CHANGED_SHARE = 0.3  # more than this means the model rewrote the text instead of fixing it

PROMPT = """You proofread text that a handwriting recognizer read from a photo of a student's notebook page.
The text is mostly Lithuanian. The recognizer often confuses similar letters (a/ą, e/ę/ė, i/į/y, u/ų/ū,
s/š, z/ž, c/č, n/u, m/n, l/t, o/a) and sometimes misreads a whole word.

Fix only words that are clearly misread: words that don't exist in Lithuanian, or forms that don't fit
the grammar or meaning of the sentence. Keep everything else exactly as it is: the same words in the same
order, line breaks, punctuation, names, numbers, formulas and the Markdown marks (#, -, 1., **, __).
Do not rephrase, translate, add or remove words.
{hints}
Output only the corrected text.

Text:
{text}"""

_WORD = re.compile(r"\w+", re.UNICODE)

_lock = threading.Lock()
_model = None
_tokenizer = None
_worker_started = False


def enabled() -> bool:
    flag = os.environ.get("HANDAPP_CONTEXT_FIX")
    if flag is not None:
        return flag == "1"
    if BACKEND == "mock":
        return True
    try:
        import torch

        return torch.cuda.is_available()  # Gemma 12B needs the NVIDIA card; on a Mac turn it on with HANDAPP_CONTEXT_FIX=1
    except ImportError:
        return False


def model_name() -> str:
    if os.environ.get("HANDAPP_FIX_MODEL"):
        return os.environ["HANDAPP_FIX_MODEL"]
    if BACKEND == "mock":
        return "словарь (тестовый режим)"
    from . import local_model

    # Already 4-bit and not gated (no Hugging Face login needed). The 1B one is for trying it on a Mac.
    return "unsloth/gemma-3-12b-it-bnb-4bit" if local_model.device() == "cuda" else "unsloth/gemma-3-1b-it"


# ---------- model ----------

def _load():
    global _model, _tokenizer
    if _model is not None:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    from . import local_model

    name = model_name()
    kwargs = {"dtype": torch.bfloat16} if local_model.device() == "cuda" else {"dtype": torch.float16}
    if local_model.device() == "cuda":
        kwargs["device_map"] = "auto"
    try:  # Gemma 3 4B/12B/27B are image+text checkpoints; 1B is text-only
        model = AutoModelForImageTextToText.from_pretrained(name, **kwargs)
    except ValueError:
        model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if local_model.device() == "mps":
        model.to("mps")
    model.eval()
    _tokenizer = AutoTokenizer.from_pretrained(name)
    _model = model
    print(f"Корректор загружен: {name}", flush=True)


def unload():
    """Free the GPU for the reader. Call with the GPU lock held."""
    global _model, _tokenizer
    if _model is None:
        return
    _model = _tokenizer = None
    gpu.free_cache()


gpu.register("proofreader", unload)


def is_loaded() -> bool:
    return _model is not None


def _generate(prompt: str, max_new_tokens: int) -> str:
    import torch

    _load()
    inputs = _tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(_model.device)
    with torch.inference_mode():
        out = _model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return _tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# ---------- proofreading ----------

def build_hints(uncertain: list[str], unknown: list[str]) -> str:
    lines = []
    if uncertain:
        lines.append("The recognizer was unsure about: " + ", ".join(uncertain[:40]) + ".")
    if unknown:
        guesses = []
        for w in unknown[:30]:
            s = spell.suggestions(w)
            guesses.append(f"{w} (maybe: {', '.join(s)})" if s else w)
        lines.append("Not in the Lithuanian dictionary (may still be names or terms): " + "; ".join(guesses) + ".")
    subs = learning.top_substitutions(20)
    if subs:
        lines.append("This writer's words that were misread before (wrong → correct): "
                     + ", ".join(f"{s['wrong']} → {s['right']}" for s in subs) + ".")
    return ("\n" + "\n".join(lines) + "\n") if lines else ""


def _plain(word: str) -> str:
    """Lowercase without diacritics: ė, ę and e all count as the same letter when comparing look-alikes."""
    return "".join(c for c in unicodedata.normalize("NFD", word.lower()) if not unicodedata.combining(c))


def _looks_like_misread(old: str, new: str) -> bool:
    if old == new or any(c.isdigit() for c in old + new):
        return False
    if abs(len(old) - len(new)) > 3:
        return False
    return difflib.SequenceMatcher(a=_plain(old), b=_plain(new)).ratio() >= 0.6


def merge(original: str, proposed: str) -> tuple[str, list[dict]]:
    """Applies only the word-for-word fixes from `proposed` to `original`, keeping the original layout.

    Returns (text, [{"from": old, "to": new}, ...]).
    """
    proposed = re.sub(r"^```\w*\n|\n```\s*$", "", proposed.strip())
    spans = [(m.start(), m.end(), m.group(0)) for m in _WORD.finditer(original)]
    old_words = [w for _, _, w in spans]
    new_words = _WORD.findall(proposed)
    if not old_words or not new_words:
        return original, []
    edits = {}  # word index -> replacement
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=old_words, b=new_words, autojunk=False).get_opcodes():
        if op != "replace" or (i2 - i1) != (j2 - j1):
            continue  # added or dropped words: the model rewrote something, keep the original
        for k in range(i2 - i1):
            old, new = old_words[i1 + k], new_words[j1 + k]
            if _looks_like_misread(old, new) and (not spell.available() or not spell.unknown_words(new)):
                edits[i1 + k] = new
    if not edits or len(edits) > MAX_CHANGED_SHARE * len(old_words):
        return original, []
    out, pos, changes = [], 0, []
    for idx, (start, end, word) in enumerate(spans):
        if idx in edits:
            out.append(original[pos:start] + edits[idx])
            pos = end
            changes.append({"from": word, "to": edits[idx]})
    out.append(original[pos:])
    return "".join(out), changes


def _dictionary_proofread(text: str, unknown: list[str]) -> str:
    """Test mode without a model: take the dictionary's first suggestion for each unknown word."""
    for w in unknown:
        s = spell.suggestions(w, 1)
        if s:
            text = re.sub(rf"\b{re.escape(w)}\b", s[0], text)
    return text


def proofread(text: str, uncertain: list[str], unknown: list[str]) -> tuple[str, list[dict]]:
    if BACKEND == "mock":
        proposed = _dictionary_proofread(text, unknown)
    else:
        prompt = PROMPT.format(hints=build_hints(uncertain, unknown), text=text)
        with gpu.use("proofreader"):
            proposed = _generate(prompt, max_new_tokens=int(len(text) * 0.8) + 64)
    return merge(text, proposed)


def fix_page(page_id: str):
    page = storage.get_page(page_id)
    if not page or page.get("fix_status") != "pending":
        return
    text = page["ocr_text"]
    failure = None
    try:
        fixed, changes = proofread(text, page.get("uncertain", []), page.get("unknown", []))
    except Exception as e:  # out of memory, model download failed: the page keeps the reader's text
        failure = f"{type(e).__name__}: {e}"
    if failure:
        print(f"Корректор не сработал: {failure}", flush=True)
        gpu.free_cache()
        page = storage.get_page(page_id) or page
        page.update(fix_status="error", fix_error=failure)
        storage.save_page(page)
        return
    page = storage.get_page(page_id)  # re-read: the user may have saved or re-recognized it meanwhile
    if not page or page.get("fix_status") != "pending" or page["ocr_text"] != text:
        return
    if page["status"] != "recognized" or page["text"] != text:
        page.update(fix_status="skipped")  # the user already edited and saved it; don't overwrite their text
    else:
        page.update(text=fixed, changes=changes, unknown=spell.unknown_words(fixed), fix_status="done")
    storage.save_page(page)


def _pending() -> list[str]:
    return [p["id"] for p in storage.list_pages() if p.get("fix_status") == "pending"]


def _worker():
    while True:
        time.sleep(2)
        try:
            ids = _pending()
            if not ids or time.time() - gpu.last_used("reader") < IDLE_SECONDS:
                continue
            for page_id in ids:
                if gpu.waiting("reader"):  # a new photo arrived: let the reader go first
                    break
                fix_page(page_id)
        except Exception as e:
            print(f"Корректор: {type(e).__name__}: {e}", flush=True)


def start_worker():
    global _worker_started
    with _lock:
        if _worker_started:
            return
        _worker_started = True
    threading.Thread(target=_worker, daemon=True, name="proofreader").start()
