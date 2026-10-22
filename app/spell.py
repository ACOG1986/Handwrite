"""Lithuanian spelling check (Hunspell lt_LT from LibreOffice) plus the writer's own vocabulary.

Words the dictionary doesn't know are shown to the user to double-check, and their suggestions are passed
to the proofreader as hints. The dictionary is downloaded once into data/dict/.
"""

import re
import threading
import time
import urllib.request

from . import learning, storage

DICT_URL = "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/lt_LT/"
DICT_DIR = storage.DATA_DIR / "dict"

_WORD = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*", re.UNICODE)  # letters only: numbers and formulas aren't checked

_lock = threading.Lock()
_dict = None
_failed = False
_own_cache = (0.0, set())  # (when computed, words): reading every verified page per word was slow
OWN_WORDS_TTL = 30


def _load():
    global _dict, _failed
    if _dict is not None or _failed:
        return _dict
    with _lock:
        if _dict is not None or _failed:
            return _dict
        try:
            from spylls.hunspell import Dictionary

            DICT_DIR.mkdir(parents=True, exist_ok=True)
            for name in ("lt.aff", "lt.dic"):
                path = DICT_DIR / name
                if not path.exists():
                    tmp = path.with_suffix(".part")
                    urllib.request.urlretrieve(DICT_URL + name, tmp)
                    tmp.replace(path)
            _dict = Dictionary.from_files(str(DICT_DIR / "lt"))
        except Exception as e:  # no internet on first run, missing spylls: the app works without the check
            print(f"Литовский словарь недоступен, проверка слов отключена: {type(e).__name__}: {e}", flush=True)
            _failed = True
    return _dict


def available() -> bool:
    return _load() is not None


def forget_own_words():
    """Call after the user saves a page so its words count as known right away."""
    global _own_cache
    _own_cache = (0.0, set())


def _own_words() -> set[str]:
    """Words the writer used in verified pages or typed as corrections: their names, terms, abbreviations."""
    global _own_cache
    if time.time() - _own_cache[0] < OWN_WORDS_TTL:
        return _own_cache[1]
    words = set()
    for page in storage.verified_pages(10_000):
        words.update(w.lower() for w in _WORD.findall(page["text"]))
    for s in learning.top_substitutions(10_000):
        words.add(s["right"].lower())
    _own_cache = (time.time(), words)
    return words


def unknown_words(text: str) -> list[str]:
    """Words neither the dictionary nor the writer's own pages know, in order of appearance."""
    d = _load()
    if d is None:
        return []
    own = _own_words()
    out = []
    for word in _WORD.findall(text):
        if len(word) < 2 or word in out or word.lower() in own:
            continue
        parts = [p for p in word.split("-") if not (p.isupper() and len(p) <= 5)]  # abbreviations: LR, KTU-ą
        if not all(d.lookup(part) or len(part) < 2 for part in parts):
            out.append(word)
    return out


def suggestions(word: str, limit: int = 4) -> list[str]:
    d = _load()
    if d is None:
        return []
    try:
        return list(d.suggest(word))[:limit]
    except Exception:
        return []
