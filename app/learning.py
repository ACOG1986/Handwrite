"""Learns from the user's corrections: which words the recognizer tends to misread, and how."""

import difflib
import re

from . import storage

_WORD = re.compile(r"\w+", re.UNICODE)


def _words(text: str) -> list[str]:
    return _WORD.findall(text)


def correction_pairs(ocr_text: str, corrected: str) -> list[tuple[str, str]]:
    """Word-level substitutions between what was recognized and what the user fixed it to."""
    a, b = _words(ocr_text), _words(corrected)
    pairs = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op != "replace":
            continue
        if (i2 - i1) == (j2 - j1):
            pairs.extend((w, r) for w, r in zip(a[i1:i2], b[j1:j2]) if w != r)
        else:
            pairs.extend(_pair_similar(a[i1:i2], b[j1:j2]))
    return pairs


def _pair_similar(wrong: list[str], right: list[str]) -> list[tuple[str, str]]:
    """In an uneven block (words added or removed too), keep only pairs that look like misreads of each other."""
    pairs, start = [], 0
    for w in wrong:
        best, best_j = 0.0, None
        for j in range(start, len(right)):
            ratio = difflib.SequenceMatcher(a=w.lower(), b=right[j].lower()).ratio()
            if ratio > best:
                best, best_j = ratio, j
        if best_j is not None and best > 0.6 and w != right[best_j]:
            pairs.append((w, right[best_j]))
            start = best_j + 1
    return pairs


def record_correction(ocr_text: str, corrected: str) -> int:
    pairs = correction_pairs(ocr_text, corrected)
    if not pairs:
        return 0
    data = storage.load_learning()
    subs = data.setdefault("substitutions", {})
    for wrong, right in pairs:
        key = f"{wrong}→{right}"
        subs[key] = subs.get(key, 0) + 1
    storage.save_learning(data)
    return len(pairs)


def top_substitutions(limit: int = 40) -> list[dict]:
    subs = storage.load_learning().get("substitutions", {})
    ranked = sorted(subs.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    out = []
    for key, count in ranked:
        wrong, right = key.split("→", 1)
        out.append({"wrong": wrong, "right": right, "count": count})
    return out


def character_confusions(limit: int = 15) -> list[dict]:
    """Single-character confusions (e.g. e→ė, a→ą) aggregated from all word corrections."""
    counts: dict[tuple[str, str], int] = {}
    for item in top_substitutions(limit=10_000):
        w, r = item["wrong"], item["right"]
        if len(w) != len(r):
            continue
        for cw, cr in zip(w, r):
            if cw != cr:
                counts[(cw, cr)] = counts.get((cw, cr), 0) + item["count"]
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:limit]
    return [{"wrong": w, "right": r, "count": c} for (w, r), c in ranked]
