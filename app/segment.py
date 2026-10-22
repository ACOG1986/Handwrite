"""Cuts a photo of a handwritten line (alphabet, separate letters or words) into pieces and pairs them with the typed text.

Works on ink projections, so dots and hooks that sit above or below a letter (ė, į, š, ž, ą) stay with it.
Only PIL and numpy, no model needed.
"""

import io

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

MIN_CROP_HEIGHT = 96  # tiny crops are upscaled so the vision model sees the strokes
MAX_CROP_EDGE = 768


class SegmentError(Exception):
    pass


def labels_for(text: str, mode: str) -> list[str]:
    """"letters": every non-space character is one piece; "words": every space-separated word."""
    if mode == "letters":
        return [ch for ch in text if not ch.isspace()]
    return text.split()


def ink_mask(img: Image.Image) -> np.ndarray:
    """True where there is pen ink. Evens out lighting first and drops ruled or squared notebook lines."""
    gray = ImageOps.grayscale(img)
    w, h = gray.size
    # Paper brightness: a max filter on a small copy removes the ink, then blur and scale back.
    small = gray.resize((max(1, w // 8), max(1, h // 8)), Image.BILINEAR)
    bg = small.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(3)).resize((w, h), Image.BILINEAR)
    norm = np.asarray(gray, np.float32) / np.maximum(np.asarray(bg, np.float32), 1.0)
    # Among everything darker than paper, split pen ink from the paler notebook lines and shadows.
    darker = norm[norm < 0.92]
    threshold = float(np.clip(_otsu(darker), 0.45, 0.8)) if darker.size > 50 else 0.7
    ink = norm < threshold
    # Ruled lines: long straight horizontal runs, which handwriting doesn't have.
    _clear_long_runs(ink, w // 6)
    _clear_long_runs(ink.T, h // 2)
    # Photo edges (table, shadow of the phone).
    m = max(2, min(w, h) // 100)
    ink[:m, :] = ink[-m:, :] = False
    ink[:, :m] = ink[:, -m:] = False
    # Remove single-pixel specks.
    cleaned = Image.fromarray(ink.astype(np.uint8) * 255).filter(ImageFilter.MedianFilter(3))
    return np.asarray(cleaned) > 127


def _clear_long_runs(ink: np.ndarray, length: int):
    """Erases every horizontal run of ink at least `length` pixels long (in place; pass .T for vertical)."""
    for y in np.flatnonzero(ink.sum(axis=1) >= length):
        for a, b in _runs(ink[y]):
            if b - a >= length:
                ink[y, a:b] = False


def _otsu(values: np.ndarray) -> float:
    hist, edges = np.histogram(np.clip(values, 0, 1), bins=256, range=(0, 1))
    centers = (edges[:-1] + edges[1:]) / 2
    weight = np.cumsum(hist)
    total = weight[-1]
    mean = np.cumsum(hist * centers)
    between = (mean[-1] * weight - mean * total) ** 2 / np.maximum(weight * (total - weight), 1)
    return float(centers[np.argmax(between)])


def _runs(active: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of each run of True values."""
    padded = np.concatenate([[False], active, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(changes[::2], changes[1::2]))


def _smooth(profile: np.ndarray, size: int) -> np.ndarray:
    size = max(1, size)
    return np.convolve(profile, np.ones(size) / size, mode="same")


def find_lines(ink: np.ndarray) -> list[tuple[int, int]]:
    """Rows of writing, top to bottom. Diacritics above a line are merged into it."""
    rows = ink.sum(axis=1).astype(np.float32)
    if rows.max() == 0:
        return []
    bands = _runs(_smooth(rows, 3) > rows.max() * 0.02)
    bands = [b for b in bands if rows[b[0]:b[1]].sum() > rows.sum() * 0.01]
    if not bands:
        return []
    typical = np.median([e - s for s, e in bands])
    merged = [list(bands[0])]
    for s, e in bands[1:]:
        if s - merged[-1][1] < typical * 0.6 or (e - s) < typical * 0.4 or (merged[-1][1] - merged[-1][0]) < typical * 0.4:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [tuple(b) for b in merged]


def _pieces_in_line(ink: np.ndarray, top: int, bottom: int) -> list[list[int]]:
    """Column runs of ink inside one line: [x0, x1)."""
    cols = ink[top:bottom].sum(axis=0)
    pieces = [list(r) for r in _runs(cols > 0)]
    if not pieces:
        return []
    amounts = [cols[a:b].sum() for a, b in pieces]
    typical = np.median(amounts)
    return [p for p, amount in zip(pieces, amounts) if amount >= typical * 0.04]


def _split_widest(pieces: list[dict], ink: np.ndarray, expected: float) -> bool:
    """Cuts the widest piece where it has the least ink (letters that touch). Returns False if nothing to cut."""
    candidates = [p for p in pieces if p["x1"] - p["x0"] > expected * 1.3]
    if not candidates:
        return False
    p = max(candidates, key=lambda p: p["x1"] - p["x0"])
    cols = _smooth(ink[p["top"]:p["bottom"], p["x0"]:p["x1"]].sum(axis=0).astype(np.float32), 3)
    width = p["x1"] - p["x0"]
    lo, hi = int(width * 0.25), max(int(width * 0.75), int(width * 0.25) + 1)
    cut = p["x0"] + lo + int(np.argmin(cols[lo:hi]))
    i = pieces.index(p)
    pieces[i:i + 1] = [{**p, "x1": cut, "gap_after": 0}, {**p, "x0": cut}]
    return True


def segment(img: Image.Image, count: int, mode: str = "letters") -> tuple[list[dict], list[str]]:
    """Boxes (x0, y0, x1, y1) of `count` pieces in reading order, plus warnings for the user."""
    ink = ink_mask(img)
    lines = find_lines(ink)
    if not lines:
        raise SegmentError("На фото не нашлось написанного. Снимите ближе, при хорошем свете.")
    if len(lines) > count:
        raise SegmentError(f"Нашёл {len(lines)} строк(и), а в подписи всего {count} элемент(ов). Снимите по одной строке.")

    pieces = []
    for top, bottom in lines:
        line_pieces = _pieces_in_line(ink, top, bottom)
        for j, (x0, x1) in enumerate(line_pieces):
            nxt = line_pieces[j + 1][0] if j + 1 < len(line_pieces) else None
            # The end of a line is always a boundary.
            gap = (nxt - x1) if nxt is not None else float("inf")
            pieces.append({"x0": x0, "x1": x1, "top": top, "bottom": bottom, "gap_after": gap})
    if not pieces:
        raise SegmentError("На фото не нашлось написанного.")

    warnings = []
    found = len(pieces)
    # Too many pieces: keep only the widest gaps as boundaries (parts of one letter or word are close together).
    if len(pieces) > count:
        # A letter can fall apart into two or three strokes, but many more pieces than letters means a wrong label.
        if mode == "letters" and found > count * 1.5 and found - count >= 3:
            warnings.append(f"На фото больше отдельных значков ({found}), чем букв в подписи ({count}). "
                            "Проверьте, что подпись совпадает со строкой.")
        gaps = sorted(range(len(pieces) - 1), key=lambda i: -pieces[i]["gap_after"])
        keep = set(gaps[:count - 1])
        groups, cur = [], [pieces[0]]
        for i in range(1, len(pieces)):
            if i - 1 in keep:
                groups.append(cur)
                cur = []
            cur.append(pieces[i])
        groups.append(cur)
        pieces = [{"x0": g[0]["x0"], "x1": g[-1]["x1"], "top": g[0]["top"], "bottom": g[0]["bottom"],
                   "gap_after": g[-1]["gap_after"]} for g in groups]
    # Too few: letters touch each other, cut the widest pieces.
    elif len(pieces) < count:
        widths = sorted(p["x1"] - p["x0"] for p in pieces)
        expected = sum(widths) / count
        while len(pieces) < count and _split_widest(pieces, ink, expected):
            pass
        if len(pieces) < count:
            raise SegmentError(f"Нашёл только {found} отдельных фрагментов, а в подписи {count}. "
                               "Пишите с пробелами побольше или проверьте подпись.")
        warnings.append(f"Некоторые буквы или слова слиплись ({found} фрагментов на {count}), я разрезал их сам. Проверьте нарезку.")

    boxes = []
    for p in pieces:
        region = ink[p["top"]:p["bottom"], p["x0"]:p["x1"]]
        ys = np.flatnonzero(region.any(axis=1))
        y0, y1 = (p["top"] + ys[0], p["top"] + ys[-1] + 1) if len(ys) else (p["top"], p["bottom"])
        boxes.append({"x0": int(p["x0"]), "y0": int(y0), "x1": int(p["x1"]), "y1": int(y1)})
    return boxes, warnings


def crop(img: Image.Image, box: dict, line_height: int) -> Image.Image:
    """The piece with a white margin, upscaled if small."""
    pad = max(4, int(line_height * 0.15))
    w, h = img.size
    piece = img.crop((max(0, box["x0"] - pad), max(0, box["y0"] - pad), min(w, box["x1"] + pad), min(h, box["y1"] + pad)))
    if piece.height < MIN_CROP_HEIGHT:
        scale = MIN_CROP_HEIGHT / piece.height
        piece = piece.resize((max(1, round(piece.width * scale)), MIN_CROP_HEIGHT), Image.LANCZOS)
    piece.thumbnail((MAX_CROP_EDGE, MAX_CROP_EDGE), Image.LANCZOS)
    return piece


def overlay(img: Image.Image, boxes: list[dict]) -> Image.Image:
    """The photo with numbered boxes, so the user can see where the cuts went."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    width = max(2, img.width // 400)
    for i, b in enumerate(boxes):
        color = (47, 91, 211) if i % 2 == 0 else (211, 47, 91)
        draw.rectangle((b["x0"], b["y0"], b["x1"], b["y1"]), outline=color, width=width)
    return out


def jpeg(img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    return buf.getvalue()
