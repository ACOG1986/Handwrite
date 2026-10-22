"""Synthetic training lines: Lithuanian text drawn in handwriting fonts on notebook paper.

They teach the model the Lithuanian letters (ą č ę ė į š ų ū ž) in many hand-like shapes before it learns
your own handwriting from your verified pages. train.py mixes them in automatically.

    python -m app.synth --count 3000      # or synth.bat on Windows

Text comes from your verified pages, any .txt files you put in data/texts/, and random Lithuanian Wikipedia
articles (downloaded once). Fonts (SIL Open Font License) are downloaded from Google Fonts into data/fonts/.
"""

import argparse
import io
import json
import random
import re
import time
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import storage

SYNTH_DIR = storage.DATA_DIR / "synthetic"
FONTS_DIR = storage.DATA_DIR / "fonts"
TEXTS_DIR = storage.DATA_DIR / "texts"
INDEX = SYNTH_DIR / "lines.json"

FONTS_URL = "https://raw.githubusercontent.com/google/fonts/main/ofl/"
# Handwriting fonts that have all Lithuanian letters. Playwrite CZ/HU/PL are school cursive, the closest
# to how notebooks are written; the rest are print-like hands.
FONTS = [
    "playwritepl/PlaywritePL[wght].ttf",
    "playwritecz/PlaywriteCZ[wght].ttf",
    "playwritehu/PlaywriteHU[wght].ttf",
    "caveat/Caveat[wght].ttf",
    "caveatbrush/CaveatBrush-Regular.ttf",
    "kalam/Kalam-Regular.ttf",
    "patrickhand/PatrickHand-Regular.ttf",
    "shadowsintolighttwo/ShadowsIntoLightTwo-Regular.ttf",
    "marckscript/MarckScript-Regular.ttf",
    "badscript/BadScript-Regular.ttf",
    "mali/Mali-Regular.ttf",
    "sriracha/Sriracha-Regular.ttf",
    "playpensans/PlaypenSans[wght].ttf",
    "dancingscript/DancingScript[wght].ttf",
    "pangolin/Pangolin-Regular.ttf",
    "itim/Itim-Regular.ttf",
    "architectsdaughter/ArchitectsDaughter-Regular.ttf",
    "indieflower/IndieFlower-Regular.ttf",
    "reeniebeanie/ReenieBeanie.ttf",
    "cedarvillecursive/Cedarville-Cursive.ttf",
    "zeyada/Zeyada.ttf",
    "comforter/Comforter-Regular.ttf",
    "merienda/Merienda[wght].ttf",
]
LT_LETTERS = "ąčęėįšųūžĄČĘĖĮŠŲŪŽ"
WIKI_API = "https://lt.wikipedia.org/w/api.php"
USER_AGENT = "handapp/1.0 (https://github.com/ACOG1986/Handwrite; handwriting recognition training data)"

# Characters a notebook line may contain; lines with anything else (foreign scripts, symbols) are skipped.
_ALLOWED = re.compile(r"^[0-9A-Za-zĄČĘĖĮŠŲŪŽąčęėįšųūž .,;:!?\-–()\"„“'%+=/^*]+$")


# ---------- fonts ----------

class Font:
    def __init__(self, path):
        from fontTools.ttLib import TTFont

        self.path = path
        tt = TTFont(path)
        self.chars = {chr(c) for c in tt.getBestCmap()}
        self.weights = None
        if "fvar" in tt:
            for axis in tt["fvar"].axes:
                if axis.axisTag == "wght":
                    self.weights = (axis.minValue, axis.maxValue)

    def covers(self, text: str) -> bool:
        return all(c in self.chars or c == " " for c in text)

    def get(self, size: int, rng: random.Random) -> ImageFont.FreeTypeFont:
        font = ImageFont.truetype(str(self.path), size)
        if self.weights:
            lo, hi = self.weights
            try:
                font.set_variation_by_axes([rng.uniform(lo, lo + (hi - lo) * 0.6)])
            except (OSError, AttributeError):
                pass
        return font


def ensure_fonts() -> list[Font]:
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    fonts = []
    for rel in FONTS:
        path = FONTS_DIR / rel.split("/")[-1].replace("[", "_").replace("]", "")
        if not path.exists():
            try:
                with urllib.request.urlopen(FONTS_URL + urllib.parse.quote(rel), timeout=30) as r:
                    path.write_bytes(r.read())
            except Exception as e:
                print(f"  шрифт {rel} не скачался: {e}")
                continue
        try:
            font = Font(path)
        except Exception as e:
            print(f"  шрифт {path.name} не открывается: {e}")
            continue
        if font.covers(LT_LETTERS):
            fonts.append(font)
    return fonts


# ---------- text ----------

def _wikipedia_text(articles: int) -> str:
    """Plain text of random Lithuanian Wikipedia articles."""
    chunks = []
    for _ in range(articles):  # the API gives a full-page extract for only one page per request
        query = urllib.parse.urlencode({
            "action": "query", "format": "json", "generator": "random", "grnnamespace": 0, "grnlimit": 1,
            "prop": "extracts", "explaintext": 1, "exchars": 2000,
        })
        req = urllib.request.Request(f"{WIKI_API}?{query}", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
        chunks += [p.get("extract", "") for p in data.get("query", {}).get("pages", {}).values()]
        time.sleep(0.2)
    return "\n".join(chunks)


def load_texts(wiki_articles: int) -> list[str]:
    TEXTS_DIR.mkdir(parents=True, exist_ok=True)
    wiki = TEXTS_DIR / "wikipedia.txt"
    if wiki_articles and not wiki.exists():
        print(f"Скачиваю {wiki_articles} случайных статей литовской Википедии…")
        try:
            wiki.write_text(_wikipedia_text(wiki_articles), encoding="utf-8")
        except Exception as e:
            print(f"  Википедия недоступна ({e}), беру только ваши тексты.")
    texts = [p.read_text(encoding="utf-8", errors="ignore") for p in sorted(TEXTS_DIR.glob("*.txt"))]
    texts += [p["text"] for p in storage.verified_pages(10_000)]
    return texts


def _clean(text: str) -> str:
    text = re.sub(r"[#*_]+", "", text)  # Markdown marks from verified pages
    return re.sub(r"\s+", " ", text).strip()


def make_lines(texts: list[str], max_chars: int = 42) -> list[str]:
    """Cuts the texts into notebook-length lines."""
    lines = []
    for text in texts:
        for para in re.split(r"\n\s*\n|\n(?=[-#\d])", text):
            words = _clean(para).split(" ")
            line = []
            for w in words:
                if line and len(" ".join(line + [w])) > max_chars:
                    lines.append(" ".join(line))
                    line = []
                line.append(w)
            if line:
                lines.append(" ".join(line))
    return [l for l in dict.fromkeys(lines) if len(l) >= 3 and _ALLOWED.match(l)]


# ---------- drawing ----------

def _elastic(mask: np.ndarray, rng: random.Random, amount: float) -> np.ndarray:
    """Wobbles the strokes a little, like a hand that isn't a printer."""
    h, w = mask.shape
    nrng = np.random.default_rng(rng.randrange(1 << 30))
    gh, gw = max(2, h // 24), max(2, w // 24)
    fields = []
    for _ in range(2):
        coarse = Image.fromarray((nrng.standard_normal((gh, gw)) * 40 + 128).clip(0, 255).astype(np.uint8))
        fields.append((np.asarray(coarse.resize((w, h), Image.BICUBIC), dtype=np.float32) - 128) / 40 * amount)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sx = np.clip(xx + fields[0], 0, w - 1)
    sy = np.clip(yy + fields[1], 0, h - 1)
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = sx - x0, sy - y0
    m = mask.astype(np.float32)
    top = m[y0, x0] * (1 - fx) + m[y0, x1] * fx
    bottom = m[y1, x0] * (1 - fx) + m[y1, x1] * fx
    return (top * (1 - fy) + bottom * fy).astype(np.uint8)


def _ink_mask(text: str, font: Font, size: int, rng: random.Random) -> Image.Image:
    """White-on-black mask of the line, drawn word by word with small jumps in size, height and angle."""
    words = text.split(" ")
    pad = size
    width = int(sum(len(w) + 1 for w in words) * size * 0.9) + 2 * pad
    mask = Image.new("L", (width, int(size * 2.4)), 0)
    x, baseline = pad, int(size * 1.5)
    drift = rng.uniform(-0.04, 0.04) * size  # the line slowly rises or falls
    for i, word in enumerate(words):
        s = max(10, int(size * rng.uniform(0.93, 1.07)))
        f = font.get(s, rng)
        w = int(f.getlength(word)) + 4
        tile = Image.new("L", (w + s, int(s * 2.4)), 0)
        ImageDraw.Draw(tile).text((s // 2, int(s * 1.5)), word, font=f, fill=255, anchor="ls")
        tile = tile.rotate(rng.uniform(-2.0, 2.0), resample=Image.BICUBIC, expand=False)
        y = int(baseline + drift * i / max(len(words), 1) + rng.uniform(-0.05, 0.05) * size) - int(s * 1.5)
        mask.paste(255, (x - s // 2, y), tile)
        x += w + int(f.getlength(" ") * rng.uniform(0.7, 1.6))
    return mask.crop((0, 0, min(width, x + pad // 2), mask.height))


def _slant(mask: Image.Image, shear: float) -> Image.Image:
    extra = int(abs(shear) * mask.height) + 2
    w = mask.width + extra
    offset = -extra if shear > 0 else 0
    return mask.transform((w, mask.height), Image.AFFINE, (1, shear, offset, 0, 1, 0),
                          resample=Image.BICUBIC)


def _paper(w: int, h: int, size: int, rng: random.Random) -> np.ndarray:
    v, warm = rng.randint(222, 252), rng.randint(0, 14)  # white to slightly yellowish paper, never coloured
    base = np.array([v, v - warm // 3, v - warm], dtype=np.float32)
    paper = np.ones((h, w, 3), dtype=np.float32) * base
    img = Image.fromarray(paper.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    kind = rng.choices(["grid", "lines", "blank"], weights=[5, 3, 1])[0]  # Lithuanian notebooks: mostly squares
    dark = rng.randint(35, 80)  # faint blue-grey rules
    line_color = tuple(int(c) for c in (base - np.array([dark, dark * 0.75, rng.randint(0, dark // 3)])))
    if kind == "grid":
        cell = int(size * rng.uniform(0.55, 0.75))
        ox, oy = rng.randrange(cell), rng.randrange(cell)
        for gx in range(ox, w, cell):
            draw.line([(gx, 0), (gx, h)], fill=line_color, width=1)
        for gy in range(oy, h, cell):
            draw.line([(0, gy), (w, gy)], fill=line_color, width=1)
    elif kind == "lines":
        gy = int(size * 1.5) + rng.randint(2, int(size * 0.25))  # writing sits on the rule
        draw.line([(0, gy), (w, gy)], fill=line_color, width=1)
        draw.line([(0, gy - int(size * 1.1)), (w, gy - int(size * 1.1))], fill=line_color, width=1)
    if rng.random() < 0.15:  # red margin
        mx = rng.randint(2, max(3, w // 12))
        draw.line([(mx, 0), (mx, h)], fill=(210, 80, 80), width=2)
    return np.asarray(img, dtype=np.float32)


def render_line(text: str, font: Font, rng: random.Random) -> Image.Image:
    size = rng.randint(44, 76)
    mask = _ink_mask(text, font, size, rng)
    mask = _slant(mask, rng.uniform(-0.28, 0.12))
    if rng.random() < 0.35:
        mask = mask.filter(ImageFilter.MaxFilter(3))  # thicker pen
    m = _elastic(np.asarray(mask), rng, rng.uniform(0.6, 2.2)).astype(np.float32) / 255
    h, w = m.shape
    paper = _paper(w, h, size, rng)
    ink = rng.choice([
        (rng.randint(15, 45), rng.randint(30, 70), rng.randint(120, 190)),  # blue ballpoint
        (rng.randint(10, 40), rng.randint(10, 40), rng.randint(15, 50)),  # black
        (rng.randint(40, 70), rng.randint(30, 60), rng.randint(110, 150)),  # violet-blue
    ])
    alpha = (m * rng.uniform(0.75, 1.0))[..., None]
    img = paper * (1 - alpha) + np.array(ink, dtype=np.float32) * alpha
    # Uneven light across the photo, and sensor noise.
    gx = np.linspace(rng.uniform(0.78, 1.0), rng.uniform(0.9, 1.05), w, dtype=np.float32)
    gy = np.linspace(rng.uniform(0.9, 1.03), rng.uniform(0.85, 1.03), h, dtype=np.float32)
    img = img * (gy[:, None] * gx[None, :])[..., None]
    img += np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(1.5, 6), img.shape)
    out = Image.fromarray(img.clip(0, 255).astype(np.uint8))
    out = out.rotate(rng.uniform(-1.5, 1.5), resample=Image.BICUBIC, expand=True, fillcolor=tuple(int(c) for c in paper[0, 0]))
    if rng.random() < 0.6:
        out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.1)))
    scale = rng.uniform(0.55, 1.0)
    out = out.resize((max(1, int(out.width * scale)), max(1, int(out.height * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=rng.randint(55, 90))
    return Image.open(io.BytesIO(buf.getvalue()))


# ---------- dataset ----------

def list_lines() -> list[dict]:
    try:
        return json.loads(INDEX.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def line_image_path(line_id: str):
    return SYNTH_DIR / f"{line_id}.jpg"


def _pick(lines: list[str], rng: random.Random) -> str:
    """Prefers lines with Lithuanian letters: those are the ones the model gets wrong."""
    for _ in range(4):
        line = rng.choice(lines)
        if any(c in LT_LETTERS for c in line):
            return line
    return line


def generate(count: int, fresh: bool = False, wiki_articles: int = 300, seed: int | None = None) -> int:
    rng = random.Random(seed)
    fonts = ensure_fonts()
    if not fonts:
        raise SystemExit("Не удалось скачать ни одного шрифта. Проверьте интернет.")
    print(f"Шрифтов с литовскими буквами: {len(fonts)}")
    lines = make_lines(load_texts(wiki_articles))
    if len(lines) < 50:
        raise SystemExit(f"Мало текста: {len(lines)} строк. Положите литовские тексты (.txt) в {TEXTS_DIR} "
                         f"или проверьте интернет для Википедии.")
    print(f"Строк текста: {len(lines)}")
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    index = [] if fresh else list_lines()
    if fresh:
        for old in SYNTH_DIR.glob("*.jpg"):
            old.unlink()
    started = time.time()
    for i in range(count):
        text = _pick(lines, rng)
        usable = [f for f in fonts if f.covers(text)]
        if not usable:
            continue
        font = rng.choice(usable)
        line_id = storage.new_id()
        render_line(text, font, rng).save(line_image_path(line_id), "JPEG", quality=92)
        index.append({"id": line_id, "label": text, "font": font.path.stem})
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{count}  {time.time() - started:.0f} с", flush=True)
            INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    return len(index)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=3000, help="how many lines to draw")
    ap.add_argument("--fresh", action="store_true", help="delete the lines drawn before")
    ap.add_argument("--wiki", type=int, default=300, help="random Wikipedia articles to download (0 = none)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    total = generate(args.count, args.fresh, args.wiki, args.seed)
    print(f"Готово. Синтетических строк: {total} в {SYNTH_DIR}. train.bat подмешает их к вашим страницам.")


if __name__ == "__main__":
    main()
