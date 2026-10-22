"""File-based storage: everything lives under data/ so it is easy to back up or inspect."""

import io
import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()  # iPhone photos (HEIC)
except ImportError:
    pass

DATA_DIR = Path(os.environ.get("HANDAPP_DATA", Path(__file__).resolve().parent.parent / "data"))
PAGES_DIR = DATA_DIR / "pages"
SAMPLES_DIR = DATA_DIR / "samples"
LINES_DIR = SAMPLES_DIR / "lines"  # photos of a written line waiting to be cut into samples
LEARNING_FILE = DATA_DIR / "learning.json"

# Line photos keep more detail: one letter is a small part of the photo.
LINE_MAX_EDGE = 3000

# Long edge of a stored page photo; the reader shrinks it further to HANDAPP_MAX_PIXELS (local_model.py).
MAX_EDGE = 1568

_lock = threading.Lock()
_ID = re.compile(r"[0-9A-Za-z-]+")  # what new_id() makes; anything else (.., \ on Windows) is refused


def valid_id(item_id: str) -> bool:
    return bool(_ID.fullmatch(item_id))


def _ensure_dirs():
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write_json(path: Path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(20):
        try:
            tmp.replace(path)
            return
        except PermissionError:  # Windows: another thread is reading the file right now
            if attempt == 19:
                raise
            time.sleep(0.05)


def normalize_image(raw: bytes, max_edge: int = MAX_EDGE) -> bytes:
    """Rotate per EXIF, convert to RGB, shrink to max_edge and re-encode as JPEG."""
    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((max_edge, max_edge), Image.LANCZOS)
    out = io.BytesIO()
    img.save(out, "JPEG", quality=90)
    return out.getvalue()


def new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


# ---------- pages ----------

def create_page(raw_image: bytes) -> dict:
    _ensure_dirs()
    page_id = new_id()
    folder = PAGES_DIR / page_id
    folder.mkdir()
    (folder / "image.jpg").write_bytes(normalize_image(raw_image))
    page = {
        "id": page_id,
        "created": time.time(),
        "status": "pending",  # pending -> recognized -> verified | error
        "ocr_text": "",
        "text": "",
        "uncertain": [],
        "error": None,
    }
    save_page(page)
    return page


def page_image_path(page_id: str) -> Path:
    return PAGES_DIR / page_id / "image.jpg"


def get_page(page_id: str) -> dict | None:
    if not valid_id(page_id):
        return None
    return _read_json(PAGES_DIR / page_id / "page.json", None)


def save_page(page: dict):
    with _lock:
        _write_json(PAGES_DIR / page["id"] / "page.json", page)


def list_pages() -> list[dict]:
    _ensure_dirs()
    pages = [_read_json(p / "page.json", None) for p in sorted(PAGES_DIR.iterdir()) if p.is_dir()]
    return [p for p in pages if p]


def delete_page(page_id: str):
    if get_page(page_id) is None:
        return
    shutil.rmtree(PAGES_DIR / page_id)


def verified_pages(limit: int) -> list[dict]:
    """Most recent pages whose text the user confirmed; used as personal few-shot examples."""
    pages = [p for p in list_pages() if p["status"] == "verified" and p["text"].strip()]
    return pages[-limit:] if limit > 0 else []


# ---------- handwriting samples ----------

def _samples_index() -> list[dict]:
    return _read_json(SAMPLES_DIR / "samples.json", [])


def list_samples() -> list[dict]:
    _ensure_dirs()
    return _samples_index()


def add_sample(raw_image: bytes, label: str) -> dict:
    return add_samples([(normalize_image(raw_image), label)])[0]


def add_samples(items: list[tuple[bytes, str]], **extra) -> list[dict]:
    """Stores ready JPEGs with their labels; `extra` fields (e.g. which line they were cut from) go into the index."""
    _ensure_dirs()
    added = []
    for jpeg, label in items:
        sample = {"id": new_id(), "label": label.strip(), "created": time.time(), **extra}
        (SAMPLES_DIR / f"{sample['id']}.jpg").write_bytes(jpeg)
        added.append(sample)
    with _lock:
        index = _samples_index()
        index.extend(added)
        _write_json(SAMPLES_DIR / "samples.json", index)
    return added


def sample_image_path(sample_id: str) -> Path:
    return SAMPLES_DIR / f"{sample_id}.jpg"


def save_line_photo(raw_image: bytes) -> str:
    """Keeps an uploaded line photo until the user confirms the cut. Old unconfirmed ones are dropped."""
    LINES_DIR.mkdir(parents=True, exist_ok=True)
    for old in LINES_DIR.glob("*.jpg"):
        if time.time() - old.stat().st_mtime > 24 * 3600:
            old.unlink(missing_ok=True)
    line_id = new_id()
    (LINES_DIR / f"{line_id}.jpg").write_bytes(normalize_image(raw_image, LINE_MAX_EDGE))
    return line_id


def line_photo_path(line_id: str) -> Path | None:
    path = LINES_DIR / f"{line_id}.jpg"
    return path if valid_id(line_id) and path.exists() else None


def delete_sample(sample_id: str):
    if not valid_id(sample_id):
        return
    with _lock:
        index = [s for s in _samples_index() if s["id"] != sample_id]
        _write_json(SAMPLES_DIR / "samples.json", index)
    sample_image_path(sample_id).unlink(missing_ok=True)


# ---------- learned corrections ----------

def load_learning() -> dict:
    return _read_json(LEARNING_FILE, {"substitutions": {}})


def save_learning(data: dict):
    _ensure_dirs()
    with _lock:
        _write_json(LEARNING_FILE, data)
