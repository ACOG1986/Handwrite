"""File-based storage: everything lives under data/ so it is easy to back up or inspect."""

import io
import json
import os
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
LEARNING_FILE = DATA_DIR / "learning.json"

# Claude downsizes anything larger than ~1568px on the long edge, so sending more only costs upload time.
MAX_EDGE = 1568

_lock = threading.Lock()


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
    tmp.replace(path)


def normalize_image(raw: bytes) -> bytes:
    """Rotate per EXIF, convert to RGB, shrink to MAX_EDGE and re-encode as JPEG."""
    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((MAX_EDGE, MAX_EDGE), Image.LANCZOS)
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
    if "/" in page_id or ".." in page_id:
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
    folder = PAGES_DIR / page_id
    if get_page(page_id) is None:
        return
    for f in folder.iterdir():
        f.unlink()
    folder.rmdir()


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
    _ensure_dirs()
    sample = {"id": new_id(), "label": label.strip(), "created": time.time()}
    (SAMPLES_DIR / f"{sample['id']}.jpg").write_bytes(normalize_image(raw_image))
    with _lock:
        index = _samples_index()
        index.append(sample)
        _write_json(SAMPLES_DIR / "samples.json", index)
    return sample


def sample_image_path(sample_id: str) -> Path:
    return SAMPLES_DIR / f"{sample_id}.jpg"


def delete_sample(sample_id: str):
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
