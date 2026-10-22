import base64
import io
import ipaddress
import threading
import time
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from . import context_fix, docx_export, gpu, learning, ocr, segment, spell, storage, synth

STATIC = Path(__file__).parent / "static"

def _preload_model():
    """Load the model in the background so the first photo doesn't wait for it."""
    from . import local_model

    try:
        with gpu.use("reader"):
            local_model._load()
        adapter = " + ваш адаптер почерка" if local_model.has_adapter() else ""
        print(f"Модель загружена: {local_model.model_name()} ({local_model.device()}){adapter}", flush=True)
    except Exception as e:
        print(f"Не удалось загрузить модель: {type(e).__name__}: {e}", flush=True)


def _lan_addresses() -> list[str]:
    """IPv4 addresses the phone might reach; the default-route one first."""
    import socket

    found = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.168.0.1", 80))  # no packet is sent; this just picks the main interface
            found.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in found and not ip.startswith("127."):
                found.append(ip)
    except OSError:
        pass
    # Home networks use private ranges; VPN adapters (e.g. Radmin VPN, 26.x.x.x) don't, so list them last.
    found.sort(key=lambda ip: not ipaddress.ip_address(ip).is_private)
    return found or ["IP-адрес-этого-компьютера"]


def _quiet_windows_disconnects():
    """On Windows, a phone closing its connection logs a harmless ConnectionResetError traceback; hide those."""
    import asyncio
    import sys

    if sys.platform != "win32":
        return
    loop = asyncio.get_running_loop()

    def handler(loop, context):
        if isinstance(context.get("exception"), ConnectionResetError):
            return
        loop.default_exception_handler(context)

    loop.set_exception_handler(handler)


@asynccontextmanager
async def lifespan(_app):
    _quiet_windows_disconnects()
    addresses = _lan_addresses()
    print(f"\n  Откройте на телефоне (та же Wi-Fi сеть): http://{addresses[0]}:8000", flush=True)
    for ip in addresses[1:]:
        print(f"  Если не открывается, попробуйте: http://{ip}:8000", flush=True)
    print("  Телефон не подключается? Запустите allow_phone.bat (разрешит порт 8000 в брандмауэре).\n", flush=True)
    if ocr.BACKEND == "local":
        threading.Thread(target=_preload_model, daemon=True).start()
    threading.Thread(target=spell.available, daemon=True).start()  # downloads the dictionary on first run
    if context_fix.enabled():
        context_fix.start_worker()
    yield


app = FastAPI(title="Handapp", lifespan=lifespan)


def _page_or_404(page_id: str) -> dict:
    page = storage.get_page(page_id)
    if page is None:
        raise HTTPException(404, "Страница не найдена")
    return page


def _run_ocr(page: dict) -> dict:
    try:
        text, uncertain = ocr.recognize(page["id"])
        page.update(status="recognized", ocr_text=text, text=text, uncertain=uncertain, error=None,
                    unknown=spell.unknown_words(text), changes=[],
                    fix_status="pending" if context_fix.enabled() else None, fix_error=None)
    except ocr.OCRError as e:
        page.update(status="error", error=str(e))
    storage.save_page(page)
    return page


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(STATIC / "favicon.ico")


@app.get("/apple-touch-icon.png", include_in_schema=False)
@app.get("/apple-touch-icon-precomposed.png", include_in_schema=False)
def touch_icon():
    return FileResponse(STATIC / "icon.png")


# ---------- pages ----------

@app.get("/api/pages")
def list_pages():
    return storage.list_pages()


@app.post("/api/pages")
def upload_page(file: UploadFile = File(...)):
    try:
        page = storage.create_page(file.file.read())
    except Exception:
        raise HTTPException(400, "Не удалось открыть изображение")
    return _run_ocr(page)


@app.get("/api/pages/{page_id}/image")
def page_image(page_id: str):
    _page_or_404(page_id)
    return FileResponse(storage.page_image_path(page_id), media_type="image/jpeg")


class PageText(BaseModel):
    text: str


@app.put("/api/pages/{page_id}")
def save_page_text(page_id: str, body: PageText):
    page = _page_or_404(page_id)
    # Diff against the last saved text so re-saving the same fixes doesn't count them twice.
    learned = learning.record_correction(page["text"], body.text)
    page.update(text=body.text, status="verified", verified_at=time.time())
    storage.save_page(page)
    return {**page, "learned": learned}


@app.post("/api/pages/{page_id}/retry")
def retry_page(page_id: str):
    return _run_ocr(_page_or_404(page_id))


@app.delete("/api/pages/{page_id}")
def delete_page(page_id: str):
    _page_or_404(page_id)
    storage.delete_page(page_id)
    return {"ok": True}


# ---------- handwriting samples ----------

@app.get("/api/samples")
def list_samples():
    return storage.list_samples()


@app.post("/api/samples")
def add_sample(file: UploadFile = File(...), label: str = Form(...)):
    if not label.strip():
        raise HTTPException(400, "Напишите, что изображено на образце")
    try:
        return storage.add_sample(file.file.read(), label)
    except Exception:
        raise HTTPException(400, "Не удалось открыть изображение")


def _data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


@app.post("/api/samples/line")
def cut_line(file: UploadFile = File(...), label: str = Form(...), mode: str = Form("letters")):
    """Cuts a photo of one written line into letters or words and shows the result; nothing is saved as a sample yet."""
    labels = segment.labels_for(label, mode)
    if not labels:
        raise HTTPException(400, "Напишите, что написано в строке")
    try:
        line_id = storage.save_line_photo(file.file.read())
    except Exception:
        raise HTTPException(400, "Не удалось открыть изображение")
    path = storage.line_photo_path(line_id)
    with Image.open(path) as img:
        img.load()
    try:
        boxes, warnings = segment.segment(img, len(labels), mode)
    except segment.SegmentError as e:
        path.unlink(missing_ok=True)
        raise HTTPException(400, str(e))
    height = _line_height(boxes)
    preview = segment.overlay(img, boxes)
    preview.thumbnail((1200, 1200))
    return {
        "id": line_id,
        "label": label.strip(),
        "warnings": warnings,
        "overlay": _data_url(segment.jpeg(preview, 80)),
        "pieces": [{"box": b, "label": t, "image": _data_url(segment.jpeg(segment.crop(img, b, height), 85))}
                   for b, t in zip(boxes, labels)],
    }


def _line_height(boxes: list[dict]) -> int:
    heights = sorted(b["y1"] - b["y0"] for b in boxes)
    return heights[len(heights) // 2] if heights else 50


class Box(BaseModel):
    x0: int
    y0: int
    x1: int
    y1: int


class Piece(BaseModel):
    box: Box
    label: str


class LineSave(BaseModel):
    pieces: list[Piece]
    whole_line: str = ""  # also keep the whole photo with this label, if not empty


@app.post("/api/samples/line/{line_id}")
def save_line(line_id: str, body: LineSave):
    path = storage.line_photo_path(line_id)
    if path is None:
        raise HTTPException(404, "Фото строки не найдено, загрузите его заново")
    with Image.open(path) as img:
        img.load()
    boxes = [p.box.model_dump() for p in body.pieces if p.label.strip()]
    height = _line_height(boxes)
    items = [(segment.jpeg(segment.crop(img, p.box.model_dump(), height)), p.label)
             for p in body.pieces if p.label.strip()]
    if body.whole_line.strip():
        items.append((storage.normalize_image(path.read_bytes()), body.whole_line))
    added = storage.add_samples(items, line=line_id)
    path.unlink(missing_ok=True)
    return {"added": len(added)}


@app.get("/api/samples/{sample_id}/image")
def sample_image(sample_id: str):
    path = storage.sample_image_path(sample_id)
    if "/" in sample_id or not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/jpeg")


@app.delete("/api/samples/{sample_id}")
def delete_sample(sample_id: str):
    storage.delete_sample(sample_id)
    return {"ok": True}


# ---------- learning stats & export ----------

@app.get("/api/stats")
def stats():
    pages = storage.list_pages()
    return {
        "proofreader": {"enabled": context_fix.enabled(), "model": context_fix.model_name() if context_fix.enabled() else "",
                        "loaded": context_fix.is_loaded(), "dictionary": spell.available()},
        "synthetic": len(synth.list_lines()),
        "pages": len(pages),
        "verified": sum(p["status"] == "verified" for p in pages),
        "samples": len(storage.list_samples()),
        "backend": ocr.BACKEND,
        **_model_info(),
        "substitutions": learning.top_substitutions(30),
        "letters": learning.character_confusions(15),
    }


def _model_info() -> dict:
    if ocr.BACKEND != "local":
        return {"model": "тестовый режим", "loaded": True, "adapter": False, "device": "-"}
    from . import local_model

    return {"model": local_model.model_name(), "loaded": local_model.is_loaded(),
            "adapter": local_model.has_adapter(), "device": local_model.device()}


class ExportRequest(BaseModel):
    page_ids: list[str]
    title: str = ""
    page_breaks: bool = False


@app.post("/api/export")
def export_docx(body: ExportRequest):
    pages = [_page_or_404(pid) for pid in body.page_ids]
    if not pages:
        raise HTTPException(400, "Нет страниц для экспорта")
    data = docx_export.build_docx([p["text"] for p in pages], body.title, body.page_breaks)
    name = (body.title.strip() or time.strftime("konspektas-%Y-%m-%d")) + ".docx"
    return Response(
        data,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}"},
    )


@app.get("/api/dataset.zip")
def dataset_zip():
    """Verified (image, text) pairs: the training set for a future locally fine-tuned model."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in storage.list_pages():
            if p["status"] == "verified":
                z.write(storage.page_image_path(p["id"]), f"pages/{p['id']}.jpg")
                z.writestr(f"pages/{p['id']}.txt", p["text"])
        for s in storage.list_samples():
            z.write(storage.sample_image_path(s["id"]), f"samples/{s['id']}.jpg")
            z.writestr(f"samples/{s['id']}.txt", s["label"])
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": "attachment; filename=handapp-dataset.zip"})


app.mount("/static", StaticFiles(directory=STATIC), name="static")
