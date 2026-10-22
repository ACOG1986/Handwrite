import io
import threading
import time
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import docx_export, learning, ocr, storage

STATIC = Path(__file__).parent / "static"

def _preload_model():
    """Load the model in the background so the first photo doesn't wait for it."""
    from . import local_model

    try:
        with local_model._lock:
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
    return found or ["IP-адрес-этого-компьютера"]


@asynccontextmanager
async def lifespan(_app):
    addresses = _lan_addresses()
    print(f"\n  Откройте на телефоне (та же Wi-Fi сеть): http://{addresses[0]}:8000", flush=True)
    for ip in addresses[1:]:
        print(f"  Если не открывается, попробуйте: http://{ip}:8000", flush=True)
    print("  Телефон не подключается? Запустите allow_phone.bat (разрешит порт 8000 в брандмауэре).\n", flush=True)
    if ocr.BACKEND == "local":
        threading.Thread(target=_preload_model, daemon=True).start()
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
        page.update(status="recognized", ocr_text=text, text=text, uncertain=uncertain, error=None)
    except ocr.OCRError as e:
        page.update(status="error", error=str(e))
    storage.save_page(page)
    return page


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


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
