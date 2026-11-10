"""Try the stick-figure gesture model with your own camera (a test page, not part of the app).

    uv run main.py skeleton-test          ->  http://127.0.0.1:8002

Serves flymnist/static/skeleton-test.html, the browser brain file the app already built
(data/web/brain_lamina.bin.gz, made by `uv run main.py serve` on first start) and the decoders of
data/pretrained_skeleton.joblib (made by `uv run main.py pretrain --step skeleton`). This computer only
(127.0.0.1: browsers allow the camera there without https). The decoders are derived from HaGRIDv2,
so they stay on this computer: they are not in flymnist/static/ and the static site does not include them.
"""

import gzip
import json
import mimetypes
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from .camera import CLASSES, DURATION_MS, IMG_SHAPE, export_decoder
from .data import DATA_DIR
from .pretrain_skeleton import OUT
from .webmodel import WEB_DIR

STATIC = Path(__file__).resolve().parent / "static"
mimetypes.add_type("text/javascript", ".js")     # as in server.py (Windows registry can say text/plain)
BRAIN = WEB_DIR / "brain_lamina.bin.gz"
REGION_NAMES = ["eye input", "optic lobe", "visual projection", "central brain", "descending", "other"]  # server.REGIONS


def create_app():
    if not BRAIN.exists():
        raise SystemExit(f"No browser brain file at {BRAIN}: start the app once (uv run main.py serve), then stop it.")
    if not OUT.exists():
        raise SystemExit(f"No stick-figure model at {OUT}: run  uv run main.py pretrain --step skeleton")
    saved = joblib.load(OUT)
    rate_hz = saved.get("meta", {}).get("rate_hz", 500.0)   # stick figures are shown at 1000 Hz (skeleton.RATE_HZ)
    with open(BRAIN, "rb") as f:   # unpacked size: the last 4 bytes of a gzip file
        f.seek(-4, 2)
        raw = int.from_bytes(f.read(4), "little")
    info = gzip.compress(json.dumps({
        "classes": CLASSES, "image_shape": list(IMG_SHAPE), "duration_ms": DURATION_MS, "rate_hz": rate_hz,
        "region_names": REGION_NAMES, "metrics": saved["metrics"], "meta": saved.get("meta", {}),
        "model": {"url": f"/api/brain/model.bin?v={BRAIN.stat().st_mtime_ns}", "bytes": BRAIN.stat().st_size, "raw_bytes": raw},
        "decoders": {k: export_decoder(v) for k, v in saved["models"].items()},
    }).encode())

    app = FastAPI(title="Stick-figure gesture test")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return RedirectResponse("/static/skeleton-test.html")

    @app.get("/api/skeleton")
    def skeleton_info():
        return Response(info, media_type="application/json", headers={"Content-Encoding": "gzip", "Cache-Control": "no-store"})

    @app.get("/api/brain/model.bin")
    def brain_model():
        if not BRAIN.exists():
            raise HTTPException(404, "brain file missing")
        return FileResponse(BRAIN, media_type="application/octet-stream")

    return app


def serve(port=8002):
    import uvicorn

    app = create_app()
    print(f"\n  stick-figure test page:  http://127.0.0.1:{port}   (this computer only)\n", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)
