"""Web app: draw a digit or use your webcam, and watch a fruit fly brain respond.

    uv run main.py serve      ->  http://127.0.0.1:8000  (and https://<this computer's LAN IP>:8443)

Every visitor runs their own copy of the brain in their browser (static/brainsim.js): the
server sends the connectome once and keeps the parts that are cheap or shared: image
preprocessing, knockout lookup, decoder training, everyone's camera recordings, and one
simulated fly body per visitor. Visitors are told apart by a cookie; whoever uses the app
on this computer is the teacher, whose recordings make up the class model.

The camera tab has two inputs with separate recordings and decoders: camera pictures
(/api/cam/*) and stick figures of the person's pose (/api/stick/*, flymnist/stick.py), whose
decoders start from a pre-trained model.
"""

import base64
import gzip
import io
import json
import mimetypes
import secrets
import socket
import threading
import time
from http.cookies import SimpleCookie
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp
import segno
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from scipy import ndimage

from .body import BodyPool
from .camera import (CLASSES as CAM_CLASSES, DURATION_MS as CAM_MS, IMG_SHAPE, check_trainable,
                     export_decoder, preprocess_frame, train_decoders)
from .data import load_annotations, load_connectome, load_mnist
from .lesion import Lesions, cross_hemisphere_mask
from .progress import Progress
from .readout import readouts_path
from .retina import build_retina, images_to_rates
from .sessions import _SID, SessionStore
from .simulate import feature_path, load_features, shuffle_connectome
from .stick import Pretrained, StickSession, landmarks_json, parse_landmarks, spec_version
from .stick import train as stick_train
from .webmodel import brain_file

# Windows takes file types from the registry, where .js is sometimes "text/plain"; browsers
# then refuse to run the page's JavaScript modules. Pin the types the static files need.
for _type, _ext in [("text/javascript", ".js"), ("text/javascript", ".mjs"), ("text/css", ".css"),
                    ("text/html", ".html"), ("image/gif", ".gif"), ("video/mp4", ".mp4")]:
    mimetypes.add_type(_type, _ext)

SID_COOKIE = "flysid"
STATIC = Path(__file__).resolve().parent / "static"

# Display regions (every neuron gets exactly one). Input neurons first, so they win.
REGIONS = [
    ("eye input", "#f5c542"),
    ("optic lobe", "#4fc3f7"),
    ("visual projection", "#b388ff"),
    ("central brain", "#5be3a0"),
    ("descending", "#ff6b6b"),
    ("other", "#8a9099"),
]
_SUPER_TO_REGION = {"optic": 1, "visual_projection": 2, "central": 3, "descending": 4}


# Neuropil labels for the 3D map, placed at the median position of neurons that live there.
# (column, values, paired per hemisphere?)
NEUROPILS = [
    ("Lamina", "cell_type", ["R1-6"], True),            # photoreceptors R1-6 end in the lamina
    ("Medulla", "cell_class", ["ME"], True),
    ("Lobula", "cell_class", ["LO"], True),
    ("Lobula plate", "cell_class", ["LOP"], True),
    ("Mushroom body", "cell_class", ["Kenyon_Cell"], True),
    ("Lateral horn", "cell_class", ["LHLN"], True),
    ("Antennal lobe", "cell_class", ["ALLN"], True),
    ("Central complex", "cell_class", ["CX"], False),
    ("SEZ", "cell_class", ["gustatory"], False),      # taste neurons end in the subesophageal zone
]


def neuropil_anchors(ann, pos_um):
    anchors = []
    for name, col, values, paired in NEUROPILS:
        m = ann[col].isin(values).to_numpy() & ~np.isnan(pos_um).any(1)
        for side in (["left", "right"] if paired else [None]):
            mm = m & (ann.side == side).to_numpy() if side else m
            if mm.sum() >= 10:
                anchors.append({"name": name, "side": side, "pos": np.round(np.median(pos_um[mm], 0), 1).tolist()})
    return anchors


def lan_ip():
    """This computer's address on the local network (the interface used for outgoing traffic)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.254.254.254", 1))  # UDP connect sends nothing; it only picks a route
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def preprocess(png_bytes):
    """Turn a canvas drawing (white ink on black) into an MNIST-style 28x28 image.

    Same recipe as MNIST: fit the digit into a 20x20 box (keeping aspect ratio),
    then centre its centre of mass in a 28x28 frame.
    """
    a = np.asarray(Image.open(io.BytesIO(png_bytes)).convert("L"), dtype=np.float32) / 255.0
    ys, xs = np.nonzero(a > 0.1)
    if len(ys) == 0:
        return np.zeros((28, 28), np.float32)
    crop = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = 20.0 / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    small = np.asarray(Image.fromarray((crop * 255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS), np.float32) / 255
    out = np.zeros((28, 28), np.float32)
    top, left = (28 - nh) // 2, (28 - nw) // 2
    out[top:top + nh, left:left + nw] = small
    cy, cx = ndimage.center_of_mass(out)
    out = ndimage.shift(out, (14 - cy, 14 - cx), order=1, mode="constant")
    return np.clip(out, 0, 1).astype(np.float32)


class ImageRequest(BaseModel):
    image: str  # data URL (PNG drawing or JPEG webcam frame)


class SessionCookie:
    """Give every browser a random id (cookie), so each visitor gets their own session and fly."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        cookie = SimpleCookie()
        for k, v in scope.get("headers", []):
            if k == b"cookie":
                try:
                    cookie.load(v.decode("latin-1"))
                except Exception:  # noqa: BLE001
                    pass
        sid = cookie[SID_COOKIE].value if SID_COOKIE in cookie else None
        new = not (sid and _SID.match(sid))
        if new:
            sid = secrets.token_urlsafe(12)
        scope.setdefault("state", {})["sid"] = sid

        async def send_with_cookie(msg):
            if new and msg["type"] == "http.response.start":
                c = f"{SID_COOKIE}={sid}; Path=/; Max-Age=2592000; SameSite=Lax".encode()
                msg = {**msg, "headers": [*msg.get("headers", []), (b"set-cookie", c)]}
            await send(msg)

        await self.app(scope, receive, send_with_cookie)


def _gz_json(obj):
    return gzip.compress(json.dumps(obj).encode(), 6)


def _gz_response(data, cache="no-store"):
    return Response(data, media_type="application/json", headers={"Content-Encoding": "gzip", "Cache-Control": cache})


def create_app(layer="lamina", rate_hz=500.0, duration_ms=100.0, shuffle=False, host="0.0.0.0", port=8000, https_port=0):
    fpath = feature_path(layer, rate_hz, duration_ms, shuffle)
    rpath = readouts_path(fpath)
    if not rpath.exists():
        raise SystemExit(f"No trained readouts at {rpath}.\nRun first:  uv run main.py all --train 3000 --test 1000")
    saved = joblib.load(rpath)
    readouts = saved["readouts"]

    print("loading fly brain ...", flush=True)
    pre, post, weight, root_ids = load_connectome()
    if shuffle:
        pre, post, weight = shuffle_connectome(pre, post, weight)
    ann = load_annotations(root_ids)
    input_idx, uv = build_retina(ann, layer)
    n = len(root_ids)

    region = np.full(n, len(REGIONS) - 1, np.uint8)
    for sc, r in _SUPER_TO_REGION.items():
        region[(ann.super_class == sc).to_numpy()] = r
    region[input_idx] = 0
    cell_type = ann.cell_type.fillna("untyped " + ann.super_class.fillna("neuron")).to_numpy()

    # The brain every browser downloads (built once, ~20 s, then cached on disk).
    model_path, model_version = brain_file(pre, post, weight, n, input_idx, uv, ann.side.to_numpy(), region, cell_type,
                                           layer + ("_shuffled" if shuffle else ""))

    with open(model_path, "rb") as f:   # unpacked size: the last 4 bytes of a gzip file
        f.seek(-4, 2)
        model_raw_bytes = int.from_bytes(f.read(4), "little")

    # Positions in micrometres (FlyWire voxels are 4 x 4 x 40 nm); neurons without one are hidden.
    pos = ann[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * np.array([4, 4, 40]) / 1000.0
    anchors = neuropil_anchors(ann, pos)
    has_pos = ~np.isnan(pos).any(1)
    pos = np.where(has_pos[:, None], pos, -1)
    pos_q = np.round(pos * 10).astype(np.int16)  # 0.1 um resolution

    _, _, x_test, y_test = load_mnist()
    app = FastAPI(title="Fly brain")
    app.add_middleware(SessionCookie)

    order = ["whole brain", "descending (brain -> body)", "central brain",
             "visual projection (optic lobe -> central brain)", "optic lobe",
             "eye input (stimulated neurons, noisy pixels)", "pixels"]
    readout_keys = [k for k in order if k in readouts]
    readouts_gz = _gz_json({k: {"name": readouts[k]["name"], **export_decoder(readouts[k])} for k in readout_keys})

    # Who is who: the teacher is whoever uses the app on this computer, unless they opened it
    # with ?as=student (a cookie) to see and test what students see.
    own_addresses = {"127.0.0.1", "::1", lan_ip()}
    teacher = lambda req: (req.client is not None and req.client.host in own_addresses
                           and req.cookies.get("flyrole") != "student")
    sid = lambda req: req.state.sid
    store = SessionStore(n)
    bodies = BodyPool()
    progresses = {}
    progress_of = lambda key: progresses.setdefault(key, Progress())
    session_key = lambda req: "class" if teacher(req) else sid(req)
    trainers = threading.Semaphore(3)   # decoder trainings at the same time (the rest wait their turn)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/slides")
    def slides():
        """The lecture: an interactive slide deck with live demos on the same brain."""
        return FileResponse(STATIC / "slides.html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/progress")
    def get_progress(request: Request):
        return progress_of(session_key(request)).get()

    @app.get("/api/network")
    def network():
        """Address other devices on the local network can use, with a QR code for it."""
        ip = lan_ip() if host in ("0.0.0.0", "::", "") else (None if host.startswith("127.") or host == "localhost" else host)
        if not ip:
            return {"shared": False, "reason": "The server only listens on this computer. Restart with --host 0.0.0.0."}
        http_url = f"http://{ip}:{port}"
        https_url = f"https://{ip}:{https_port}" if https_port else None
        url = https_url or http_url  # https so phones can use their camera
        svg = segno.make(url, error="m").svg_inline(scale=1, border=2, dark="#000", light="#fff", omitsize=True)
        return {"shared": True, "url": url, "http_url": http_url, "https_url": https_url, "ip": ip, "qr_svg": svg}

    @app.get("/api/brain")
    def brain_info(request: Request):
        return {
            "n": n,
            "pos": _b64(pos_q),
            "region": _b64(region),
            "anchors": anchors,
            "regions": [{"name": nm, "color": c, "neurons": int((region == i).sum())} for i, (nm, c) in enumerate(REGIONS)],
            "config": {"layer": layer, "rate_hz": rate_hz, "duration_ms": duration_ms, "shuffled": shuffle,
                       "n_inputs": int(len(input_idx)), "n_train": int(saved["n_train"]), "synapses": int(len(pre)),
                       "camera_ms": CAM_MS},
            "readouts": [{"key": k, "name": readouts[k]["name"], "accuracy": float(readouts[k]["accuracy"])} for k in readout_keys],
            "model": {"url": f"/api/brain/model.bin?v={model_version}", "bytes": model_path.stat().st_size,
                      "raw_bytes": model_raw_bytes, "version": model_version},
            "me": {"teacher": teacher(request)},
        }

    @app.get("/api/brain/model.bin")
    def brain_model():
        """The connectome and everything else a browser needs to simulate the brain (~40 MB gzip file;
        sent as-is and unpacked by the browser, so it can keep the small packed copy)."""
        return FileResponse(model_path, media_type="application/gzip",
                            headers={"Cache-Control": "public, max-age=31536000, immutable"})

    @app.get("/api/readouts")
    def get_readouts():
        """The trained MNIST readouts as plain weights, applied in the browser to its own spikes."""
        return _gz_response(readouts_gz, "public, max-age=3600")

    @app.get("/api/sample")
    def sample(i: int | None = None):
        i = int(np.random.randint(len(y_test))) if i is None else i % len(y_test)
        return {"index": i, "label": int(y_test[i]), "pixels": np.round(x_test[i].ravel(), 3).tolist()}

    def decode_image(data_url):
        try:
            return base64.b64decode(data_url.split(",", 1)[-1])
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"bad image: {e}")

    @app.post("/api/digit")
    def digit(req: ImageRequest):
        """A drawing -> the 28x28 MNIST-style image the eye sees."""
        try:
            x = preprocess(decode_image(req.image))
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"bad image: {e}")
        if x.max() == 0:
            raise HTTPException(400, "empty drawing")
        return {"image": np.round(x.ravel(), 3).tolist(), "image_shape": [28, 28]}

    test_cache = {}

    @app.get("/api/mnist/test")
    def mnist_test(n: int = 200):
        """The first n MNIST test digits, plus each readout's accuracy on them with the intact brain."""
        k = int(np.clip(n, 20, 1000))
        if k not in test_cache:
            _, _, Xte, yte, _ = load_features(fpath)
            acc = {}
            for key in readout_keys:
                r = readouts[key]
                X = x_test[:k].reshape(k, -1) if r["cols"] is None else np.log1p(Xte[:k][:, r["cols"]].toarray().astype(np.float32))
                acc[key] = float((r["model"].predict(X) == yte[:k]).mean())
            test_cache[k] = _gz_json({"n": k, "images": _b64(np.round(x_test[:k] * 255).astype(np.uint8)),
                                      "labels": yte[:k].astype(int).tolist(), "intact": acc})
        return _gz_response(test_cache[k], "public, max-age=3600")

    # ---------------- knockouts (rules are resolved here, applied in each browser) ----------------
    lesions = Lesions(ann, region, pre, post)
    cross = cross_hemisphere_mask(ann, pre, post)

    class LesionRequest(BaseModel):
        rules: list[dict] = []
        cut_hemispheres: bool = False

    @app.get("/api/lesion/options")
    def lesion_options():
        return lesions.options()

    @app.post("/api/lesion/resolve")
    def lesion_resolve(req: LesionRequest):
        try:
            silenced, _, per_rule = lesions.build(req.rules, False)
        except ValueError as e:
            raise HTTPException(400, str(e))
        removed = silenced[pre] | silenced[post]
        if req.cut_hemispheres:
            removed |= cross
        return {"rules": req.rules, "cut_hemispheres": req.cut_hemispheres, "per_rule": list(per_rule),
                "silenced": int(silenced.sum()), "connections_removed": int(removed.sum()), "connections_total": int(len(pre)),
                "silenced_idx": _b64(np.flatnonzero(silenced).astype(np.int32)),
                "by_region": [{"name": nm, "silenced": int((silenced & (region == i)).sum()), "neurons": int((region == i).sum())}
                              for i, (nm, _) in enumerate(REGIONS)]}

    # ---------------- camera: recordings, decoders, the class model ----------------
    not_input = region != 0
    masks = {
        "descending": (ann.super_class == "descending").to_numpy() & not_input,
        "central brain": (ann.super_class == "central").to_numpy() & not_input,
        "whole brain": not_input,
    }

    def class_counts_job():
        """Older class recordings have no spike counts yet: simulate them once on this computer."""
        cls = store.cls
        miss = cls.missing_counts
        if miss <= 0:
            return
        from .brain import FlyBrain
        prog = progress_of("class")
        prog.start("Preparing the class model", "Simulating your earlier recordings once", miss)
        try:
            brain = FlyBrain(pre, post, weight, n, input_idx)
            start = cls.counts.shape[0]
            rows = []
            for i in range(start, len(cls.images), 32):
                imgs = np.stack(cls.images[i:i + 32])
                c = brain.run(images_to_rates(imgs, uv, rate_hz), CAM_MS, on_progress=prog.advance).numpy()
                rows.append(sp.csr_matrix(np.minimum(c, 255).astype(np.uint8)))
            with cls.lock:
                cls.counts = sp.vstack([cls.counts, *rows]).tocsr()
                cls.save()
            print(f"class model: spike counts ready for {miss} earlier recordings", flush=True)
        finally:
            prog.finish()

    threading.Thread(target=class_counts_job, daemon=True, name="class-counts").start()
    threading.Thread(target=bodies.start, daemon=True, name="bodies-build").start()

    exports = {}

    def decoders_in_use(req):
        """(source, session) whose decoders this visitor uses: their own once trained, else the class model's."""
        s = store.get(sid(req), teacher(req))
        if teacher(req) or s.models:
            return ("class" if teacher(req) else "own"), s
        return "class", store.cls

    def cam_status(req):
        s = store.get(sid(req), teacher(req))
        cls = store.cls
        source, used = decoders_in_use(req)
        return {"classes": CAM_CLASSES, "duration_ms": CAM_MS, "image_shape": list(IMG_SHAPE),
                "teacher": teacher(req),
                "counts": s.label_counts(len(CAM_CLASSES)), "class_counts": cls.label_counts(len(CAM_CLASSES)),
                "trained": bool(used.models), "using": source, "metrics": used.metrics,
                "decoders_version": f"{source}:{used.version}",
                "include_class": s.include_class,
                "class_updated": bool(not teacher(req) and s.models and s.trained_on is not None and s.trained_on != cls.version),
                "class_preparing": cls.missing_counts > 0,
                "users": store.active_users(), "bodies": bodies.stats()}

    @app.get("/api/cam/status")
    def cam_status_ep(request: Request):
        return cam_status(request)

    @app.post("/api/cam/frame")
    def cam_frame(req: ImageRequest):
        """Webcam frame -> the local-contrast image the fly's eyes get."""
        img = preprocess_frame(decode_image(req.image))
        return {"image": np.round(img.ravel(), 3).tolist(), "image_shape": list(img.shape)}

    class Samples(BaseModel):
        label: int
        images: list[str]                 # base64 uint8, 36 x 48 each
        counts: list[dict]                # {"idx": base64 int32, "val": base64 uint8}: non-zero spike counts

    @app.post("/api/cam/samples")
    def cam_samples(req: Samples, request: Request):
        """Store recordings and the spike counts the visitor's own brain produced for them."""
        if not 0 <= req.label < len(CAM_CLASSES):
            raise HTTPException(400, f"label must be 0..{len(CAM_CLASSES) - 1}")
        if len(req.images) != len(req.counts) or not req.images:
            raise HTTPException(400, "need one spike-count entry per image")
        s = store.get(sid(request), teacher(request))
        if s is store.cls and s.missing_counts > 0:
            raise HTTPException(409, "The class model is still being prepared. Try again in a minute.")
        imgs, counts = parse_frames(req.images, req.counts)
        s.add(imgs, [req.label] * len(imgs), counts)
        return cam_status(request)

    def parse_frames(images, counts_in):
        """base64 uint8 36x48 images + sparse spike counts -> (images in [0, 1], sparse counts)."""
        imgs, rows, cols, vals = [], [], [], []
        for k, (im, c) in enumerate(zip(images, counts_in)):
            a = np.frombuffer(base64.b64decode(im), np.uint8)
            if a.size != IMG_SHAPE[0] * IMG_SHAPE[1]:
                raise HTTPException(400, "bad image size")
            imgs.append((a.astype(np.float32) / 255).reshape(IMG_SHAPE))
            idx = np.frombuffer(base64.b64decode(c["idx"]), np.int32)
            val = np.frombuffer(base64.b64decode(c["val"]), np.uint8)
            if len(idx) != len(val) or (len(idx) and (idx.min() < 0 or idx.max() >= n)):
                raise HTTPException(400, "bad spike counts")
            rows.append(np.full(len(idx), k)); cols.append(idx); vals.append(val)
        counts = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(imgs), n), dtype=np.uint8)
        return imgs, counts

    class TrainRequest(BaseModel):
        include_class: bool = True

    @app.post("/api/cam/train")
    def cam_train(req: TrainRequest, request: Request):
        s = store.get(sid(request), teacher(request))
        cls = store.cls
        if cls.missing_counts > 0:
            raise HTTPException(409, "The class model is still being prepared. Try again in a minute.")
        use_class = s is cls or req.include_class
        parts = [cls] + ([s] if s is not cls else []) if use_class else [s]
        images = [im for p in parts for im in p.images]
        labels = [lb for p in parts for lb in p.labels]
        try:
            check_trainable(labels)
        except ValueError as e:
            raise HTTPException(400, str(e))
        counts = sp.vstack([p.counts for p in parts]).tocsr()
        prog = progress_of(session_key(request))
        prog.start("Training the motor decoder", "Waiting for a free slot (others are training)", 1)
        with trainers:
            try:
                t0 = time.time()
                models, metrics = train_decoders(images, labels, counts, masks, prog)
            except ValueError as e:
                raise HTTPException(400, str(e))
            finally:
                prog.finish()
        s.set_models(models, metrics, trained_on=cls.version if (use_class and s is not cls) else None,
                     include_class=req.include_class)
        return {**cam_status(request), "frames": len(labels), "train_seconds": round(time.time() - t0, 1)}

    @app.post("/api/cam/clear")
    def cam_clear(request: Request):
        """Delete this visitor's recordings and decoders (for the teacher: the class model)."""
        store.get(sid(request), teacher(request)).clear()
        bodies.set_levels(sid(request), [])
        return cam_status(request)

    @app.get("/api/cam/decoders")
    def cam_decoders(request: Request):
        """The decoders this visitor uses, as plain weights their browser applies to its own spikes."""
        source, s = decoders_in_use(request)
        version, data = exports.get(id(s), (None, None))   # keep only the latest export per session
        if version != s.version:
            data = _gz_json({"version": f"{source}:{s.version}", "source": source,
                             "decoders": {k: export_decoder(v) for k, v in s.models.items()}})
            exports[id(s)] = (s.version, data)
        return _gz_response(data)

    # ---------------- stick figures: the camera tab's other input (flymnist/stick.py) ----------------
    # Same flow as camera pictures, in separate sessions (data/skeleton_session.joblib for the class,
    # data/sessions/skeleton/ for students). Out of the box everyone uses the pre-trained decoders.
    stick_store = SessionStore(n, kind="skeleton", session_cls=StickSession, last_seen=store.last_seen)
    pretrained = Pretrained(masks)
    threading.Thread(target=pretrained.load, daemon=True, name="stick-pretrained").start()

    def stick_in_use(req):
        """(source, session or the pre-trained model) whose stick-figure decoders this visitor uses:
        their own once trained, else the class's (once the teacher trained), else the pre-trained ones.
        Decoders made for another stick-figure spec are skipped."""
        spec = spec_version()
        s = stick_store.get(sid(req), teacher(req))
        if s.models and s.model_spec == spec:
            return ("class" if teacher(req) else "own"), s
        cls = stick_store.cls
        if cls.models and cls.model_spec == spec:
            return "class", cls
        pretrained.ready.wait(60)
        return "pretrained", pretrained

    def stick_status(req):
        s = stick_store.get(sid(req), teacher(req))
        cls = stick_store.cls
        spec = spec_version()
        loading = not pretrained.ready.is_set()
        source, used = ("pretrained", pretrained) if loading else stick_in_use(req)
        own_old = bool(s.models) and s.model_spec != spec
        return {"input": "stick", "classes": CAM_CLASSES, "duration_ms": CAM_MS, "image_shape": list(IMG_SHAPE),
                "rate_hz": pretrained.rate_hz, "spec_version": spec, "teacher": teacher(req),
                "counts": s.label_counts(len(CAM_CLASSES)), "class_counts": cls.label_counts(len(CAM_CLASSES)),
                "stale": len(s.stale(spec)),
                "trained": bool(used.models), "using": source, "metrics": used.metrics,
                "decoders_version": f"{source}:{used.version}", "include_class": s.include_class,
                "own_outdated": own_old,
                "class_updated": bool(not teacher(req) and source == "own" and s.trained_on is not None
                                      and s.trained_on != cls.version),
                "class_preparing": False, "preparing": loading, "pretrained": pretrained.info(),
                "users": store.active_users(), "bodies": bodies.stats()}

    @app.get("/api/stick/status")
    def stick_status_ep(request: Request):
        return stick_status(request)

    class StickSamples(BaseModel):
        label: int
        spec: int                         # stick-figure spec version the browser drew them with
        images: list[str]                 # base64 uint8, 36 x 48 each
        counts: list[dict]                # {"idx": base64 int32, "val": base64 uint8}: non-zero spike counts
        landmarks: list[dict | None] = [] # {pose, hands: {left, right}, size} per frame (tracker.js)

    def check_spec(spec):
        if spec != spec_version():
            raise HTTPException(409, f"This page draws stick figures version {spec}, the app is at version "
                                     f"{spec_version()}: reload the page.")

    @app.post("/api/stick/samples")
    def stick_samples(req: StickSamples, request: Request):
        """Stick-figure recordings: the drawings, their spike counts and the landmarks they were drawn from."""
        if not 0 <= req.label < len(CAM_CLASSES):
            raise HTTPException(400, f"label must be 0..{len(CAM_CLASSES) - 1}")
        if len(req.images) != len(req.counts) or not req.images:
            raise HTTPException(400, "need one spike-count entry per image")
        if req.landmarks and len(req.landmarks) != len(req.images):
            raise HTTPException(400, "need one landmark entry per image")
        check_spec(req.spec)
        try:
            lms = [parse_landmarks(x) for x in req.landmarks] if req.landmarks else None
        except (ValueError, TypeError) as e:
            raise HTTPException(400, f"bad landmarks: {e}")
        imgs, counts = parse_frames(req.images, req.counts)
        stick_store.get(sid(request), teacher(request)).add(imgs, [req.label] * len(imgs), counts, lms, req.spec)
        return stick_status(request)

    @app.post("/api/stick/train")
    def stick_train_ep(req: TrainRequest, request: Request):
        """Pre-trained examples (a fixed subset) + the class's stick recordings + the visitor's own."""
        s = stick_store.get(sid(request), teacher(request))
        cls = stick_store.cls
        pretrained.ready.wait(120)
        use_class = s is not cls and req.include_class
        parts = [cls, s] if use_class else [s]
        prog = progress_of(session_key(request))
        prog.start("Training the motor decoder", "Waiting for a free slot (others are training)", 1)
        with trainers:
            try:
                t0 = time.time()
                pre = pretrained if pretrained.models and pretrained.spec == spec_version() else None   # same drawing only
                models, metrics, frames = stick_train(pre, parts, masks, prog)
            except ValueError as e:
                raise HTTPException(400, str(e))
            finally:
                prog.finish()
        s.set_models(models, metrics, trained_on=cls.version if use_class else None,
                     include_class=req.include_class, spec=spec_version())
        return {**stick_status(request), "frames": frames, "train_seconds": round(time.time() - t0, 1)}

    @app.post("/api/stick/clear")
    def stick_clear(request: Request):
        """Delete this visitor's stick-figure recordings and decoders (for the teacher: the class's).
        The pre-trained model is never touched."""
        stick_store.get(sid(request), teacher(request)).clear()
        bodies.set_levels(sid(request), [])
        return stick_status(request)

    @app.get("/api/stick/decoders")
    def stick_decoders(request: Request):
        source, s = stick_in_use(request)
        if s is pretrained:
            return _gz_response(pretrained.export())
        version, data = exports.get(id(s), (None, None))
        if version != s.version:
            data = _gz_json({"version": f"{source}:{s.version}", "source": source,
                             "decoders": {k: export_decoder(v) for k, v in s.models.items()}})
            exports[id(s)] = (s.version, data)
        return _gz_response(data)

    @app.get("/api/stick/stale")
    def stick_stale(request: Request):
        """Recordings drawn with an older stick-figure spec, with their landmarks, for the browser to redraw."""
        s = stick_store.get(sid(request), teacher(request))
        spec = spec_version()
        with s.lock:
            frames = [{"i": i, "label": int(s.labels[i]), "landmarks": landmarks_json(s.landmarks[i])}
                      for i in s.stale(spec)]
        return {"spec": spec, "frames": frames}

    class Redrawn(BaseModel):
        spec: int
        rows: list[int]
        images: list[str]
        counts: list[dict]

    @app.post("/api/stick/redraw")
    def stick_redraw(req: Redrawn, request: Request):
        """Replace old-spec recordings by the browser's redrawn stick figures and their new spike counts."""
        check_spec(req.spec)
        if not (len(req.rows) == len(req.images) == len(req.counts)) or not req.rows:
            raise HTTPException(400, "need one image and spike-count entry per row")
        imgs, counts = parse_frames(req.images, req.counts)
        try:
            stick_store.get(sid(request), teacher(request)).replace(req.rows, imgs, counts, req.spec)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return stick_status(request)

    # ---------------- one fly body per visitor ----------------
    @app.get("/api/fly/model")
    def fly_model():
        """The fly's 3D meshes, once per page load (~1 MB gzipped)."""
        return _gz_response(bodies.model_payload(), "max-age=3600")

    @app.get("/api/fly/stream")
    async def fly_stream(request: Request):
        """This visitor's fly: body pose ~25 times a second (server-sent events) while their page shows it."""
        await run_in_threadpool(bodies.body, sid(request))   # first call builds the model; don't block the server
        return StreamingResponse(bodies.pose_stream(sid(request)), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    class FlyCommand(BaseModel):
        levels: list[float] | None = None   # one level (0..1) per action
        action: int | None = None           # or: do just this action (0 = rest)

    @app.post("/api/fly/command")
    def fly_command(req: FlyCommand, request: Request):
        """Move this visitor's fly: action levels decoded from their brain, or one action by hand."""
        if req.levels is not None:
            levels = np.asarray(req.levels[:len(CAM_CLASSES)], float)
        else:
            levels = np.zeros(len(CAM_CLASSES))
            if req.action and 0 < req.action < len(CAM_CLASSES):
                levels[req.action] = 1
        levels[0] = 0  # "rest" is the absence of every other action
        bodies.set_levels(sid(request), levels)
        return {"ok": True}

    print("ready", flush=True)
    return app
