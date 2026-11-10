"""Build the static "student" version of the app (e.g. for GitHub Pages):

    uv run main.py site [--out site] [--url https://<user>.github.io/<repo>/]

No Python server is needed to use it: everything the server would answer is exported here as
files (brain, readouts, test digits, knockout groups, the class model's decoders as a frozen
snapshot, the stick-figure decoders plus a compact subset of the pre-trained stick-figure
examples, the fly's meshes), and the rest runs in the browser (static/api.js). All links are
relative, so the folder works under any sub-path. No teacher mode, no join QR from the LAN,
and no fly physics (the body is animated by pose blending, static/flypose.js).

The payloads come from the server's own endpoints, called in-process, so they are exactly what
`serve` sends. Nothing in data/ is changed (session files are opened read-only).
"""

import base64
import gzip
import inspect
import json
import shutil
import time
from pathlib import Path

import numpy as np

from . import sessions, stick

STATIC = Path(__file__).resolve().parent / "static"
SKIP = {"index.html", "pose-test.html", "skeleton-test.html", ".DS_Store"}   # the app page goes to the site's root
FLAG = '<meta name="flybrain-static" content="1">'
TEST_DIGITS = 1000      # MNIST test digits shipped for "MNIST sample" (the first 200 also for "Test on 200 digits")


def _no_save(self):
    raise RuntimeError("the site build never writes sessions")


def _read_brain_file(path):
    """The arrays of the packed brain (flymnist/webmodel.py)."""
    buf = gzip.decompress(Path(path).read_bytes())
    hl = int.from_bytes(buf[:4], "little")
    meta = json.loads(buf[4:4 + hl])
    base = 4 + hl
    arrays = {s["name"]: np.frombuffer(buf, np.dtype(s["dtype"]), s["length"], base + s["offset"]) for s in meta["sections"]}
    return meta, arrays


def _with_flag(html):
    if FLAG in html:
        return html
    return html.replace('<meta charset="utf-8">', '<meta charset="utf-8">\n' + FLAG, 1)


def build(out="site", layer="lamina", rate_hz=500.0, duration_ms=100.0, shuffle=False, url=None):
    from starlette.requests import Request

    from .server import create_app

    t0 = time.time()
    sessions.Session.save = _no_save        # read-only: the class model and student sessions stay untouched
    app = create_app(layer, rate_hz, duration_ms, shuffle, host="127.0.0.1", port=0, https_port=0)
    ep = {r.path: r.endpoint for r in app.routes if hasattr(r, "endpoint")}
    # a visitor from another computer: a student with an empty session -> the class model's decoders
    req = Request({"type": "http", "method": "GET", "path": "/", "headers": [], "query_string": b"",
                   "client": ("192.0.2.1", 1), "state": {"sid": "static-site-build"}})
    unzip = lambda resp: json.loads(gzip.decompress(resp.body))

    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    data = out / "data"
    data.mkdir(parents=True)
    write = lambda name, obj: (data / name).write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")

    # brain info + the packed brain (the version in the name, so browsers fetch a new one when it changes)
    info = ep["/api/brain"](req)
    model_path = Path(ep["/api/brain/model.bin"]().path)
    ver = info["model"]["version"]
    shutil.copyfile(model_path, data / f"brain-{ver}.bin")   # gzip inside; the brain worker unzips it
    info["model"] = {"file": f"brain-{ver}.bin", "bytes": model_path.stat().st_size,
                     "raw_bytes": info["model"]["raw_bytes"], "version": ver}
    info["me"] = {"teacher": False}
    write("brain.json", info)
    write("readouts.json", unzip(ep["/api/readouts"]()))

    # test digits: images of the first TEST_DIGITS, intact accuracies for 200 and all of them
    tests = {n: unzip(ep["/api/mnist/test"](n)) for n in (200, TEST_DIGITS)}
    big = tests[TEST_DIGITS]
    write("mnist-test.json", {"n": big["n"], "images": big["images"], "labels": big["labels"], "default_n": 200,
                              "intact": {str(n): t["intact"] for n, t in tests.items()}})

    # knockouts: each group as a bit set, plus side and cell type per neuron for the other rules
    options = ep["/api/lesion/options"]()
    resolve = ep["/api/lesion/resolve"]
    LesionRequest = inspect.signature(resolve).parameters["req"].annotation
    bmeta, arrays = _read_brain_file(model_path)
    groups = {}
    for g in options["groups"]:
        r = resolve(LesionRequest(rules=[{"group": g["key"], "side": "both", "fraction": 1.0}]))
        idx = np.frombuffer(base64.b64decode(r["silenced_idx"]), np.int32)
        mask = np.zeros(info["n"], bool)
        mask[idx] = True
        groups[g["key"]] = base64.b64encode(np.packbits(mask, bitorder="little").tobytes()).decode()
    names = bmeta["cell_type_names"]
    # the brain file names unannotated neurons "untyped <class>"; the knockout lookup treats them as untyped ("")
    type_names = ["" if t.startswith("untyped ") else t.lower() for t in names]
    write("lesion.json", {"n": info["n"], "synapses": info["config"]["synapses"], "options": options, "groups": groups,
                          "side": base64.b64encode(arrays["side"].tobytes()).decode(),
                          "types": base64.b64encode(arrays["cell_type"].astype(np.uint16).tobytes()).decode(),
                          "type_names": type_names})

    # the class model, frozen: its decoders and metrics (its recordings stay on the teacher's computer)
    st = ep["/api/cam/status"](req)
    dec = unzip(ep["/api/cam/decoders"](req))
    write("cam-class.json", {"classes": st["classes"], "duration_ms": st["duration_ms"], "image_shape": st["image_shape"],
                             "class_counts": st["class_counts"], "trained": st["trained"], "metrics": st["metrics"],
                             "version": dec["version"].split(":", 1)[-1], "decoders": dec["decoders"]})

    # stick figures: the decoders everyone starts with (the class's stick-figure model if the teacher
    # trained one, else the pre-trained one), frozen; and a compact subset of the pre-trained examples
    # (stick figures + spike counts derived from HaGRIDv2, CC BY-SA 4.0; no photos) for Train in the browser
    sdec = unzip(ep["/api/stick/decoders"](req))     # first: waits for the pre-trained model to load
    sst = ep["/api/stick/status"](req)
    pre = sst["pretrained"]
    write("stick-class.json", {"classes": sst["classes"], "duration_ms": sst["duration_ms"], "image_shape": sst["image_shape"],
                               "rate_hz": sst["rate_hz"], "spec_version": sst["spec_version"], "class_counts": sst["class_counts"],
                               "trained": sst["trained"], "metrics": sst["metrics"], "source": sdec["source"],
                               "version": sdec["version"].split(":", 1)[-1], "decoders": sdec["decoders"],
                               "pretrained": {k: pre.get(k) for k in ("available", "spec_version", "classes", "frames_per_class")}
                                             | {"frames_in_train": None, "license": stick.LICENSE}})
    if stick.PRETRAINED.exists():
        sub = stick.static_export()
        write("stick-train.json", sub)
        pre_bytes = (data / "stick-train.json").stat().st_size
        print(f"  stick figures: {sub['n']} pre-trained examples, {len(base64.b64decode(sub['cols'])) // 4} neurons, "
              f"stick-train.json {pre_bytes / 1e6:.1f} MB ({len(gzip.compress((data / 'stick-train.json').read_bytes())) / 1e6:.1f} MB gzipped), "
              f"stick-class.json {(data / 'stick-class.json').stat().st_size / 1e6:.1f} MB", flush=True)

    # the fly's meshes (the poses come from static/fly-poses.json)
    try:
        write("fly-model.json", unzip(ep["/api/fly/model"]()))
    except Exception as e:  # noqa: BLE001
        print(f"  fly body skipped ({e}); the camera tab works without it", flush=True)

    if url:
        import segno
        svg = segno.make(url, error="m").svg_inline(scale=1, border=2, dark="#000", light="#fff", omitsize=True)
        write("network.json", {"shared": True, "url": url, "qr_svg": svg})

    # pages and scripts: the app at the root, everything else in static/ as on the server
    shutil.copytree(STATIC, out / "static", ignore=lambda d, fs: [f for f in fs if f in SKIP])
    (out / "index.html").write_text(_with_flag((STATIC / "index.html").read_text(encoding="utf-8")), encoding="utf-8")
    sl = out / "static" / "slides.html"
    sl.write_text(_with_flag(sl.read_text(encoding="utf-8")), encoding="utf-8")
    (out / "slides.html").write_text('<!doctype html><meta charset="utf-8"><title>Slides</title>'
                                     '<meta http-equiv="refresh" content="0; url=static/slides.html">'
                                     '<a href="static/slides.html">Slides</a>\n', encoding="utf-8")
    (out / ".nojekyll").write_text("")

    files = [p for p in out.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    biggest = max(files, key=lambda p: p.stat().st_size)
    print(f"\n  static site in {out}/: {len(files)} files, {total / 1e6:.1f} MB "
          f"(largest: {biggest.relative_to(out)}, {biggest.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f} s", flush=True)
    print(f"  try it:  python -m http.server 8200 -d {out}   ->  http://127.0.0.1:8200/", flush=True)
    return out
