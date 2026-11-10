// Where the app gets its data: the Python server (`uv run main.py serve`), or, in the static
// student version (`uv run main.py site`, e.g. on GitHub Pages), files exported at build time
// plus code that runs in the browser. The build marks its pages with
//   <meta name="flybrain-static" content="1">
// Server mode makes exactly the requests the app always made.
//
// Static mode, in short: brain info, readouts, test digits, knockout groups, the class model's
// decoders and the fly's meshes are JSON files in ../data/; drawings and camera frames are
// preprocessed here (imageproc.js, same results as the server); the student's recordings and
// decoder stay in this browser (IndexedDB) and the decoder is trained here (logreg.js); the fly
// body is animated by pose blending (flypose.js) instead of the server's physics. The camera tab's
// stick-figure input works the same way, with its own data (stick-class.json, stick-train.json) and
// its own database ("flybrain-stick").

import { canvasGray, preprocessDigit, preprocessFrame, CAM_SHAPE } from "./imageproc.js";

export const STATIC = !!document.querySelector('meta[name="flybrain-static"]');
const HERE = import.meta.url;
/** URL of a file next to this one (the app's static folder). */
export const staticURL = (path) => new URL(path, HERE).href;
/** URL of the app's main page. */
export const appURL = new URL("../", HERE).href;
const dataURL = (f) => new URL(`../data/${f}`, HERE).href;

const once = (fn) => { let p = null; return () => (p ||= fn().catch((e) => { p = null; throw e; })); };
const round3 = (v) => Math.round(v * 1000) / 1000;

export function decode(b64, T) {
  const bin = atob(b64), u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return new T(u.buffer);
}
/** A base64 string (from the server or a JSON file) or an array that is already decoded. */
export const arr = (v, T) => (typeof v === "string" ? decode(v, T) : v);
export function b64(typed) {
  const u = new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength);
  let s = "";
  for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000));
  return btoa(s);
}

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.detail || `${url}: ${res.status} ${res.statusText}`); }
  return res.json();
}
async function post(path, body) {
  const res = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
  if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.detail || res.statusText); }
  return res.json();
}
// Static files, each fetched once per page (callers get their own copy, as they may change it).
const files = {};
const dataText = (f) => (files[f] ||= fetch(dataURL(f)).then((r) => {
  if (!r.ok) throw new Error(`${f}: ${r.status} ${r.statusText}`);
  return r.text();
}).catch((e) => { delete files[f]; throw e; }));
const data = async (f) => JSON.parse(await dataText(f));

// ---------------- brain, MNIST readouts and digits ----------------
/** /api/brain: brain info; model.url is where the packed brain file is. */
export async function brain() {
  if (!STATIC) return getJSON("/api/brain");
  const b = await data("brain.json");
  b.model.url = dataURL(b.model.file);   // absolute: the brain worker fetches it from its own folder
  return b;
}
export async function readouts() { return STATIC ? data("readouts.json") : getJSON("/api/readouts"); }

const testDigits = once(async () => {
  const t = await data("mnist-test.json");
  return { ...t, px: decode(t.images, Uint8Array) };
});
/** A test digit: {index, label, pixels} (random without i). */
export async function sample(i = null) {
  if (!STATIC) return getJSON("/api/sample" + (i != null ? `?i=${i}` : ""));
  const t = await testDigits(), n = t.labels.length;
  const k = i == null ? Math.floor(Math.random() * n) : ((i % n) + n) % n;
  return { index: k, label: t.labels[k], pixels: Array.from(t.px.subarray(k * 784, (k + 1) * 784), (v) => round3(v / 255)) };
}
/** The first n test digits and each readout's accuracy on them with the intact brain. */
export async function mnistTest(n = 200) {
  if (!STATIC) return getJSON(`/api/mnist/test?n=${n}`);
  const t = await testDigits(), k = Math.min(n, t.labels.length);
  return { n: k, images: t.px.subarray(0, k * 784), labels: t.labels.slice(0, k), intact: t.intact[k] || t.intact[String(t.default_n)] };
}
/** A drawing (white on black canvas) -> {image (28x28 in [0,1]), image_shape}. */
export async function digit(canvas) {
  if (!STATIC) return post("/api/digit", { image: canvas.toDataURL("image/png") });
  const { gray, w, h } = canvasGray(canvas), x = preprocessDigit(gray, w, h);
  if (!x) throw new Error("empty drawing");
  return { image: Array.from(x, round3), image_shape: [28, 28] };
}

// ---------------- knockouts ----------------
const lesionData = once(async () => {
  const L = await data("lesion.json");
  const groups = Object.fromEntries(Object.entries(L.groups).map(([k, v]) => [k, decode(v, Uint8Array)]));
  return { ...L, groups, side: decode(L.side, Uint8Array), types: decode(L.types, Uint16Array) };
});
export async function lesionOptions() { return STATIC ? (await lesionData()).options : getJSON("/api/lesion/options"); }

// fnmatch.fnmatchcase as a regular expression
function globRegex(p) {
  let s = "";
  for (let i = 0; i < p.length; i++) {
    const c = p[i];
    if (c === "*") s += ".*";
    else if (c === "?") s += ".";
    else if (c === "[") {
      const j = p.indexOf("]", i + 2);
      if (j < 0) s += "\\[";
      else { let body = p.slice(i + 1, j).replace(/\\/g, "\\\\"); if (body[0] === "!") body = "^" + body.slice(1); s += `[${body}]`; i = j; }
    } else s += c.replace(/[.*+?^${}()|[\]\\/-]/g, "\\$&");
  }
  return new RegExp(`^${s}$`, "s");
}
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const roundHalfEven = (x) => { const r = Math.round(x); return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r; };

// Same rules as flymnist/lesion.py; the random subset for "amount" < 100% is seeded the same way
// every time, but is not the same subset numpy picks on the server.
function resolveRule(L, rule) {
  const n = L.n, m = new Uint8Array(n);
  if (rule.group) {
    const bits = L.groups[rule.group];
    if (!bits) throw new Error(`unknown group ${rule.group}`);
    for (let i = 0; i < n; i++) m[i] = (bits[i >> 3] >> (i & 7)) & 1;
  } else {
    const pats = String(rule.cell_types || "").split(",").map((p) => p.trim().toLowerCase()).filter(Boolean);
    if (!pats.length) throw new Error("give a group or cell type names (e.g. Tm1, LPLC2, DN*)");
    const res = pats.map(globRegex), hit = L.type_names.map((t) => !!t && res.some((r) => r.test(t)));
    for (let i = 0; i < n; i++) m[i] = hit[L.types[i]] ? 1 : 0;
  }
  const side = { left: 1, right: 2 }[rule.side];
  if (side) for (let i = 0; i < n; i++) if (L.side[i] !== side) m[i] = 0;
  const frac = rule.fraction == null ? 1 : +rule.fraction;
  if (frac < 1) {
    const idx = []; for (let i = 0; i < n; i++) if (m[i]) idx.push(i);
    const rnd = mulberry32(+rule.seed || 0);
    for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    m.fill(0);
    for (const i of idx.slice(0, roundHalfEven(frac * idx.length))) m[i] = 1;
  }
  return m;
}
/** Knockout rules -> silenced neurons. In static mode connections_removed is null: the brain worker counts them. */
export async function lesionResolve(body) {
  if (!STATIC) return post("/api/lesion/resolve", body);
  const L = await lesionData(), silenced = new Uint8Array(L.n), perRule = [];
  for (const r of body.rules || []) {
    const m = resolveRule(L, r);
    let c = 0; for (let i = 0; i < L.n; i++) if (m[i]) { c++; silenced[i] = 1; }
    perRule.push(c);
  }
  const idx = []; for (let i = 0; i < L.n; i++) if (silenced[i]) idx.push(i);
  return { rules: body.rules || [], cut_hemispheres: !!body.cut_hemispheres, per_rule: perRule, silenced: idx.length,
           connections_removed: null, connections_total: L.synapses, silenced_idx: Int32Array.from(idx) };
}

// ---------------- camera: frames, recordings, decoders ----------------
/** A webcam frame drawn on a canvas -> {image (36x48 local contrast in [0,1]), image_shape}. */
export async function camFrame(canvas) {
  if (!STATIC) return post("/api/cam/frame", { image: canvas.toDataURL("image/jpeg", 0.85) });
  const { gray, w, h } = canvasGray(canvas);
  return { image: Array.from(preprocessFrame(gray, w, h), round3), image_shape: [...CAM_SHAPE] };
}

const classModel = once(() => data("cam-class.json"));

// The student's own recordings and decoders, kept in this browser (IndexedDB; memory only if that fails).
// One database per camera-tab input, so camera pictures and stick figures never mix.
function ownStore(dbName) {
  const o = { loaded: null, frames: [], keys: [], state: { models: null, metrics: {}, version: 0 } };
  const idb = once(() => new Promise((resolve, reject) => {
    const r = indexedDB.open(dbName, 1);
    r.onupgradeneeded = () => { r.result.createObjectStore("frames", { autoIncrement: true }); r.result.createObjectStore("state"); };
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  }));
  o.do = (store, fn) => idb().then((db) => new Promise((resolve, reject) => {
    const tx = db.transaction(store, "readwrite"), req = fn(tx.objectStore(store));
    tx.oncomplete = () => resolve(req && req.result);
    tx.onerror = tx.onabort = () => reject(tx.error);
  })).catch((e) => console.warn("browser storage:", e));
  o.load = () => (o.loaded ||= (async () => {
    const frames = await o.do("frames", (s) => s.getAll());
    const keys = await o.do("frames", (s) => s.getAllKeys());
    const state = await o.do("state", (s) => s.get("model"));
    if (frames) { o.frames = frames; o.keys = keys || []; }
    if (state) o.state = state;
  })());
  o.saveState = () => o.do("state", (s) => s.put(o.state, "model"));
  return o;
}
const own = ownStore("flybrain-camera");
const ownLoad = () => own.load();
const idbDo = (store, fn) => own.do(store, fn);
const saveState = () => own.saveState();

async function staticCamStatus() {
  const [cls] = await Promise.all([classModel(), ownLoad()]);
  const counts = new Array(cls.classes.length).fill(0);
  for (const f of own.frames) counts[f.label]++;
  const mine = !!own.state.models;
  return { classes: cls.classes, duration_ms: cls.duration_ms, image_shape: cls.image_shape, teacher: false,
           counts, class_counts: cls.class_counts, trained: mine || cls.trained, using: mine ? "own" : "class",
           metrics: mine ? own.state.metrics : cls.metrics,
           decoders_version: mine ? `own:${own.state.version}` : `class:${cls.version}`,
           include_class: false, class_updated: false, class_preparing: false, users: 0, static: true };
}
export async function camStatus() { return STATIC ? staticCamStatus() : getJSON("/api/cam/status"); }

/** Store recordings: frames = [{image (values in [0,1]), sparse: {idx, val}}] (spike counts from this device's brain). */
export async function camSamples(label, frames) {
  const images = frames.map((f) => Uint8Array.from(f.image, (v) => Math.round(v * 255)));
  if (!STATIC) return post("/api/cam/samples", { label, counts: frames.map((f) => ({ idx: b64(f.sparse.idx), val: b64(f.sparse.val) })), images: images.map(b64) });
  await ownLoad();
  const recs = frames.map((f, k) => ({ label, image: images[k], idx: Int32Array.from(f.sparse.idx), val: Uint8Array.from(f.sparse.val) }));
  own.frames.push(...recs);
  await idbDo("frames", (s) => { for (const r of recs) s.add(r); });
  return staticCamStatus();
}

export async function camClear() {
  if (!STATIC) return post("/api/cam/clear");
  await ownLoad();
  own.frames = []; own.state = { models: null, metrics: {}, version: own.state.version + 1 };
  await idbDo("frames", (s) => s.clear());
  await saveState();
  return staticCamStatus();
}

/** The decoders this visitor uses: {version, source, decoders: {name: weights}}. */
export async function camDecoders() {
  if (!STATIC) return getJSON("/api/cam/decoders");
  const [cls] = await Promise.all([classModel(), ownLoad()]);
  return own.state.models ? { version: `own:${own.state.version}`, source: "own", decoders: own.state.models }
                          : { version: `class:${cls.version}`, source: "class", decoders: cls.decoders };
}

// Decoder training in the browser: every decoder's 6 fits (5 cross-validation folds + the final
// fit) are separate jobs, run by a pool of workers (one per spare CPU core).
function runJob(msg) {
  return new Promise((resolve, reject) => {
    const w = new Worker(staticURL("train-worker.js"), { type: "module" });
    w.onmessage = (e) => {
      const m = e.data;
      if (m.type === "progress") return;
      w.terminate(); if (m.type === "done") resolve(m); else reject(new Error(m.message));
    };
    w.onerror = (e) => { w.terminate(); reject(new Error(e.message || "training failed")); };
    w.postMessage(msg);
  });
}
async function pool(tasks, size) {
  const out = new Array(tasks.length);
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(size, tasks.length) }, async () => {
    while (next < tasks.length) { const k = next++; out[k] = await tasks[k](); }
  }));
  return out;
}
/** Fit every decoder (descending, central brain, whole brain, pixels) on recordings
 *  rows = [{label, image (uint8, 0..255), idx (neuron indices), val (spike counts)}].
 *  A neuron is read if it fires in at least max(1, minFrac * size) rows of one of `parts`
 *  ([[start, end], ...] row ranges; default: all rows) and, with `allowed`, is in that set. */
async function trainHere(rows, imageShape, onProgress, { weights = null, groups = null, allowed = null, minFrac = 0, parts = null } = {}) {
  const b = await data("brain.json");
  const labels = rows.map((f) => f.label);
  const per = new Map(); for (const l of labels) per.set(l, (per.get(l) || 0) + 1);
  if (per.size < 2 || Math.max(...per.values()) < 5) throw new Error("Record at least 5 frames of at least two actions first.");
  const region = decode(b.region, Uint8Array), n = rows.length;
  const masks = { descending: (r) => r === 4, "central brain": (r) => r === 3, "whole brain": (r) => r !== 0 };
  const jobs = {};
  for (const [name, inMask] of Object.entries(masks)) {          // log(1 + spike count) of the active neurons
    const active = new Set();
    for (const [p0, p1] of parts || [[0, n]]) {
      const need = Math.max(1, minFrac * (p1 - p0)), cnt = new Map();
      for (let i = p0; i < p1; i++) for (const j of rows[i].idx) if (inMask(region[j]) && (!allowed || allowed.has(j))) cnt.set(j, (cnt.get(j) || 0) + 1);
      for (const [j, c] of cnt) if (c >= need) active.add(j);
    }
    const cols = Int32Array.from([...active].sort((a, c) => a - c)), pos = new Map(Array.from(cols, (j, k) => [j, k]));
    const indptr = new Int32Array(n + 1), idx = [], val = [];
    rows.forEach((f, i) => {
      f.idx.forEach((j, k) => { const p = pos.get(j); if (p !== undefined) { idx.push(p); val.push(Math.log1p(f.val[k])); } });
      indptr[i + 1] = idx.length;
    });
    jobs[name] = cols.length ? { X: { n, F: cols.length, indptr, idx: Int32Array.from(idx), val: Float32Array.from(val) }, cols } : null;
  }
  {                                                                // pixels baseline
    const F = imageShape[0] * imageShape[1], indptr = new Int32Array(n + 1), idx = [], val = [];
    rows.forEach((f, i) => { f.image.forEach((v, k) => { if (v) { idx.push(k); val.push(Math.fround(v / 255)); } }); indptr[i + 1] = idx.length; });
    jobs.pixels = { X: { n, F, indptr, idx: Int32Array.from(idx), val: Float32Array.from(val) }, cols: null };
  }
  const names = Object.keys(jobs), fitsEach = 6, total = names.length * fitsEach, t0 = performance.now();
  let done = 0;
  const report = () => {
    const el = (performance.now() - t0) / 1000;
    onProgress({ active: true, task: "Training the motor decoder", stage: "Fitting decoders on this device (5-fold cross-validation)",
                 step: 1, steps: 1, done, total, eta: done > 0.05 * total ? el / done * (total - done) : null });
  };
  report();
  const tasks = [], where = [];
  for (const name of names) {
    const j = jobs[name];
    if (!j) { done += fitsEach; continue; }
    for (const fold of [-1, 0, 1, 2, 3, 4]) {
      where.push([name, fold]);
      tasks.push(() => runJob({ X: j.X, labels, cols: j.cols, weights, groups, fold }).then((r) => { done++; report(); return r; }));
    }
  }
  report();
  const cores = (typeof navigator !== "undefined" && navigator.hardwareConcurrency) || 4;
  const order = tasks.map((_, k) => k).sort((a, c) => (where[a][0] === "whole brain" ? 0 : 1) - (where[c][0] === "whole brain" ? 0 : 1));
  const res = await pool(order.map((k) => tasks[k]), Math.max(2, Math.min(cores - 1, 8)));
  const results = new Array(tasks.length); order.forEach((k, i) => (results[k] = res[i]));
  const models = {}, metrics = {}, accs = {};
  where.forEach(([name, fold], k) => {
    const r = results[k];
    if (fold < 0) { models[name] = r.decoder; metrics[name] = { cv_accuracy: null, neurons: r.neurons }; }
    else if (r.cv) (accs[name] ||= []).push(r.cv.acc);
  });
  for (const name of names) {
    if (!jobs[name]) { metrics[name] = { cv_accuracy: null, neurons: 0 }; continue; }
    const a = accs[name] || [];
    metrics[name].cv_accuracy = a.length ? a.reduce((x, y) => x + y, 0) / a.length : null;
  }
  return { models, metrics, seconds: Math.round((performance.now() - t0) / 100) / 10 };
}
async function staticTrain(onProgress) {
  const [cls] = await Promise.all([classModel(), ownLoad()]);
  const frames = own.frames;
  const { models, metrics, seconds } = await trainHere(frames, cls.image_shape, onProgress);
  own.state = { models, metrics, version: own.state.version + 1 };
  await saveState();
  return { ...(await staticCamStatus()), frames: frames.length, train_seconds: seconds };
}
/** Train the visitor's decoders; onProgress gets the same progress objects as /api/progress. */
export async function camTrain(body, onProgress = () => {}) {
  if (STATIC) return staticTrain(onProgress);
  const tick = async () => {
    try { const p = await getJSON("/api/progress"); if (p.active) onProgress(p); } catch { /* keep the last state */ }
  };
  const timer = setInterval(tick, 500); tick();
  try { return await post("/api/cam/train", body); } finally { clearInterval(timer); }
}

// ---------------- stick figures (the camera tab's other input) ----------------
// Same flow as camera frames, kept apart: server routes /api/stick/*, or in the static site the
// "flybrain-stick" database. Frames: [{image (values in [0,1]), sparse: {idx, val}, landmarks}].
// Out of the box everyone uses the pre-trained decoders (stick figures of HaGRIDv2 people); in the
// static site, Train mixes a compact subset of those examples (stick-train.json, fetched on the
// first Train) with the student's own recordings, own frames weighing more.
const stickClass = once(() => data("stick-class.json"));
const stickTrainData = once(async () => {
  const t = await data("stick-train.json");
  const cols = decode(t.cols, Int32Array), indptr = decode(t.indptr, Int32Array), idx = decode(t.idx, Uint16Array);
  const val = decode(t.val, Uint8Array), images = decode(t.images, Uint8Array), P = t.image_shape[0] * t.image_shape[1];
  const rows = t.labels.map((label, i) => ({
    label, image: images.subarray(i * P, (i + 1) * P),
    idx: Int32Array.from(idx.subarray(indptr[i], indptr[i + 1]), (k) => cols[k]), val: val.subarray(indptr[i], indptr[i + 1]),
  }));
  return { ...t, rows, allowed: new Set(cols) };
});
const ownStick = ownStore("flybrain-stick");

async function staticStickStatus(spec) {
  const [cls] = await Promise.all([stickClass(), ownStick.load()]);
  const counts = new Array(cls.classes.length).fill(0);
  let stale = 0;
  for (const f of ownStick.frames) { counts[f.label]++; if (spec != null && f.spec !== spec) stale++; }
  const st = ownStick.state, mine = !!st.models && (spec == null || st.spec === spec);
  return { input: "stick", classes: cls.classes, duration_ms: cls.duration_ms, image_shape: cls.image_shape, rate_hz: cls.rate_hz,
           spec_version: cls.spec_version, teacher: false, counts, class_counts: cls.class_counts, stale,
           trained: mine || cls.trained, using: mine ? "own" : cls.source, metrics: mine ? st.metrics : cls.metrics,
           decoders_version: mine ? `own:${st.version}` : `${cls.source}:${cls.version}`,
           own_outdated: !!st.models && !mine, include_class: false, class_updated: false, class_preparing: false,
           preparing: false, pretrained: cls.pretrained, users: 0, static: true };
}
/** Status of the stick-figure input. spec: the stick-figure version this page draws (static site:
 *  recordings and decoders of other versions count as outdated). */
export async function stickStatus(spec = null) { return STATIC ? staticStickStatus(spec) : getJSON("/api/stick/status"); }

const lmJSON = (lm) => (lm ? { pose: lm.pose, hands: lm.hands || { left: null, right: null }, ...(lm.size ? { size: lm.size } : {}) } : null);
/** Store stick-figure recordings (drawn with stick-figure version `spec`) for one action. */
export async function stickSamples(label, frames, spec) {
  const images = frames.map((f) => Uint8Array.from(f.image, (v) => Math.round(v * 255)));
  if (!STATIC) return post("/api/stick/samples", { label, spec, images: images.map(b64), landmarks: frames.map((f) => lmJSON(f.landmarks)),
                                                   counts: frames.map((f) => ({ idx: b64(f.sparse.idx), val: b64(f.sparse.val) })) });
  await ownStick.load();
  const recs = frames.map((f, k) => ({ label, spec, image: images[k], idx: Int32Array.from(f.sparse.idx), val: Uint8Array.from(f.sparse.val),
                                       landmarks: lmJSON(f.landmarks) }));
  const keys = await ownStick.do("frames", (s) => { const ks = []; for (const r of recs) s.add(r).onsuccess = (e) => ks.push(e.target.result); return { get result() { return ks; } }; });
  ownStick.frames.push(...recs); ownStick.keys.push(...(keys || recs.map(() => null)));
  return staticStickStatus(spec);
}

export async function stickClear(spec = null) {
  if (!STATIC) return post("/api/stick/clear");
  await ownStick.load();
  ownStick.frames = []; ownStick.keys = []; ownStick.state = { models: null, metrics: {}, version: ownStick.state.version + 1 };
  await ownStick.do("frames", (s) => s.clear());
  await ownStick.saveState();
  return staticStickStatus(spec);
}

/** The stick-figure decoders this visitor uses: {version, source, decoders}. */
export async function stickDecoders(spec = null) {
  if (!STATIC) return getJSON("/api/stick/decoders");
  const [cls] = await Promise.all([stickClass(), ownStick.load()]);
  const st = ownStick.state;
  return st.models && (spec == null || st.spec === spec) ? { version: `own:${st.version}`, source: "own", decoders: st.models }
                                                         : { version: `${cls.source}:${cls.version}`, source: cls.source, decoders: cls.decoders };
}

/** Recordings drawn with another stick-figure version: [{i, label, landmarks}] to redraw (stickRedraw). */
export async function stickStale(spec) {
  if (!STATIC) return getJSON("/api/stick/stale");
  await ownStick.load();
  return { spec, frames: ownStick.frames.map((f, i) => ({ i, label: f.label, landmarks: f.landmarks, spec: f.spec })).filter((f) => f.spec !== spec) };
}
/** Replace those recordings by redrawn ones: frames = [{i, image, sparse}]. */
export async function stickRedraw(spec, frames) {
  const images = frames.map((f) => Uint8Array.from(f.image, (v) => Math.round(v * 255)));
  if (!STATIC) return post("/api/stick/redraw", { spec, rows: frames.map((f) => f.i), images: images.map(b64),
                                                  counts: frames.map((f) => ({ idx: b64(f.sparse.idx), val: b64(f.sparse.val) })) });
  await ownStick.load();
  frames.forEach((f, k) => Object.assign(ownStick.frames[f.i], { spec, image: images[k], idx: Int32Array.from(f.sparse.idx), val: Uint8Array.from(f.sparse.val) }));
  await ownStick.do("frames", (s) => { for (const f of frames) if (ownStick.keys[f.i] != null) s.put(ownStick.frames[f.i], ownStick.keys[f.i]); });
  return staticStickStatus(spec);
}

// In the browser: the pre-trained subset + own recordings of this stick-figure version, own frames
// weighted like the server does (flymnist/stick.py: own_weights).
async function staticStickTrain(spec, onProgress) {
  const [cls, pre] = await Promise.all([stickClass(), stickTrainData(), ownStick.load()]);
  const mine = ownStick.frames.filter((f) => f.spec === spec);
  const usePre = pre.spec_version === spec;      // the shipped examples are only useful for the same drawing
  const base = usePre ? pre.rows : [], rows = [...base, ...mine];
  const nOwn = new Map(); for (const f of mine) nOwn.set(f.label, (nOwn.get(f.label) || 0) + 1);
  const weights = rows.map((f, i) => (i < base.length || !usePre ? 1 : Math.min(pre.max_weight, Math.max(1, pre.per_class / nOwn.get(f.label)))));
  const groups = rows.map((f, i) => (i < base.length ? f.label : 1000 + f.label));
  const parts = [[0, base.length], [base.length, rows.length]].filter(([a, b]) => b > a);
  const { models, metrics, seconds } = await trainHere(rows, cls.image_shape, onProgress,
    { weights, groups, parts, minFrac: pre.min_active, allowed: usePre ? pre.allowed : null });
  for (const m of Object.values(metrics)) m.frames = { pretrained: base.length, recorded: mine.length };
  ownStick.state = { models, metrics, version: ownStick.state.version + 1, spec };
  await ownStick.saveState();
  return { ...(await staticStickStatus(spec)), frames: rows.length, train_seconds: seconds };
}
export async function stickTrain(body, onProgress = () => {}, spec = null) {
  if (STATIC) return staticStickTrain(spec, onProgress);
  const tick = async () => {
    try { const p = await getJSON("/api/progress"); if (p.active) onProgress(p); } catch { /* keep the last state */ }
  };
  const timer = setInterval(tick, 500); tick();
  try { return await post("/api/stick/train", body); } finally { clearInterval(timer); }
}

// ---------------- the fly body ----------------
export async function flyModel() { return STATIC ? data("fly-model.json") : getJSON("/api/fly/model"); }
/** Static mode: the pose blender that moves the fly in this browser (flypose.js). */
export const poser = once(async () => {
  const { loadPoser } = await import("./flypose.js");
  return loadPoser(staticURL("fly-poses.json"));
});
/** Move the fly: {levels: one per action} or {action: i} (0 = rest). */
export async function flyCommand(body) {
  if (!STATIC) return post("/api/fly/command", body);
  const p = await poser(), n = 10;
  const levels = body.levels ? Array.from({ length: n }, (_, i) => Math.min(1, Math.max(0, +body.levels[i] || 0)))
                             : Array.from({ length: n }, (_, i) => (i === body.action && i > 0 ? 1 : 0));
  levels[0] = 0;   // "rest" is the absence of every other action
  p.setLevels(levels);
  return { ok: true };
}

// ---------------- network (join QR on the slides) ----------------
/** Server: this computer's LAN address + QR. Static: the published address + QR if the build was given --url. */
export async function network() {
  if (!STATIC) return getJSON("/api/network");
  try { return { ...(await data("network.json")), static: true }; }
  catch { return { shared: false, static: true }; }
}
