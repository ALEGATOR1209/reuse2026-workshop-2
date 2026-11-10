// Interactive widgets for the lecture slides, running on the real fruit fly brain in the browser.
//
//   import { brainExplorer, retinaDemo, mnistDemo, joinBrain } from "/static/slides/brain.js";
//   const w = brainExplorer(el);   // el: an empty, sized container
//   w.start(); ... w.stop();       // start = animate/render, stop = no requestAnimationFrame at all
//
// All widgets on a page share ONE /api/brain fetch, ONE /api/readouts fetch and ONE brain Web Worker
// (the 40 MB connectome is downloaded once and then kept in the browser's Cache Storage, shared with
// the main app). Everything is loaded lazily, the first time a widget needs it.
// Styles are injected once and scoped with the `bd-` prefix.

// ============================================================================================
// Shared state: brain info, readouts, the brain worker
// ============================================================================================

const THREE_URL = "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js";
const ORBIT_URL = "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/controls/OrbitControls.js";
// OrbitControls.js imports the bare specifier "three", which only resolves with an import map.
// Without one we use jsDelivr's bundled copies, which import each other by full URL.
const THREE_ESM = "https://cdn.jsdelivr.net/npm/three@0.170.0/+esm";
const ORBIT_ESM = "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/controls/OrbitControls.js/+esm";
const WORKER_URL = new URL("../brain-worker.js", import.meta.url);   // = /static/brain-worker.js
// Data comes from the server's API or, in the static student site, from exported files (see ../api.js).
import * as A from "../api.js";

const once = (fn) => { let p = null; return () => (p ||= fn()); };
const fmt = (v) => Math.round(v).toLocaleString("en-US");
const mb = (x) => (x / 1e6).toFixed(0);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function decode(b64, T) {
  const bin = atob(b64), u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return new T(u.buffer);
}
function hexToRgb(h) { const n = parseInt(h.slice(1), 16); return [n >> 16, (n >> 8) & 255, n & 255]; }

/** /api/brain, with pos (Int16Array, 0.1 µm, -10 = unknown) and region (Uint8Array) decoded. */
export const getBrainInfo = once(async () => {
  const b = await A.brain();
  b.pos = decode(b.pos, Int16Array);
  b.region = decode(b.region, Uint8Array);
  return b;
});

// A trained readout as plain weights: p = softmax(W · standardize(features) + b) (copied from index.html).
function prepDecoder(d) {
  return { cols: d.cols ? A.arr(d.cols, Int32Array) : null, F: d.n_features, mean: A.arr(d.mean, Float32Array),
           scale: A.arr(d.scale, Float32Array), coef: A.arr(d.coef, Float32Array), intercept: d.intercept, classes: d.classes };
}
function decoderProba(d, counts, image, nClasses = 10) {
  const F = d.F, z = new Float64Array(F);
  for (let f = 0; f < F; f++) z[f] = ((d.cols ? Math.log1p(counts[d.cols[f]]) : image[f]) - d.mean[f]) / d.scale[f];
  const K = d.coef.length / F, logit = new Float64Array(K);
  for (let k = 0; k < K; k++) {
    let s = d.intercept[k];
    for (let f = 0, o = k * F; f < F; f++) s += d.coef[o + f] * z[f];
    logit[k] = s;
  }
  const p = new Array(nClasses).fill(0);
  if (K === 1) { const p1 = 1 / (1 + Math.exp(-logit[0])); p[d.classes[0]] = 1 - p1; p[d.classes[1]] = p1; return p; }
  const mx = Math.max(...logit);
  let sum = 0;
  const ex = Array.from(logit, (v) => { const e = Math.exp(v - mx); sum += e; return e; });
  d.classes.forEach((c, k) => (p[c] = ex[k] / sum));
  return p;
}
/** The MNIST readouts, {key: decoder}. */
export const getReadouts = once(async () => {
  const r = await A.readouts();
  return Object.fromEntries(Object.entries(r).map(([k, v]) => [k, prepDecoder(v)]));
});
/** {key: probabilities over 0-9} for every readout. */
export async function predict(counts, image) {
  const R = await getReadouts();
  return Object.fromEntries(Object.entries(R).map(([k, d]) => [k, decoderProba(d, counts, image)]));
}

// ---- the brain worker (one per page) ----
// status: idle | loading | ready | error; loaded/total bytes while loading.
const BRAIN = { status: "idle", loaded: 0, total: 0, cached: false, info: null, error: null,
                worker: null, ready: null, pending: new Map(), next: 1, subs: new Set() };
function setBrain(patch) { Object.assign(BRAIN, patch); for (const f of BRAIN.subs) f(BRAIN); }
/** Subscribe to the brain's loading status; calls fn right away. Returns an unsubscribe function. */
export function onBrain(fn) { BRAIN.subs.add(fn); fn(BRAIN); return () => BRAIN.subs.delete(fn); }

/** Start loading this device's brain (once). Resolves with {n, inputs, synapses} when it can run. */
export function loadBrain() {
  if (BRAIN.ready) return BRAIN.ready;
  BRAIN.ready = (async () => {
    const b = await getBrainInfo();
    setBrain({ status: "loading", total: b.model.bytes, loaded: 0 });
    const w = (BRAIN.worker = new Worker(WORKER_URL, { type: "module" }));
    return new Promise((resolve, reject) => {
      w.onmessage = (e) => {
        const m = e.data;
        if (m.type === "progress") setBrain({ loaded: m.loaded, total: m.total, cached: m.cached });
        else if (m.type === "ready") { setBrain({ status: "ready", info: m, loaded: BRAIN.total }); resolve(m); }
        else {
          const p = BRAIN.pending.get(m.id);
          if (!p) { if (m.type === "error") reject(new Error(m.message)); return; }
          BRAIN.pending.delete(m.id);
          if (m.type === "error") p.reject(new Error(m.message)); else p.resolve(m);
        }
      };
      w.onerror = (e) => reject(new Error(e.message || "the brain could not start on this device"));
      w.postMessage({ type: "load", url: b.model.url, gzBytes: b.model.bytes, rawBytes: b.model.raw_bytes,
                      regionNames: b.regions.map((r) => r.name) });
    });
  })();
  BRAIN.ready.catch((e) => setBrain({ status: "error", error: e.message }));
  return BRAIN.ready;
}
/** Send a message to the brain worker and wait for its reply. */
export async function brainCall(msg, transfer = []) {
  await loadBrain();
  const id = BRAIN.next++;
  return new Promise((resolve, reject) => {
    BRAIN.pending.set(id, { resolve, reject });
    BRAIN.worker.postMessage({ ...msg, id }, transfer);
  });
}
/** Run one 28x28 image (values in [0,1]) through this device's brain. */
export async function runImage(image, { ms = null, describe = true, H = 28, W = 28 } = {}) {
  const b = await getBrainInfo();
  const img = Float32Array.from(image);
  return brainCall({ type: "run", image: img, H, W, rateHz: b.config.rate_hz, ms: ms || b.config.duration_ms, describe }, [img.buffer]);
}
/** The eye mapping: {uv, inputIdx, region, side, cellType, cellTypeNames} per input neuron. */
export const getRetina = once(() => brainCall({ type: "retina" }));

async function getSample(i) { return A.sample(i); }
async function preprocess(canvas) { return A.digit(canvas); }

// ============================================================================================
// Small UI helpers
// ============================================================================================

const CSS = `
/* Typography follows the page: the base size is 0.7em of whatever font size the widget's container has
   (the deck's --fs, ~30px on a 1080p projector -> 21px here), never below 14px. Everything else is in em. */
/* overflow: on narrow screens the widgets stack and scroll inside their box */
.bd-wrap { container-type: inline-size; position: relative; width: 100%; height: 100%; box-sizing: border-box; overflow-x: hidden; overflow-y: auto;
  --bd-bg: #0b0d11; --bd-panel: #13161c; --bd-panel-2: #1a1e26; --bd-line: #262b35; --bd-text: #e6e8ec;
  --bd-muted: #8b93a1; --bd-accent: #f5c542; --bd-red: #ff6b6b;
  color: var(--bd-text); font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; line-height: 1.3; text-align: left; }
.bd-wrap *, .bd-wrap *::before, .bd-wrap *::after { box-sizing: border-box; }
.bd-wrap > * { font-size: max(14px, 0.7em); }
.bd-wrap [hidden] { display: none !important; }
.bd-wrap button { background: var(--bd-panel-2); color: var(--bd-text); border: 1px solid var(--bd-line); border-radius: 0.45em;
  padding: 0.35em 0.7em; font: inherit; font-size: 1.02em; cursor: pointer; line-height: 1.2; }
.bd-wrap button:hover { border-color: #3a4150; }
.bd-wrap button:disabled { opacity: 0.5; cursor: default; }
.bd-wrap button.bd-primary { background: var(--bd-accent); color: #17130a; border-color: var(--bd-accent); font-weight: 650; }
.bd-panel { background: var(--bd-panel); border: 1px solid var(--bd-line); border-radius: 0.55em; padding: 0.6em 0.7em; min-width: 0; min-height: 0; }
.bd-h { font-size: 1.02em; font-weight: 700; color: var(--bd-text); margin: 0 0 0.4em; line-height: 1.2; }
.bd-h small { font-size: 0.86em; font-weight: 500; color: var(--bd-muted); }
.bd-row { display: flex; gap: 0.35em; align-items: center; flex-wrap: wrap; }
.bd-note { color: var(--bd-muted); font-size: 0.88em; }
.bd-note b { color: var(--bd-text); font-weight: 600; }
.bd-dot { display: inline-block; width: 0.6em; height: 0.6em; border-radius: 50%; flex: none; }
.bd-spin { display: inline-block; width: 0.8em; height: 0.8em; border: 2px solid var(--bd-muted); border-top-color: transparent;
  border-radius: 50%; animation: bd-spin 0.8s linear infinite; vertical-align: -0.1em; margin-right: 0.4em; }
@keyframes bd-spin { to { transform: rotate(360deg); } }
.bd-err { color: var(--bd-red); }

/* loading bar for the brain download */
.bd-load { font-size: 0.88em; color: var(--bd-muted); }
.bd-load .bd-lab { display: flex; justify-content: space-between; gap: 0.8em; margin-bottom: 0.3em; flex-wrap: wrap; }
.bd-load .bd-lab span:last-child { font-variant-numeric: tabular-nums; white-space: nowrap; color: var(--bd-text); }
.bd-bar { height: 0.35em; min-height: 5px; background: var(--bd-panel-2); border-radius: 3px; overflow: hidden; }
.bd-bar i { display: block; height: 100%; width: 0; background: var(--bd-accent); transition: width 0.3s; }

/* probability cells over 0-9 */
.bd-cells { display: grid; grid-template-columns: repeat(10, 1fr); gap: 2px; }
.bd-cells div { height: 1.45em; border-radius: 3px; font-size: 0.88em; display: grid; place-items: center;
  font-variant-numeric: tabular-nums; color: #cfd3da; background: #1c2029; transition: background 0.3s; }
.bd-cells div.bd-max { outline: 2px solid var(--bd-text); outline-offset: -2px; color: #fff; font-weight: 700; }

/* horizontal bars per region (retina + mnist) */
.bd-rbar { display: grid; grid-template-columns: 6.6em minmax(0, 1fr) auto; gap: 0.4em; align-items: center; font-size: 0.88em; }
.bd-rbar > span:first-child { display: flex; align-items: center; gap: 0.35em; white-space: nowrap; overflow: hidden; }
.bd-rbar .bd-track { position: relative; height: 1.15em; background: #1a1e26; border-radius: 3px; overflow: hidden; }
.bd-rbar .bd-track i { position: absolute; left: 0; top: 0; bottom: 0; width: 0; border-radius: 3px; opacity: 0.85; }
.bd-rbar .bd-track em { position: absolute; left: 0.4em; top: 0; bottom: 0; display: flex; align-items: center; font-style: normal;
  font-variant-numeric: tabular-nums; color: #fff; text-shadow: 0 0 3px #000, 0 0 3px #000; white-space: nowrap; }
.bd-rbar > span:last-child { text-align: right; color: var(--bd-muted); font-variant-numeric: tabular-nums; white-space: nowrap; min-width: 2.6em; }
.bd-rbar.bd-hot > span:first-child { color: var(--bd-accent); }
.bd-rbars { display: flex; flex-direction: column; gap: 0.22em; }

/* ---------- brainExplorer ---------- */
.bd-ex { display: grid; grid-template-rows: minmax(0, 1fr) auto; gap: 0.5em; height: 100%; padding: 0.4em; }
.bd-ex-map { position: relative; background: #05070a; border-radius: 0.55em; overflow: hidden; min-height: 0; }
.bd-ex-map canvas { position: absolute; inset: 0; width: 100%; height: 100%; display: block; cursor: grab; touch-action: none; }
.bd-ex-map canvas:active { cursor: grabbing; }
.bd-ex-labels { position: absolute; inset: 0; pointer-events: none; overflow: hidden; }
.bd-wrap button.bd-ex-lbl { position: absolute; left: 0; top: 0; white-space: nowrap; font-size: 0.9em; font-weight: 650; color: #f0f2f5; pointer-events: auto;
  text-shadow: 0 0 3px #000, 0 0 6px #000, 0 0 10px #000; display: flex; align-items: center; gap: 0.3em; cursor: pointer;
  background: none; border: 0; padding: 0.1em 0.2em; font-family: inherit; }
.bd-wrap .bd-ex-lbl::before { content: ""; width: 0.3em; height: 0.3em; border-radius: 50%; background: #fff; box-shadow: 0 0 4px #000; flex: none; }
.bd-wrap button.bd-ex-lbl:hover, .bd-wrap button.bd-ex-lbl.bd-on { color: var(--bd-accent); }
.bd-wrap .bd-ex-lbl.bd-on::before { background: var(--bd-accent); }
.bd-wrap .bd-ex-lbl.bd-dim { opacity: 0.45; }
.bd-ex-help { position: absolute; right: 0.6em; bottom: 0.4em; font-size: 0.86em; color: #6f7682; pointer-events: none; }
.bd-ex-msg { position: absolute; inset: 0; display: grid; place-items: center; color: var(--bd-muted); pointer-events: none; padding: 1em; text-align: center; }
.bd-ex-side { display: grid; grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr); gap: 0.4em 1em; padding: 0 0.2em; }
.bd-ex-side .bd-h { margin-bottom: 0.25em; }
.bd-ex-list { display: grid; grid-template-columns: 1fr 1fr; gap: 0 0.4em; }
.bd-wrap button.bd-ex-item { display: flex; align-items: center; gap: 0.45em; background: none; border: 1px solid transparent; text-align: left;
  padding: 0.18em 0.4em; border-radius: 0.4em; font-size: 0.95em; width: 100%; }
.bd-wrap button.bd-ex-item:hover { background: var(--bd-panel-2); }
.bd-wrap button.bd-ex-item.bd-on { background: var(--bd-panel-2); border-color: var(--bd-accent); }
.bd-ex-item span { flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.bd-ex-item b { font-weight: 500; color: var(--bd-muted); font-variant-numeric: tabular-nums; font-size: 0.92em; }
.bd-ex-nps { display: flex; flex-wrap: wrap; gap: 0.25em; }
.bd-wrap .bd-ex-nps button { font-size: 0.88em; padding: 0.2em 0.5em; border-radius: 999px; }
.bd-wrap .bd-ex-nps button.bd-on { border-color: var(--bd-accent); color: var(--bd-accent); }
.bd-ex-info { position: absolute; left: 0.5em; top: 0.5em; max-width: min(21em, 60%); background: rgba(19, 22, 28, 0.82); border: 1px solid var(--bd-line);
  border-radius: 0.5em; padding: 0.45em 0.6em; font-size: 0.95em; pointer-events: none; }
.bd-ex-info h4 { margin: 0 0 0.2em; font-size: 1.05em; display: flex; align-items: baseline; gap: 0.4em; flex-wrap: wrap; }
.bd-ex-info h4 small { font-size: 0.8em; font-weight: 500; color: var(--bd-muted); }
.bd-ex-info h4 .bd-dot { align-self: center; }
.bd-ex-info p { margin: 0; color: #cfd3da; }
.bd-ex-info small { display: block; color: var(--bd-muted); margin-top: 0.25em; font-size: 0.92em; }
@container (max-width: 620px) {
  .bd-ex { grid-template-rows: none; height: auto; }
  .bd-ex-map { height: auto; aspect-ratio: 1 / 0.85; }
  .bd-ex-side { grid-template-columns: minmax(0, 1fr); }
  .bd-ex-info { max-width: 80%; }
}

/* ---------- retinaDemo ---------- */
.bd-rt { display: grid; grid-template-columns: minmax(9em, 12.5em) minmax(0, 1fr); grid-template-rows: minmax(0, 1fr) auto;
  gap: 0.45em; height: 100%; padding: 0.4em; position: relative; }
.bd-rt > section { display: flex; flex-direction: column; min-height: 0; min-width: 0; }
.bd-rt-digit { grid-row: 1 / 3; }
.bd-rt-imgbox { position: relative; width: 100%; aspect-ratio: 1; background: #000; border-radius: 0.4em; border: 1px solid var(--bd-line); overflow: hidden; touch-action: none; flex: none; }
.bd-rt-imgbox canvas { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
.bd-rt-img { image-rendering: pixelated; cursor: crosshair; }
.bd-rt-pad { cursor: crosshair; }
.bd-rt-digit .bd-row { margin-top: 0.45em; }
.bd-rt-px { margin-top: 0.5em; }
.bd-rt-eyes { position: relative; }
.bd-rt-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 0.6em; flex: 1; min-height: 0; }
.bd-rt-pair figure { margin: 0; display: flex; flex-direction: column; min-width: 0; min-height: 0; }
.bd-rt-eyebox { position: relative; flex: 1; min-height: 0; }
.bd-rt-eyebox canvas { position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); display: block; cursor: crosshair; border-radius: 50% / 46%;
  background: radial-gradient(ellipse at 45% 40%, #151922 0%, #0d1015 70%); box-shadow: 0 0 0 1px var(--bd-line); }
.bd-rt-pair figcaption { margin-top: 0.25em; font-size: 0.88em; color: var(--bd-muted); text-align: center; white-space: nowrap; }
.bd-rt-pair figcaption b { color: var(--bd-text); font-weight: 600; }
.bd-rt-scale { margin-top: 0.35em; display: flex; align-items: center; gap: 0.45em; font-size: 0.88em; color: var(--bd-muted); justify-content: center; flex-wrap: wrap; }
.bd-rt-scale i { display: inline-block; width: 5em; height: 0.55em; border-radius: 3px; background: linear-gradient(90deg, #2a2f3a, #c8782a, #f5c542, #fff5d0); }
.bd-rt-loadhost { position: absolute; left: 8%; right: 8%; top: 42%; }
.bd-rt-brain { display: grid !important; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 0.4em 0.9em; }
.bd-rt-brain .bd-primary { width: 100%; }
.bd-rt-status { margin: 0.3em 0 0.35em; min-height: 1.2em; }
.bd-rt-out { display: flex; flex-direction: column; justify-content: center; gap: 0.35em; }
.bd-rt-arrow { font-size: 0.88em; color: var(--bd-muted); }
.bd-rt-arrow b { color: var(--bd-accent); font-weight: 650; }
.bd-rt-arrow small { font-size: 1em; }
.bd-rt-pred { display: flex; gap: 0.6em; align-items: center; background: var(--bd-panel-2); border-radius: 0.5em; padding: 0.45em 0.55em; }
.bd-rt-big { font-size: 3em; font-weight: 750; line-height: 1; width: 1.1em; text-align: center; font-variant-numeric: tabular-nums; color: var(--bd-accent); flex: none; }
.bd-rt-pred > div:last-child { flex: 1; min-width: 0; }
.bd-rt-pred .bd-note { margin-bottom: 0.25em; }
.bd-rt-overlay { position: absolute; left: 0; top: 0; pointer-events: none; z-index: 2; }
@container (max-width: 620px) {
  .bd-rt { grid-template-columns: minmax(0, 1fr); grid-template-rows: none; height: auto; }
  .bd-rt-digit { grid-row: auto; }
  .bd-rt-imgbox { max-width: 14em; margin: 0 auto; }
  .bd-rt-eyebox { flex: none; aspect-ratio: 1; }
  .bd-rt-brain { grid-template-columns: minmax(0, 1fr); }
}

/* ---------- mnistDemo ---------- */
.bd-mn { display: grid; grid-template-columns: minmax(9em, 12.5em) minmax(0, 1fr) minmax(0, 1fr); grid-template-rows: minmax(0, 1fr) auto;
  gap: 0.45em; height: 100%; padding: 0.4em; }
.bd-mn > section { display: flex; flex-direction: column; min-height: 0; min-width: 0; }
.bd-mn-in { grid-row: 1 / 3; }
.bd-mn-act { grid-column: 2 / 4; }
.bd-mn-in .bd-row button { padding-left: 0.5em; padding-right: 0.5em; }
.bd-mn-pad { width: 100%; aspect-ratio: 1; display: block; background: #000; border-radius: 0.4em; border: 1px solid var(--bd-line); touch-action: none; cursor: crosshair; flex: none; }
.bd-mn-in .bd-row { margin-top: 0.45em; }
.bd-mn-in .bd-primary { flex: 1; }
.bd-mn-seen canvas { display: block; width: 2.4em; height: 2.4em; image-rendering: pixelated; background: #000; border-radius: 0.25em; border: 1px solid var(--bd-line); flex: none; }
.bd-mn-head { display: flex; align-items: center; gap: 0.4em; margin-top: 0.5em; }
.bd-mn-headtxt { line-height: 1.2; }
.bd-mn-big { font-size: 3em; font-weight: 750; line-height: 1; width: 1.05em; text-align: center; font-variant-numeric: tabular-nums; color: var(--bd-accent); flex: none; }
.bd-mn-status { margin-top: 0.45em; }
.bd-mn-chartbox { position: relative; flex: 1; min-height: 6em; }
.bd-mn-chartbox canvas { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
.bd-mn-regions .bd-rbar { grid-template-columns: 9.2em minmax(0, 1fr) auto; }
.bd-mn-ro + .bd-mn-ro { margin-top: 0.3em; }
.bd-mn-ro.bd-base { border-top: 1px dashed var(--bd-line); margin-top: 0.4em; padding-top: 0.35em; }
.bd-mn-ro .bd-top { display: flex; justify-content: space-between; font-size: 0.88em; margin-bottom: 0.15em; gap: 0.5em; }
.bd-mn-ro .bd-top span:first-child { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.bd-mn-ro .bd-top span:last-child { color: var(--bd-muted); font-variant-numeric: tabular-nums; white-space: nowrap; }
.bd-mn-ro .bd-cells div { height: 1.35em; }
@container (max-width: 620px) {
  .bd-mn { grid-template-columns: minmax(0, 1fr); grid-template-rows: none; height: auto; }
  .bd-mn-in, .bd-mn-act { grid-row: auto; grid-column: auto; }
  .bd-mn-pad { max-width: 14em; margin: 0 auto; }
  .bd-mn-chartbox { flex: none; height: 12em; }
}

/* ---------- joinBrain (inherits the size of its container) ---------- */
.bd-wrap > .bd-join { font-size: max(13px, 1em); }
.bd-join { display: inline-flex; align-items: center; gap: 0.55em; color: var(--bd-muted); padding: 0.35em 0.75em;
  border: 1px solid var(--bd-line); border-radius: 0.8em; background: var(--bd-panel); max-width: 100%; }
.bd-join .bd-txt { min-width: 0; }
.bd-join b { color: var(--bd-text); font-weight: 600; }
.bd-join .bd-bar { width: 6em; flex: none; }
.bd-join .bd-st { width: 0.55em; height: 0.55em; border-radius: 50%; background: var(--bd-muted); flex: none; }
.bd-join.bd-ready .bd-st { background: #5be3a0; box-shadow: 0 0 6px #5be3a0; }
.bd-join.bd-loading .bd-st { background: var(--bd-accent); animation: bd-pulse 1s ease-in-out infinite alternate; }
.bd-join.bd-error .bd-st { background: var(--bd-red); }
@keyframes bd-pulse { to { opacity: 0.3; } }
`;
const injectCSS = once(() => { const s = document.createElement("style"); s.id = "bd-brain-css"; s.textContent = CSS; document.head.appendChild(s); return s; });

/** Replace el's content with a scoped wrapper; returns {wrap, q} (q = querySelector inside it). */
function mount(el, html) {
  injectCSS();
  el.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.className = "bd-wrap";
  wrap.innerHTML = html;
  el.appendChild(wrap);
  return { wrap, q: (s) => wrap.querySelector(s) };
}

/** A requestAnimationFrame loop that only runs between start() and stop(). frame(dtSeconds, now). */
function rafLoop(frame) {
  let id = 0, last = 0;
  const tick = (ts) => {
    id = requestAnimationFrame(tick);
    const dt = last ? Math.min((ts - last) / 1000, 0.1) : 0;
    last = ts;
    frame(dt, ts);
  };
  return {
    get running() { return id !== 0; },
    start() { if (!id) { last = 0; id = requestAnimationFrame(tick); } },
    stop() { cancelAnimationFrame(id); id = 0; },
  };
}

function loadText(s) {
  if (s.status === "error") return null;
  if (s.loaded >= s.total && s.total) return ["Unpacking the fly brain…", ""];
  const what = s.cached ? "Loading the fly brain stored on this device" : "Downloading the fly brain to this device";
  return [what, `${mb(s.loaded)} / ${mb(s.total)} MB`];
}
/** A progress bar in `host` while the brain loads; hidden once it is ready. Returns unsubscribe. */
function brainProgress(host) {
  host.innerHTML = `<div class="bd-load"><div class="bd-lab"><span></span><span></span></div><div class="bd-bar"><i></i></div></div>`;
  const [a, b] = host.querySelectorAll(".bd-lab span"), bar = host.querySelector("i");
  return onBrain((s) => {
    host.hidden = s.status === "ready";
    if (s.status === "error") { host.innerHTML = `<div class="bd-load bd-err">This device could not run the fly brain: ${s.error}</div>`; return; }
    if (s.status === "idle") { a.textContent = "Getting ready to download the fly brain…"; b.textContent = ""; return; }
    const [t1, t2] = loadText(s);
    a.textContent = t1; b.textContent = t2;
    bar.style.width = `${(100 * s.loaded / Math.max(1, s.total)).toFixed(1)}%`;
  });
}

/** Probability cells over 0-9 (like the main app). */
function cellsHTML(p) {
  const best = p ? p.indexOf(Math.max(...p)) : -1;
  return [...Array(10).keys()].map((d) => {
    const v = p ? p[d] : 0;
    return `<div class="${d === best ? "bd-max" : ""}" style="${p ? `background:rgba(245,197,66,${(0.08 + 0.92 * v).toFixed(3)});` : ""}${v > 0.55 ? "color:#17130a" : ""}"
      title="${p ? (v * 100).toFixed(1) + "%" : ""}">${d}</div>`;
  }).join("");
}

/** Show a 28x28 image (values 0-1) on a canvas. */
function putImage(canvas, px, w = 28, h = 28) {
  canvas.width = w; canvas.height = h;
  const c = canvas.getContext("2d"), img = c.createImageData(w, h);
  for (let i = 0; i < w * h; i++) { const g = Math.round(clamp(px[i], 0, 1) * 255); img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = g; img.data[i * 4 + 3] = 255; }
  c.putImageData(img, 0, 0);
}

/** A 280x280 drawing pad (white ink on black). onStroke() is called when the user draws. */
function drawingPad(canvas, onStroke) {
  canvas.width = canvas.height = 280;
  const ctx = canvas.getContext("2d");
  const pad = { dirty: false, clear() { ctx.fillStyle = "#000"; ctx.fillRect(0, 0, 280, 280); pad.dirty = false; } };
  pad.clear();
  let drawing = false, last = null;
  const pos = (e) => { const r = canvas.getBoundingClientRect(); return [(e.clientX - r.left) * 280 / r.width, (e.clientY - r.top) * 280 / r.height]; };
  const stroke = (a, b) => {
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 20; ctx.lineCap = "round"; ctx.lineJoin = "round";
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); pad.dirty = true;
  };
  canvas.addEventListener("pointerdown", (e) => { drawing = true; last = pos(e); canvas.setPointerCapture(e.pointerId); onStroke && onStroke(); stroke(last, last); e.preventDefault(); });
  canvas.addEventListener("pointermove", (e) => { if (!drawing) return; const p = pos(e); stroke(last, p); last = p; });
  const up = () => { drawing = false; };
  canvas.addEventListener("pointerup", up); canvas.addEventListener("pointercancel", up);
  // Draw an MNIST image (28x28) upscaled onto the pad.
  pad.showImage = (px) => {
    const tmp = document.createElement("canvas"); putImage(tmp, px);
    pad.clear(); ctx.imageSmoothingEnabled = true; ctx.drawImage(tmp, 0, 0, 280, 280); pad.dirty = true;
  };
  return pad;
}

/** Spikes per ms per region on a log scale (as in the main app), revealed up to `upto` ms. */
function drawTimeline(canvas, timeline, regions, upto = Infinity) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  const c = canvas.getContext("2d"), W = canvas.width, H = canvas.height;
  // text follows the widget's font size (which follows the deck), like the rest of the widget
  const fs = Math.max(11, 0.88 * (parseFloat(getComputedStyle(canvas).fontSize) || 14));
  const L = fs * 2.6 * dpr, B = fs * 1.6 * dpr, T = fs * 0.6 * dpr, R = fs * 1.9 * dpr;
  c.clearRect(0, 0, W, H);
  c.font = `${fs * dpr}px system-ui, sans-serif`; c.fillStyle = "#8b93a1"; c.strokeStyle = "#262b35"; c.lineWidth = dpr;
  if (!timeline) {
    c.textAlign = "center"; c.fillText("Run a digit to see the activity", W / 2, H / 2);
    return;
  }
  const n = timeline[0].length;
  let mx = 1; timeline.forEach((s) => s.forEach((v) => { if (v > mx) mx = v; }));
  const ly = (v) => H - B - (Math.log10(1 + v) / Math.log10(1 + mx)) * (H - B - T);
  const lx = (i) => L + (i / (n - 1)) * (W - L - R);
  c.textAlign = "right";
  for (let p = 0; Math.pow(10, p) <= mx; p++) {
    const v = Math.pow(10, p), yy = ly(v);
    c.beginPath(); c.moveTo(L, yy); c.lineTo(W - R, yy); c.stroke();
    c.fillText(v >= 1000 ? v / 1000 + "k" : v, L - fs * 0.35 * dpr, yy + fs * 0.35 * dpr);
  }
  c.textAlign = "center";
  const step = (W - L - R) / dpr / fs > 22 ? 20 : 50;    // fewer ticks when narrow; labels in seconds
  for (let ms = 0; ms <= n; ms += step) c.fillText(ms ? `${+(ms / 1000).toFixed(2)} s` : "0", lx(Math.min(ms, n - 1)), H - fs * 0.35 * dpr);
  const last = Math.min(n - 1, upto);
  timeline.forEach((s, k) => {
    c.strokeStyle = regions[k].color; c.lineWidth = Math.max(1.6, fs / 9) * dpr; c.beginPath();
    for (let i = 0; i <= last; i++) (i ? c.lineTo(lx(i), ly(s[i])) : c.moveTo(lx(i), ly(s[i])));
    c.stroke();
  });
  if (upto < n - 1) { c.strokeStyle = "#e6e8ec"; const x = lx(upto); c.beginPath(); c.moveTo(x, T); c.lineTo(x, H - B); c.stroke(); }
}

// Plain-language names (the audience is high school students; the lecturer explains the rest out loud).
// Readers (= readouts, the trained logistic regressions), by readout key.
const RO_NAMES = {   // the MNIST slide uses the scientific names
  "whole brain": "Whole brain",
  "optic lobe": "Optic lobe",
  "central brain": "Central brain",
  "descending (brain -> body)": "Descending neurons",
  "visual projection (optic lobe -> central brain)": "Visual projection",
  "eye input (stimulated neurons, noisy pixels)": "Eye input",
  "pixels": "Raw pixels (no brain)",
};
// Region row labels on the MNIST slide (scientific, short).
const MN_REGION_NAMES = { "eye input": "Eye input", "optic lobe": "Optic lobe", "visual projection": "Visual projection",
  "central brain": "Central brain", "descending": "Descending", "other": "Other" };
// Brain regions, by the name /api/brain uses: plain name, short label for bars, scientific name, one line.
const REGIONS_PLAIN = {
  "eye input": { name: "Eye cells", short: "eye cells", sci: "photoreceptors, lamina", info: "Where the picture comes in." },
  "optic lobe": { name: "Seeing", short: "seeing", sci: "optic lobe", info: "Works out what the fly sees. Half of all neurons!" },
  "visual projection": { name: "Vision messengers", short: "messengers", sci: "visual projection neurons", info: "Carry what was seen to the centre." },
  "central brain": { name: "Centre of the brain", short: "centre", sci: "central brain", info: "Smell, memory, finding the way, deciding." },
  "descending": { name: "Wires to the body", short: "to body", sci: "descending neurons", info: "Send commands to the legs and wings." },
  "other": { name: "Other", short: "other", sci: "", info: "Senses from the body, and more." },
};
const plain = (name) => REGIONS_PLAIN[name] || { name, short: name, sci: "", info: "" };
// 1234 -> "1,234", 23456 -> "23k", 123456 -> "123k"
const fmtK = (v) => (v >= 1e4 ? `${Math.round(v / 1000)}k` : fmt(v));

// ============================================================================================
// 1. brainExplorer: the whole brain in 3D, regions and neuropils
// ============================================================================================

// Main neuropils. `optic`: belongs to the optic lobe (only optic neurons are highlighted).
// r: radius in µm around the anchor (the median position of the neurons that belong there).
// name: as in /api/brain anchors; label: shown on screen (scientific name); full: title in the info box;
// fn: what it does, in plain words.
const NEUROPILS = [
  { name: "Lamina", label: "Lamina", optic: true, r: 55, fn: "First stop for the eye's signals." },
  { name: "Medulla", label: "Medulla", optic: true, r: 75, fn: "Local motion and colour." },
  { name: "Lobula", label: "Lobula", optic: true, r: 50, fn: "Shapes and looming objects." },
  { name: "Lobula plate", label: "Lobula plate", optic: true, r: 50, fn: "Wide-field motion (self-motion)." },
  { name: "Mushroom body", label: "Mushroom body", r: 55, fn: "Learning and memory." },
  { name: "Lateral horn", label: "Lateral horn", r: 45, fn: "Inborn reactions to smells." },
  { name: "Antennal lobe", label: "Antennal lobe", r: 45, fn: "First stop for smells from the antennae." },
  { name: "Central complex", label: "Central complex", r: 50, fn: "Inner compass, navigation." },
  { name: "SEZ", label: "SEZ", full: "SEZ (subesophageal zone)", r: 60, fn: "Taste and feeding." },
];
// The brain map uses the scientific region names, each with a plain description.
const EXPLORER_REGIONS = {
  "eye input": { name: "Eye input", info: "Stimulated photoreceptors and lamina: where the picture comes in." },
  "optic lobe": { name: "Optic lobe", info: "Processes what the eyes see. Half of all neurons." },
  "visual projection": { name: "Visual projection neurons", info: "Carry detected features (motion, looming) to the central brain." },
  "central brain": { name: "Central brain", info: "Smell, memory, navigation, decisions." },
  "descending": { name: "Descending neurons", info: "Wires from the brain to the body." },
  "other": { name: "Other", info: "Sensory, ascending and other neurons." },
};
const exRegion = (name) => EXPLORER_REGIONS[name] || { name, info: "" };
const OPTIC_REGIONS = new Set(["eye input", "optic lobe", "visual projection"]);
// The explorer's slow rotation: swing around the vertical axis (radians, seconds), so the brain stays framed.
const SWAY = { az0: 0.3, amp: 0.5, period: 22 };

const VERT = `
  attribute vec3 aColor; attribute float aRegion; attribute float aNp; attribute float aNpD;
  uniform float uPx, uSelRegion, uSelNp, uMix, uOnA, uOnSize;
  varying vec3 vColor; varying float vA;
  void main() {
    vColor = aColor;
    float on = 1.0;                                       // 1 = part of the selection
    if (uSelRegion >= 0.0) on = abs(aRegion - uSelRegion) < 0.5 ? 1.0 : 0.0;
    if (uSelNp >= 0.0) on = abs(aNp - uSelNp) < 0.5 ? 1.0 - 0.45 * aNpD : 0.0;
    vA = mix(0.15, on > 0.0 ? uOnA * on : 0.022, uMix);   // additive: big selections need less alpha
    float size = mix(2.4, on > 0.0 ? uOnSize : 2.0, uMix);   // micrometres
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = clamp(size * uPx / -mv.z, 1.0, 24.0);
  }`;
const FRAG = `
  varying vec3 vColor; varying float vA;
  void main() {
    vec2 d = gl_PointCoord - 0.5;
    if (dot(d, d) > 0.25) discard;
    gl_FragColor = vec4(vColor * vA, 1.0);                // additive blending
  }`;

const loadThree = once(async () => {
  let mapped = false;   // does the page have an import map for "three" (like the main app)?
  try { mapped = !!import.meta.resolve && !!import.meta.resolve("three"); } catch { mapped = false; }
  const [THREE, O] = await Promise.all(mapped ? [import(THREE_URL), import(ORBIT_URL)] : [import(THREE_ESM), import(ORBIT_ESM)]);
  return { THREE, OrbitControls: O.OrbitControls };
});

export function brainExplorer(el) {
  const { wrap, q } = mount(el, `
    <div class="bd-ex">
      <div class="bd-ex-map"><canvas></canvas><div class="bd-ex-labels"></div>
        <div class="bd-ex-msg"><span><span class="bd-spin"></span>Loading neuron positions…</span></div>
        <div class="bd-ex-info"></div>
        <div class="bd-ex-help">drag to turn · scroll to zoom</div></div>
      <div class="bd-ex-side">
        <div><div class="bd-h">Regions</div><div class="bd-ex-list"></div></div>
        <div><div class="bd-h">Neuropils</div><div class="bd-ex-nps"></div></div>
      </div>
    </div>`);
  const canvas = q("canvas"), mapEl = q(".bd-ex-map"), labelsEl = q(".bd-ex-labels"), info = q(".bd-ex-info");
  let B = null, G = null;             // brain info; GL state once built
  let started = false, building = null;
  let pinned = null, hover = null;    // {kind: "region"|"np", i}
  let npCount = [], lastSel = { r: -1, n: -1 };

  // ---- side panel ----
  getBrainInfo().then((b) => {
    B = b;
    q(".bd-ex-list").innerHTML = b.regions.map((r, i) => `<button class="bd-ex-item" data-kind="region" data-i="${i}">
      <i class="bd-dot" style="background:${r.color}"></i><span>${exRegion(r.name).name}</span><b>${fmt(r.neurons)}</b></button>`).join("");
    q(".bd-ex-nps").innerHTML = NEUROPILS.map((p, i) => `<button data-kind="np" data-i="${i}" title="${p.full || p.name}">${p.label}</button>`).join("");
    wrap.querySelectorAll(".bd-ex-side [data-kind]").forEach(bindSelect);
    showInfo();
    if (started) build();
  }).catch((e) => { q(".bd-ex-msg").innerHTML = `<span class="bd-err">${e.message}</span>`; });

  function bindSelect(btn) {
    const sel = () => ({ kind: btn.dataset.kind, i: +btn.dataset.i });
    btn.addEventListener("pointerenter", (e) => { if (e.pointerType === "mouse") { hover = sel(); applySel(); } });
    btn.addEventListener("pointerleave", () => { hover = null; applySel(); });
    btn.addEventListener("click", () => {
      const s = sel();
      pinned = pinned && pinned.kind === s.kind && pinned.i === s.i ? null : s;
      hover = null; applySel();
    });
  }
  const current = () => hover || pinned;

  function showInfo() {
    const s = current();
    if (!s) {
      info.innerHTML = `<h4>${fmt(B.n)} neurons</h4><p>Each dot is one neuron.</p>`;
    } else if (s.kind === "region") {
      const r = B.regions[s.i];
      const er = exRegion(r.name);
      info.innerHTML = `<h4><i class="bd-dot" style="background:${r.color}"></i>${er.name}</h4><p>${er.info}</p>`;
    } else {
      const p = NEUROPILS[s.i];
      info.innerHTML = `<h4><i class="bd-dot" style="background:var(--bd-accent)"></i>${p.full || p.label}</h4><p>${p.fn}</p>`;
    }
  }
  function applySel() {
    const s = current();
    wrap.querySelectorAll("[data-kind]").forEach((b) => b.classList.toggle("bd-on", !!s && b.dataset.kind === s.kind && +b.dataset.i === s.i));
    if (G) for (const L of G.labels) {
      L.el.classList.toggle("bd-on", !!s && s.kind === "np" && L.np === s.i);
      L.el.classList.toggle("bd-dim", !!s && !(s.kind === "np" && L.np === s.i));
    }
    if (s) {
      lastSel = { r: s.kind === "region" ? s.i : -1, n: s.kind === "np" ? s.i : -1 };
      // brightness of the selection: dense selections would saturate to white with additive blending
      const count = s.kind === "region" ? B.regions[s.i].neurons : npCount[s.i] || 1000;
      if (G) { G.uniforms.uOnA.value = clamp(0.55 * Math.sqrt(2500 / count), 0.2, 0.6); G.uniforms.uOnSize.value = clamp(3.6 - count / 25000, 2.4, 3.6); }
    }
    if (B) showInfo();
    if (G && !started) render();
  }

  // ---- 3D scene (built on first start) ----
  function build() {
    if (building || !B) return building;
    building = (async () => {
      const { THREE, OrbitControls } = await loadThree();
      const { n, pos, region, regions, anchors } = B;
      // positions in µm; centre; flip so dorsal is +Y, the front of the head +Z and the fly's left +X
      const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      let nv = 0;
      for (let i = 0; i < n; i++) {
        if (pos[i * 3] < 0) continue;
        nv++;
        for (let k = 0; k < 3; k++) { const v = pos[i * 3 + k] / 10; if (v < lo[k]) lo[k] = v; if (v > hi[k]) hi[k] = v; }
      }
      const center = lo.map((v, k) => (v + hi[k]) / 2), extent = Math.max(...hi.map((v, k) => v - lo[k]));
      const toScene = (x, y, z) => [-(x - center[0]), -(y - center[1]), -(z - center[2])];

      // Each neuron belongs to the nearest neuropil anchor within that neuropil's radius (and of the right kind).
      const npIdx = anchors.map((a) => NEUROPILS.findIndex((p) => p.name === a.name));
      const P = new Float32Array(nv * 3), C = new Float32Array(nv * 3), AR = new Float32Array(nv), AN = new Float32Array(nv), AD = new Float32Array(nv);
      const rgb = regions.map((r) => hexToRgb(r.color).map((c) => c / 255));
      const optic = regions.map((r) => OPTIC_REGIONS.has(r.name));
      npCount = new Array(NEUROPILS.length).fill(0);
      let v = 0;
      for (let i = 0; i < n; i++) {
        if (pos[i * 3] < 0) continue;
        const x = pos[i * 3] / 10, y = pos[i * 3 + 1] / 10, z = pos[i * 3 + 2] / 10;
        P.set(toScene(x, y, z), v * 3); C.set(rgb[region[i]], v * 3); AR[v] = region[i];
        let best = -1, bd = Infinity;
        for (let a = 0; a < anchors.length; a++) {
          const ap = anchors[a].pos, d2 = (x - ap[0]) ** 2 + (y - ap[1]) ** 2 + (z - ap[2]) ** 2;
          if (d2 < bd) { bd = d2; best = a; }
        }
        // optic neuropils: optic-lobe-side neurons only; central ones: central neurons (visual projection neurons count for both)
        const np = best >= 0 ? npIdx[best] : -1, d = Math.sqrt(bd);
        const ok = np >= 0 && d < NEUROPILS[np].r && (NEUROPILS[np].optic ? optic[region[i]] : !optic[region[i]] || regions[region[i]].name === "visual projection");
        AN[v] = ok ? np : -1; AD[v] = ok ? d / NEUROPILS[np].r : 1;
        if (ok) npCount[np]++;
        v++;
      }
      const renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
      renderer.setClearColor(0x05070a, 1);
      const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(35, 1, 1, 50000);
      const controls = new OrbitControls(camera, canvas);
      controls.enableDamping = true; controls.dampingFactor = 0.09; controls.rotateSpeed = 0.7;
      controls.enablePan = false;
      // The brain slowly swings left and right (see SWAY); dragging or zooming pauses it for a few seconds.
      controls.addEventListener("start", () => { G.dragging = true; });
      controls.addEventListener("end", () => { G.dragging = false; G.lastInteract = performance.now(); });
      controls.addEventListener("change", () => { if (!started) render(); });

      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(P, 3));
      g.setAttribute("aColor", new THREE.BufferAttribute(C, 3));
      g.setAttribute("aRegion", new THREE.BufferAttribute(AR, 1));
      g.setAttribute("aNp", new THREE.BufferAttribute(AN, 1));
      g.setAttribute("aNpD", new THREE.BufferAttribute(AD, 1));
      const uniforms = { uPx: { value: 1 }, uSelRegion: { value: -1 }, uSelNp: { value: -1 }, uMix: { value: 0 }, uOnA: { value: 0.5 }, uOnSize: { value: 3 } };
      const points = new THREE.Points(g, new THREE.ShaderMaterial({ uniforms, vertexShader: VERT, fragmentShader: FRAG,
        transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending }));
      scene.add(points);

      // Clickable neuropil labels; earlier in `priority` wins when two would overlap on screen.
      const priority = ["Medulla", "Central complex", "Mushroom body", "Antennal lobe", "Lobula", "SEZ", "Lamina", "Lateral horn", "Lobula plate"];
      const labels = [...anchors].sort((a, b) => priority.indexOf(a.name) - priority.indexOf(b.name)).map((a) => {
        const np = NEUROPILS.findIndex((p) => p.name === a.name);
        const btn = document.createElement("button");
        btn.className = "bd-ex-lbl"; btn.textContent = NEUROPILS[np] ? NEUROPILS[np].label : a.name; btn.dataset.kind = "np"; btn.dataset.i = np;
        labelsEl.appendChild(btn); bindSelect(btn);
        return { el: btn, np, p: new THREE.Vector3(...toScene(...a.pos)) };
      });
      // every 25th neuron, to fit the camera to the point cloud
      const sample = [];
      for (let k = 0; k < nv; k += 25) sample.push(new THREE.Vector3(P[k * 3], P[k * 3 + 1], P[k * 3 + 2]));
      G = { THREE, renderer, scene, camera, controls, points, uniforms, labels, extent, sample, viewSet: false, tmp: new THREE.Vector3() };
      q(".bd-ex-msg").hidden = true;
      // a click on empty space (not a drag) clears the selection
      let down = null;
      canvas.addEventListener("pointerdown", (e) => (down = [e.clientX, e.clientY]));
      canvas.addEventListener("pointerup", (e) => {
        if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 4 && pinned) { pinned = null; applySel(); }
        down = null;
      });
      resize(); applySel();
    })().catch((e) => { q(".bd-ex-msg").innerHTML = `<span class="bd-err">Could not draw the brain: ${e.message}</span>`; });
    return building;
  }

  // Fit the camera to the point cloud: front view, a little from the side and above. The distance is
  // the smallest that keeps (almost) every neuron on screen at every angle of the slow rotation, and
  // the target is moved so the brain is centred on screen.
  function setView() {
    const { THREE, camera, controls, sample } = G;
    const tanH = Math.tan((camera.fov / 2) * Math.PI / 180), margin = 0.97;
    const tx = tanH * camera.aspect * margin, ty = tanH * margin;
    const elev = 0.3, az0 = SWAY.az0, cam = new THREE.PerspectiveCamera(), q = new THREE.Vector3();
    // fit for one direction: returns the needed distance and the screen-centre offset (camera space)
    const fit = (az, target) => {
      cam.position.set(Math.sin(az) * Math.cos(elev), Math.sin(elev), Math.cos(az) * Math.cos(elev)).add(target);
      cam.lookAt(target); cam.updateMatrixWorld(); cam.matrixWorldInverse.copy(cam.matrixWorld).invert();
      const need = [], xs = [], ys = [];
      for (const p of sample) {
        q.copy(p).applyMatrix4(cam.matrixWorldInverse);
        const d = -q.z - 1;                                  // depth relative to the target plane
        need.push(Math.max(Math.abs(q.x) / tx, Math.abs(q.y) / ty) - d);
        xs.push(q.x); ys.push(q.y);
      }
      const pct = (a, f) => { a.sort((u, v) => u - v); return a[Math.min(a.length - 1, Math.floor(f * a.length))]; };
      return { D: pct(need, 0.997), cx: (pct(xs, 0.003) + pct(xs, 0.997)) / 2, cy: (pct(ys, 0.003) + pct(ys, 0.997)) / 2, cam };
    };
    // centre: shift the target so the front view is centred
    const target = new THREE.Vector3();
    const f0 = fit(az0, target);
    const right = new THREE.Vector3().setFromMatrixColumn(f0.cam.matrixWorld, 0), up = new THREE.Vector3().setFromMatrixColumn(f0.cam.matrixWorld, 1);
    target.addScaledVector(right, f0.cx).addScaledVector(up, f0.cy);
    let D = 0;
    for (let k = -3; k <= 3; k++) D = Math.max(D, fit(az0 + (k / 3) * SWAY.amp, target).D);   // every angle of the sway
    const dir = new THREE.Vector3(Math.sin(az0) * Math.cos(elev), Math.sin(elev), Math.cos(az0) * Math.cos(elev));
    controls.target.copy(target);
    camera.position.copy(target).addScaledVector(dir, D); camera.lookAt(target); controls.update();
  }
  function resize() {
    if (!G) return;
    const w = mapEl.clientWidth, h = mapEl.clientHeight;
    if (!w || !h) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    G.renderer.setPixelRatio(dpr); G.renderer.setSize(w, h, false);
    G.camera.aspect = w / h; G.camera.updateProjectionMatrix();
    G.uniforms.uPx.value = (h * dpr) / (2 * Math.tan((G.camera.fov / 2) * Math.PI / 180));
    if (!G.viewSet || Math.abs(G.viewAspect - G.camera.aspect) > 0.15) { setView(); G.viewSet = true; G.viewAspect = G.camera.aspect; }
    render();
  }
  function updateLabels() {
    const W = mapEl.clientWidth, H = mapEl.clientHeight, v = G.tmp, placed = [];
    for (const L of G.labels) {
      v.copy(L.p).project(G.camera);
      let vis = v.z < 1 && Math.abs(v.x) < 1.05 && Math.abs(v.y) < 1.05;
      if (vis) {
        if (!L.w) { L.el.style.display = ""; L.w = L.el.offsetWidth; L.h = L.el.offsetHeight; }
        const x = ((v.x + 1) / 2) * W - 4, y = ((1 - v.y) / 2) * H;
        const r = [x - 2, y - L.h / 2, x + L.w + 2, y + L.h / 2];
        vis = r[0] >= 0 && r[1] >= 0 && r[2] <= W && r[3] <= H && !placed.some((o) => r[0] < o[2] && r[2] > o[0] && r[1] < o[3] && r[3] > o[1]);
        if (vis) { placed.push(r); L.el.style.transform = `translate(${x}px, ${y}px) translateY(-50%)`; }
      }
      L.el.style.display = vis ? "" : "none";
    }
  }
  function render() {
    if (!G) return;
    const target = current() ? 1 : 0;
    // instant when stopped (nothing animates), eased while running
    const u = G.uniforms;
    u.uSelRegion.value = lastSel.r; u.uSelNp.value = lastSel.n;
    if (!started) u.uMix.value = target;
    G.renderer.render(G.scene, G.camera);
    updateLabels();
  }
  const off = { v: null, s: null };
  const loop = rafLoop((dt) => {
    if (!G) return;
    const u = G.uniforms, target = current() ? 1 : 0;
    u.uMix.value += (target - u.uMix.value) * (1 - Math.exp(-dt * 12));
    // slow sway around the vertical axis; eases back in after the user lets go
    if (!G.dragging && performance.now() - (G.lastInteract || 0) > 4000) {
      if (!off.v) { off.v = new G.THREE.Vector3(); off.s = new G.THREE.Spherical(); }
      G.swayT = (G.swayT || 0) + dt;
      const want = SWAY.az0 + SWAY.amp * Math.sin((G.swayT * 2 * Math.PI) / SWAY.period);
      off.v.copy(G.camera.position).sub(G.controls.target); off.s.setFromVector3(off.v);
      let d = want - off.s.theta; d = Math.atan2(Math.sin(d), Math.cos(d));
      off.s.theta += d * (1 - Math.exp(-dt * 1.5));
      off.v.setFromSpherical(off.s); G.camera.position.copy(G.controls.target).add(off.v);
    }
    G.controls.update(dt);
    render();
  });
  new ResizeObserver(() => resize()).observe(mapEl);

  return {
    start() { started = true; build(); loop.start(); },
    stop() { started = false; loop.stop(); },
  };
}

// ============================================================================================
// 2. retinaDemo: how an image enters the brain
// ============================================================================================

// rate (0..1 of max) -> colour: dark grey, orange, accent yellow, near white
const RATE_LUT = (() => {
  const stops = [[0, [42, 47, 58]], [0.35, [200, 120, 42]], [0.75, [245, 197, 66]], [1, [255, 245, 208]]];
  const lut = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255; let k = 0;
    while (k < stops.length - 2 && t > stops[k + 1][0]) k++;
    const [t0, c0] = stops[k], [t1, c1] = stops[k + 1], f = (t - t0) / (t1 - t0);
    for (let j = 0; j < 3; j++) lut[i * 3 + j] = Math.round(c0[j] + (c1[j] - c0[j]) * f);
  }
  return lut;
})();
const SLOW = 60;   // spikes in the eye mosaics are shown this many times slower than real time

export function retinaDemo(el) {
  const { wrap, q } = mount(el, `
    <div class="bd-rt">
      <section class="bd-panel bd-rt-digit">
        <div class="bd-h">1 · The picture</div>
        <div class="bd-rt-imgbox"><canvas class="bd-rt-img" width="28" height="28"></canvas><canvas class="bd-rt-pad" hidden></canvas></div>
        <div class="bd-row bd-rt-btns"><button data-a="another">Another digit</button><button data-a="draw">✎ Draw</button></div>
        <div class="bd-row bd-rt-dbtns" hidden><button class="bd-primary" data-a="use">Show the fly</button><button data-a="cancel">Cancel</button></div>
        <div class="bd-note bd-rt-px"></div>
      </section>
      <section class="bd-panel bd-rt-eyes">
        <div class="bd-h">2 · The fly's two eyes <small>· <span class="bd-rt-nin">18,266</span> eye cells</small></div>
        <div class="bd-rt-pair">
          <figure><div class="bd-rt-eyebox"><canvas data-side="1"></canvas></div><figcaption><b>left eye</b></figcaption></figure>
          <figure><div class="bd-rt-eyebox"><canvas data-side="2"></canvas></div><figcaption><b>right eye</b></figcaption></figure>
        </div>
        <div class="bd-rt-scale"><span>How often each eye cell fires:</span><span>never</span><i></i><span><span class="bd-rt-maxhz">500</span> times a second</span></div>
        <div class="bd-rt-loadhost"></div>
      </section>
      <section class="bd-panel bd-rt-brain">
        <div>
          <button class="bd-primary" data-a="run" disabled>Send it through the brain</button>
          <div class="bd-note bd-rt-status"></div>
          <div class="bd-rbars bd-rt-bars"></div>
        </div>
        <div class="bd-rt-out">
          <div class="bd-rt-arrow"><b>reader</b> → guess <small>(the only part that learns)</small></div>
          <div class="bd-rt-pred"><div class="bd-rt-big">?</div><div><div class="bd-note bd-rt-predtxt"></div><div class="bd-cells">${cellsHTML(null)}</div></div></div>
          <div class="bd-note bd-rt-ronote"></div>
        </div>
      </section>
      <canvas class="bd-rt-overlay"></canvas>
    </div>`);
  const root = q(".bd-rt"), imgCv = q(".bd-rt-img"), padCv = q(".bd-rt-pad"), overlay = q(".bd-rt-overlay");
  const eyeCv = [...wrap.querySelectorAll(".bd-rt-eyebox canvas")], runBtn = q('[data-a="run"]');
  const pad = drawingPad(padCv);
  let B = null, RT = null;                 // brain info; retina mapping
  let image = null, label = null;          // the current 28x28 image
  let eyes = null;                         // per eye: {k: indices of input neurons, sorted by rate}
  let rate = null, flash = null, arrive = null, arriveT = null, flyT = -1;
  let hoverPix = -1, started = false, busy = false, result = null, barT = -1;
  let pixBuckets = null;                   // pixel -> input neurons (CSR)
  const geo = { s: 1, img: null, eyes: [] };

  brainProgress(q(".bd-rt-loadhost"));
  getBrainInfo().then((b) => {
    B = b;
    wrap.querySelectorAll(".bd-rt-maxhz").forEach((e) => (e.textContent = `${b.config.rate_hz}`));
    const ro = b.readouts.find((r) => r.key === "whole brain");
    if (ro) q(".bd-rt-ronote").innerHTML = `Right <b>${(ro.accuracy * 100).toFixed(0)}%</b> of the time`;
    runBtn.textContent = `Send it through the brain (${b.config.duration_ms / 1000} s)`;
    q(".bd-rt-nin").textContent = fmt(b.config.n_inputs);
    renderBars(null);
  });
  getRetina().then((r) => {
    RT = r;
    const nIn = r.uv.length / 2;
    const side = [[], []];
    for (let k = 0; k < nIn; k++) if (r.side[k] === 1 || r.side[k] === 2) side[r.side[k] - 1].push(k);
    eyes = side.map((ks) => ({ k: Int32Array.from(ks) }));
    // pixel of each input neuron (nearest; the brain samples bilinearly) -> buckets
    const pix = new Int32Array(nIn), cnt = new Int32Array(785);
    for (let k = 0; k < nIn; k++) { pix[k] = Math.round(r.uv[2 * k + 1] * 27) * 28 + Math.round(r.uv[2 * k] * 27); cnt[pix[k] + 1]++; }
    for (let i = 0; i < 784; i++) cnt[i + 1] += cnt[i];
    const fill = cnt.slice(0, 784), list = new Int32Array(nIn);
    for (let k = 0; k < nIn; k++) list[fill[pix[k]]++] = k;
    pixBuckets = { start: cnt, list, pix };
    flash = new Float32Array(nIn); arrive = new Float32Array(nIn); arriveT = new Float32Array(nIn);
    runBtn.disabled = false;
    if (image) setRates(true);
  }).catch(() => {});
  getSample(0).then((s) => { if (!image) setImage(s.pixels, s.label); }).catch((e) => (q(".bd-rt-px").innerHTML = `<span class="bd-err">${e.message}</span>`));

  // ---- the image ----
  function setImage(px, lab) {
    image = Float32Array.from(px); label = lab;
    putImage(imgCv, image);
    result = null; barT = -1; renderBars(null); renderPred(null);
    q(".bd-rt-status").textContent = "";
    if (RT) setRates(true);
  }
  // firing rate of every input neuron for the current image (same bilinear sampling as the brain)
  function setRates(animate) {
    const uv = RT.uv, nIn = uv.length / 2;
    rate = new Float32Array(nIn);
    for (let k = 0; k < nIn; k++) {
      const x = uv[2 * k] * 27, y = uv[2 * k + 1] * 27, x0 = Math.floor(x), y0 = Math.floor(y);
      const x1 = Math.min(x0 + 1, 27), y1 = Math.min(y0 + 1, 27), fx = x - x0, fy = y - y0;
      rate[k] = image[y0 * 28 + x0] * (1 - fx) * (1 - fy) + image[y0 * 28 + x1] * fx * (1 - fy) + image[y1 * 28 + x0] * (1 - fx) * fy + image[y1 * 28 + x1] * fx * fy;
    }
    for (const e of eyes) e.k = Int32Array.from([...e.k].sort((a, b) => rate[a] - rate[b]));   // bright on top
    // fly-in: bright neurons leave their pixel row by row, with a little jitter
    for (let k = 0; k < nIn; k++) arriveT[k] = 0.15 + uv[2 * k + 1] * 0.55 + Math.random() * 0.25;
    flyT = animate && started ? 0 : 99;
    arrive.fill(animate && started ? 0 : 1);
    drawStatic();
  }

  // ---- layout: canvas sizes and positions relative to the widget ----
  function layout() {
    // eyes: the largest square that fits in each eye's box
    eyeCv.forEach((cv) => {
      const box = cv.parentElement, side = Math.max(1, Math.floor(Math.min(box.clientWidth, box.clientHeight || box.clientWidth)));
      cv.style.width = cv.style.height = side + "px";
    });
    const R = root.getBoundingClientRect(), s = R.width / Math.max(1, root.offsetWidth) || 1;   // the deck may scale the slide
    const rel = (e) => { const r = e.getBoundingClientRect(); return { x: (r.left - R.left) / s, y: (r.top - R.top) / s, w: r.width / s, h: r.height / s }; };
    geo.s = s; geo.img = rel(imgCv); geo.eyes = eyeCv.map(rel);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    overlay.width = Math.round(root.offsetWidth * dpr); overlay.height = Math.round(root.offsetHeight * dpr);
    overlay.style.width = root.offsetWidth + "px"; overlay.style.height = root.offsetHeight + "px";
    eyeCv.forEach((cv) => { cv.width = Math.max(1, Math.round(cv.clientWidth * dpr)); cv.height = Math.max(1, Math.round(cv.clientHeight * dpr)); cv._img = null; });
    drawStatic();
  }
  const eyeXY = (k, g) => [g.x + RT.uv[2 * k] * g.w, g.y + RT.uv[2 * k + 1] * g.h];

  // ---- drawing the eyes (ImageData; ~9k dots each) ----
  function drawEyes() {
    if (!RT || !rate) return;
    eyeCv.forEach((cv, e) => {
      const W = cv.width, H = cv.height;
      if (W < 2) return;
      const ctx = cv.getContext("2d");
      if (!cv._img || cv._img.width !== W) cv._img = ctx.createImageData(W, H);
      const img = cv._img, d = img.data;
      d.fill(0);
      const ds = Math.max(2, Math.round(W / 115)), ks = eyes[e].k, uv = RT.uv;
      for (let j = 0; j < ks.length; j++) {
        const k = ks[j], a = arrive[k], v = rate[k] * a;
        const li = Math.round(v * 255) * 3, f = flash[k] * a * 0.9;
        const r = RATE_LUT[li] + (255 - RATE_LUT[li]) * f, g = RATE_LUT[li + 1] + (255 - RATE_LUT[li + 1]) * f, b = RATE_LUT[li + 2] + (255 - RATE_LUT[li + 2]) * f;
        const x0 = Math.round(uv[2 * k] * (W - ds)), y0 = Math.round(uv[2 * k + 1] * (H - ds));
        for (let y = y0; y < y0 + ds; y++) for (let x = x0, o = (y * W + x0) * 4; x < x0 + ds; x++, o += 4) { d[o] = r; d[o + 1] = g; d[o + 2] = b; d[o + 3] = 255; }
      }
      ctx.putImageData(img, 0, 0);
    });
  }
  function drawOverlay() {
    const dpr = overlay.width / Math.max(1, root.offsetWidth);
    const c = overlay.getContext("2d");
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, overlay.width, overlay.height);
    if (!RT || !rate || !geo.img) return;
    const gi = geo.img, cell = gi.w / 28;
    // pixels flying to their neurons
    if (flyT >= 0 && flyT < 1.6) {
      for (let e = 0; e < 2; e++) {
        const ks = eyes[e].k, ge = geo.eyes[e];
        for (let j = ks.length - 1; j >= 0; j--) {
          const k = ks[j], v = rate[k];
          if (v < 0.12) break;                                   // sorted by rate: the rest are dark
          const t = (flyT - (arriveT[k] - 0.55)) / 0.55;
          if (t <= 0 || t >= 1) continue;
          const ease = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
          const sx = gi.x + RT.uv[2 * k] * gi.w, sy = gi.y + RT.uv[2 * k + 1] * gi.h;
          const [ex, ey] = eyeXY(k, ge);
          const x = sx + (ex - sx) * ease, y = sy + (ey - sy) * ease - Math.sin(Math.PI * ease) * 40;
          const li = Math.round(v * 255) * 3;
          c.fillStyle = `rgb(${RATE_LUT[li]},${RATE_LUT[li + 1]},${RATE_LUT[li + 2]})`;
          c.fillRect(x - 1.2, y - 1.2, 2.4, 2.4);
        }
      }
    }
    // hover link: one pixel <-> the neurons that sample it
    if (hoverPix >= 0 && pixBuckets) {
      const px = hoverPix % 28, py = Math.floor(hoverPix / 28);
      const cx = gi.x + (px + 0.5) * cell, cy = gi.y + (py + 0.5) * cell;
      c.strokeStyle = "#f5c542"; c.lineWidth = 2; c.strokeRect(gi.x + px * cell, gi.y + py * cell, cell, cell);
      const { start, list } = pixBuckets;
      c.lineWidth = 1; c.strokeStyle = "rgba(245,197,66,0.35)";
      c.beginPath();
      const ends = [];
      for (let j = start[hoverPix]; j < start[hoverPix + 1]; j++) {
        const k = list[j], e = RT.side[k] - 1;
        if (e < 0 || e > 1) continue;
        const [x, y] = eyeXY(k, geo.eyes[e]);
        c.moveTo(cx, cy); c.lineTo(x, y); ends.push([x, y]);
      }
      c.stroke();
      c.strokeStyle = "#fff"; c.lineWidth = 1.5;
      for (const [x, y] of ends) { c.beginPath(); c.arc(x, y, 4, 0, 2 * Math.PI); c.stroke(); }
    }
  }
  function drawStatic() { drawEyes(); drawOverlay(); }

  function setHover(p) {
    if (p === hoverPix) return;
    hoverPix = p;
    const box = q(".bd-rt-px");
    if (p < 0 || !pixBuckets || !image) { box.innerHTML = ""; if (!started) drawOverlay(); return; }
    const { start, list } = pixBuckets, types = new Map(), per = [0, 0];
    for (let j = start[p]; j < start[p + 1]; j++) {
      const k = list[j]; per[RT.side[k] - 1]++;
      const t = RT.cellTypeNames[RT.cellType[k]]; types.set(t, (types.get(t) || 0) + 1);
    }
    const hz = Math.round(image[p] * (B ? B.config.rate_hz : 500));
    box.innerHTML = `${per[0] + per[1]} eye cells see this pixel.<br>They fire <b>${hz}</b> times a second.`;
    if (!started) drawOverlay();
  }
  imgCv.addEventListener("pointermove", (e) => {
    const r = imgCv.getBoundingClientRect();
    const x = Math.floor((e.clientX - r.left) / r.width * 28), y = Math.floor((e.clientY - r.top) / r.height * 28);
    setHover(x >= 0 && x < 28 && y >= 0 && y < 28 ? y * 28 + x : -1);
  });
  imgCv.addEventListener("pointerleave", () => setHover(-1));
  // hovering a neuron in an eye: link it to its pixel
  eyeCv.forEach((cv, e) => {
    cv.addEventListener("pointermove", (ev) => {
      if (!RT || !pixBuckets) return;
      const r = cv.getBoundingClientRect(), u = (ev.clientX - r.left) / r.width, v = (ev.clientY - r.top) / r.height;
      let best = -1, bd = Infinity;
      for (const k of eyes[e].k) { const d = (RT.uv[2 * k] - u) ** 2 + (RT.uv[2 * k + 1] - v) ** 2; if (d < bd) { bd = d; best = k; } }
      setHover(best >= 0 && bd < 0.003 ? pixBuckets.pix[best] : -1);
    });
    cv.addEventListener("pointerleave", () => setHover(-1));
  });

  // ---- buttons ----
  const setDrawMode = (on) => {
    padCv.hidden = !on; imgCv.hidden = on;
    q(".bd-rt-btns").hidden = on; q(".bd-rt-dbtns").hidden = !on;
    if (on) { pad.clear(); q(".bd-rt-px").textContent = "Draw a digit."; setHover(-1); }
  };
  q('[data-a="another"]').onclick = async () => { try { const s = await getSample(); setImage(s.pixels, s.label); } catch (e) { q(".bd-rt-status").innerHTML = `<span class="bd-err">${e.message}</span>`; } };
  q('[data-a="draw"]').onclick = () => setDrawMode(true);
  q('[data-a="cancel"]').onclick = () => setDrawMode(false);
  q('[data-a="use"]').onclick = async () => {
    if (!pad.dirty) return;
    try { const d = await preprocess(padCv); setDrawMode(false); setImage(d.image, null); }
    catch (e) { q(".bd-rt-px").innerHTML = `<span class="bd-err">${e.message}</span>`; }
  };
  runBtn.onclick = async () => {
    if (busy || !image) return;
    busy = true; runBtn.disabled = true;
    const st = q(".bd-rt-status");
    try {
      st.innerHTML = `<span class="bd-spin"></span>Running the fly brain…`;
      const img = image;
      const [r] = await Promise.all([runImage(img), getReadouts()]);
      const p = (await predict(r.counts, img))["whole brain"];
      const statusHTML = "How much each part fired";
      st.innerHTML = statusHTML;
      result = { payload: r.payload, p, statusHTML };
      barT = started ? 0 : 99;
      renderBars(result, started ? 0 : Infinity);
      if (!started) renderPred(p);
    } catch (e) { st.innerHTML = `<span class="bd-err">${e.message}</span>`; }
    finally { busy = false; runBtn.disabled = !RT; }
  };

  // spikes per region (log scale), counted up to `upto` ms
  function renderBars(res, upto = Infinity) {
    const box = q(".bd-rt-bars");
    if (!B) return;
    if (!box.children.length) {
      box.innerHTML = B.regions.map((r) => `<div class="bd-rbar"><span title="${plain(r.name).name}"><i class="bd-dot" style="background:${r.color}"></i>${plain(r.name).short}</span>
        <div class="bd-track"><i style="background:${r.color}"></i><em></em></div><span></span></div>`).join("");
    }
    const rows = box.querySelectorAll(".bd-rbar");
    const tl = res && res.payload.timeline, sum = res ? res.payload.summary : null;
    const max = sum ? Math.max(...sum.map((s) => s.spikes)) : 1;
    rows.forEach((row, k) => {
      let v = 0;
      if (tl) { const n = Math.min(tl[k].length, Math.floor(upto) + 1); for (let i = 0; i < n; i++) v += tl[k][i]; }
      row.querySelector(".bd-track i").style.width = res ? `${(100 * Math.log10(1 + v) / Math.log10(1 + max)).toFixed(1)}%` : "0";
      row.lastElementChild.textContent = res ? fmtK(v) : "";
    });
  }
  function renderPred(p) {
    const big = q(".bd-rt-big"), txt = q(".bd-rt-predtxt");
    q(".bd-rt-pred .bd-cells").innerHTML = cellsHTML(p);
    if (!p) { big.textContent = "?"; txt.textContent = ""; return; }
    const d = p.indexOf(Math.max(...p));
    big.textContent = d;
    txt.innerHTML = `<b>${(p[d] * 100).toFixed(0)}%</b> sure` + (label != null ? (d === label ? " · ✓ right" : ` · ✗ it was a ${label}`) : "");
  }

  // ---- animation ----
  const loop = rafLoop((dt) => {
    if (!RT || !rate) return;
    if (flyT >= 0 && flyT < 99) {
      flyT += dt;
      for (let k = 0; k < arrive.length; k++) arrive[k] = clamp((flyT - arriveT[k]) / 0.12, 0, 1);
      if (flyT > 1.6) { flyT = 99; arrive.fill(1); }
    }
    // Poisson spikes, slowed down: each neuron flashes at rate/SLOW
    const hz = (B ? B.config.rate_hz : 500) / SLOW, decay = Math.exp(-dt / 0.09);
    for (let k = 0; k < flash.length; k++) {
      let f = flash[k] * decay;
      if (rate[k] > 0 && Math.random() < rate[k] * hz * dt) f = 1;
      flash[k] = f;
    }
    if (result && barT >= 0 && barT < 99) {
      barT += dt;
      const ms = result.payload.duration_ms, t = (barT / 1.6) * ms;
      renderBars(result, t);
      q(".bd-rt-status").innerHTML = `${result.statusHTML} · <b>${(Math.min(ms, t) / 1000).toFixed(2)} s</b>`;
      if (t >= ms) { barT = 99; renderBars(result); renderPred(result.p); q(".bd-rt-status").innerHTML = result.statusHTML; }
    }
    drawStatic();
  });
  new ResizeObserver(() => layout()).observe(root);

  return {
    start() { started = true; layout(); loop.start(); },
    stop() { started = false; loop.stop(); if (flash) flash.fill(0); if (flyT < 99) { flyT = 99; arrive.fill(1); } if (barT >= 0 && barT < 99 && result) { barT = 99; renderBars(result); renderPred(result.p); } drawStatic(); },
  };
}

// ============================================================================================
// 3. mnistDemo: draw a digit, run it, see activity and every readout's prediction
// ============================================================================================

const MN_READOUTS = ["whole brain", "optic lobe", "central brain", "descending (brain -> body)", "pixels"];

export function mnistDemo(el) {
  const { wrap, q } = mount(el, `
    <div class="bd-mn">
      <section class="bd-panel bd-mn-in">
        <div class="bd-h">Draw a digit</div>
        <canvas class="bd-mn-pad"></canvas>
        <div class="bd-row"><button class="bd-primary" data-a="run">Run</button><button data-a="clear">Clear</button>
          <button data-a="sample" style="flex:1 1 100%">Example digit</button></div>
        <div class="bd-mn-head"><div class="bd-mn-seen" title="What the eye gets: cropped and centred like MNIST, 28×28"><canvas width="28" height="28"></canvas></div>
          <div class="bd-mn-big">?</div><div class="bd-note bd-mn-headtxt"></div></div>
        <div class="bd-note bd-mn-status"></div>
        <div class="bd-mn-loadhost"></div>
      </section>
      <section class="bd-panel bd-mn-act">
        <div class="bd-h">Brain activity over time</div>
        <div class="bd-mn-chartbox"><canvas></canvas></div>
      </section>
      <section class="bd-panel bd-mn-reg">
        <div class="bd-h">How much each part fired</div>
        <div class="bd-rbars bd-mn-regions"></div>
        <div class="bd-note bd-mn-take" style="margin-top:0.4em"></div>
      </section>
      <section class="bd-panel bd-mn-pred">
        <div class="bd-h">Guesses from each brain part</div>
        <div class="bd-mn-ros"></div>
      </section>
    </div>`);
  const padCv = q(".bd-mn-pad"), seenCv = q(".bd-mn-seen canvas"), chart = q(".bd-mn-chartbox canvas"), runBtn = q('[data-a="run"]');
  let sample = null;                          // {pixels, label} while the pad shows an untouched MNIST sample
  const pad = drawingPad(padCv, () => { if (sample) { sample = null; pad.clear(); } });
  let B = null, res = null, started = false, busy = false, sweep = -1;
  const status = (h) => (q(".bd-mn-status").innerHTML = h);
  putImage(seenCv, new Float32Array(784));

  brainProgress(q(".bd-mn-loadhost"));
  loadBrain().catch(() => {});
  getReadouts().catch(() => {});
  getBrainInfo().then((b) => {
    B = b;
    q(".bd-mn-regions").innerHTML = b.regions.map((r) => `<div class="bd-rbar"><span title="${MN_REGION_NAMES[r.name] || r.name}"><i class="bd-dot" style="background:${r.color}"></i>${MN_REGION_NAMES[r.name] || r.name}</span>
      <div class="bd-track"><i style="background:${r.color}"></i><em></em></div><span></span></div>`).join("");
    renderReadouts(null);
    draw();
  });

  function renderReadouts(pred) {
    const acc = Object.fromEntries(B.readouts.map((r) => [r.key, r.accuracy]));
    q(".bd-mn-ros").innerHTML = MN_READOUTS.filter((k) => k in acc).map((k) => `<div class="bd-mn-ro${k === "pixels" ? " bd-base" : ""}">
      <div class="bd-top"><span>${RO_NAMES[k]}</span><span>right ${(acc[k] * 100).toFixed(0)}%</span></div>
      <div class="bd-cells">${cellsHTML(pred ? pred[k] : null)}</div></div>`).join("");
    const big = q(".bd-mn-big"), txt = q(".bd-mn-headtxt");
    if (!pred) { big.textContent = "?"; txt.textContent = ""; return; }
    const p = pred["whole brain"], d = p.indexOf(Math.max(...p));
    big.textContent = d;
    txt.innerHTML = `<b>${(p[d] * 100).toFixed(0)}%</b> sure` + (res && res.label != null ? (d === res.label ? "<br>✓ right" : `<br>✗ it was a ${res.label}`) : "");
  }
  // Spikes per region (log bars) and how many of its neurons fired. The two regions after the eye
  // with the most spikes are outlined: normally the optic lobe and the visual projection neurons.
  function renderRegions() {
    const rows = q(".bd-mn-regions").children;
    const sum = res ? res.payload.summary : null;
    let hot = new Set();
    const max = sum ? Math.max(...sum.map((s) => s.spikes)) : 1;
    if (sum) {
      const ranked = sum.map((s, k) => ({ k, sp: s.spikes, name: s.name }))
        .filter((s) => s.name !== "eye input" && s.name !== "other").sort((a, b) => b.sp - a.sp);
      hot = new Set(ranked.slice(0, 2).map((s) => s.k));
      q(".bd-mn-take").innerHTML = "Most firing stays in the <b>optic lobe</b>.";
    } else q(".bd-mn-take").textContent = "";
    [...rows].forEach((row, k) => {
      const s = sum && sum[k];
      row.classList.toggle("bd-hot", hot.has(k));
      row.querySelector(".bd-track i").style.width = s ? `${(100 * Math.log10(1 + s.spikes) / Math.log10(1 + max)).toFixed(1)}%` : "0";
      row.lastElementChild.textContent = s ? fmtK(s.spikes) : "";
    });
  }
  function draw() {
    if (!B) return;
    drawTimeline(chart, res && res.payload.timeline, B.regions, sweep >= 0 ? sweep : Infinity);
  }

  q('[data-a="clear"]').onclick = () => { pad.clear(); sample = null; };
  q('[data-a="sample"]').onclick = async () => {
    try {
      const s = await getSample();
      pad.showImage(s.pixels); sample = { pixels: s.pixels, label: s.label, index: s.index };
      status("");
      run();
    } catch (e) { status(`<span class="bd-err">${e.message}</span>`); }
  };
  runBtn.onclick = () => run();

  async function run() {
    if (busy) return;
    if (!pad.dirty) { status("Draw a digit first."); return; }
    busy = true; runBtn.disabled = true;
    try {
      const smp = sample;
      // an untouched MNIST sample is used as is; a drawing is cropped and centred by the server
      const image = smp ? Float32Array.from(smp.pixels) : Float32Array.from((await preprocess(padCv)).image);
      putImage(seenCv, image);
      if (BRAIN.status !== "ready") status(`<span class="bd-spin"></span>Waiting for the fly brain to load…`);
      await loadBrain();
      status(`<span class="bd-spin"></span>Running the fly brain…`);
      const r = await runImage(image);
      const pred = await predict(r.counts, image);
      res = { payload: r.payload, pred, label: smp ? smp.label : null, seconds: r.seconds };
      status(`Done in <b>${r.seconds.toFixed(1)} s</b> on this device`);
      renderRegions(); renderReadouts(pred);
      sweep = started ? 0 : -1;
      if (started) loop.start(); else draw();
    } catch (e) { status(`<span class="bd-err">${e.message}</span>`); }
    finally { busy = false; runBtn.disabled = false; }
  }

  // the timeline is drawn in over ~1.2 s after a run; no animation frames otherwise
  const loop = rafLoop((dt) => {
    if (sweep < 0 || !res) { loop.stop(); return; }
    sweep += dt * (res.payload.duration_ms / 1.2);
    if (sweep >= res.payload.duration_ms) { sweep = -1; draw(); loop.stop(); return; }
    draw();
  });
  new ResizeObserver(() => draw()).observe(q(".bd-mn-chartbox"));

  return {
    start() { started = true; if (sweep >= 0) loop.start(); draw(); },
    stop() { started = false; loop.stop(); if (sweep >= 0) { sweep = -1; draw(); } },
  };
}

// ============================================================================================
// 4. joinBrain: start downloading the brain on the title slide, with a tiny status line
// ============================================================================================

export function joinBrain(el) {
  const { q } = mount(el, `<div class="bd-join"><span class="bd-st"></span><span class="bd-txt">Fly brain on this device: …</span><span class="bd-bar" hidden><i></i></span></div>`);
  const box = q(".bd-join"), txt = q(".bd-txt"), bar = q(".bd-bar");
  loadBrain().catch(() => {});
  onBrain((s) => {
    box.className = "bd-join bd-" + s.status;
    bar.hidden = s.status !== "loading";
    if (s.status === "ready") {
      txt.innerHTML = `Fly brain on this device: <b>ready</b>`;
    } else if (s.status === "error") {
      txt.textContent = `Fly brain on this device: could not load (${s.error})`;
    } else if (s.status === "loading") {
      const done = s.total && s.loaded >= s.total;
      txt.innerHTML = done ? `Fly brain on this device: <b>unpacking…</b>`
        : `Fly brain on this device: ${s.cached ? "loading" : "downloading"} <b>${mb(s.loaded)} / ${mb(s.total)} MB</b>`;
      bar.firstChild.style.width = `${(100 * s.loaded / Math.max(1, s.total)).toFixed(1)}%`;
    } else txt.textContent = "Fly brain on this device: starting…";
  });
  return { start() {}, stop() {} };
}
