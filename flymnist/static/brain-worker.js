// Runs this visitor's own fly brain in a background thread, so the page stays responsive.
// Messages in:  load {url, gzBytes, rawBytes, regionNames} | lesion {silenced, cut} | run {image, H, W, rateHz, ms, record, describe} | retina {}
// Messages out: progress {loaded, total} | ready {...} | reply {id, ...} | error {id, message}
import { FlyBrainSim } from "./brainsim.js";

let brain = null, regionNames = [], regionSize = null;
const CACHE = "flybrain-model";

async function download(url, gzBytes, rawBytes) {
  // The file is gzip, unpacked here as it streams in. The packed file (~40 MB) is kept on the
  // device between visits (Cache Storage needs https or localhost).
  let res = null, cache = null;
  try {
    cache = await caches.open(CACHE);
    res = await cache.match(url);
    for (const req of await cache.keys()) if (!req.url.endsWith(url)) await cache.delete(req);  // older versions
  } catch { cache = null; }
  const cached = !!res;
  if (!res) {
    res = await fetch(url);
    if (!res.ok) throw new Error(`download failed (${res.status})`);
    if (cache) cache.put(url, res.clone()).catch(() => {});
  }
  const total = +res.headers.get("Content-Length") || gzBytes;
  let loaded = 0, lastPost = 0;
  const counter = new TransformStream({
    transform(chunk, ctl) {
      loaded += chunk.length;
      if (loaded - lastPost > 1 << 20 || loaded === total) { postMessage({ type: "progress", loaded, total, cached }); lastPost = loaded; }
      ctl.enqueue(chunk);
    },
  });
  const reader = res.body.pipeThrough(counter).pipeThrough(new DecompressionStream("gzip")).getReader();
  let buf = new Uint8Array(rawBytes), size = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (size + value.length > buf.length) { const b = new Uint8Array(Math.max(buf.length * 2, size + value.length)); b.set(buf.subarray(0, size)); buf = b; }
    buf.set(value, size); size += value.length;
  }
  return size === buf.length ? buf.buffer : buf.buffer.slice(0, size);
}

// Everything the page draws for one run (same fields the server used to send).
function describe(counts, events, durMs) {
  const n = brain.n, region = brain.region, R = regionNames.length;
  const stepsPerMs = Math.round(1 / brain.dt), nFrames = Math.round(durMs);
  // spikes binned to 1 ms frames, each neuron at most once per frame
  const { steps, neurons } = events, seen = new Int32Array(n).fill(-1);
  const frameN = new Int32Array(neurons.length), offsets = new Array(nFrames + 1).fill(0);
  const timeline = Array.from({ length: R }, () => new Array(nFrames).fill(0));
  let m = 0;
  for (let e = 0; e < steps.length; e++) {
    const f = Math.min(Math.floor(steps[e] / stepsPerMs), nFrames - 1), j = neurons[e];
    timeline[region[j]][f]++;
    if (seen[j] !== f) { seen[j] = f; frameN[m++] = j; offsets[f + 1]++; }
  }
  for (let f = 0; f < nFrames; f++) offsets[f + 1] += offsets[f];
  const active = [], summary = regionNames.map((name, i) => ({ name, neurons: regionSize[i], active: 0, spikes: 0 }));
  for (let i = 0; i < n; i++) if (counts[i]) { active.push(i); summary[region[i]].active++; summary[region[i]].spikes += counts[i]; }
  const activeIdx = Int32Array.from(active), activeCounts = new Uint8Array(active.length);
  for (let k = 0; k < active.length; k++) activeCounts[k] = Math.min(counts[active[k]], 255);
  // most active cell types outside the stimulated eye input
  const byType = new Map();
  for (const i of active) {
    if (region[i] === 0) continue;
    const t = brain.cellType[i];
    let o = byType.get(t);
    if (!o) byType.set(t, (o = { spikes: 0, neurons: 0, regions: new Array(R).fill(0) }));
    o.spikes += counts[i]; o.neurons++; o.regions[region[i]]++;
  }
  const top_types = [...byType.entries()].sort((a, b) => b[1].spikes - a[1].spikes).slice(0, 12).map(([t, o]) => ({
    type: brain.cellTypeNames[t], neurons: o.neurons, spikes: o.spikes,
    region: regionNames[o.regions.indexOf(Math.max(...o.regions))],
  }));
  return {
    duration_ms: durMs, frames: { neurons: frameN.slice(0, m), offsets }, timeline, summary, top_types,
    active: { neurons: activeIdx, counts: activeCounts },
  };
}

// Non-zero spike counts outside the eye input, to send to the server with a recording.
function sparseCounts(counts) {
  const idx = [];
  for (let i = 0; i < brain.n; i++) if (counts[i] && brain.region[i] !== 0) idx.push(i);
  const val = new Uint8Array(idx.length);
  idx.forEach((i, k) => (val[k] = Math.min(counts[i], 255)));
  return { idx: Int32Array.from(idx), val };
}

onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "load") {
      regionNames = msg.regionNames;
      const buf = await download(msg.url, msg.gzBytes, msg.rawBytes);
      brain = new FlyBrainSim(buf);
      regionSize = new Array(regionNames.length).fill(0);
      for (let i = 0; i < brain.n; i++) regionSize[brain.region[i]]++;
      postMessage({ type: "ready", n: brain.n, inputs: brain.nIn, synapses: brain.post.length });
    } else if (msg.type === "lesion") {
      const removed = brain.setLesion(msg.silenced, msg.cut);
      postMessage({ type: "reply", id: msg.id, removed });
    } else if (msg.type === "run") {
      const t0 = performance.now();
      const rates = brain.rates(msg.image, msg.H, msg.W, msg.rateHz);
      const { counts, events } = brain.run(rates, msg.ms, { record: !!msg.describe });
      const seconds = (performance.now() - t0) / 1000;
      const out = { type: "reply", id: msg.id, counts, seconds };
      const transfer = [counts.buffer];
      if (msg.describe) {
        out.payload = describe(counts, events, msg.ms);
        transfer.push(out.payload.frames.neurons.buffer, out.payload.active.neurons.buffer, out.payload.active.counts.buffer);
      }
      if (msg.sparse) { out.sparse = sparseCounts(counts); transfer.push(out.sparse.idx.buffer, out.sparse.val.buffer); }
      postMessage(out, transfer);
    } else if (msg.type === "retina") {
      // The eye mapping, for drawing it (used by the lecture slides): per input neuron its image
      // position (u, v in [0,1]), simulation index, region, side (1 left, 2 right) and cell type.
      const nIn = brain.nIn, inputIdx = Int32Array.from(brain.inputIdx), uv = Float32Array.from(brain.uv);
      const region = new Uint8Array(nIn), side = new Uint8Array(nIn), cellType = new Uint16Array(nIn), names = new Map();
      for (let k = 0; k < nIn; k++) {
        const i = inputIdx[k], t = brain.cellTypeNames[brain.cellType[i]];
        region[k] = brain.region[i]; side[k] = brain.side[i];
        if (!names.has(t)) names.set(t, names.size);
        cellType[k] = names.get(t);
      }
      postMessage({ type: "reply", id: msg.id, uv, inputIdx, region, side, cellType, cellTypeNames: [...names.keys()] },
                  [uv.buffer, inputIdx.buffer, region.buffer, side.buffer, cellType.buffer]);
    }
  } catch (err) {
    postMessage({ type: "error", id: msg.id, message: err.message || String(err) });
  }
};
