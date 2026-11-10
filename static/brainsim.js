// The fly brain (Shiu et al. 2024 leaky integrate-and-fire model) in plain JavaScript.
// Same equations, parameters and step order as flymnist/brain.py, for one image at a time.
// Event-driven: only the outgoing connections of neurons that spiked are touched.

export function parseBrainFile(buf) {
  const headerLen = new DataView(buf).getUint32(0, true);
  const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, headerLen)));
  const base = 4 + headerLen;
  const T = { int32: Int32Array, int16: Int16Array, uint8: Uint8Array, uint16: Uint16Array, float32: Float32Array };
  const arrays = {};
  for (const s of meta.sections) arrays[s.name] = new T[s.dtype](buf, base + s.offset, s.length);
  return { meta, arrays };
}

// Small fast random generator (xorshift128+ is overkill; mulberry32 is enough for Poisson input).
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

// One time step for every neuron. Kept in its own small, branch-free function: the CPU can't
// predict which neurons are active, and branches made this loop 2.4x slower.
// refr[i] is the number of steps since neuron i last spiked (-1 = it spiked in the previous step).
function stepNeurons(n, v, g, refr, rs, ring, off, cur, counts, sd, mem, vRest, vTh, vReset) {
  let nCur = 0;
  for (let i = 0; i < n; i++) {
    const r = ring[off + i]; ring[off + i] = 0;
    const gi = g[i], vi0 = v[i], rf = refr[i] + 1;
    const gn = gi * sd + r * ((rf >= rs[i]) | 0);        // refractory neurons ignore input
    const vi = vi0 + mem * (gi - (vi0 - vRest));
    v[i] = vi; g[i] = gn; refr[i] = rf;
    if (vi > vTh) {                                      // rare: a spike
      v[i] = vReset; g[i] = 0; refr[i] = -1; counts[i]++;
      if (nCur < cur.length) cur[nCur] = i;
      nCur++;
    }
  }
  return nCur;
}

function scatter(prev, nPrev, ip, post, w, ring, off, wScale) {
  for (let s = 0; s < nPrev; s++) {
    const j = prev[s];
    for (let k = ip[j], e = ip[j + 1]; k < e; k++) ring[off + post[k]] += w[k] * wScale;
  }
}

export class FlyBrainSim {
  constructor(buf) {
    const { meta, arrays: a } = parseBrainFile(buf);
    this.meta = meta;
    this.n = meta.n;
    this.p = meta.params;
    this.dt = meta.dt;
    this.indptr = a.indptr;
    // Targets are stored as differences within each row: turn them back into indices (in place).
    const post = a.post_delta;
    for (let j = 0; j < this.n; j++) {
      for (let k = a.indptr[j] + 1; k < a.indptr[j + 1]; k++) post[k] += post[k - 1];
    }
    this.post = post;
    this.wIntact = a.weight;          // signed synapse counts; multiplied by wScale when used
    this.w = a.weight;
    this.inputIdx = a.input_idx;
    this.uv = a.uv;
    this.side = a.side;
    this.region = a.region;
    this.cellType = a.cell_type;
    this.cellTypeNames = meta.cell_type_names;
    this.nIn = a.input_idx.length;
    this.refracSteps = new Uint8Array(this.n).fill(Math.round(this.p.tRefrac / this.dt));
    for (let k = 0; k < this.nIn; k++) this.refracSteps[a.input_idx[k]] = 0;  // stimulated neurons
    this.delaySteps = Math.trunc(this.p.tDelay / this.dt);                     // int() in Python
    this.inputAlive = new Uint8Array(this.nIn).fill(1);
    this.lesioned = { silenced: 0, removed: 0 };
  }

  // Knock out neurons (all their connections in and out, and their eye input) and/or cut
  // every connection between the left and right hemispheres. Pass nothing to restore.
  setLesion(silencedIdx = null, cutHemispheres = false) {
    const n = this.n, S = this.post.length;
    const dead = new Uint8Array(n);
    if (silencedIdx) for (const i of silencedIdx) dead[i] = 1;
    const any = (silencedIdx && silencedIdx.length) || cutHemispheres;
    let removed = 0;
    if (!any) { this.w = this.wIntact; }
    else {
      const w = new Int16Array(this.wIntact), side = this.side, post = this.post, ip = this.indptr;
      for (let j = 0; j < n; j++) {
        const dj = dead[j], sj = side[j];
        for (let k = ip[j]; k < ip[j + 1]; k++) {
          const t = post[k];
          if (dj || dead[t] || (cutHemispheres && sj && side[t] && sj !== side[t])) { if (w[k]) removed++; w[k] = 0; }
        }
      }
      this.w = w;
    }
    for (let k = 0; k < this.nIn; k++) this.inputAlive[k] = dead[this.inputIdx[k]] ? 0 : 1;
    this.lesioned = { silenced: silencedIdx ? silencedIdx.length : 0, removed };
    return removed;
  }

  // Image (H x W, values in [0,1]) -> firing rate of every input neuron (Hz).
  // Bilinear sampling with corners aligned, like torch grid_sample(align_corners=True).
  rates(img, H, W, maxRateHz) {
    const out = new Float32Array(this.nIn), uv = this.uv;
    for (let k = 0; k < this.nIn; k++) {
      const x = uv[2 * k] * (W - 1), y = uv[2 * k + 1] * (H - 1);
      const x0 = Math.min(Math.floor(x), W - 1), y0 = Math.min(Math.floor(y), H - 1);
      const x1 = Math.min(x0 + 1, W - 1), y1 = Math.min(y0 + 1, H - 1);
      const fx = x - x0, fy = y - y0;
      const v = img[y0 * W + x0] * (1 - fx) * (1 - fy) + img[y0 * W + x1] * fx * (1 - fy)
              + img[y1 * W + x0] * (1 - fx) * fy + img[y1 * W + x1] * fx * fy;
      out[k] = v * maxRateHz;
    }
    return out;
  }

  // Simulate `durationMs` with Poisson input at `ratesHz` (one rate per input neuron).
  // Returns spike counts per neuron (Uint16Array) and, with record, every spike as (step, neuron).
  run(ratesHz, durationMs, { record = false, seed = (Math.random() * 2 ** 32) >>> 0, onProgress = null } = {}) {
    const n = this.n, p = this.p, dt = this.dt;
    const steps = Math.round(durationMs / dt);
    const L = this.delaySteps + 1;
    const v = new Float32Array(n).fill(p.vRest), g = new Float32Array(n);
    const refr = new Int32Array(n), rs = this.refracSteps;
    for (let i = 0; i < n; i++) refr[i] = rs[i];
    const counts = new Uint16Array(n);
    const ring = new Float32Array(L * n);
    let prev = new Int32Array(1 << 14), nPrev = 0, cur = new Int32Array(1 << 14), nCur = 0;
    const kick = p.wScale * p.scalePoisson, sd = 1 - dt / p.tauSyn, mem = dt / p.tauMem;
    const vRest = p.vRest, vTh = p.vThreshold, vReset = p.vReset, wScale = p.wScale;
    const post = this.post, w = this.w, ip = this.indptr, inIdx = this.inputIdx;

    // Poisson input: draw each input neuron's spike steps directly (geometric gaps), then
    // bucket them by step. Same distribution as a coin flip per neuron per step.
    const rnd = mulberry32(seed);
    let evN = new Int32Array(1 << 16), evS = new Int32Array(1 << 16), nEv = 0;
    for (let k = 0; k < this.nIn; k++) {
      const q = ratesHz[k] * (dt / 1000) * this.inputAlive[k];
      if (q <= 0) continue;
      const lq = Math.log1p(-Math.min(q, 0.999999));
      let t = Math.floor(Math.log(1 - rnd()) / lq);
      while (t < steps) {
        if (nEv === evN.length) { const a = new Int32Array(nEv * 2); a.set(evN); evN = a; const b = new Int32Array(nEv * 2); b.set(evS); evS = b; }
        evN[nEv] = inIdx[k]; evS[nEv++] = t;
        t += 1 + Math.floor(Math.log(1 - rnd()) / lq);
      }
    }
    const evStart = new Int32Array(steps + 1);
    for (let e = 0; e < nEv; e++) evStart[evS[e] + 1]++;
    for (let s = 0; s < steps; s++) evStart[s + 1] += evStart[s];
    const fill = evStart.slice(0, steps), evByStep = new Int32Array(nEv);
    for (let e = 0; e < nEv; e++) evByStep[fill[evS[e]]++] = evN[e];

    let recS = record ? new Int32Array(1 << 16) : null, recN = record ? new Int32Array(1 << 16) : null, nRec = 0;
    let ptr = 0;
    for (let step = 0; step < steps; step++) {
      for (let e = evStart[step]; e < evStart[step + 1]; e++) v[evByStep[e]] += kick;
      const off = ptr * n;
      nCur = stepNeurons(n, v, g, refr, rs, ring, off, cur, counts, sd, mem, vRest, vTh, vReset);
      if (nCur > cur.length) {  // rare: more spikes than the buffer holds; collect them again
        cur = new Int32Array(nCur * 2);
        let c = 0;
        for (let i = 0; i < n; i++) if (refr[i] === -1) cur[c++] = i;
      }
      // spikes of the previous step arrive in this slot, read again delaySteps + 1 steps later
      scatter(prev, nPrev, ip, post, w, ring, off, wScale);
      if (record) {
        if (nRec + nCur > recS.length) {
          const size = Math.max(recS.length * 2, nRec + nCur);
          const a = new Int32Array(size); a.set(recS); recS = a; const b = new Int32Array(size); b.set(recN); recN = b;
        }
        for (let s = 0; s < nCur; s++) { recS[nRec] = step; recN[nRec++] = cur[s]; }
      }
      const tmp = prev; prev = cur; cur = tmp; nPrev = nCur;
      ptr = (ptr + 1) % L;
      if (onProgress && (step + 1) % 50 === 0) onProgress((step + 1) / steps);
    }
    return { counts, events: record ? { steps: recS.slice(0, nRec), neurons: recN.slice(0, nRec) } : null };
  }
}
