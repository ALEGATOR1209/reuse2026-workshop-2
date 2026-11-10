// Teaching widgets for the "neurons" part of the lecture deck (audience: high school students,
// so on-screen text is kept to labels; the lecturer explains out loud).
//
//   neuronDemo(el)   "A neuron is an electrical gate": one leaky integrate-and-fire neuron
//   circuitDemo(el)  "Neurons wire up into circuits": a one-detector looming-escape circuit
//   synapseDemo(el)  "How the brain model works": A → B (excitatory), C → B (inhibitory)
//
// Each takes an empty container and returns { start(), stop() }. Nothing animates (no
// requestAnimationFrame, no timers) while stopped. Plain browser APIs only.
//
// Every neuron here is simulated with exactly the rules of the whole-brain model
// (flymnist/brain.py, flymnist/static/brainsim.js), with the same step order:
//   per 0.1 ms step: g ← g·(1 − dt/τsyn) + arriving input (ignored while refractory)
//                    v ← v + dt/τmem·(g − (v − vRest))
//                    v > −45 mV → spike: v ← −52 mV, g ← 0, 2.2 ms refractory
//   a spike reaches its targets after 1.8 ms and adds its weight (mV) to g.
// On screen a connection's strength is shown as its "push": how many mV one spike raises the
// target's voltage at most (PSP peak). In the model the weight added to g is push / PSP.

// ---------------------------------------------------------------------------------------------
// Model parameters (Shiu et al. 2024, as in brain.py PARAMS)
// ---------------------------------------------------------------------------------------------
const P = Object.freeze({
  dt: 0.1,          // ms, time step
  tauSyn: 5,        // ms, synaptic current decay
  tDelay: 1.8,      // ms, spike travel time to the next neuron
  vRest: -52,       // mV, where the membrane leaks back to
  vReset: -52,      // mV, after a spike
  vTh: -45,         // mV, spike threshold
  tauMem: 20,       // ms, membrane leak time constant
  tRefrac: 2.2,     // ms, after a spike the neuron ignores its input
  wScale: 0.275,    // mV per synapse
  scalePoisson: 250 // an input ("eye") spike counts as 250 synapses
});
const KICK = P.scalePoisson * P.wScale;  // 68.75 mV: one input spike, always enough to fire

const COL = {
  bg: '#0b0d11', panel: '#13161c', panel2: '#1a1e26', line: '#262b35',
  text: '#e6e8ec', muted: '#8b93a1', accent: '#f5c542', exc: '#5be3a0', inh: '#ff6b6b',
};
const FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif';

// ---------------------------------------------------------------------------------------------
// Small shared helpers
// ---------------------------------------------------------------------------------------------
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

// Number of Poisson events with mean lam (Knuth for small means, normal approximation for big).
function poisson(lam, rnd) {
  if (lam <= 0) return 0;
  if (lam > 30) {
    const u = rnd() || 1e-12, v = rnd();
    return Math.max(0, Math.round(lam + Math.sqrt(lam) * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)));
  }
  const L = Math.exp(-lam);
  let k = 0, p = 1;
  do { k++; p *= rnd(); } while (p > L);
  return k - 1;
}

// A tiny network of leaky integrate-and-fire neurons with the brain model's rules and step order.
class LIFNet {
  constructor(n, seed = 1) {
    this.n = n;
    this.rnd = mulberry32(seed);
    this.L = Math.trunc(P.tDelay / P.dt) + 1;      // delay line slots (as in brainsim.js)
    this.rs = Math.round(P.tRefrac / P.dt);         // refractory length in steps (22)
    this.v = new Float64Array(n).fill(P.vRest);     // membrane voltage, mV
    this.g = new Float64Array(n);                   // synaptic current, mV
    this.refr = new Int32Array(n).fill(this.rs);    // steps since the last spike
    this.ring = new Float64Array(this.L * n);       // spikes in flight (the 1.8 ms delay)
    this.ptr = 0;
    this.out = Array.from({ length: n }, () => []); // outgoing connections [{post, w}]
    this.prev = [];
    this.kickHz = new Float64Array(n);  // Poisson input spikes straight onto v (like the eyes' input)
    this.synHz = new Float64Array(n);   // Poisson synaptic input into g ...
    this.synMv = new Float64Array(n);   // ... with this weight per input spike
    this.addV = new Float64Array(n);    // one-off kicks onto v (next step)
    this.addG = new Float64Array(n);    // one-off synaptic input into g (next step)
    this.t = 0;                         // ms
  }
  connect(pre, post, synapses) { this.out[pre].push({ post, w: synapses * P.wScale }); }
  connectMv(pre, post, w) { this.out[pre].push({ post, w }); }   // weight straight in mV of g
  disconnectAll() { for (const o of this.out) o.length = 0; }
  // Advance one 0.1 ms step; returns the indices of neurons that spiked.
  step() {
    const n = this.n, off = this.ptr * n, rnd = this.rnd;
    const sd = 1 - P.dt / P.tauSyn, mem = P.dt / P.tauMem, pk = P.dt / 1000;
    for (let i = 0; i < n; i++) {
      if (this.kickHz[i] > 0 && rnd() < this.kickHz[i] * pk) this.v[i] += KICK;
      if (this.addV[i]) { this.v[i] += this.addV[i]; this.addV[i] = 0; }
      if (this.synHz[i] > 0) this.ring[off + i] += this.synMv[i] * poisson(this.synHz[i] * pk, rnd);
      if (this.addG[i]) { this.ring[off + i] += this.addG[i]; this.addG[i] = 0; }
    }
    const spk = [];
    for (let i = 0; i < n; i++) {
      const input = this.ring[off + i]; this.ring[off + i] = 0;
      const rf = this.refr[i] + 1;
      const g0 = this.g[i], v0 = this.v[i];
      this.g[i] = g0 * sd + (rf >= this.rs ? input : 0);  // refractory neurons ignore input
      this.v[i] = v0 + mem * (g0 - (v0 - P.vRest));       // leak towards rest, pushed by g
      this.refr[i] = rf;
      if (this.v[i] > P.vTh) {                            // all-or-none spike
        this.v[i] = P.vReset; this.g[i] = 0; this.refr[i] = -1; spk.push(i);
      }
    }
    // Spikes of the previous step go into the delay line; they are read L steps from now.
    for (const j of this.prev) for (const c of this.out[j]) this.ring[off + c.post] += c.w;
    this.prev = spk;
    this.ptr = (this.ptr + 1) % this.L;
    this.t += P.dt;
    return spk;
  }
}

// PSP: the peak voltage bump (mV above rest) that 1 mV added to g produces, measured with the
// model's own step rule (≈0.158). A "push" of X mV per spike is a weight of X / PSP in g.
const PSP = (() => {
  const n = new LIFNet(1, 1);
  n.addG[0] = 1;
  let m = P.vRest;
  for (let k = 0; k < 1500; k++) { n.step(); m = Math.max(m, n.v[0]); }
  return m - P.vRest;
})();

function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const k of kids) if (k != null) e.append(k);
  return e;
}

const fmtMv = (x) => (x < 0 ? '−' : '+') + (Math.abs(x) < 10 && Math.abs(x) % 1 ? Math.abs(x).toFixed(1) : String(Math.abs(x)));

const CSS = `
/* Text inherits the deck's base size (the slide sets it with clamp()); everything is in em,
   so at 1920x1080 (deck ~30 px) the widget text is ~21 px, small labels >= 18 px. */
.nd-wrap{width:100%;height:100%;container-type:inline-size;overflow:auto;box-sizing:border-box;-webkit-tap-highlight-color:transparent}
.nd-root{--nd-accent:var(--accent,${COL.accent});box-sizing:border-box;height:100%;min-height:20em;display:grid;
  grid-template-columns:minmax(0,1fr);grid-template-rows:minmax(0,1fr) auto;gap:.6em;font-family:${FONT};
  font-size:max(13px,.72em);line-height:1.3;color:var(--text,${COL.text});text-align:left}
.nd-root *{box-sizing:border-box}
.nd-main{position:relative;min-width:0;min-height:0;background:var(--panel,${COL.panel});
  border:1px solid var(--line,${COL.line});border-radius:.6em;overflow:hidden}
.nd-cv{position:relative;width:100%;height:100%;min-height:0;min-width:0}
.nd-cv>canvas{position:absolute;inset:0;width:100%;height:100%;display:block;touch-action:manipulation}
.nd-bar{min-width:0;background:var(--panel,${COL.panel});border:1px solid var(--line,${COL.line});border-radius:.6em;
  padding:.55em .9em;display:flex;flex-wrap:wrap;align-items:center;gap:.6em 2em}
.nd-ctl{flex:1 1 13em;min-width:min(100%,12em);display:flex;flex-direction:column;gap:.1em}
.nd-head{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:baseline;gap:0 .7em;font-weight:650}
.nd-v{font-weight:750;font-size:1.1em;color:var(--nd-accent);font-variant-numeric:tabular-nums;white-space:nowrap}
.nd-stack{flex:2 1 22em;min-width:min(100%,16em);display:flex;flex-direction:column;gap:.25em}
.nd-stack>.nd-ctl{flex:none}
.nd-out{flex:0 0 auto;display:flex;flex-direction:column;align-items:flex-start;line-height:1.1;min-width:8em}
.nd-out .nd-k{font-weight:650}
.nd-out .nd-huge{font-size:2.7em;font-weight:800;color:var(--nd-accent);font-variant-numeric:tabular-nums;line-height:1.05;letter-spacing:-.02em}
.nd-out .nd-u{color:var(--muted,${COL.muted})}
.nd-chart{flex:1 1 15em;min-width:min(100%,13em);height:7.6em;position:relative}
.nd-row{display:flex;flex-wrap:wrap;gap:.45em;align-items:center}
.nd-btn{font:inherit;font-size:1.05em;font-weight:650;color:var(--text,${COL.text});background:var(--panel-2,${COL.panel2});
  border:1px solid var(--line,${COL.line});border-radius:.45em;padding:.32em .7em;cursor:pointer;line-height:1.2}
.nd-btn:hover{border-color:var(--muted,${COL.muted})}
.nd-btn:focus-visible,.nd-range:focus-visible,.nd-check input:focus-visible{outline:2px solid var(--nd-accent);outline-offset:2px}
.nd-btn.nd-on{border-color:var(--nd-accent);color:var(--nd-accent);background:rgba(245,197,66,.12)}
.nd-btn.nd-primary{background:var(--nd-accent);color:#16130a;border-color:var(--nd-accent);font-weight:750}
.nd-btn:active{transform:translateY(1px)}
.nd-btn.nd-red{border-color:${COL.inh};color:${COL.inh}}
.nd-range{width:100%;accent-color:var(--nd-accent);margin:.15em 0;height:1.3em}
.nd-range.nd-exc-r{accent-color:${COL.exc}} .nd-range.nd-inh-r{accent-color:${COL.inh}}
.nd-check{display:flex;gap:.45em;align-items:center;cursor:pointer;font-weight:650;line-height:1.2}
.nd-check input{width:1.1em;height:1.1em;margin:0;accent-color:${COL.inh};flex:0 0 auto}
.nd-glabel{font-weight:650;margin-right:.2em}
.nd-exc{color:${COL.exc}} .nd-inh{color:${COL.inh}}
@container (max-width: 720px){
  .nd-root{grid-template-rows:auto auto;height:auto;min-height:0}
  .nd-main{height:max(15em,66cqw)}
  .nd-main.nd-tall{height:max(24em,125cqw)}
  .nd-main.nd-mid{height:max(21em,100cqw)}
  .nd-main.nd-port{height:max(27em,150cqw)}
  .nd-bar{flex-direction:column;align-items:stretch;gap:.7em}
  .nd-ctl,.nd-out,.nd-stack{flex:0 0 auto}
  .nd-chart{flex:0 0 auto;height:8.5em}
}`;

function injectCSS() {
  let s = document.getElementById('nd-style');
  if (!s) { s = document.createElement('style'); s.id = 'nd-style'; document.head.append(s); }
  if (s.textContent !== CSS) s.textContent = CSS;   // also refreshes after a module reload
}

// The standard layout: a big canvas area on top, a bar of controls below it.
function mountLayout(el) {
  injectCSS();
  el.replaceChildren();
  const wrap = h('div', { class: 'nd-wrap' });
  const root = h('div', { class: 'nd-root' });
  const main = h('div', { class: 'nd-main' });
  const bar = h('div', { class: 'nd-bar' });
  root.append(main, bar);
  wrap.append(root);
  el.append(wrap);
  return { wrap, root, main, bar };
}

// A slider with its name and current value above it.
function control(name, input, cls = '') {
  const val = h('span', { class: 'nd-v' });
  const box = h('label', { class: 'nd-ctl ' + cls }, h('span', { class: 'nd-head' }, h('span', {}, name), val), input);
  return { el: box, val };
}

// A devicePixelRatio-sharp canvas that fills its parent. Also sharp when the deck scales the
// slide with a CSS transform (the extra scale is folded into the pixel ratio).
function makeCanvas(parent, cls = '') {
  const wrap = h('div', { class: 'nd-cv ' + cls });
  const cv = document.createElement('canvas');
  wrap.append(cv);
  parent.append(wrap);
  const o = {
    wrap, cv, ctx: cv.getContext('2d'), w: 0, h: 0, ratio: 1,
    fit() {
      const w = wrap.clientWidth, hh = wrap.clientHeight;
      const r = wrap.getBoundingClientRect();
      const css = w > 0 ? r.width / w : 1;
      const ratio = Math.min(3, Math.max(1, (window.devicePixelRatio || 1) * (css || 1)));
      if (w !== o.w || hh !== o.h || ratio !== o.ratio) {
        cv.width = Math.max(1, Math.round(w * ratio));
        cv.height = Math.max(1, Math.round(hh * ratio));
        o.w = w; o.h = hh; o.ratio = ratio;
      }
      o.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      return o.w > 2 && o.h > 2;
    },
    // pointer position in canvas (CSS layout) pixels
    pos(e) {
      const r = cv.getBoundingClientRect();
      return [(e.clientX - r.left) * (o.w / (r.width || 1)), (e.clientY - r.top) * (o.h / (r.height || 1))];
    },
  };
  return o;
}

// requestAnimationFrame loop that only runs between start() and stop().
function animLoop(frame) {
  let raf = 0, last = 0, running = false;
  const tick = (now) => {
    if (!running) return;
    const dtw = last ? Math.min(0.05, (now - last) / 1000) : 0;  // wall seconds, clamped
    last = now;
    frame(dtw, now / 1000);
    if (running) raf = requestAnimationFrame(tick);
  };
  return {
    start() { if (running) return; running = true; last = 0; raf = requestAnimationFrame(tick); },
    stop() { running = false; if (raf) cancelAnimationFrame(raf); raf = 0; },
    get running() { return running; },
  };
}

// Segmented buttons. items: [[label, value]]; returns {el, set(value)}.
function segmented(items, value, onChange, ariaLabel) {
  const box = h('div', { class: 'nd-row', role: 'group', 'aria-label': ariaLabel || '' });
  const btns = items.map(([label, v]) => {
    const b = h('button', { class: 'nd-btn', type: 'button', onclick: () => { set(v); onChange(v); } }, label);
    b.dataset.v = String(v);
    box.append(b);
    return b;
  });
  function set(v) { for (const b of btns) { const on = b.dataset.v === String(v); b.classList.toggle('nd-on', on); b.setAttribute('aria-pressed', on); } }
  set(value);
  return { el: box, set };
}

function fontPx(root) { return parseFloat(getComputedStyle(root).fontSize) || 16; }
const font = (px, weight = 400) => `${weight} ${Math.round(px * 10) / 10}px ${FONT}`;

function label(ctx, text, x, y, px, color, weight = 400, align = 'center', base = 'middle') {
  ctx.font = font(px, weight); ctx.fillStyle = color; ctx.textAlign = align; ctx.textBaseline = base; ctx.fillText(text, x, y);
}

// Split text into lines no wider than maxW (ctx.font must be set).
function wrapText(ctx, text, maxW) {
  const lines = []; let cur = '';
  for (const w of text.split(' ')) {
    const t = cur ? cur + ' ' + w : w;
    if (cur && ctx.measureText(t).width > maxW) { lines.push(cur); cur = w; } else cur = t;
  }
  if (cur) lines.push(cur);
  return lines;
}
// Same number of lines as wrapText, but with line lengths as even as possible.
function wrapBalanced(ctx, text, maxW) {
  let best = wrapText(ctx, text, maxW);
  if (best.length < 2) return best;
  for (let w = maxW - 4; w > 20; w -= 4) {
    const l = wrapText(ctx, text, w);
    if (l.length > best.length) break;
    best = l;
  }
  return best;
}

// Voltage axis shared by the voltage plots: the sub-threshold range gets most of the height so
// the 7 mV between rest and threshold is easy to see; spikes are drawn into a band above it.
function voltScale(top, bottom, vLo, vHi = -43.6, spikeFrac = 0.27) {
  const yHi = top + (bottom - top) * spikeFrac;
  return {
    y: (v) => bottom - (Math.min(v, vHi) - vLo) / (vHi - vLo) * (bottom - yHi),
    peak: top + 2,
    yHi,
  };
}

// Draw one action potential for illustration: up from threshold to a fixed peak, back to reset.
// The model itself just resets v; every spike is drawn with the same height on purpose.
function drawSpike(ctx, x, ys, width) {
  ctx.beginPath();
  ctx.moveTo(x - width * 0.15, ys.y(P.vTh));
  ctx.lineTo(x + width * 0.25, ys.peak);
  ctx.lineTo(x + width * 0.75, ys.y(P.vReset) + 0.5);
  ctx.stroke();
}

function voltAxis(ctx, x0, x1, ys, fs) {
  // threshold and rest lines with labels on the left
  ctx.save();
  ctx.lineWidth = 1.2;
  ctx.setLineDash([6, 5]);
  ctx.strokeStyle = 'rgba(245,197,66,.7)';
  ctx.beginPath(); ctx.moveTo(x0, ys.y(P.vTh)); ctx.lineTo(x1, ys.y(P.vTh)); ctx.stroke();
  ctx.setLineDash([]);
  ctx.strokeStyle = 'rgba(139,147,161,.55)';
  ctx.beginPath(); ctx.moveTo(x0, ys.y(P.vRest)); ctx.lineTo(x1, ys.y(P.vRest)); ctx.stroke();
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  const lx = x0 - fs * 0.45;
  ctx.fillStyle = COL.accent; ctx.font = font(fs * 0.9, 650);
  ctx.fillText('threshold', lx, ys.y(P.vTh) - fs * 0.42);
  ctx.font = font(fs * 0.86); ctx.fillText('−45 mV', lx, ys.y(P.vTh) + fs * 0.42);
  ctx.fillStyle = COL.text; ctx.font = font(fs * 0.9, 650);
  ctx.fillText('rest', lx, ys.y(P.vRest) - fs * 0.42);
  ctx.fillStyle = COL.muted; ctx.font = font(fs * 0.86); ctx.fillText('−52 mV', lx, ys.y(P.vRest) + fs * 0.42);
  ctx.fillStyle = COL.accent; ctx.font = font(fs * 0.9, 650);
  ctx.fillText('spike', lx, ys.peak + fs * 0.45);
  ctx.restore();
}

// Horizontal time scale bar, label to its left.
function scaleBar(ctx, xRight, y, pxLen, text, fs) {
  ctx.save();
  ctx.strokeStyle = COL.text; ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(xRight - pxLen, y - 4); ctx.lineTo(xRight - pxLen, y); ctx.lineTo(xRight, y); ctx.lineTo(xRight, y - 4);
  ctx.stroke();
  ctx.fillStyle = COL.text; ctx.font = font(fs * 0.86, 600); ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  ctx.fillText(text, xRight - pxLen - fs * 0.5, y - 2);
  ctx.restore();
}

// =============================================================================================
// 1. neuronDemo — "A neuron is an electrical gate"
// =============================================================================================
// One LIF neuron gets random (Poisson) incoming spikes. Two sliders: how OFTEN they arrive and how
// BIG each one is (its push: the voltage bump it causes). Rare or weak input never reaches the
// threshold: no spikes. More frequent or stronger input: more spikes, never bigger ones. The
// 2.2 ms refractory period caps the rate. The small chart: firing rate vs incoming spikes per
// second for the chosen strength, measured with the same neuron. A third slider sets the firing
// rate directly: the inputs move to values that produce it (that curve, inverted), and moving the
// inputs moves it. Default slow motion 50× (also 20× and 100×); rates are always in brain time.

export function neuronDemo(el) {
  const { root, main, bar } = mountLayout(el);
  main.classList.add('nd-mid');
  const plot = makeCanvas(main);

  const PUSHES = [0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 4, 5, 7, 10];
  const CEIL = 1000 / ((Math.round(P.tRefrac / P.dt) + 2) * P.dt); // ≈417 Hz: refractory + 2 steps
  const MAX_TARGET = 300;                   // highest firing rate one can ask for (reachable: ≈330)
  // incoming spikes per second: slider 0..1000, log scale 10..10,000 (0 = none), 2 significant digits
  const nice = (x) => { const e = 10 ** (Math.floor(Math.log10(x)) - 1); return Math.round(x / e) * e; };
  const rateOfSlider = (v) => (v <= 0 ? 0 : nice(10 ** (1 + 3 * v / 1000)));
  const sliderOfRate = (r) => (r <= 0 ? 0 : Math.max(1, Math.min(1000, Math.round(1000 * (Math.log10(r) - 1) / 3))));
  let inHz = 50, si = 4;                    // start: 50 per second, +1 mV each → no spikes
  let simMsPerSec = 20;                     // slow motion: 1 s on screen = 20 ms of neuron time (50×)
  const WINDOW_S = 4;                       // the trace shows the last 4 s of screen time
  const windowMs = () => WINDOW_S * simMsPerSec;
  const KEEP_MS = 600;                      // spike history kept (the rate uses the last 300 ms)
  const RATE_WIN_MS = 300;

  // --- controls ---
  const sRate = h('input', { class: 'nd-range nd-exc-r', type: 'range', min: 0, max: 1000, step: 1, value: sliderOfRate(inHz), 'aria-label': 'Incoming spikes per second' });
  const sPush = h('input', { class: 'nd-range nd-exc-r', type: 'range', min: 0, max: PUSHES.length - 1, step: 1, value: si, 'aria-label': 'Strength of each incoming spike' });
  const sOut = h('input', { class: 'nd-range', type: 'range', min: 0, max: MAX_TARGET, step: 1, value: 0, 'aria-label': 'Firing rate, spikes per second' });
  const cRate = control('Incoming spikes per second', sRate);
  const cPush = control('Strength of each incoming spike', sPush);
  const cOut = control(h('span', { style: `color:${COL.accent}` }, 'Firing rate (spikes per second)'), sOut);
  const rateNum = h('span', { class: 'nd-huge', 'aria-live': 'polite' }, '0');
  const out = h('div', { class: 'nd-out' }, h('span', { class: 'nd-k' }, 'Measured'), rateNum, h('span', { class: 'nd-u' }, 'spikes per second'));
  const chartBox = h('div', { class: 'nd-chart' });
  const speed = segmented([['20×', 50], ['50×', 20], ['100×', 10]], simMsPerSec, (v) => { simMsPerSec = v; draw(); }, 'Slow motion');
  const speedBox = h('div', { class: 'nd-out' }, h('span', { class: 'nd-k' }, 'Slower than real'), speed.el);
  bar.append(h('div', { class: 'nd-stack' }, cRate.el, cPush.el, cOut.el), out, chartBox, speedBox);
  const fi = makeCanvas(chartBox);

  // --- simulation state ---
  const net = new LIFNet(1, 20240601);
  const CAP = 1 << 14;                      // history: last 1638 ms of voltage, one value per step
  const hist = new Float32Array(CAP).fill(P.vRest);
  let stepN = 0, acc = 0, changeStep = 0, spikeTotal = 0;
  let spikes = [];                          // step indices of output spikes
  let inputs = [];                          // [step, push] of incoming spikes

  const rateHz = () => inHz, push = () => PUSHES[si];

  // The neuron's rate curve (firing rate vs incoming spikes per second) for one strength: the
  // same neuron simulated for 4 s of brain time at 31 input rates (10 .. 10,000 per second, log
  // spaced), made non-decreasing. Computed once per strength, when first needed (~50 ms).
  const GRID = Array.from({ length: 31 }, (_, i) => 10 ** (1 + i / 10));
  const curves = new Map();
  function curve(s) {
    if (curves.has(s)) return curves.get(s);
    const w = PUSHES[s] / PSP, T = 4000, c = [];
    GRID.forEach((r, i) => {
      const m = new LIFNet(1, 777 + 100 * s + i), lam = r * P.dt / 1000;
      let n = 0;
      for (let k = 0; k < T / P.dt; k++) { const q = poisson(lam, m.rnd); if (q) m.addG[0] += q * w; n += m.step().length; }
      c.push(Math.max(n / (T / 1000), i ? c[i - 1] : 0));
    });
    curves.set(s, c);
    return c;
  }
  // expected firing rate for an input rate and strength (interpolated in log input rate)
  function expectedRate(r, s) {
    if (r < GRID[0]) return 0;
    const c = curve(s), x = (Math.log10(r) - 1) * 10, i = Math.min(GRID.length - 2, Math.floor(x)), f = x - i;
    return c[i] + (c[i + 1] - c[i]) * f;
  }
  // inverse: input rate that gives firing rate `hz` at strength s (null if out of reach)
  function inputFor(hz, s) {
    const c = curve(s);
    if (hz > c[c.length - 1] || hz < c[0]) return null;
    let i = 1; while (c[i] < hz) i++;
    const f = c[i] > c[i - 1] ? (hz - c[i - 1]) / (c[i] - c[i - 1]) : 0;
    return 10 ** (1 + (i - 1 + f) / 10);
  }

  // Inputs → the firing-rate slider follows the expected rate.
  function setInput() {
    inHz = rateOfSlider(+sRate.value); si = +sPush.value;
    sOut.value = String(Math.round(Math.min(MAX_TARGET, expectedRate(inHz, si))));
    showValues();
  }
  // Firing-rate slider → keep the strength and adjust the incoming rate; if that cannot reach the
  // target, change the strength (stronger for high targets, weaker for very low ones).
  function setTarget() {
    const hz = +sOut.value;
    if (hz <= 0) inHz = 0;
    else {
      // binary search over the strengths (each new curve costs a moment to compute)
      const top = (s) => curve(s)[GRID.length - 1], bottom = (s) => curve(s)[0];
      let s = si;
      if (hz > top(s)) {            // the smallest stronger push that can reach it
        let lo = s + 1, hi = PUSHES.length - 1;
        while (lo < hi) { const m = (lo + hi) >> 1; if (top(m) >= hz) hi = m; else lo = m + 1; }
        s = Math.min(lo, PUSHES.length - 1);
      } else if (hz < bottom(s)) {  // the largest weaker push that can go this low
        let lo = 0, hi = s - 1;
        while (lo < hi) { const m = (lo + hi + 1) >> 1; if (bottom(m) <= hz) lo = m; else hi = m - 1; }
        s = Math.max(0, hi);
      }
      const r = inputFor(hz, s);
      si = s;
      inHz = r == null ? GRID[GRID.length - 1] : nice(r);
    }
    sRate.value = String(sliderOfRate(inHz)); sPush.value = String(si);
    showValues();
  }
  function showValues() {
    cRate.val.textContent = inHz.toLocaleString('en-US');
    cPush.val.textContent = fmtMv(push()) + ' mV';
    cOut.val.textContent = sOut.value;
    cOut.val.style.color = COL.accent;
    changeStep = stepN;
    rateNum.textContent = String(Math.round(liveRate()));
    if (!loop.running) draw();
  }

  function liveRate() {
    // spikes in the last 300 ms of model time; right after a change the missing part of the
    // window is filled with the expected rate so the number responds at once
    const win = RATE_WIN_MS / P.dt, since = Math.min(win, stepN - changeStep);
    let c = 0;
    for (let i = spikes.length - 1; i >= 0 && spikes[i] > stepN - since; i--) c++;
    const exp = expectedRate(inHz, si);
    return (c + exp * (win - since) * P.dt / 1000) / (win * P.dt / 1000);
  }

  function simulate(n) {
    const lam = rateHz() * P.dt / 1000, w = push() / PSP;
    for (let k = 0; k < n; k++) {
      const m = poisson(lam, net.rnd);
      if (m) { net.addG[0] += m * w; inputs.push([stepN, push()]); }
      const fired = net.step().length > 0;
      hist[stepN % CAP] = fired ? P.vTh : net.v[0];  // a spike step is drawn as reaching threshold
      if (fired) { spikes.push(stepN); spikeTotal++; }
      stepN++;
    }
    const keep = stepN - Math.round(KEEP_MS / P.dt);
    if (spikes.length && spikes[0] < keep) spikes = spikes.filter((s) => s >= keep);
    if (inputs.length && inputs[0][0] < keep) inputs = inputs.filter((s) => s[0] >= keep);
  }

  sRate.addEventListener('input', setInput);
  sPush.addEventListener('input', setInput);
  sOut.addEventListener('input', setTarget);

  // --- drawing ---
  function draw() {
    if (!plot.fit()) return;
    const ctx = plot.ctx, W = plot.w, H = plot.h;
    const fs = Math.min(fontPx(root), W / 30, H / 16);
    ctx.clearRect(0, 0, W, H);
    const x0 = fs * 5.6, x1 = W - fs * 0.9;
    const inTop = fs * 0.7, inBot = inTop + fs * 2.4;            // incoming spikes lane
    const outH = fs * 1.8, scaleH = fs * 1.7;
    const top = inBot + fs * 0.9, bottom = H - outH - scaleH - fs * 0.9;
    const ys = voltScale(top, bottom, -53.2);
    const winSteps = Math.round(windowMs() / P.dt);
    const pxPerStep = (x1 - x0) / winSteps;
    const xOf = (step) => x1 - (stepN - 1 - step) * pxPerStep;
    const first = Math.max(0, stepN - winSteps, stepN - CAP + 1);

    voltAxis(ctx, x0, x1, ys, fs);

    // incoming spikes: one green tick each, taller = stronger
    ctx.fillStyle = COL.panel2; ctx.fillRect(x0, inTop, x1 - x0, inBot - inTop);
    label(ctx, 'incoming', x0 - fs * 0.45, (inTop + inBot) / 2, fs * 0.9, COL.exc, 650, 'right');
    ctx.save(); ctx.beginPath(); ctx.rect(x0, 0, x1 - x0, H); ctx.clip();
    ctx.strokeStyle = COL.exc; ctx.lineWidth = Math.max(1.3, fs * 0.08);
    ctx.beginPath();
    for (const [s, p] of inputs) if (s >= first) {
      const x = xOf(s), hh = (inBot - inTop - 4) * (0.18 + 0.82 * Math.sqrt(p / 10));
      ctx.moveTo(x, inBot - 2); ctx.lineTo(x, inBot - 2 - hh);
    }
    ctx.stroke();

    // voltage trace
    ctx.strokeStyle = 'rgba(230,232,236,.92)'; ctx.lineWidth = Math.max(1.6, fs * 0.1); ctx.lineJoin = 'round';
    ctx.beginPath();
    for (let s = first; s < stepN; s++) {
      const x = xOf(s), y = ys.y(hist[s % CAP]);
      if (s === first) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    // spikes: identical, tall action potentials (illustration)
    const apW = Math.min(fs * 0.8, Math.max(4, 0.9 / P.dt * pxPerStep));
    ctx.strokeStyle = COL.accent; ctx.lineWidth = Math.max(1.8, fs * 0.12);
    for (const s of spikes) if (s >= first) drawSpike(ctx, xOf(s), ys, apW);
    ctx.restore();

    // outgoing spikes: one tick each, all the same
    const ry0 = bottom + fs * 0.8, ry1 = ry0 + outH;
    ctx.fillStyle = COL.panel2; ctx.fillRect(x0, ry0, x1 - x0, ry1 - ry0);
    label(ctx, 'outgoing', x0 - fs * 0.45, (ry0 + ry1) / 2, fs * 0.9, COL.accent, 650, 'right');
    ctx.save(); ctx.beginPath(); ctx.rect(x0, 0, x1 - x0, H); ctx.clip();
    ctx.strokeStyle = COL.accent; ctx.lineWidth = Math.max(1.8, fs * 0.12);
    ctx.beginPath();
    for (const s of spikes) if (s >= first) { const x = xOf(s); ctx.moveTo(x, ry0 + 3); ctx.lineTo(x, ry1 - 3); }
    ctx.stroke();
    ctx.restore();

    // time scale
    const barMs = windowMs() >= 150 ? 20 : windowMs() >= 60 ? 10 : 5;
    scaleBar(ctx, x1, H - fs * 0.85, barMs / P.dt * pxPerStep, `${barMs} ms`, fs);

    drawChart();
  }

  // Firing rate vs incoming spikes per second (log), for the chosen strength.
  function drawChart() {
    if (!fi.fit()) return;
    const ctx = fi.ctx, W = fi.w, H = fi.h;
    const fs = Math.min(fontPx(root), W / 13, H / 6);
    ctx.clearRect(0, 0, W, H);
    const x0 = fs * 2.3, x1 = W - fs * 1.2, y0 = H - fs * 2.45, y1 = fs * 1.1;
    const lx = (r) => x0 + (r > 0 ? Math.log10(r) / 4 : 0) * (x1 - x0);   // 1 .. 10,000
    const ly = (hz) => y0 - hz / CEIL * (y0 - y1);
    ctx.fillStyle = COL.panel2; ctx.fillRect(x0, y1, x1 - x0, y0 - y1);
    // the maximum rate
    ctx.setLineDash([4, 4]); ctx.strokeStyle = 'rgba(255,107,107,.8)'; ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(x0, ly(CEIL)); ctx.lineTo(x1, ly(CEIL)); ctx.stroke(); ctx.setLineDash([]);
    label(ctx, 'max', x1, ly(CEIL) - 3, fs * 0.86, COL.inh, 700, 'right', 'bottom');
    // axes
    for (const hz of [0, 200, 400]) label(ctx, String(hz), x0 - 4, ly(hz), fs * 0.86, COL.muted, 400, 'right');
    for (const r of [10, 100, 1000, 10000]) label(ctx, r >= 1000 ? r / 1000 + 'k' : String(r), lx(r), y0 + 2, fs * 0.86, COL.muted, 400, 'center', 'top');
    label(ctx, 'incoming spikes per second', (x0 + x1) / 2, H - 1, fs * 0.86, COL.muted, 400, 'center', 'bottom');
    label(ctx, 'firing rate', x0, ly(CEIL) - 3, fs * 0.86, COL.text, 600, 'left', 'bottom');
    // measured curve for this strength
    const c = curve(si), pts = GRID.map((r, i) => [r, c[i]]);
    ctx.strokeStyle = 'rgba(245,197,66,.75)'; ctx.lineWidth = Math.max(1.5, fs * 0.1); ctx.lineJoin = 'round';
    ctx.beginPath(); pts.forEach(([r, hz], i) => (i ? ctx.lineTo(lx(r), ly(hz)) : ctx.moveTo(lx(r), ly(hz)))); ctx.stroke();
    // where we are now
    const hz = expectedRate(inHz, si);
    ctx.fillStyle = COL.accent; ctx.strokeStyle = COL.text; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(lx(rateHz()), ly(hz), Math.max(4, fs * 0.36), 0, 7); ctx.fill(); ctx.stroke();
  }

  let lastRateUpdate = 0;
  const loop = animLoop((dtw, now) => {
    acc += dtw * simMsPerSec / P.dt;
    const n = Math.min(2000, Math.floor(acc));
    acc -= n;
    simulate(n);
    if (now - lastRateUpdate > 0.2) {          // refresh the number 5× per second
      lastRateUpdate = now;
      rateNum.textContent = String(Math.round(liveRate()));
    }
    draw();
  });

  const ro = new ResizeObserver(() => draw());
  ro.observe(plot.wrap); ro.observe(fi.wrap);
  setInput();
  simulate(Math.round(KEEP_MS / P.dt));      // pre-fill the first screen and the rate window
  changeStep = 0;
  rateNum.textContent = String(Math.round(liveRate()));
  draw();
  const api = { start: () => loop.start(), stop: () => loop.stop() };
  // exposed for testing: expected firing rate (Hz) for the current sliders, and the live number
  Object.defineProperty(api, '_rate', { enumerable: false, value: () => ({ expected: expectedRate(inHz, si), live: liveRate(), shown: +rateNum.textContent, target: +sOut.value, inHz, push: push(), t: net.t, spk: spikeTotal }) });
  // exposed for testing: firing rate of a fresh neuron with the current inputs over T ms of brain time
  Object.defineProperty(api, '_measure', { enumerable: false, value: (T = 10000, seed = 5) => {
    const m = new LIFNet(1, seed), lam = inHz * P.dt / 1000, w = push() / PSP; let n = 0;
    for (let k = 0; k < T / P.dt; k++) { const q = poisson(lam, m.rnd); if (q) m.addG[0] += q * w; n += m.step().length; }
    return n / (T / 1000); } });
  return api;
}

// =============================================================================================
// 2. circuitDemo — "Neurons wire up into circuits": the looming escape circuit
// =============================================================================================
// Two mirror-image halves of the brain. In each: eye → looming detector → giant fiber → motor
// neuron → that side's legs. Each detector also excites the OTHER side's giant fiber through a
// crossing connection with few synapses (one crossing spike alone is not enough; two in a row
// are). Each eye also drives a brake neuron (inhibitory) that holds its giant fiber back, so a
// slow object, which makes the detector fire only now and then, does not trigger an escape; the
// fast, regular firing for a looming object overcomes the brake.
// Measured with these weights: right eye looming → right giant fiber ≈185 Hz, left ≈95 Hz;
// crossing cut → left 0 Hz; both eyes → both ≈185 Hz; slow object → right 0 Hz (≈40 Hz with the
// brakes silenced). The fly's legs on each side follow that side's motor neuron: both sides
// active → jump; one side only → it stumbles.

export function circuitDemo(el) {
  const { root, main, bar } = mountLayout(el);
  main.classList.add('nd-port');           // narrow screens: a tall canvas for the portrait layout
  const cvs = makeCanvas(main);

  const W = { ipsi: 400, cross: 110, brake: 100, toMotor: 400 };   // synapse counts (×0.275 mV)
  const EYE = { loom: 10000, slow: 600, mv: 4 };   // eye → detector: synaptic input (spikes/s, mV)
  const BRAKE_IN = { hz: 20000, mv: 8 };           // eye → brake neuron, whenever it sees an object
  const NODE = ['detL', 'detR', 'brL', 'brR', 'gfL', 'gfR', 'mnL', 'mnR'];
  const U = Object.fromEntries(NODE.map((k, i) => [k, i]));
  const net = new LIFNet(NODE.length, 4242);

  let stim = 'right', cut = false, noBrake = false;
  const SIM_MS_PER_SEC = 100;              // slow motion: 1 s on screen = 100 ms of brain time
  const sees = (s) => (s === 'L' ? stim === 'left' || stim === 'both' : stim === 'right' || stim === 'both' || stim === 'slow');
  const looming = (s) => sees(s) && stim !== 'slow';

  function wire() {
    net.disconnectAll();
    net.connect(U.detL, U.gfL, W.ipsi); net.connect(U.detR, U.gfR, W.ipsi);
    if (!cut) { net.connect(U.detL, U.gfR, W.cross); net.connect(U.detR, U.gfL, W.cross); }
    if (!noBrake) { net.connect(U.brL, U.gfL, -W.brake); net.connect(U.brR, U.gfR, -W.brake); }
    net.connect(U.gfL, U.mnL, W.toMotor); net.connect(U.gfR, U.mnR, W.toMotor);
  }
  function setStim() {
    for (const s of ['L', 'R']) {
      net.synHz[U['det' + s]] = looming(s) ? EYE.loom : sees(s) ? EYE.slow : 0;
      net.synMv[U['det' + s]] = EYE.mv;
      net.synHz[U['br' + s]] = sees(s) && !noBrake ? BRAKE_IN.hz : 0;
      net.synMv[U['br' + s]] = BRAKE_IN.mv;
    }
  }

  // --- controls ---
  const stimSeg = segmented([['none', 'none'], ['left eye', 'left'], ['right eye', 'right'], ['both eyes', 'both'], ['slow object', 'slow']],
    stim, (v) => setStimMode(v), 'Object coming at the fly');
  const cutBox = h('input', { type: 'checkbox' });
  const brakeBox = h('input', { type: 'checkbox' });
  bar.append(
    h('div', { class: 'nd-row' }, h('span', { class: 'nd-glabel' }, 'Object'), stimSeg.el),
    h('label', { class: 'nd-check' }, cutBox, h('span', {}, 'Cut left ↔ right')),
    h('label', { class: 'nd-check' }, brakeBox, h('span', {}, 'Silence brake neurons')),
  );
  function setStimMode(v) { stim = v; stimSeg.set(v); setStim(); if (!loop.running) draw(); }
  cutBox.addEventListener('change', () => { cut = cutBox.checked; wire(); if (!loop.running) draw(); });
  brakeBox.addEventListener('change', () => { noBrake = brakeBox.checked; wire(); setStim(); if (!loop.running) draw(); });

  // --- state for drawing ---
  const flash = new Float32Array(net.n);   // brightness after a spike (wall-time decay)
  const spikeT = Array.from({ length: net.n }, () => []);   // spike times (ms) for the rates
  const RATE_WIN = 300;                    // ms
  const legs = { L: 0, R: 0 };             // leg muscle activation (filtered motor spikes)
  let dots = [];                           // spikes in flight on the diagram
  let acc = 0, wallNow = 0, clickBoxes = [];
  let act = { t0: -10, kind: '', side: '' };   // the fly's reaction: 'jump' or 'stumble'
  const DOT_T = 0.5;                       // s on screen for a dot to cross an edge
  const ACT_HZ = 22;                       // a giant fiber above this makes the fly react

  const EDGES = [
    { a: 'detL', b: 'gfL', type: 'exc' }, { a: 'detR', b: 'gfR', type: 'exc' },
    { a: 'detL', b: 'gfR', type: 'exc', cross: true }, { a: 'detR', b: 'gfL', type: 'exc', cross: true },
    { a: 'brL', b: 'gfL', type: 'inh' }, { a: 'brR', b: 'gfR', type: 'inh' },
    { a: 'gfL', b: 'mnL', type: 'exc' }, { a: 'gfR', b: 'mnR', type: 'exc' },
    { a: 'mnL', b: 'legsL', type: 'motor' }, { a: 'mnR', b: 'legsR', type: 'motor' },
  ];
  const edgeOn = (e) => !(e.cross && cut) && !(e.type === 'inh' && noBrake);

  function rate(u) {
    const t0 = net.t - RATE_WIN, arr = spikeT[u];
    let i = 0; while (i < arr.length && arr[i] < t0) i++;
    if (i) arr.splice(0, i);
    return arr.length / (Math.min(RATE_WIN, net.t) / 1000 || 1);
  }

  function simulate(nSteps, visual) {
    const decay = Math.exp(-P.dt / 20);
    for (let k = 0; k < nSteps; k++) {
      const spk = net.step();
      legs.L *= decay; legs.R *= decay;
      for (const u of spk) {
        spikeT[u].push(net.t);
        if (u === U.mnL) legs.L += 0.3;
        if (u === U.mnR) legs.R += 0.3;
        if (!visual) continue;
        flash[u] = 1;
        for (const e of EDGES) if (e.a === NODE[u] && edgeOn(e)) dots.push({ e, t0: wallNow });
      }
    }
  }

  // --- layout (in canvas pixels) ---
  // The fly faces the incoming object (towards the eyes). Seen from above and facing left, its
  // right side is at the top: so on wide screens the right half of the brain is the top row.
  // On narrow screens the fly faces up: left half on the left, right half on the right.
  function layout(Wd, Hd) {
    const port = Wd < 640 && Hd > Wd * 0.9;
    const padX = Wd * 0.02, padY = Hd * 0.03;
    const X = (f) => padX + f * (Wd - 2 * padX), Y = (f) => padY + f * (Hd - 2 * padY);
    const p = {}, place = {};
    let r, flyS, mid;
    if (port) {
      r = Math.max(9, Math.min(Wd * 0.045, Hd * 0.03));
      flyS = Math.min(Wd * 0.13, Hd * 0.075);
      mid = X(0.5);
      for (const [s, x, bx, out] of [['L', X(0.33), X(0.1), 'left'], ['R', X(0.67), X(0.9), 'right']]) {
        p['scr' + s] = [x, Y(0.035)]; p['eye' + s] = [x, Y(0.135)]; p['det' + s] = [x, Y(0.265)];
        p['br' + s] = [bx, Y(0.37)]; p['gf' + s] = [x, Y(0.52)]; p['mn' + s] = [x, Y(0.675)];
        Object.assign(place, { ['eye' + s]: out, ['det' + s]: out, ['gf' + s]: out, ['mn' + s]: out, ['br' + s]: 'below' });
      }
      p.fly = [X(0.5), Y(0.9)];
    } else {
      r = Math.max(10, Math.min(Wd * 0.024, Hd * 0.052));
      flyS = Math.min(Wd * 0.065, Hd * 0.15);
      const yR = Y(0.3), yL = Y(0.7);
      mid = (yR + yL) / 2;
      for (const [s, y, by, outV, inV, ne] of [['R', yR, Y(0.1), 'above', 'below', 'ne'], ['L', yL, Y(0.9), 'below', 'above', 'se']]) {
        p['scr' + s] = [X(0.035), y]; p['eye' + s] = [X(0.115), y]; p['det' + s] = [X(0.25), y];
        p['br' + s] = [X(0.385), by]; p['gf' + s] = [X(0.53), y]; p['mn' + s] = [X(0.69), y];
        Object.assign(place, { ['eye' + s]: outV, ['det' + s]: inV, ['br' + s]: ne, ['gf' + s]: ne, ['mn' + s]: ne });
      }
      p.fly = [X(0.895), mid];
    }
    // where each motor neuron's connection reaches the legs (the fly's right legs are on its right)
    const [fx, fy] = p.fly;
    if (port) { p.legsL = [fx - flyS * 0.75, fy - flyS * 0.05]; p.legsR = [fx + flyS * 0.75, fy - flyS * 0.05]; }
    else { p.legsR = [fx + flyS * 0.05, fy - flyS * 0.75]; p.legsL = [fx + flyS * 0.05, fy + flyS * 0.75]; }
    const rad = { legsL: 0, legsR: 0 };
    for (const s of ['L', 'R']) Object.assign(rad, { ['det' + s]: r * 1.05, ['br' + s]: r * 0.9, ['gf' + s]: r * 1.2, ['mn' + s]: r * 0.95 });
    // how wide a description may be next to (or under) a node
    const textW = (key) => {
      const [x] = p[key], pl = place[key], gap = (rad[key] || r) + r * 0.5;
      if (pl === 'left') return x - gap - padX;
      if (pl === 'right') return Wd - padX - (x + gap);
      if (pl === 'ne' || pl === 'se') return Math.min(Wd - padX - (x + gap), X(key.startsWith('br') ? 0.25 : 0.145) - X(0));
      return X(0.19) - X(0);
    };
    return { port, p, r, rad, place, textW, padX, padY, mid, flyS, X, Y };
  }

  // Edge geometry: start and end points, trimmed to the node outlines.
  function edgeEnds(L, e) {
    const [ax, ay] = L.p[e.a], [bx, by] = L.p[e.b];
    const dx = bx - ax, dy = by - ay, len = Math.hypot(dx, dy) || 1, ux = dx / len, uy = dy / len;
    const s = L.rad[e.a] + 3, t = L.rad[e.b] + (e.type === 'motor' ? 0 : 7);
    return [ax + ux * s, ay + uy * s, bx - ux * t, by - uy * t, ux, uy];
  }

  // Lines of text ({t, px, c, w}) as one block next to a node.
  function textBlock(ctx, x, y, rr, lines, place, fs, boxed = false) {
    const lh = (l) => l.px * 1.2;
    const total = lines.reduce((a, l) => a + lh(l), 0);
    const gap = rr + fs * 0.45;
    let yy, lx = x, align = 'center';
    if (place === 'above') yy = y - gap - total;
    else if (place === 'below') yy = y + gap;
    else if (place === 'ne' || place === 'se') {   // beside the node, above / below its outgoing line
      lx = x + rr + fs * 0.35; align = 'left';
      yy = place === 'ne' ? y - fs * 0.45 - total : y + fs * 0.45;
    } else { yy = y - total / 2; lx = place === 'right' ? x + gap : x - gap; align = place === 'right' ? 'left' : 'right'; }
    if (boxed) {                     // a dark backing, so lines passing underneath do not cut the text
      let w = 0;
      for (const l of lines) { ctx.font = font(l.px, l.w); w = Math.max(w, ctx.measureText(l.t).width); }
      const bx = align === 'left' ? lx : align === 'right' ? lx - w : lx - w / 2;
      ctx.fillStyle = 'rgba(19,22,28,.88)'; ctx.fillRect(bx - fs * 0.25, yy - fs * 0.05, w + fs * 0.5, total + fs * 0.1);
    }
    for (const l of lines) { label(ctx, l.t, lx, yy + lh(l) / 2, l.px, l.c, l.w, align); yy += lh(l); }
  }

  // Text on a dark background so it stays readable over lines.
  function boxedLines(ctx, lines, x, y, align, fs) {
    let w = 0;
    for (const [t, px, wt] of lines) { ctx.font = font(px, wt); w = Math.max(w, ctx.measureText(t).width); }
    const hh = lines.length * fs * 1.1, x0 = align === 'left' ? x : align === 'right' ? x - w : x - w / 2;
    ctx.fillStyle = COL.panel; ctx.fillRect(x0 - fs * 0.3, y - hh / 2 - fs * 0.15, w + fs * 0.6, hh + fs * 0.3);
    let yy = y - hh / 2 + fs * 0.55;
    for (const [t, px, wt, c] of lines) { label(ctx, t, x, yy, px, c, wt, align); yy += fs * 1.1; }
  }

  function draw() {
    if (!cvs.fit()) return;
    const ctx = cvs.ctx, Wd = cvs.w, Hd = cvs.h;
    const L = layout(Wd, Hd), p = L.p, r = L.r, port = L.port;
    const fs = port ? Math.min(fontPx(root), Wd / 25, Hd / 38) : Math.min(fontPx(root), Wd / 36, Hd / 23);
    ctx.clearRect(0, 0, Wd, Hd);

    // the two halves of the brain
    ctx.save();
    ctx.setLineDash([3, 6]); ctx.strokeStyle = 'rgba(139,147,161,.45)'; ctx.lineWidth = 1.2;
    ctx.beginPath();
    if (port) { ctx.moveTo(L.mid, L.padY); ctx.lineTo(L.mid, p.fly[1] - L.flyS * 1.3); }
    else { ctx.moveTo(L.padX, L.mid); ctx.lineTo(p.mnL[0] + r, L.mid); }
    ctx.stroke();
    ctx.restore();
    if (port) {   // in the top corners, beside the objects
      const room = p.scrL[0] - L.r * 1.3 - L.padX * 2;
      ctx.font = font(fs * 0.86, 650);
      const half = (t, x, align) => { const ls = wrapBalanced(ctx, t, room); ls.forEach((s, i) => label(ctx, s, x, p.scrL[1] + i * fs * 1.05, fs * 0.86, COL.muted, 650, align)); };
      half('left half of the brain', L.padX, 'left');
      half('right half of the brain', L.p.fly[0] * 2 - L.padX, 'right');
    } else {
      label(ctx, 'right half of the brain', L.padX, L.padY + fs * 0.3, fs * 0.86, COL.muted, 650, 'left');
      label(ctx, 'left half of the brain', L.padX, Hd - L.padY - fs * 0.3, fs * 0.86, COL.muted, 650, 'left');
    }

    // objects and eyes
    clickBoxes = [];
    for (const s of ['L', 'R']) {
      const [ex, ey] = p['eye' + s], er = r * 1.15, [sx, sy] = p['scr' + s], on = sees(s);
      const ss = port ? Math.min(r * 1.05, (ey - er - L.padY) * 0.5) : Math.min(r * 1.25, (ex - er - L.padX) * 0.45);
      ctx.fillStyle = on ? '#d6dae1' : '#2a2f3a';
      ctx.fillRect(sx - ss, sy - ss, 2 * ss, 2 * ss);
      if (on) {
        ctx.save(); ctx.beginPath(); ctx.rect(sx - ss, sy - ss, 2 * ss, 2 * ss); ctx.clip();
        ctx.fillStyle = '#0b0d11'; ctx.beginPath();
        if (stim === 'slow') { const ph = (wallNow * 0.25) % 1; ctx.arc(sx - ss * 0.7 + ph * ss * 1.4, sy, ss * 0.22, 0, 7); }
        else { const ph = (wallNow * 0.8) % 1; ctx.arc(sx, sy, ss * (0.08 + 1.2 * ph ** 3), 0, 7); }
        ctx.fill(); ctx.restore();
      }
      ctx.strokeStyle = on ? COL.accent : COL.line; ctx.lineWidth = 1.5; ctx.strokeRect(sx - ss, sy - ss, 2 * ss, 2 * ss);
      ctx.fillStyle = on ? '#b4473b' : '#5a2a27';
      ctx.beginPath(); ctx.ellipse(ex, ey, er * 0.85, er, 0, 0, 7); ctx.fill();
      ctx.fillStyle = 'rgba(0,0,0,.25)';
      for (let i = -2; i <= 2; i++) for (let j = -2; j <= 2; j++) {
        const fx = ex + i * er * 0.32 + (j % 2 ? er * 0.16 : 0), fy = ey + j * er * 0.34;
        if (((fx - ex) / (er * 0.8)) ** 2 + ((fy - ey) / er) ** 2 < 0.8) { ctx.beginPath(); ctx.arc(fx, fy, er * 0.09, 0, 7); ctx.fill(); }
      }
      textBlock(ctx, ex, ey, er * 0.9, [{ t: s === 'L' ? 'left eye' : 'right eye', px: fs * 0.9, c: COL.text, w: 650 }], L.place['eye' + s], fs);
      clickBoxes.push({ side: s, x0: Math.min(sx - ss, ex - er), x1: Math.max(sx + ss, ex + er), y0: Math.min(sy - ss, ey - er), y1: Math.max(sy + ss, ey + er) });
      // eye → detector and eye → brake neuron (light, not spikes)
      for (const to of ['det' + s, 'br' + s]) {
        const [dx, dy] = p[to], len = Math.hypot(dx - ex, dy - ey), ux = (dx - ex) / len, uy = (dy - ey) / len;
        const live = on && !(to.startsWith('br') && noBrake);
        ctx.save();
        ctx.strokeStyle = live ? 'rgba(214,218,225,.6)' : 'rgba(139,147,161,.22)'; ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]); ctx.lineDashOffset = live ? -wallNow * 40 : 0;
        ctx.beginPath(); ctx.moveTo(ex + ux * er * 0.95, ey + uy * er * 0.95); ctx.lineTo(dx - ux * (L.rad[to] + 4), dy - uy * (L.rad[to] + 4)); ctx.stroke();
        ctx.restore();
      }
    }

    // connections
    for (const e of EDGES) {
      const [x0, y0, x1, y1, ux, uy] = edgeEnds(L, e);
      const on = edgeOn(e);
      const col = e.type === 'motor' ? COL.muted : e.type === 'inh' ? COL.inh : COL.exc;
      const width = e.cross ? Math.max(1.3, fs * 0.09) : e.type === 'motor' ? Math.max(1.5, fs * 0.12) : Math.max(2.5, fs * 0.22);
      ctx.save();
      ctx.globalAlpha = on ? (e.cross ? 0.75 : 0.9) : 0.2;
      ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = width;
      if (!on) ctx.setLineDash([4, 6]);
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      ctx.setLineDash([]);
      if (e.type === 'inh') {        // inhibitory: a flat bar
        const bw = r * 0.5;
        ctx.lineWidth = width * 1.5;
        ctx.beginPath(); ctx.moveTo(x1 - uy * bw, y1 + ux * bw); ctx.lineTo(x1 + uy * bw, y1 - ux * bw); ctx.stroke();
      } else if (e.type === 'exc') { // excitatory: an arrow head
        const ah = Math.max(6, r * (e.cross ? 0.38 : 0.48));
        ctx.beginPath(); ctx.moveTo(x1 + ux * 4, y1 + uy * 4);
        ctx.lineTo(x1 - ux * ah - uy * ah * 0.55, y1 - uy * ah + ux * ah * 0.55);
        ctx.lineTo(x1 - ux * ah + uy * ah * 0.55, y1 - uy * ah - ux * ah * 0.55); ctx.fill();
      }
      ctx.restore();
    }

    // spikes in flight
    const keep = [];
    for (const d of dots) {
      const ph = (wallNow - d.t0) / DOT_T;
      if (ph >= 1 || !edgeOn(d.e)) continue;
      keep.push(d);
      if (ph < 0) continue;
      const [x0, y0, x1, y1] = edgeEnds(L, d.e);
      const dr = d.e.cross ? 0.12 : d.e.type === 'inh' ? 0.13 : 0.19;   // small dots: weak connections
      ctx.fillStyle = d.e.type === 'inh' ? '#ff9a9a' : COL.accent;
      ctx.beginPath(); ctx.arc(x0 + (x1 - x0) * ph, y0 + (y1 - y0) * ph, Math.max(2, r * dr), 0, 7); ctx.fill();
    }
    dots = keep.length > 2500 ? keep.slice(-2500) : keep;

    // the crossing connections: one plain label, and the cut where they cross the midline
    {
      const [ax0, ay0, ax1, ay1] = edgeEnds(L, EDGES[2]), [bx0, by0, bx1, by1] = edgeEnds(L, EDGES[3]);
      // intersection of the two crossing lines
      const d = (ax1 - ax0) * (by1 - by0) - (ay1 - ay0) * (bx1 - bx0);
      const t = d ? ((bx0 - ax0) * (by1 - by0) - (by0 - ay0) * (bx1 - bx0)) / d : 0.5;
      const cx = ax0 + (ax1 - ax0) * t, cy = ay0 + (ay1 - ay0) * t;
      const maxW = port ? (p.gfR[0] - p.gfL[0]) - L.rad.gfL * 2 - fs * 0.8 : L.X(0.24) - L.X(0);
      ctx.font = font(fs * 0.9, 700);
      const l1 = wrapBalanced(ctx, 'crossing to the other half of the brain', maxW);
      ctx.font = font(fs * 0.86, 400);
      const l2 = wrapBalanced(ctx, '(few connections → weaker)', maxW);
      const lines = [...l1.map((s) => [s, fs * 0.9, 700, cut ? 'rgba(91,227,160,.45)' : COL.exc]), ...l2.map((s) => [s, fs * 0.86, 400, COL.muted])];
      if (port) boxedLines(ctx, lines, L.mid, (p.gfL[1] + p.mnL[1]) / 2, 'center', fs);
      else boxedLines(ctx, lines, cx + Math.abs(ax1 - ax0) * 0.18 + fs * 0.6, cy, 'left', fs);
      if (cut) {
        label(ctx, '✂', cx, cy, fs * 1.7, COL.inh, 700);
        label(ctx, 'cut', cx - fs * 1.2, cy + (port ? fs * 1.3 : fs * 0.05), fs * 0.95, COL.inh, 750, port ? 'center' : 'right');
      }
    }

    // neurons: name, (a few plain words,) firing rate
    const neuron = (key, name, sub, isInh) => {
      const u = U[key], [x, y] = p[key], rr = L.rad[key], f = flash[u], hz = rate(u);
      const silenced = isInh && noBrake;
      ctx.save(); if (silenced) ctx.globalAlpha = 0.35;
      ctx.fillStyle = COL.panel2; ctx.beginPath(); ctx.arc(x, y, rr, 0, 7); ctx.fill();
      if (f > 0.01) { ctx.fillStyle = isInh ? `rgba(255,107,107,${0.9 * f})` : `rgba(245,197,66,${0.95 * f})`; ctx.beginPath(); ctx.arc(x, y, rr, 0, 7); ctx.fill(); }
      ctx.strokeStyle = isInh ? COL.inh : COL.exc; ctx.lineWidth = Math.max(2, fs * 0.15); ctx.beginPath(); ctx.arc(x, y, rr, 0, 7); ctx.stroke();
      const maxW = L.textW(key);
      ctx.font = font(fs * 0.95, 700);
      const lines = wrapBalanced(ctx, name, maxW).map((t) => ({ t, px: fs * 0.95, c: isInh ? COL.inh : COL.text, w: 700 }));
      if (sub) { ctx.font = font(fs * 0.86); for (const t of wrapBalanced(ctx, sub, maxW)) lines.push({ t, px: fs * 0.86, c: COL.muted, w: 400 }); }
      lines.push({ t: `${Math.round(hz)} Hz`, px: fs * 1.25, c: hz > 0.5 ? (isInh ? COL.inh : COL.accent) : COL.muted, w: 800 });
      textBlock(ctx, x, y, rr, lines, L.place[key], fs, key.startsWith('det') && !port);
      ctx.restore();
    };
    for (const s of ['L', 'R']) {
      const desc = !port && s === 'R';      // the plain-words descriptions once, on the top row
      neuron('det' + s, 'looming detector', desc ? 'fires when something rushes at the fly' : null);
      neuron('br' + s, port ? 'brake' : desc ? 'brake neuron (inhibitory)' : 'brake neuron', null, true);
      neuron('gf' + s, 'giant fiber', desc ? 'fastest wire to the legs' : null);
      neuron('mn' + s, 'motor neuron', null);
    }
    drawFly(ctx, L, fs);
  }

  // The fly, seen from above, facing the eyes. Its legs on each side move with that side's motor
  // neuron. Both sides strong → it jumps; one side only → it stumbles and turns.
  function drawFly(ctx, L, fs) {
    const [cx, cy] = L.p.fly, s = L.flyS;
    const ph = wallNow - act.t0, busy = ph >= 0 && ph < 0.9, env = busy ? Math.sin(Math.PI * ph / 0.9) : 0;
    const lift = act.kind === 'jump' ? env : 0;
    const turn = act.kind === 'stumble' ? env * 0.55 * (act.side === 'R' ? -1 : 1) : 0;
    const heading = L.port ? -Math.PI / 2 : Math.PI;    // head up / head left
    const aOf = { '-1': Math.min(1, legs.L), '1': Math.min(1, legs.R) };   // local −y: left, +y: right
    ctx.save();
    ctx.fillStyle = 'rgba(0,0,0,.45)';
    ctx.beginPath(); ctx.ellipse(cx, cy + s * 0.2, s * (1.05 - 0.3 * lift), s * (0.5 - 0.15 * lift), heading + turn, 0, 7); ctx.fill();
    ctx.translate(cx, cy - lift * s * (L.port ? 0.5 : 0.9));
    ctx.scale(1 + 0.4 * lift, 1 + 0.4 * lift);
    ctx.rotate(heading + turn);
    // legs: three per side, longer and brighter while that side pushes
    for (const d of [-1, 1]) {
      const a = aOf[String(d)];
      ctx.strokeStyle = `rgb(${Math.round(107 + 125 * a)},${Math.round(90 + 85 * a)},${Math.round(69 + 6 * a)})`;
      ctx.lineWidth = Math.max(1.5, s * 0.05);
      for (const lx of [-0.25, 0.05, 0.35]) {
        ctx.beginPath(); ctx.moveTo(lx * s, d * s * 0.12);
        ctx.lineTo((lx - 0.18 - 0.12 * a) * s, d * s * (0.6 + 0.3 * a)); ctx.stroke();
      }
    }
    const spread = 0.15 + 1.1 * lift;
    ctx.fillStyle = 'rgba(200,215,235,.35)'; ctx.strokeStyle = 'rgba(200,215,235,.7)'; ctx.lineWidth = 1;
    for (const d of [-1, 1]) {
      ctx.save(); ctx.translate(-s * 0.05, d * s * 0.05); ctx.rotate(Math.PI + d * spread);
      ctx.beginPath(); ctx.ellipse(s * 0.55, 0, s * 0.6, s * 0.2, 0, 0, 7); ctx.fill(); ctx.stroke(); ctx.restore();
    }
    ctx.fillStyle = '#8a7a5f'; ctx.beginPath(); ctx.ellipse(-s * 0.35, 0, s * 0.42, s * 0.26, 0, 0, 7); ctx.fill();
    ctx.fillStyle = '#6f6047'; ctx.beginPath(); ctx.ellipse(s * 0.12, 0, s * 0.26, s * 0.24, 0, 0, 7); ctx.fill();
    ctx.fillStyle = '#7a6a50'; ctx.beginPath(); ctx.arc(s * 0.45, 0, s * 0.16, 0, 7); ctx.fill();
    ctx.fillStyle = '#c0392b';
    for (const d of [-1, 1]) { ctx.beginPath(); ctx.ellipse(s * 0.47, d * s * 0.13, s * 0.11, s * 0.08, 0, 0, 7); ctx.fill(); }
    ctx.restore();
    // labels: which legs are which, and what the fly does
    if (!L.port) {
      label(ctx, 'right legs', cx, cy - s * 1.12, fs * 0.86, COL.muted, 650);
      label(ctx, 'left legs', cx, cy + s * 1.12, fs * 0.86, COL.muted, 650);
    }
    if (busy) {
      const t = act.kind === 'jump' ? 'jump!' : 'stumbles';
      if (L.port) label(ctx, t, cx, cy - s * 1.25, fs * 1.05, COL.accent, 800);
      else label(ctx, t, cx - s * 1.15, cy, fs * 1.05, COL.accent, 800, 'right');
    }
  }

  const hitEye = (x, y) => clickBoxes.find((c) => x >= c.x0 - 6 && x <= c.x1 + 6 && y >= c.y0 - 6 && y <= c.y1 + 6);
  cvs.cv.addEventListener('pointerdown', (e) => {
    const c = hitEye(...cvs.pos(e));
    if (!c) return;
    let l = stim === 'left' || stim === 'both', rr = stim === 'right' || stim === 'both';
    if (c.side === 'L') l = !l; else rr = !rr;
    setStimMode(l && rr ? 'both' : l ? 'left' : rr ? 'right' : 'none');
  });
  cvs.cv.addEventListener('pointermove', (e) => { cvs.cv.style.cursor = hitEye(...cvs.pos(e)) ? 'pointer' : ''; });

  const loop = animLoop((dtw, now) => {
    wallNow = now;
    acc += dtw * SIM_MS_PER_SEC / P.dt;
    const n = Math.min(1500, Math.floor(acc));
    acc -= n;
    simulate(n, true);
    const k = Math.exp(-dtw / 0.1);
    for (let i = 0; i < flash.length; i++) flash[i] *= k;
    // the fly reacts when a giant fiber fires strongly (and the last reaction has finished)
    const l = rate(U.gfL) > ACT_HZ, rr = rate(U.gfR) > ACT_HZ;
    if ((l || rr) && now - act.t0 > 1.3) act = { t0: now, kind: l && rr ? 'jump' : 'stumble', side: rr ? 'R' : 'L' };
    draw();
  });

  wire(); setStim();
  simulate(Math.round(RATE_WIN / P.dt), false);  // warm up so the rates are meaningful at once
  const ro = new ResizeObserver(() => draw());
  ro.observe(cvs.wrap);
  draw();
  const api = { start: () => loop.start(), stop: () => loop.stop() };
  // exposed for testing: current rates in Hz, and the fly's last reaction
  Object.defineProperty(api, '_rates', { enumerable: false, value: () => Object.fromEntries(NODE.map((k, u) => [k, rate(u)])) });
  Object.defineProperty(api, '_act', { enumerable: false, value: () => ({ ...act }) });
  return api;
}

// =============================================================================================
// 3. synapseDemo — "How the brain model works"
// =============================================================================================
// A excites B, C inhibits B. Firing A or C gives it an input kick (like an eye input spike) so it
// fires. The spike travels for 1.8 ms, then B's input g jumps (up for A, down for C) and decays
// with τ = 5 ms. g pushes B's voltage, which leaks back to −52 mV with τ = 20 ms. The sliders set
// each connection's push: the biggest voltage change one spike causes. One A spike (+2 mV) is
// not enough for B to fire; a burst of 5 is.

export function synapseDemo(el) {
  const { root, main, bar } = mountLayout(el);
  main.classList.add('nd-tall');
  const cvs = makeCanvas(main);
  const A = 0, C = 1, B = 2;
  const net = new LIFNet(3, 99);
  let pA = 2, pC = 2;                      // push per spike, mV
  const msPerSec = 4;                      // slow motion: 1 s on screen = 4 ms of brain time (250×)
  function wire() { net.disconnectAll(); net.connectMv(A, B, pA / PSP); net.connectMv(C, B, -pC / PSP); }
  wire();

  // --- controls ---
  const sA = h('input', { class: 'nd-range nd-exc-r', type: 'range', min: 0.5, max: 8, step: 0.5, value: pA, 'aria-label': 'A to B: push per spike' });
  const sC = h('input', { class: 'nd-range nd-inh-r', type: 'range', min: 0.5, max: 8, step: 0.5, value: pC, 'aria-label': 'C to B: push per spike' });
  const cA = control(h('span', { class: 'nd-exc' }, 'A → B: push per spike'), sA);
  const cC = control(h('span', { class: 'nd-inh' }, 'C → B: push per spike'), sC);
  cA.val.style.color = COL.exc; cC.val.style.color = COL.inh;
  const btnA = h('button', { class: 'nd-btn nd-primary', type: 'button' }, 'Fire A');
  const btnA5 = h('button', { class: 'nd-btn', type: 'button' }, 'A ×5');
  const btnC = h('button', { class: 'nd-btn nd-red', type: 'button' }, 'Fire C');
  bar.append(h('div', { class: 'nd-row' }, btnA, btnA5, btnC), cA.el, cC.el);
  function labels() { cA.val.textContent = fmtMv(pA) + ' mV'; cC.val.textContent = fmtMv(-pC) + ' mV'; }
  sA.addEventListener('input', () => { pA = +sA.value; wire(); labels(); if (!loop.running) draw(); });
  sC.addEventListener('input', () => { pC = +sC.value; wire(); labels(); if (!loop.running) draw(); });
  labels();

  // --- state ---
  const CAP = 1 << 13;                     // 819 ms of history per variable
  const hv = new Float32Array(CAP).fill(P.vRest), hg = new Float32Array(CAP);
  let stepN = 0, acc = 0;
  const queue = [];                        // scheduled kicks {t (ms), u}
  let spikes = [];                         // {u, t (ms), step}
  const flash = new Float32Array(3);
  let gScale = 16;                         // input plot range (mV of g), adapts smoothly

  function fire(u, count = 1, gapMs = 4) {
    for (let i = 0; i < count; i++) queue.push({ t: net.t + i * gapMs, u });
  }
  btnA.addEventListener('click', () => fire(A));
  btnA5.addEventListener('click', () => fire(A, 5, 3));
  btnC.addEventListener('click', () => fire(C));

  function simulate(n) {
    for (let k = 0; k < n; k++) {
      for (let i = queue.length - 1; i >= 0; i--) if (queue[i].t <= net.t) { net.addV[queue[i].u] += KICK; queue.splice(i, 1); }
      const spk = net.step();
      const fired = spk.includes(B);
      hv[stepN % CAP] = fired ? P.vTh : net.v[B];
      hg[stepN % CAP] = net.g[B];
      for (const u of spk) { spikes.push({ u, t: net.t, step: stepN }); flash[u] = 1; }
      stepN++;
    }
    if (spikes.length && spikes[0].step < stepN - CAP) spikes = spikes.filter((s) => s.step >= stepN - CAP);
  }

  function layout(Wd, Hd) {
    const wide = Wd > Hd * 1.15 && Wd > 560;
    const D = wide ? { x: 0, y: 0, w: Wd * 0.34, h: Hd } : { x: 0, y: 0, w: Wd, h: Hd * 0.38 };
    const Pl = wide ? { x: Wd * 0.36, y: 0, w: Wd * 0.64, h: Hd } : { x: 0, y: Hd * 0.38, w: Wd, h: Hd * 0.62 };
    const r = Math.max(12, Math.min(D.w * 0.095, D.h * 0.11));
    const pos = {
      A: [D.x + D.w * 0.17, D.y + D.h * (wide ? 0.25 : 0.24)],
      C: [D.x + D.w * 0.17, D.y + D.h * (wide ? 0.75 : 0.76)],
      B: [D.x + D.w * 0.62, D.y + D.h * 0.5],
    };
    return { wide, D, Pl, r, pos };
  }

  function draw() {
    if (!cvs.fit()) return;
    const ctx = cvs.ctx, Wd = cvs.w, Hd = cvs.h;
    const Lo = layout(Wd, Hd), { pos, r, D, Pl } = Lo;
    const fs = Math.min(fontPx(root), Pl.w / 22, Hd / 20);
    ctx.clearRect(0, 0, Wd, Hd);
    const vB = net.v[B];

    // --- diagram: axons A → B and C → B, spikes in flight ---
    const axon = (from, col, isInh) => {
      const [x0, y0] = pos[from], [x1, y1] = pos.B;
      const dx = x1 - x0, dy = y1 - y0, len = Math.hypot(dx, dy), ux = dx / len, uy = dy / len;
      const sx = x0 + ux * r, sy = y0 + uy * r, ex = x1 - ux * (r + 6), ey = y1 - uy * (r + 6);
      ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = Math.max(2, fs * 0.18);
      ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(ex, ey); ctx.stroke();
      if (isInh) { const bw = r * 0.4; ctx.lineWidth *= 1.6; ctx.beginPath(); ctx.moveTo(ex - uy * bw, ey + ux * bw); ctx.lineTo(ex + uy * bw, ey - ux * bw); ctx.stroke(); }
      else { const ah = r * 0.42; ctx.beginPath(); ctx.moveTo(ex + ux * 5, ey + uy * 5); ctx.lineTo(ex - ux * ah - uy * ah * 0.6, ey - uy * ah + ux * ah * 0.6); ctx.lineTo(ex - ux * ah + uy * ah * 0.6, ey - uy * ah - ux * ah * 0.6); ctx.fill(); }
      return { sx, sy, ex, ey, ux, uy };
    };
    const axA = axon('A', COL.exc, false), axC = axon('C', COL.inh, true);
    // push per spike, written along each axon
    const wl = (ax, txt, col, d) => {
      const mx = (ax.sx + ax.ex) / 2, my = (ax.sy + ax.ey) / 2;
      ctx.save(); ctx.translate(mx, my); ctx.rotate(Math.atan2(ax.uy, ax.ux));
      ctx.font = font(fs * 0.95, 750); ctx.fillStyle = col; ctx.textAlign = 'center'; ctx.textBaseline = d < 0 ? 'bottom' : 'top';
      ctx.fillText(txt, 0, d * fs * 0.45); ctx.restore();
    };
    wl(axA, fmtMv(pA) + ' mV', COL.exc, -1);
    wl(axC, fmtMv(-pC) + ' mV', COL.inh, 1);
    // B's output axon
    const [bx, by] = pos.B;
    const outX = D.x + D.w * 0.97;
    ctx.strokeStyle = 'rgba(230,232,236,.5)'; ctx.lineWidth = Math.max(2, fs * 0.15);
    ctx.beginPath(); ctx.moveTo(bx + r, by); ctx.lineTo(outX, by); ctx.stroke();

    // spikes travelling: they take 1.8 ms to reach B (then B's input changes)
    const t = net.t;
    for (const s of spikes) {
      const age = t - s.t;
      if (s.u === B) {
        if (age > 1.8) continue;
        const f = age / 1.8;
        ctx.fillStyle = COL.accent; ctx.beginPath(); ctx.arc(bx + r + (outX - bx - r) * f, by, Math.max(3, r * 0.2), 0, 7); ctx.fill();
        continue;
      }
      const ax = s.u === A ? axA : axC;
      if (age <= 1.8) {
        const f = age / 1.8;
        ctx.fillStyle = COL.accent;
        ctx.beginPath(); ctx.arc(ax.sx + (ax.ex - ax.sx) * f, ax.sy + (ax.ey - ax.sy) * f, Math.max(4, r * 0.24), 0, 7); ctx.fill();
      } else if (age < 6) {      // arrived: the contact point glows
        const f = 1 - (age - 1.8) / 4.2, col = s.u === A ? '91,227,160' : '255,107,107';
        ctx.fillStyle = `rgba(${col},${0.6 * f})`;
        ctx.beginPath(); ctx.arc(ax.ex, ax.ey, r * 0.55, 0, 7); ctx.fill();
      }
    }

    // neurons: A, C (clickable), B (fill shows how close its voltage is to threshold)
    const drawN = (key, u, col) => {
      const [x, y] = pos[key];
      ctx.fillStyle = COL.panel2; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill();
      if (key === 'B') {
        const a = Math.max(0, Math.min(1, (vB - P.vRest) / (P.vTh - P.vRest)));
        const b = Math.max(0, Math.min(1, (P.vRest - vB) / 4));
        if (a > 0) { ctx.fillStyle = `rgba(91,227,160,${0.55 * a})`; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill(); }
        if (b > 0) { ctx.fillStyle = `rgba(255,107,107,${0.5 * b})`; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill(); }
      }
      if (flash[u] > 0.01) { ctx.fillStyle = `rgba(245,197,66,${flash[u]})`; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill(); }
      ctx.strokeStyle = col; ctx.lineWidth = Math.max(2.5, fs * 0.18); ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.stroke();
      label(ctx, key, x, y + 1, r * 0.85, COL.text, 750);
    };
    drawN('A', A, COL.exc);
    drawN('C', C, COL.inh);
    drawN('B', B, COL.text);

    // --- plots: B's input (top) and B's voltage (bottom) ---
    const x0 = Pl.x + fs * 5.4, x1 = Pl.x + Pl.w - fs * 0.8;
    const gTop = Pl.y + fs * 1.8, gBot = Pl.y + Pl.h * 0.34;
    const vTop = gBot + fs * 2.3, vBot = Pl.y + Pl.h - fs * 2.6;
    const winMs = 10 * msPerSec;                    // 10 s on screen
    const winSteps = Math.round(winMs / P.dt), pxStep = (x1 - x0) / winSteps;
    const first = Math.max(0, stepN - winSteps, stepN - CAP + 1);
    const xOf = (s) => x1 - (stepN - 1 - s) * pxStep;

    // input plot (no numbers: its size is in g units, while the sliders speak about voltage)
    let gm = 0, vmin = P.vRest;
    for (let s = first; s < stepN; s++) { gm = Math.max(gm, Math.abs(hg[s % CAP])); vmin = Math.min(vmin, hv[s % CAP]); }
    gScale += (Math.max(8, gm * 1.15) - gScale) * 0.15;
    const gy = (g) => (gTop + gBot) / 2 - g / gScale * (gBot - gTop) / 2;
    label(ctx, 'input to B', x0, gTop - fs * 0.3, fs * 0.95, COL.text, 700, 'left', 'bottom');
    ctx.strokeStyle = 'rgba(139,147,161,.45)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x0, gy(0)); ctx.lineTo(x1, gy(0)); ctx.stroke();
    label(ctx, 'up', x0 - fs * 0.45, (gTop + gy(0)) / 2, fs * 0.86, COL.exc, 650, 'right');
    label(ctx, 'down', x0 - fs * 0.45, (gBot + gy(0)) / 2, fs * 0.86, COL.inh, 650, 'right');
    ctx.save(); ctx.beginPath(); ctx.rect(x0, gTop - 2, x1 - x0, gBot - gTop + 4); ctx.clip();
    for (const sign of [1, -1]) {          // filled area: green above 0, red below
      ctx.fillStyle = sign > 0 ? 'rgba(91,227,160,.3)' : 'rgba(255,107,107,.3)';
      ctx.beginPath(); ctx.moveTo(xOf(first), gy(0));
      for (let s = first; s < stepN; s++) { const g = hg[s % CAP]; ctx.lineTo(xOf(s), gy(sign > 0 ? Math.max(0, g) : Math.min(0, g))); }
      ctx.lineTo(xOf(stepN - 1), gy(0)); ctx.fill();
    }
    ctx.strokeStyle = COL.text; ctx.lineWidth = Math.max(1.4, fs * 0.09);
    ctx.beginPath();
    for (let s = first; s < stepN; s++) { const x = xOf(s), y = gy(hg[s % CAP]); s === first ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
    ctx.stroke();
    ctx.restore();

    // voltage plot
    const ys = voltScale(vTop + fs * 0.4, vBot, Math.min(-53.2, vmin - 0.4));
    label(ctx, 'voltage of B', x0, vTop, fs * 0.95, COL.text, 700, 'left', 'bottom');
    voltAxis(ctx, x0, x1, ys, fs);
    ctx.save(); ctx.beginPath(); ctx.rect(x0, vTop, x1 - x0, vBot - vTop + 4); ctx.clip();
    ctx.strokeStyle = COL.text; ctx.lineWidth = Math.max(1.6, fs * 0.1); ctx.lineJoin = 'round';
    ctx.beginPath();
    for (let s = first; s < stepN; s++) { const x = xOf(s), y = ys.y(hv[s % CAP]); s === first ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
    ctx.stroke();
    ctx.strokeStyle = COL.accent; ctx.lineWidth = Math.max(1.8, fs * 0.12);
    for (const s of spikes) if (s.u === B && s.step >= first) drawSpike(ctx, xOf(s.step), ys, Math.min(fs * 0.9, Math.max(4, 9 * pxStep)));
    ctx.restore();
    // arrival markers for A and C spikes (when they reach B)
    for (const s of spikes) if (s.u !== B) {
      const arr = s.step + Math.round(P.tDelay / P.dt) + 2;   // g changes 18 + 2 steps later
      if (arr < first || arr >= stepN) continue;
      const x = xOf(arr);
      ctx.fillStyle = s.u === A ? COL.exc : COL.inh;
      ctx.beginPath(); ctx.moveTo(x, vBot + 3); ctx.lineTo(x - fs * 0.35, vBot + fs * 0.75); ctx.lineTo(x + fs * 0.35, vBot + fs * 0.75); ctx.fill();
    }

    // time scale
    scaleBar(ctx, x1, Pl.y + Pl.h - fs * 0.9, 1 / P.dt * pxStep, '1 ms', fs);
  }

  const near = (x, y) => { const { pos, r } = layout(cvs.w, cvs.h); return [['A', A], ['C', C]].find(([k]) => Math.hypot(x - pos[k][0], y - pos[k][1]) < r * 1.35); };
  cvs.cv.addEventListener('pointerdown', (e) => { const hit = near(...cvs.pos(e)); if (hit) fire(hit[1]); });
  cvs.cv.addEventListener('pointermove', (e) => { cvs.cv.style.cursor = near(...cvs.pos(e)) ? 'pointer' : ''; });

  const loop = animLoop((dtw) => {
    acc += dtw * msPerSec / P.dt;
    const n = Math.min(500, Math.floor(acc));
    acc -= n;
    simulate(n);
    const k = Math.exp(-dtw / 0.25);
    for (let i = 0; i < 3; i++) flash[i] *= k;
    draw();
  });
  const ro = new ResizeObserver(() => draw());
  ro.observe(cvs.wrap);
  simulate(Math.round(10 * msPerSec / P.dt));
  draw();
  const api = { start: () => loop.start(), stop: () => loop.stop() };
  // exposed for testing: B's spike count so far
  Object.defineProperty(api, '_bSpikes', { enumerable: false, value: () => spikes.filter((s) => s.u === B).length });
  return api;
}
