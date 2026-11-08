// Interactive schematic of mushroom-body learning for the lecture slides.
//
//   import { learningDiagram } from "/static/slides/learning.js";
//   const w = learningDiagram(el);   // el: an empty, sized container
//   w.start(); ... w.stop();         // start = draw/animate, stop = no requestAnimationFrame, no timers
//
// A toy model, no server calls: 40 Kenyon cells (KCs), two odors with sparse, partly overlapping KC sets,
// two compartments, each with one output neuron (MBON) and one dopamine neuron (DAN):
//   γ1pedc  : MBON drives approach, DAN = PPL1 (punishment, shock)
//   γ5β′2a  : MBON drives avoidance, DAN = PAM (reward, sugar)
// Rule: a KC→MBON synapse weakens only if that KC is active AND its compartment gets dopamine.
// Test: per odor, net = approach drive − avoid drive; 20 flies in a T-maze choose with a logistic rule.
// Styles are injected once and scoped with the `ld-` prefix.

const C = {
  panel: "#13161c", panel2: "#1a1e26", line: "#262b35", text: "#e6e8ec", muted: "#8b93a1", accent: "#f5c542",
  pam: "#5be3a0", ppl1: "#ff6b6b", kc: "#4fc3f7", mbon: "#b388ff", banana: "#f2d04b", vinegar: "#e39a5b",
};
const FONT = 'ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif';
const ROWS = 8, COLS = 5, NKC = ROWS * COLS;
const ODORS = [
  { name: "banana", color: C.banana, kcs: [2, 9, 17, 24, 33] },
  { name: "vinegar", color: C.vinegar, kcs: [6, 17, 21, 28, 38] },   // KC 17 is shared: memories generalize a little
];
// On screen only plain words (audience: high school). Scientific names stay in comments and speaker notes:
//   "go toward" = MBON-γ1pedc>α/β, its dopamine = PPL1-γ1pedc;  "go away" = MBON-γ5β′2a, its dopamine = PAM-γ5/β′2a.
const COMPS = [
  { role: "go toward", danWord: "punishment", danColor: C.ppl1, us: "shock" },
  { role: "go away", danWord: "reward", danColor: C.pam, us: "sugar" },
];
const KEEP = 0.45, FLOOR = 0.05;        // one pairing leaves 45% of the synapse (Hige et al. 2015 saw large drops)
const NFLY = 20, GAIN = 4;              // T-maze: P(banana) = 1 / (1 + exp(-GAIN * (net_banana - net_vinegar)))
const T_TRAIN = 2700, T_ODOR = 1500, T_FLY = 1900;

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const ph = (t, a, b) => clamp((t - a) / (b - a), 0, 1);
const ease = (p) => p * p * (3 - 2 * p);
const cap1 = (s) => s[0].toUpperCase() + s.slice(1);
function rgba(hex, a) { const n = parseInt(hex.slice(1), 16); return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`; }
function rng(seed) { let a = seed | 0; return () => { a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

const CSS = `
/* Typography follows the page: 0.7em of the container's font size (the deck's --fs, ~30px at 1080p -> 21px), never below 14px. */
.ld-wrap { container-type: inline-size; position: relative; width: 100%; height: 100%; box-sizing: border-box; display: flex; flex-direction: column;
  gap: 0.5em; padding: 0.6em 0.7em 0.55em; overflow: hidden; color: #e6e8ec; font-family: ${FONT}; line-height: 1.3; text-align: left; }
.ld-wrap *, .ld-wrap *::before, .ld-wrap *::after { box-sizing: border-box; }
.ld-wrap > * { font-size: max(14px, 0.7em); }
.ld-ctl { display: flex; flex-wrap: wrap; gap: 0.45em 1.1em; align-items: center; flex: none; }
.ld-grp { display: flex; gap: 0.35em; align-items: center; flex-wrap: wrap; }
.ld-lab { color: #8b93a1; margin-right: 0.15em; }
.ld-wrap button { background: #1a1e26; color: #e6e8ec; border: 1px solid #262b35; border-radius: 0.45em; padding: 0.32em 0.65em;
  font: inherit; cursor: pointer; line-height: 1.2; display: inline-flex; align-items: center; gap: 0.4em; }
.ld-wrap button:hover { border-color: #3a4150; }
.ld-wrap button.ld-on { border-color: #f5c542; color: #f5c542; }
.ld-wrap button.ld-primary { background: #f5c542; color: #17130a; border-color: #f5c542; font-weight: 650; }
.ld-dot { width: 0.65em; height: 0.65em; border-radius: 50%; flex: none; display: inline-block; }
.ld-stage { position: relative; flex: 1 1 auto; min-height: 0; }
.ld-stage canvas { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
.ld-cap { flex: none; min-height: 2.6em; color: #c4c9d2; }
.ld-cap b { color: #e6e8ec; font-weight: 650; }
/* narrow (phones): the stage gets an explicit height from JS and the widget grows instead of squeezing */
@container (max-width: 559px) { .ld-ctl { gap: 0.45em 0.8em; } .ld-cap { min-height: 0; } }
`;
let cssDone = false;
function injectCSS() {
  if (cssDone || document.getElementById("ld-learning-css")) return;
  const s = document.createElement("style"); s.id = "ld-learning-css"; s.textContent = CSS; document.head.appendChild(s); cssDone = true;
}

// ---------------------------------------------------------------- icons (centred at x, y; s = size in px)
function banana(ctx, x, y, s) {
  ctx.save(); ctx.translate(x, y + 0.02 * s); ctx.rotate(-0.4);
  ctx.beginPath(); ctx.moveTo(-0.5 * s, -0.12 * s);
  ctx.quadraticCurveTo(0, 0.62 * s, 0.5 * s, -0.12 * s);
  ctx.quadraticCurveTo(0, 0.22 * s, -0.5 * s, -0.12 * s);
  ctx.closePath(); ctx.fillStyle = C.banana; ctx.fill();
  ctx.fillStyle = "#7a5b22"; ctx.beginPath(); ctx.arc(-0.5 * s, -0.12 * s, 0.05 * s, 0, 7); ctx.arc(0.5 * s, -0.12 * s, 0.035 * s, 0, 7); ctx.fill();
  ctx.restore();
}
function bottle(ctx, x, y, s) {
  ctx.save(); ctx.translate(x, y);
  ctx.fillStyle = C.vinegar; rr(ctx, -0.21 * s, -0.12 * s, 0.42 * s, 0.6 * s, 0.08 * s); ctx.fill();
  ctx.fillRect(-0.08 * s, -0.36 * s, 0.16 * s, 0.26 * s);
  ctx.fillStyle = "#8a4f22"; ctx.fillRect(-0.1 * s, -0.46 * s, 0.2 * s, 0.11 * s);
  ctx.fillStyle = "#f4e3cf"; ctx.fillRect(-0.21 * s, 0.06 * s, 0.42 * s, 0.2 * s);
  ctx.restore();
}
function bolt(ctx, x, y, s, col) {
  const P = [[0.12, -0.5], [-0.26, 0.06], [-0.02, 0.06], [-0.12, 0.5], [0.28, -0.1], [0.04, -0.1], [0.16, -0.5]];
  ctx.beginPath(); P.forEach(([a, b], i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, x + a * s, y + b * s)); ctx.closePath();
  ctx.fillStyle = col; ctx.fill();
}
function cube(ctx, x, y, s, col) {
  const a = 0.36 * s, h = 0.2 * s;
  const face = (pts, f) => { ctx.beginPath(); pts.forEach(([p, q], i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, x + p, y + q)); ctx.closePath(); ctx.fillStyle = f; ctx.fill(); ctx.stroke(); };
  ctx.lineWidth = Math.max(1, s * 0.04); ctx.strokeStyle = col; ctx.lineJoin = "round";
  face([[0, -h - 0.16 * s], [a, -0.16 * s], [0, h - 0.16 * s], [-a, -0.16 * s]], "#f4f6f9");
  face([[-a, -0.16 * s], [0, h - 0.16 * s], [0, h + 0.36 * s], [-a, 0.36 * s]], "#cdd2da");
  face([[a, -0.16 * s], [0, h - 0.16 * s], [0, h + 0.36 * s], [a, 0.36 * s]], "#aab1bd");
}
function fly(ctx, x, y, s, ang) {
  ctx.save(); ctx.translate(x, y); ctx.rotate(ang);
  ctx.fillStyle = "rgba(205,214,228,0.5)";
  for (const k of [-1, 1]) { ctx.beginPath(); ctx.ellipse(-0.12 * s, k * 0.2 * s, 0.34 * s, 0.14 * s, k * 0.45, 0, 7); ctx.fill(); }
  ctx.fillStyle = "#e6e8ec"; ctx.beginPath(); ctx.ellipse(-0.04 * s, 0, 0.34 * s, 0.15 * s, 0, 0, 7); ctx.fill();
  ctx.fillStyle = "#b5524a"; ctx.beginPath(); ctx.arc(0.33 * s, 0, 0.13 * s, 0, 7); ctx.fill();
  ctx.restore();
}
function rr(ctx, x, y, w, h, r) { ctx.beginPath(); ctx.roundRect ? ctx.roundRect(x, y, w, h, r) : ctx.rect(x, y, w, h); }
function odorIcon(ctx, i, x, y, s) { (i === 0 ? banana : bottle)(ctx, x, y, s); }

// ---------------------------------------------------------------- the widget
export function learningDiagram(el) {
  injectCSS();
  el.innerHTML = `<div class="ld-wrap">
    <div class="ld-ctl">
      <div class="ld-grp"><span class="ld-lab">Smell</span>
        ${ODORS.map((o, i) => `<button data-odor="${i}"><i class="ld-dot" style="background:${o.color}"></i>${o.name}</button>`).join("")}</div>
      <div class="ld-grp"><span class="ld-lab">together with</span>
        <button data-us="sugar"><i class="ld-dot" style="background:${C.pam}"></i>sugar</button>
        <button data-us="shock"><i class="ld-dot" style="background:${C.ppl1}"></i>shock</button>
        <button data-us="none">nothing</button></div>
      <div class="ld-grp"><button class="ld-primary" data-a="train">Train</button><button data-a="test">Test in T-maze</button><button data-a="reset">Reset memory</button></div>
    </div>
    <div class="ld-stage"><canvas></canvas></div>
    <div class="ld-cap" aria-live="polite"></div>
  </div>`;
  const root = el.firstElementChild, stage = root.querySelector(".ld-stage"), cv = stage.querySelector("canvas"),
        capEl = root.querySelector(".ld-cap"), ctx = cv.getContext("2d");

  // ---- state
  const w = COMPS.map(() => new Float32Array(NKC).fill(1));
  let odor = 1, us = "shock", anim = null, result = null, trials = 0;
  let running = false, raf = 0, L = null, dpr = 1;
  const rand = rng(5);

  const drive = (c, o, W = w[c]) => ODORS[o].kcs.reduce((a, k) => a + W[k], 0) / ODORS[o].kcs.length;
  const net = (o) => drive(0, o) - drive(1, o);
  function caption(html) { capEl.innerHTML = html; }
  caption(`Pick a smell and what comes with it, then press <b>Train</b>.`);

  function syncButtons() {
    root.querySelectorAll("[data-odor]").forEach((b) => b.classList.toggle("ld-on", +b.dataset.odor === odor));
    root.querySelectorAll("[data-us]").forEach((b) => b.classList.toggle("ld-on", b.dataset.us === us));
  }
  syncButtons();

  // ---- actions
  function train() {
    const o = odor, c = COMPS.findIndex((k) => k.us === us);
    const prev = c >= 0 ? Float32Array.from(w[c]) : null;
    const changed = new Set();
    if (c >= 0) for (const k of ODORS[o].kcs) { const nw = Math.max(FLOOR, w[c][k] * KEEP); if (nw < w[c][k]) changed.add(k); w[c][k] = nw; }
    result = null; trials++;
    const r = rng(trials * 97 + o * 13 + (c + 1));
    const puff = Array.from({ length: 16 }, (_, i) => ({ k: ODORS[o].kcs[i % ODORS[o].kcs.length], t: r() * 320, dx: r() - 0.5, dy: r() - 0.5 }));
    const da = c < 0 ? [] : Array.from({ length: 70 }, () => ({ t: 520 + r() * 950, u: 0.06 + r() * 0.88, v: 0.08 + r() * 0.86, sx: r() - 0.5 }));
    anim = { kind: "train", t0: performance.now(), dur: T_TRAIN, o, c, prev, changed, puff, da };
    const O = ODORS[o].name;
    if (c < 0) caption(`<b>${cap1(O)} alone</b>, no dopamine: nothing changes. Learning needs both at the same time.`);
    else {
      const K = COMPS[c];
      caption(`<b>${cap1(O)} + ${K.us}</b>: <b style="color:${K.danColor}">${K.danWord} dopamine</b> arrives while the ${O} neurons are active, ` +
        `so their connections to “${K.role}” get <b>weaker</b>.`);
    }
    kick();
  }
  function test() {
    const nets = ODORS.map((_, o) => net(o));
    const p = 1 / (1 + Math.exp(-GAIN * (nets[0] - nets[1])));
    const nA = Math.round(NFLY * p), counts = [nA, NFLY - nA];
    const pi = (counts[0] - counts[1]) / NFLY;
    const r = rng(1234 + Math.round(p * 1000));
    const sides = Array.from({ length: NFLY }, (_, i) => (i < nA ? 0 : 1));
    for (let i = NFLY - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [sides[i], sides[j]] = [sides[j], sides[i]]; }
    const slot = [0, 0];
    const flies = sides.map((side) => ({ side, j: slot[side]++, d: r() * 0.35, sp: 0.75 + r() * 0.25, jx: r() - 0.5, jy: r() - 0.5 }));
    flies.forEach((f) => (f.n = counts[f.side]));
    result = { nets, counts, pi, flies, drives: ODORS.map((_, o) => COMPS.map((__, c) => drive(c, o))) };
    anim = { kind: "test", t0: performance.now(), dur: 2 * T_ODOR + T_FLY };
    caption(`Each smell pulls two ways: <b>go toward</b> or <b>go away</b>. The stronger pull wins.`);
    kick();
  }
  function endAnim() {
    const a = anim; anim = null;
    if (a && a.kind === "test" && result) {
      const [nA, nB] = result.counts, win = nA === nB ? -1 : nA > nB ? 0 : 1;
      if (win < 0) caption(`No favourite: nothing has been learned yet.`);
      else {
        const lose = 1 - win, n = result.nets;
        const why = n[lose] < -0.05 && -n[lose] >= n[win] ? `${ODORS[lose].name} now means danger` : `${ODORS[win].name} now means food`;
        caption(`Most flies chose <b>${ODORS[win].name}</b>: ${why}.`);
      }
    }
  }
  function reset() {
    w.forEach((a) => a.fill(1)); anim = null; result = null;
    caption(`Memory erased: every connection is back to full strength.`); kick();
  }
  root.querySelector(".ld-ctl").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    if (b.dataset.odor != null) { odor = +b.dataset.odor; syncButtons(); kick(); }
    else if (b.dataset.us) { us = b.dataset.us; syncButtons(); kick(); }
    else if (b.dataset.a === "train") train();
    else if (b.dataset.a === "test") test();
    else if (b.dataset.a === "reset") reset();
  });

  // ---- layout
  function measure() {
    const fs = parseFloat(getComputedStyle(capEl).fontSize) || 16;
    const W = stage.clientWidth;
    if (W < 2) return false;
    const tall = W < 560;
    let H;
    if (tall) {
      H = Math.round(W * 0.62 + fs * 33);
      if (stage.style.height !== H + "px") { stage.style.flex = "none"; stage.style.height = H + "px"; }
    } else {
      if (stage.style.height) { stage.style.height = ""; stage.style.flex = ""; }
      H = stage.clientHeight;
    }
    if (H < 2) return false;
    L = layout(W, H, fs, tall);
    dpr = Math.min(2, window.devicePixelRatio || 1);
    const bw = Math.round(W * dpr), bh = Math.round(H * dpr);
    if (cv.width !== bw || cv.height !== bh) { cv.width = bw; cv.height = bh; }
    return true;
  }
  function layout(W, H, fs, tall) {
    const pad = Math.round(fs * 0.5);
    const o = { W, H, fs, tall, pad, sm: tall ? fs : fs * 0.88 };   // sm = small label size (≥18 px at 1080p; full size on phones)
    let mainW, mainH;
    if (!tall) {
      const pw = clamp(W * 0.28, fs * 11, fs * 15);
      mainW = W - pw - pad; mainH = H;
      o.panel = { x: mainW + pad * 0.5, y: 0, w: pw - pad * 0.5, h: H };
    } else {
      mainW = W; mainH = H - fs * 17.8;
      o.panel = { x: 0, y: mainH, w: W, h: H - mainH };
    }
    o.mainW = mainW;
    // columns
    const ow = Math.max(mainW * 0.11, fs * 2.7);
    const kx0 = ow + fs * 0.3, kw = mainW * (tall ? 0.18 : 0.17);
    const bx0 = kx0 + kw + mainW * 0.045, gap = mainW * 0.04;
    const bw = (mainW - bx0 - gap) / 2;
    const stacked = bw < fs * 10.5;
    // rows
    const tb = stacked ? fs * 4.3 : fs * 2.6, bb = stacked ? fs * 5.0 : fs * 4.2, th = fs * 0.4;
    const boxTop = tb + fs * 0.35, boxBot = mainH - bb;
    const gy0 = boxTop + th, gy1 = boxBot - fs * 0.35, rowSp = (gy1 - gy0) / ROWS;
    o.rowY = (r) => gy0 + (r + 0.5) * rowSp;
    o.boxTop = boxTop; o.boxBot = boxBot; o.th = th; o.stacked = stacked;
    o.kc = { x0: kx0, w: kw, colX: (c) => kx0 + (c + 0.5) * (kw / COLS), r: Math.max(2.5, Math.min(kw / COLS, rowSp) * 0.31) };
    o.odorX = ow / 2;
    o.odorY = [gy0 + (gy1 - gy0) * 0.27, gy0 + (gy1 - gy0) * 0.73];
    o.icon = Math.min(fs * 2.3, ow * 0.8);
    o.boxes = COMPS.map((_, i) => {
      const x = bx0 + i * (bw + gap), ip = fs * 0.35, cw = (bw - 2 * ip) / COLS;
      return { x, w: bw, colX: (c) => x + ip + (c + 0.5) * cw, rMax: Math.max(3, Math.min(cw, rowSp) * 0.42) };
    });
    o.axEnd = bx0 + 2 * bw + gap;
    return o;
  }

  // ---- text helpers
  function font(px, wt = 500) { ctx.font = `${wt} ${px}px ${FONT}`; }
  function text(s, x, y, px, { wt = 500, col = C.text, align = "left", maxW = 0, base = "middle" } = {}) {
    font(px, wt);
    if (maxW > 0) { const m = ctx.measureText(s).width; if (m > maxW) { px = Math.max(8, px * maxW / m); font(px, wt); } }
    ctx.fillStyle = col; ctx.textAlign = align; ctx.textBaseline = base; ctx.fillText(s, x, y);
    return px;
  }

  // ---- drawing
  function dispW(c, k, t) {
    const a = anim;
    if (a && a.kind === "train" && a.c === c && a.changed.has(k)) { const p = ease(ph(t, 1150, 1950)); return a.prev[k] + (w[c][k] - a.prev[k]) * p; }
    return w[c][k];
  }
  // which odor is being "smelled" now, and how strongly (0..1)
  function smell(t) {
    const a = anim;
    if (!a) return { o: odor, v: 0.45, live: false };
    if (a.kind === "train") return { o: a.o, v: ph(t, 250, 480) * (1 - ph(t, 2250, 2650)), live: true, t0: 0 };
    if (t < 2 * T_ODOR) { const i = t < T_ODOR ? 0 : 1, tt = t - i * T_ODOR; return { o: i, v: ph(tt, 150, 350) * (1 - ph(tt, 1200, 1450)), live: true, t0: i * T_ODOR }; }
    return { o: odor, v: 0.45 * ph(t, 2 * T_ODOR, 2 * T_ODOR + 300), live: false };
  }
  function danLevel(c, t) {
    const a = anim;
    if (!a || a.kind !== "train" || a.c !== c) return 0;
    return ph(t, 450, 650) * (1 - ph(t, 1700, 2300));
  }

  function draw(now) {
    if (!L) return;
    const t = anim ? now - anim.t0 : 0, fs = L.fs, sm = smell(t), active = new Set(ODORS[sm.o].kcs);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, L.W, L.H);

    // KC block background + title
    const K = L.kc;
    ctx.fillStyle = C.panel; rr(ctx, K.x0 - fs * 0.15, L.boxTop, K.w + fs * 0.3, L.boxBot - L.boxTop, fs * 0.45); ctx.fill();
    ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.stroke();
    {  // "smell neurons" under the column: one line if it fits, else two
      const kcx = K.x0 + K.w / 2, mw = K.w + fs * 0.9, o2 = { wt: 650, col: C.kc, align: "center", maxW: mw };
      font(L.sm, 650);
      if (ctx.measureText("smell neurons").width <= mw) text("smell neurons", kcx, L.boxBot + fs * 0.85, L.sm, o2);
      else { text("smell", kcx, L.boxBot + fs * 0.8, L.sm, o2); text("neurons", kcx, L.boxBot + fs * 1.85, L.sm, o2); }
    }

    // compartments
    L.boxes.forEach((B, c) => {
      const lvl = danLevel(c, t), D = COMPS[c];
      ctx.fillStyle = C.panel2; rr(ctx, B.x, L.boxTop, B.w, L.boxBot - L.boxTop, fs * 0.45); ctx.fill();
      if (lvl > 0) { ctx.fillStyle = rgba(D.danColor, 0.13 * lvl); ctx.fill(); }
      ctx.strokeStyle = lvl > 0 ? rgba(D.danColor, 0.35 + 0.6 * lvl) : C.line; ctx.lineWidth = lvl > 0 ? 2 : 1; ctx.stroke();
    });
    // legend in the free corner above the odors / smell neurons
    {
      const B = L.boxes[0], lx = fs * 0.2, rr0 = Math.min(B.rMax, fs * 0.6), mw = B.x - fs * 0.4 - lx;
      const lines = L.stacked ? ["bigger dot =", "stronger", "connection"] : ["bigger dot =", "stronger connection"];
      const ly = L.stacked ? fs * 0.6 : L.boxTop - fs * 1.75 - rr0;
      lines.forEach((s, i) => text(s, lx, ly + i * fs * 0.95, L.sm, { col: C.muted, maxW: mw }));
      const dy = ly + lines.length * fs * 0.95 + rr0 * 0.4;
      ctx.fillStyle = rgba(C.mbon, 0.85);
      ctx.beginPath(); ctx.arc(lx + rr0, dy, rr0, 0, 7); ctx.fill();
      ctx.beginPath(); ctx.arc(lx + rr0 * 2.7, dy, rr0 * 0.45, 0, 7); ctx.fill();
    }

    // KC axons: one faint line per row, through both compartments
    ctx.strokeStyle = "#252c38"; ctx.lineWidth = Math.max(1, fs * 0.07);
    for (let r = 0; r < ROWS; r++) { const y = L.rowY(r); ctx.beginPath(); ctx.moveTo(K.colX(0), y); ctx.lineTo(L.axEnd - fs * 0.2, y); ctx.stroke(); }

    // spikes travelling along the active KCs' axons
    if (sm.live && sm.v > 0.05) {
      const tt = t - sm.t0, x1 = L.axEnd - fs * 0.2;
      for (const k of active) {
        const r = Math.floor(k / COLS), x0 = K.colX(k % COLS), y = L.rowY(r);
        for (let j = 0; j < 5; j++) {
          const p = (tt - 380 - j * 170 - (k % 3) * 40) / 650;
          if (p < 0 || p > 1) continue;
          const x = x0 + (x1 - x0) * p;
          const g = ctx.createLinearGradient(x - fs * 1.2, 0, x, 0); g.addColorStop(0, rgba(C.kc, 0)); g.addColorStop(1, rgba(C.kc, 0.9 * sm.v));
          ctx.strokeStyle = g; ctx.lineWidth = Math.max(2, fs * 0.14); ctx.beginPath(); ctx.moveTo(Math.max(x0, x - fs * 1.2), y); ctx.lineTo(x, y); ctx.stroke();
          ctx.fillStyle = rgba("#bfefff", sm.v); ctx.beginPath(); ctx.arc(x, y, Math.max(2, fs * 0.12), 0, 7); ctx.fill();
        }
      }
    }

    // KC dots
    for (let k = 0; k < NKC; k++) {
      const x = K.colX(k % COLS), y = L.rowY(Math.floor(k / COLS)), on = active.has(k) ? sm.v : 0;
      ctx.fillStyle = "#2f3744"; ctx.beginPath(); ctx.arc(x, y, K.r, 0, 7); ctx.fill();
      if (on > 0) {
        if (sm.live) { ctx.fillStyle = rgba(C.kc, 0.25 * on); ctx.beginPath(); ctx.arc(x, y, K.r * 2, 0, 7); ctx.fill(); }
        ctx.fillStyle = rgba(C.kc, Math.min(1, 0.35 + on)); ctx.beginPath(); ctx.arc(x, y, K.r, 0, 7); ctx.fill();
      }
    }

    // synapse dots (area ∝ strength)
    L.boxes.forEach((B, c) => {
      const a = anim, coin = a && a.kind === "train" && a.c === c ? ph(t, 1000, 1200) * (1 - ph(t, 1900, 2400)) : 0;
      for (let k = 0; k < NKC; k++) {
        const x = B.colX(k % COLS), y = L.rowY(Math.floor(k / COLS)), wt = dispW(c, k, t), on = active.has(k) ? sm.v : 0;
        const r = Math.max(1.5, B.rMax * wt);   // radius ∝ strength: easy to see on a projector
        ctx.fillStyle = on > 0 ? rgba(C.mbon, 0.55 + 0.45 * on) : rgba(C.mbon, 0.55);
        ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill();
        if (on > 0.05) { ctx.strokeStyle = rgba(C.kc, on); ctx.lineWidth = Math.max(1.5, fs * 0.1); ctx.beginPath(); ctx.arc(x, y, B.rMax + fs * 0.08, 0, 7); ctx.stroke(); }
        if (coin > 0 && a.changed.has(k)) {
          const pulse = 0.5 + 0.5 * Math.sin(t / 70);
          ctx.strokeStyle = rgba(COMPS[c].danColor, coin * (0.6 + 0.4 * pulse)); ctx.lineWidth = Math.max(2, fs * 0.13);
          ctx.beginPath(); ctx.arc(x, y, B.rMax + fs * 0.3, 0, 7); ctx.stroke();
        }
      }
    });

    drawDANs(t, fs);
    drawMBONs(t, fs, sm);
    drawOdors(t, fs, sm);
    drawPanel(t, fs);
  }

  function drawDANs(t, fs) {
    L.boxes.forEach((B, c) => {
      const D = COMPS[c], lvl = danLevel(c, t), sel = us === D.us;
      let ix, dx, dy, iy;
      const dr = fs * 0.8;
      if (!L.stacked) {
        dy = L.boxTop - fs * 0.35 - fs * 1.15; iy = dy; ix = B.x + fs * 0.9; dx = B.x + fs * 2.9;
        text(D.danWord, dx + dr + fs * 0.45, dy - fs * 0.42, fs, { wt: 750, col: D.danColor, maxW: B.x + B.w - dx - dr - fs * 0.5 });
        text("dopamine", dx + dr + fs * 0.45, dy + fs * 0.55, L.sm, { col: C.muted, maxW: B.x + B.w - dx - dr - fs * 0.5 });
      } else {
        const cx = B.x + B.w / 2;
        text(D.danWord, cx, fs * 0.6, fs, { wt: 750, col: D.danColor, align: "center", maxW: B.w });
        text("dopamine", cx, fs * 1.6, L.sm, { col: C.muted, align: "center", maxW: B.w });
        dy = iy = L.boxTop - fs * 0.35 - fs * 1.2; ix = cx - fs * 1.15; dx = cx + fs * 0.9;
      }
      // US icon (lit when it is the chosen pairing, flashes while training)
      const usA = sel ? 1 : 0.32;
      ctx.globalAlpha = usA;
      if (D.us === "shock") bolt(ctx, ix, iy, fs * 1.55, C.ppl1); else cube(ctx, ix, iy, fs * 1.3, C.pam);
      ctx.globalAlpha = 1;
      // US → DAN arrow
      ctx.strokeStyle = rgba(D.danColor, sel ? 0.7 : 0.25); ctx.lineWidth = Math.max(1.5, fs * 0.08);
      ctx.beginPath(); ctx.moveTo(ix + fs * 0.6, iy); ctx.lineTo(dx - dr - fs * 0.15, dy); ctx.stroke();
      // DAN → compartment
      ctx.beginPath(); ctx.moveTo(dx, dy + dr); ctx.lineTo(dx, L.boxTop); ctx.stroke();
      // DAN body
      if (lvl > 0) { ctx.fillStyle = rgba(D.danColor, 0.28 * lvl); ctx.beginPath(); ctx.arc(dx, dy, dr * (1.5 + 0.25 * Math.sin(t / 60)), 0, 7); ctx.fill(); }
      ctx.fillStyle = rgba(D.danColor, 0.22 + 0.78 * lvl); ctx.strokeStyle = D.danColor; ctx.lineWidth = Math.max(2, fs * 0.1);
      ctx.beginPath(); ctx.arc(dx, dy, dr, 0, 7); ctx.fill(); ctx.stroke();
      // dopamine particles: from the DAN into the whole compartment
      const a = anim;
      if (a && a.kind === "train" && a.c === c) {
        const top = L.boxTop, h = L.boxBot - L.boxTop;
        for (const p of a.da) {
          const age = (t - p.t) / 900; if (age < 0 || age > 1) continue;
          const mv = ease(Math.min(1, age / 0.55));
          const tx = B.x + p.u * B.w, ty = top + p.v * h;
          const x = dx + (tx - dx) * mv + p.sx * fs * 0.6 * Math.sin(age * 6), y = dy + (ty - dy) * mv;
          const al = (age < 0.1 ? age / 0.1 : 1) * (1 - ph(age, 0.6, 1));
          ctx.fillStyle = rgba(D.danColor, 0.9 * al); ctx.beginPath(); ctx.arc(x, y, Math.max(2.5, fs * 0.22), 0, 7); ctx.fill();
        }
      }
    });
  }

  function drawMBONs(t, fs, sm) {
    const o = sm.o;
    L.boxes.forEach((B, c) => {
      const D = COMPS[c];
      const d = ODORS[o].kcs.reduce((s, k) => s + dispW(c, k, t), 0) / ODORS[o].kcs.length;
      const glow = sm.live ? sm.v * d : 0, mr = fs * 0.85;
      let mx, my, lx, align, lw;
      if (!L.stacked) { mx = B.x + fs * 1.2; my = L.boxBot + fs * 1.45; lx = mx + mr + fs * 0.45; align = "left"; lw = B.x + B.w - lx; }
      else { mx = B.x + B.w / 2; my = L.boxBot + fs * 1.35; lx = mx; align = "center"; lw = B.w; }
      ctx.strokeStyle = rgba(C.mbon, 0.7); ctx.lineWidth = Math.max(2, fs * 0.12);
      ctx.beginPath(); ctx.moveTo(mx, L.boxBot); ctx.lineTo(mx, my - mr); ctx.stroke();
      if (glow > 0.02) { ctx.fillStyle = rgba(C.mbon, 0.3 * glow); ctx.beginPath(); ctx.arc(mx, my, mr * 1.7, 0, 7); ctx.fill(); }
      ctx.fillStyle = rgba(C.mbon, 0.25 + 0.75 * glow); ctx.strokeStyle = C.mbon; ctx.lineWidth = Math.max(2, fs * 0.1);
      ctx.beginPath(); ctx.arc(mx, my, mr, 0, 7); ctx.fill(); ctx.stroke();
      let ly, meterY;
      if (!L.stacked) { ly = my; meterY = L.boxBot + fs * 3.05; }
      else { ly = my + mr + fs * 0.75; meterY = ly + fs * 0.95; }
      text(D.role, lx, ly, fs * 1.05, { wt: 750, align, maxW: lw });
      // meter: how strongly the current smell drives this output neuron (no number: the bar says it)
      const mh = Math.max(6, fs * 0.45), mx0 = B.x + fs * 1.15, mw = Math.max(fs, B.w - fs * 1.15);
      odorIcon(ctx, o, B.x + fs * 0.45, meterY + mh / 2, fs * 1.05);
      ctx.fillStyle = "#1c2029"; rr(ctx, mx0, meterY, mw, mh, mh / 2); ctx.fill();
      ctx.fillStyle = C.mbon; rr(ctx, mx0, meterY, Math.max(mh, mw * d), mh, mh / 2); ctx.fill();
    });
  }

  function drawOdors(t, fs, sm) {
    ODORS.forEach((O, i) => {
      const x = L.odorX, y = L.odorY[i], s = L.icon, sel = i === sm.o;
      const v = sm.live && sel ? sm.v : 0;
      ctx.fillStyle = sel ? rgba(O.color, 0.1 + 0.25 * v) : "rgba(255,255,255,0.02)";
      ctx.strokeStyle = sel ? rgba(O.color, 0.5 + 0.5 * v) : C.line; ctx.lineWidth = sel ? 2 : 1;
      ctx.beginPath(); ctx.arc(x, y, s * 0.62, 0, 7); ctx.fill(); ctx.stroke();
      ctx.globalAlpha = sel ? 1 : 0.55; odorIcon(ctx, i, x, y, s * 0.85); ctx.globalAlpha = 1;
      text(O.name, x, y + s * 0.62 + fs * 0.6, L.sm, { wt: sel ? 650 : 500, col: sel ? C.text : C.muted, align: "center", maxW: L.odorX * 2 - 2 });
    });
    // odor puff: from the icon to its Kenyon cells
    const a = anim;
    if (a && (a.kind === "train" || t < 2 * T_ODOR)) {
      const list = a.kind === "train" ? a.puff : ODORS[sm.o].kcs.flatMap((k, i) => [{ k, t: i * 50, dx: 0.3, dy: -0.2 }, { k, t: 120 + i * 50, dx: -0.3, dy: 0.25 }]);
      const t0 = a.kind === "train" ? 0 : sm.t0;
      for (const p of list) {
        const age = (t - t0 - p.t) / 480; if (age < 0 || age > 1) continue;
        const x0 = L.odorX + L.icon * 0.6, y0 = L.odorY[sm.o];
        const x1 = L.kc.colX(p.k % COLS), y1 = L.rowY(Math.floor(p.k / COLS)), e = ease(age);
        const x = x0 + (x1 - x0) * e + p.dx * fs * Math.sin(age * 3.1), y = y0 + (y1 - y0) * e + p.dy * fs * Math.sin(age * 3.1);
        ctx.fillStyle = rgba(ODORS[sm.o].color, 0.85 * (1 - ph(age, 0.75, 1))); ctx.beginPath(); ctx.arc(x, y, Math.max(2, fs * 0.17), 0, 7); ctx.fill();
      }
    }
  }

  function drawPanel(t, fs) {
    const P = L.panel, px = P.x, pw = P.w, cx = px + pw / 2;
    const y0 = P.y + (L.tall ? fs * 0.6 : 0), ph0 = P.h - (y0 - P.y);
    const k = clamp(ph0 / (fs * 17.2), 0.9, 1.7), u = fs * k;     // vertical unit: spreads out when there is room
    ctx.fillStyle = C.panel; rr(ctx, px, y0, pw, ph0, fs * 0.5); ctx.fill(); ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.stroke();
    text("T-maze test", cx, y0 + u * 0.95, fs, { wt: 750, align: "center", maxW: pw - fs });
    // geometry
    const at = fs * 2.4, ay = y0 + u * 4.3 + at / 2, m = fs * 0.6, armL = px + m, armR = px + pw - m, armLen = (armR - armL) / 2 - at / 2;
    const stemB = ay + u * 4.4, cham = { w: fs * 4.8, h: fs * 2.3 };
    ctx.fillStyle = "#0b0d11"; ctx.strokeStyle = "#3a4150"; ctx.lineWidth = Math.max(1.5, fs * 0.08);
    rr(ctx, armL, ay - at / 2, armR - armL, at, at / 2); ctx.fill(); ctx.stroke();
    ctx.fillRect(cx - at / 2, ay, at, stemB - ay); ctx.beginPath(); ctx.moveTo(cx - at / 2, ay + at / 2); ctx.lineTo(cx - at / 2, stemB); ctx.moveTo(cx + at / 2, ay + at / 2); ctx.lineTo(cx + at / 2, stemB); ctx.stroke();
    rr(ctx, cx - cham.w / 2, stemB, cham.w, cham.h, fs * 0.35); ctx.fill(); ctx.stroke();
    ctx.fillStyle = "#0b0d11"; ctx.fillRect(cx - at / 2 + 1, ay, at - 2, at / 2 + 2); ctx.fillRect(cx - at / 2 + 1, stemB - 2, at - 2, 4);
    // odor at each arm end
    const ends = [armL + fs * 1.3, armR - fs * 1.3];
    ODORS.forEach((O, i) => { odorIcon(ctx, i, ends[i], ay - at / 2 - u * 1.3, fs * 1.9); });

    const R = result, a = anim, flyT = a && a.kind === "test" ? (t - 2 * T_ODOR) / T_FLY : R ? 1 : -1;
    // flies
    const fsz = Math.max(10, fs * 0.8);
    if (R) {
      for (const f of R.flies) {
        const start = { x: cx + f.jx * (cham.w - fsz * 1.4), y: stemB + cham.h / 2 + f.jy * (cham.h - fsz) };
        const dir = f.side === 0 ? -1 : 1;
        const cols = Math.max(1, Math.ceil(f.n / 3)), col = Math.floor(f.j / 3), row = f.j % 3;
        const end = { x: cx + dir * (at / 2 + fsz * 0.6 + (armLen - fsz * 1.1) * (cols > 1 ? col / (cols - 1) : 0.5)),
                      y: ay + (row - 1) * 0.3 * at + (col & 1 ? 0.06 * at : 0) };
        const pos = (q) => {
          q = clamp((q - f.d) / (1 - f.d) / f.sp, 0, 1);
          if (q < 0.45) { const e = ease(q / 0.45); return { x: start.x + (cx - start.x) * e, y: start.y + (ay - start.y) * e }; }
          const e = ease((q - 0.45) / 0.55); return { x: cx + (end.x - cx) * e, y: ay + (end.y - ay) * e };
        };
        const q = clamp(flyT, 0, 1), p = pos(q), p0 = pos(q - 0.02);
        const moving = Math.hypot(p.x - p0.x, p.y - p0.y) > 0.3;
        const ang = moving ? Math.atan2(p.y - p0.y, p.x - p0.x) : q >= 1 ? (dir > 0 ? 0 : Math.PI) : -Math.PI / 2;
        fly(ctx, p.x, p.y, fsz, ang);
      }
    } else {
      const r = rng(77);
      for (let i = 0; i < NFLY; i++) fly(ctx, cx + (r() - 0.5) * (cham.w - fsz * 1.4), stemB + cham.h / 2 + (r() - 0.5) * (cham.h - fsz), fsz, r() * 6.28);
    }
    // counts under each arm end
    const done = R && flyT >= 1;
    ODORS.forEach((O, i) => {
      const x = ends[i] + (i === 0 ? fs * 0.2 : -fs * 0.2);
      text(done ? String(R.counts[i]) : "–", x, ay + at / 2 + u * 1.15, fs * 1.6, { wt: 800, col: done ? C.text : C.muted, align: "center" });
      text(O.name, x, ay + at / 2 + u * 2.35, L.sm, { col: C.muted, align: "center", maxW: pw / 2 - at });
    });
    // the result, in plain words
    const yR = stemB + cham.h + u * 2.0, big = { wt: 800, col: C.accent, align: "center", maxW: pw - fs };
    if (done) {
      const [nA, nB] = R.counts, win = nA === nB ? -1 : nA > nB ? 0 : 1;
      text(win < 0 ? `${nA} – ${nB}` : `${R.counts[win]} of ${NFLY}`, cx, yR, fs * 2.3, big);
      text(win < 0 ? "no favourite" : `flies chose ${ODORS[win].name}`, cx, yR + u * 1.75, fs, { wt: 650, align: "center", maxW: pw - fs * 0.6 });
    } else {
      text("?", cx, yR, fs * 2.3, { ...big, col: C.muted });
      text(R ? "the flies choose…" : "press Test", cx, yR + u * 1.75, fs, { col: C.muted, align: "center", maxW: pw - fs * 0.6 });
    }
  }

  // ---- loop
  function frame(now) {
    raf = 0;
    if (!running) return;
    if (!L && !measure()) return;
    draw(now);
    if (anim && now - anim.t0 >= anim.dur) { endAnim(); draw(now); }
    if (anim) raf = requestAnimationFrame(frame);
  }
  function kick() { if (running && !raf) raf = requestAnimationFrame(frame); }
  const ro = new ResizeObserver(() => { if (measure()) { if (running) { if (!raf) draw(performance.now()); } } });
  ro.observe(stage); ro.observe(root);

  return {
    start() { running = true; measure(); kick(); },
    stop() {
      running = false;
      if (raf) cancelAnimationFrame(raf); raf = 0;
      if (anim) endAnim();   // jump to the end state: the weights are already committed
    },
    destroy() { this.stop(); ro.disconnect(); el.replaceChildren(); },
  };
}
