// Image preprocessing in the browser, for the static (no server) version of the app.
// Ports of flymnist/server.py `preprocess` (a drawing -> MNIST-style 28x28) and
// flymnist/camera.py `preprocess_frame` (a webcam frame -> 36x48 local contrast), including
// Pillow's resampling and the scipy.ndimage steps, so both give the same images as the server.

// ---- Pillow: RGB(A) -> "L" (ITU-R 601-2 luma, Pillow's fixed-point formula) ----
export function rgbaToL(rgba, n) {
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const k = i * 4;
    out[i] = (rgba[k] * 19595 + rgba[k + 1] * 38470 + rgba[k + 2] * 7471 + 0x8000) >>> 16;
  }
  return out;
}

// ---- Pillow: Image.resize for 8-bit images (src/libImaging/Resample.c) ----
const PRECISION_BITS = 32 - 8 - 2;
const FILTERS = {
  box: { support: 0.5, f: (x) => (x > -0.5 && x <= 0.5 ? 1 : 0) },
  lanczos: {
    support: 3,
    f: (x) => {
      const sinc = (v) => { if (v === 0) return 1; v *= Math.PI; return Math.sin(v) / v; };
      return -3 <= x && x < 3 ? sinc(x) * sinc(x / 3) : 0;
    },
  },
};

function coeffs(inSize, outSize, filter) {
  const scale = inSize / outSize, filterscale = Math.max(scale, 1);
  const support = filter.support * filterscale, ksize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2), kk = new Float64Array(outSize * ksize);
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale, ss = 1 / filterscale;
    let xmin = Math.trunc(center - support + 0.5); if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5); if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    let ww = 0;
    for (let x = 0; x < xmax; x++) { const w = filter.f((x + xmin - center + 0.5) * ss); kk[xx * ksize + x] = w; ww += w; }
    for (let x = 0; x < xmax; x++) if (ww !== 0) kk[xx * ksize + x] /= ww;
    bounds[xx * 2] = xmin; bounds[xx * 2 + 1] = xmax;
  }
  // to fixed point, as normalize_coeffs_8bpc
  const ik = new Float64Array(kk.length);
  for (let i = 0; i < kk.length; i++) ik[i] = kk[i] < 0 ? Math.trunc(-0.5 + kk[i] * (1 << PRECISION_BITS)) : Math.trunc(0.5 + kk[i] * (1 << PRECISION_BITS));
  return { ksize, bounds, k: ik };
}
const clip8 = (v) => (v >= 2 ** (PRECISION_BITS + 8) ? 255 : v <= 0 ? 0 : Math.floor(v / (1 << PRECISION_BITS)));

/** Resize an 8-bit grey image (w x h, row-major) to nw x nh like Pillow's Image.resize. */
export function resizeL(src, w, h, nw, nh, filterName) {
  if (nw === w && nh === h) return Uint8Array.from(src);
  const filter = FILTERS[filterName];
  const H = coeffs(w, nw, filter), V = coeffs(h, nh, filter);
  const needH = nw !== w, needV = nh !== h;
  const yFirst = V.bounds[0], yLast = V.bounds[nh * 2 - 2] + V.bounds[nh * 2 - 1];
  let img = src, iw = w;
  const vb = Int32Array.from(V.bounds);
  if (needH) {
    for (let i = 0; i < nh; i++) vb[i * 2] -= yFirst;
    const rows = yLast - yFirst, tmp = new Uint8Array(nw * rows);
    for (let yy = 0; yy < rows; yy++) {
      const row = (yy + yFirst) * w;
      for (let xx = 0; xx < nw; xx++) {
        const xmin = H.bounds[xx * 2], xmax = H.bounds[xx * 2 + 1], o = xx * H.ksize;
        let ss = 1 << (PRECISION_BITS - 1);
        for (let x = 0; x < xmax; x++) ss += src[row + x + xmin] * H.k[o + x];
        tmp[yy * nw + xx] = clip8(ss);
      }
    }
    img = tmp; iw = nw;
  }
  if (!needV) return img;
  const out = new Uint8Array(iw * nh);
  for (let yy = 0; yy < nh; yy++) {
    const ymin = vb[yy * 2], ymax = vb[yy * 2 + 1], o = yy * V.ksize;
    for (let xx = 0; xx < iw; xx++) {
      let ss = 1 << (PRECISION_BITS - 1);
      for (let y = 0; y < ymax; y++) ss += img[(y + ymin) * iw + xx] * V.k[o + y];
      out[yy * iw + xx] = clip8(ss);
    }
  }
  return out;
}

const roundHalfEven = (x) => { const r = Math.round(x); return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r; };
const f32 = Math.fround;

// scipy.ndimage.shift(order=1, mode="constant"): linear interpolation inside the image, 0 outside.
function shiftLinear(a, H, W, dy, dx) {
  const out = new Float32Array(H * W);
  const axis = (o, s, len) => {
    const c = o - s;
    if (c < 0 || c > len - 1) return null;
    const i0 = Math.floor(c), t = c - i0;
    return [i0, Math.min(i0 + 1, len - 1), 1 - t, t];
  };
  for (let y = 0; y < H; y++) {
    const ay = axis(y, dy, H); if (!ay) continue;
    for (let x = 0; x < W; x++) {
      const ax = axis(x, dx, W); if (!ax) continue;
      out[y * W + x] = ay[2] * (ax[2] * a[ay[0] * W + ax[0]] + ax[3] * a[ay[0] * W + ax[1]])
                     + ay[3] * (ax[2] * a[ay[1] * W + ax[0]] + ax[3] * a[ay[1] * W + ax[1]]);
    }
  }
  return out;
}

/** A drawing (8-bit grey, white ink on black, w x h) -> 28x28 MNIST-style image in [0, 1], or null if empty.
 *  Same recipe as server.preprocess: fit into 20x20 (keeping aspect), centre of mass to the centre. */
export function preprocessDigit(gray, w, h) {
  let y0 = h, y1 = -1, x0 = w, x1 = -1;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (f32(gray[y * w + x] / 255) > f32(0.1)) { if (y < y0) y0 = y; if (y > y1) y1 = y; if (x < x0) x0 = x; if (x > x1) x1 = x; }
  }
  if (y1 < 0) return null;
  const ch = y1 - y0 + 1, cw = x1 - x0 + 1, crop = new Uint8Array(ch * cw);
  for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) crop[y * cw + x] = Math.trunc(f32(f32(gray[(y + y0) * w + x + x0] / 255) * 255));
  const s = 20 / Math.max(ch, cw);
  const nw = Math.max(1, roundHalfEven(cw * s)), nh = Math.max(1, roundHalfEven(ch * s));
  const small = resizeL(crop, cw, ch, nw, nh, "lanczos");
  const out = new Float32Array(784), top = (28 - nh) >> 1, left = (28 - nw) >> 1;
  for (let y = 0; y < nh; y++) for (let x = 0; x < nw; x++) out[(top + y) * 28 + left + x] = f32(small[y * nw + x] / 255);
  let m = 0, my = 0, mx = 0;
  for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) { const v = out[y * 28 + x]; m += v; my += v * y; mx += v * x; }
  if (!(m > 0)) return null;
  const r = shiftLinear(out, 28, 28, 14 - my / m, 14 - mx / m);
  for (let i = 0; i < r.length; i++) r[i] = Math.min(1, Math.max(0, r[i]));
  return r;
}

// scipy.ndimage.gaussian_filter(sigma, truncate=4, mode="reflect"), float32 between the two passes.
function gaussian1d(sigma) {
  const radius = Math.trunc(4 * sigma + 0.5), w = [];
  let sum = 0;
  for (let x = -radius; x <= radius; x++) { const v = Math.exp(-0.5 * x * x / (sigma * sigma)); w.push(v); sum += v; }
  return { radius, w: w.map((v) => v / sum) };
}
const reflect = (i, n) => { while (i < 0 || i >= n) i = i < 0 ? -i - 1 : 2 * n - i - 1; return i; };
function gaussianFilter(a, H, W, sigma) {
  const { radius, w } = gaussian1d(sigma), tmp = new Float32Array(H * W), out = new Float32Array(H * W);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {       // axis 0 (rows) first, like scipy
    let s = 0; for (let k = -radius; k <= radius; k++) s += w[k + radius] * a[reflect(y + k, H) * W + x];
    tmp[y * W + x] = s;
  }
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let s = 0; for (let k = -radius; k <= radius; k++) s += w[k + radius] * tmp[y * W + reflect(x + k, W)];
    out[y * W + x] = s;
  }
  return out;
}
function percentile(a, q) {   // numpy's default (linear) method
  const s = Float64Array.from(a).sort(), pos = (q / 100) * (s.length - 1), i = Math.floor(pos), t = pos - i;
  if (i + 1 >= s.length) return s[i];
  return t >= 0.5 ? s[i + 1] - (s[i + 1] - s[i]) * (1 - t) : s[i] + (s[i + 1] - s[i]) * t;
}

export const CAM_SHAPE = [36, 48];
/** A webcam frame (8-bit grey, w x h) -> local-contrast image (36 x 48, values in [0, 1]), as camera.preprocess_frame. */
export function preprocessFrame(gray, w, h) {
  const [H, W] = CAM_SHAPE;
  const small = resizeL(gray, w, h, W, H, "box"), g = new Float32Array(H * W);
  for (let i = 0; i < g.length; i++) g[i] = f32(small[i] / 255);
  const blur = gaussianFilter(g, H, W, 2), c = new Float32Array(H * W);
  for (let i = 0; i < c.length; i++) c[i] = Math.abs(g[i] - blur[i]);
  const d = percentile(c, 99) + 1e-3, out = new Float32Array(H * W);
  for (let i = 0; i < c.length; i++) out[i] = Math.min(1, Math.max(0, c[i] / d));
  return out;
}

/** Grey pixels of a canvas. */
export function canvasGray(canvas) {
  const { width: w, height: h } = canvas;
  return { gray: rgbaToL(canvas.getContext("2d").getImageData(0, 0, w, h).data, w * h), w, h };
}
