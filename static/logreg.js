// The camera decoder's training in the browser, for the static (no server) version of the app.
// Same model as flymnist/camera.py: StandardScaler + LogisticRegression(C=0.05) (L2, multinomial;
// one weight vector for two classes, like scikit-learn), fitted with L-BFGS, plus the same
// 5-fold cross-validation with contiguous blocks of each recording. Optional sample weights (stick
// figures: own recordings weigh more than the pre-trained examples), as scikit-learn's sample_weight:
// weighted mean log-loss, ||W||^2 / (2 C sum(w)), unweighted StandardScaler, weighted CV accuracy.
//
// X is sparse (CSR over the selected features): {n, F, indptr, idx, val}. Standardizing is done
// implicitly (z = (x - mean) / scale), so the data stay sparse.

export const C_DEFAULT = 0.05, FOLDS = 5;

/** Rows `rows` of X as a new CSR matrix. */
export function subRows(X, rows) {
  let nnz = 0;
  for (const r of rows) nnz += X.indptr[r + 1] - X.indptr[r];
  const indptr = new Int32Array(rows.length + 1), idx = new Int32Array(nnz), val = new Float32Array(nnz);
  let p = 0;
  rows.forEach((r, i) => {
    for (let k = X.indptr[r]; k < X.indptr[r + 1]; k++) { idx[p] = X.idx[k]; val[p++] = X.val[k]; }
    indptr[i + 1] = p;
  });
  return { n: rows.length, F: X.F, indptr, idx, val };
}

/** StandardScaler: mean and population std per feature (std 0 -> 1). */
export function fitScaler(X) {
  const { n, F } = X, mean = new Float64Array(F), sq = new Float64Array(F);
  for (let k = 0; k < X.idx.length; k++) { mean[X.idx[k]] += X.val[k]; sq[X.idx[k]] += X.val[k] * X.val[k]; }
  const scale = new Float64Array(F);
  for (let f = 0; f < F; f++) {
    mean[f] /= n;
    const v = Math.max(0, sq[f] / n - mean[f] * mean[f]), s = Math.sqrt(v);
    scale[f] = s < 10 * 2.220446049250313e-16 * Math.max(1, Math.abs(mean[f])) ? 1 : s;
  }
  return { mean, scale };
}

// Objective: (weighted) mean log-loss + ||W||^2 / (2 C sum(w)) (the intercept is not penalised): same minimum as scikit-learn's.
function makeObjective(X, y, K, C, mean, scale, w = null) {
  const { n, F, indptr, idx, val } = X, R = K === 2 ? 1 : K;   // rows of W
  const U = new Float64Array(R * F), S = new Float64Array(n * R), G = new Float64Array(R * F + R);
  let sw = n; if (w) { sw = 0; for (let i = 0; i < n; i++) sw += w[i]; }
  const lam = 1 / (C * sw);
  return function (theta) {
    // logits = c + X U^T, with U = W / scale and c = b - U mean
    const c = new Float64Array(R);
    for (let r = 0; r < R; r++) {
      let s = theta[R * F + r];
      for (let f = 0, o = r * F; f < F; f++) { const u = theta[o + f] / scale[f]; U[o + f] = u; s -= u * mean[f]; }
      c[r] = s;
    }
    for (let i = 0; i < n; i++) {
      for (let r = 0; r < R; r++) S[i * R + r] = c[r];
      for (let k = indptr[i]; k < indptr[i + 1]; k++) { const f = idx[k], v = val[k]; for (let r = 0; r < R; r++) S[i * R + r] += v * U[r * F + f]; }
    }
    // loss and residuals w (p - y) / sum(w), in place in S
    let loss = 0;
    for (let i = 0; i < n; i++) {
      const o = i * R, wi = w ? w[i] : 1;
      if (R === 1) {
        const s = S[o], t = y[i];
        loss += wi * ((s > 0 ? s + Math.log1p(Math.exp(-s)) : Math.log1p(Math.exp(s))) - t * s);
        S[o] = wi * (1 / (1 + Math.exp(-s)) - t) / sw;
      } else {
        let mx = -Infinity; for (let r = 0; r < R; r++) mx = Math.max(mx, S[o + r]);
        let sum = 0; for (let r = 0; r < R; r++) sum += Math.exp(S[o + r] - mx);
        const lse = mx + Math.log(sum);
        loss += wi * (lse - S[o + y[i]]);
        for (let r = 0; r < R; r++) S[o + r] = wi * (Math.exp(S[o + r] - lse) - (r === y[i] ? 1 : 0)) / sw;
      }
    }
    loss /= sw;
    // gradient: dW = diag(1/scale) (X^T Res - mean * sum(Res)) + lam W ; db = sum(Res)
    G.fill(0);
    const colSum = new Float64Array(R);
    for (let i = 0; i < n; i++) {
      for (let r = 0; r < R; r++) colSum[r] += S[i * R + r];
      for (let k = indptr[i]; k < indptr[i + 1]; k++) { const f = idx[k], v = val[k]; for (let r = 0; r < R; r++) G[r * F + f] += v * S[i * R + r]; }
    }
    let reg = 0;
    for (let r = 0; r < R; r++) {
      for (let f = 0, o = r * F; f < F; f++) {
        const w = theta[o + f];
        G[o + f] = (G[o + f] - mean[f] * colSum[r]) / scale[f] + lam * w;
        reg += w * w;
      }
      G[R * F + r] = colSum[r];
    }
    return { f: loss + 0.5 * lam * reg, g: G };
  };
}

// Limited-memory BFGS with a backtracking (Armijo) line search.
function lbfgs(fg, x, { m = 10, maxIter = 1000, gtol = 1e-5 } = {}) {
  const N = x.length, S = [], Y = [], rho = [];
  let { f, g } = fg(x); g = Float64Array.from(g);
  const d = new Float64Array(N), xNew = new Float64Array(N);
  let it = 0;
  for (; it < maxIter; it++) {
    let gmax = 0; for (let i = 0; i < N; i++) gmax = Math.max(gmax, Math.abs(g[i]));
    if (gmax < gtol) break;
    // two-loop recursion: d = -H g
    d.set(g);
    const alpha = new Array(S.length);
    for (let j = S.length - 1; j >= 0; j--) { let a = 0; for (let i = 0; i < N; i++) a += S[j][i] * d[i]; a *= rho[j]; alpha[j] = a; for (let i = 0; i < N; i++) d[i] -= a * Y[j][i]; }
    if (S.length) {
      const j = S.length - 1; let sy = 0, yy = 0;
      for (let i = 0; i < N; i++) { sy += S[j][i] * Y[j][i]; yy += Y[j][i] * Y[j][i]; }
      const h = sy / yy; for (let i = 0; i < N; i++) d[i] *= h;
    } else { let gn = 0; for (let i = 0; i < N; i++) gn += g[i] * g[i]; const h = 1 / Math.max(1, Math.sqrt(gn)); for (let i = 0; i < N; i++) d[i] *= h; }
    for (let j = 0; j < S.length; j++) { let b = 0; for (let i = 0; i < N; i++) b += Y[j][i] * d[i]; b *= rho[j]; for (let i = 0; i < N; i++) d[i] += S[j][i] * (alpha[j] - b); }
    for (let i = 0; i < N; i++) d[i] = -d[i];
    let gd = 0; for (let i = 0; i < N; i++) gd += g[i] * d[i];
    if (gd >= 0) { for (let i = 0; i < N; i++) d[i] = -g[i]; gd = 0; for (let i = 0; i < N; i++) gd -= g[i] * g[i]; S.length = Y.length = rho.length = 0; }
    let step = 1, res = null;
    for (let ls = 0; ls < 40; ls++) {
      for (let i = 0; i < N; i++) xNew[i] = x[i] + step * d[i];
      res = fg(xNew);
      if (res.f <= f + 1e-4 * step * gd) break;
      step *= 0.5;
    }
    const s = new Float64Array(N), yv = new Float64Array(N);
    let sy = 0;
    for (let i = 0; i < N; i++) { s[i] = xNew[i] - x[i]; yv[i] = res.g[i] - g[i]; sy += s[i] * yv[i]; }
    const fOld = f;
    x.set(xNew); f = res.f; g.set(res.g);
    if (sy > 1e-12) { S.push(s); Y.push(yv); rho.push(1 / sy); if (S.length > m) { S.shift(); Y.shift(); rho.shift(); } }
    if (Math.abs(fOld - f) <= 1e-12 * Math.max(1, Math.abs(f))) break;
  }
  return { x, f, iterations: it };
}

/** Fit StandardScaler + logistic regression. labels: action index per row; w: optional weight per row.
 *  Returns the decoder as plain weights. */
export function fitDecoder(X, labels, C = C_DEFAULT, w = null) {
  const classes = [...new Set(labels)].sort((a, b) => a - b), K = classes.length;
  const y = Int32Array.from(labels, (l) => (K === 2 ? (l === classes[1] ? 1 : 0) : classes.indexOf(l)));
  const { mean, scale } = fitScaler(X), R = K === 2 ? 1 : K, F = X.F;
  const { x } = lbfgs(makeObjective(X, y, K, C, mean, scale, w), new Float64Array(R * F + R));
  return { n_features: F, mean: Float32Array.from(mean), scale: Float32Array.from(scale),
           coef: Float32Array.from(x.subarray(0, R * F)), intercept: Array.from(x.subarray(R * F)), classes };
}

/** Most likely class of every row. */
export function predict(d, X) {
  const F = d.n_features, R = d.coef.length / F, out = new Int32Array(X.n), base = Float64Array.from(d.intercept);
  for (let r = 0; r < R; r++) for (let f = 0; f < F; f++) base[r] -= d.coef[r * F + f] * d.mean[f] / d.scale[f];
  for (let i = 0; i < X.n; i++) {
    const s = Float64Array.from(base);
    for (let k = X.indptr[i]; k < X.indptr[i + 1]; k++) { const f = X.idx[k]; for (let r = 0; r < R; r++) s[r] += d.coef[r * F + f] * X.val[k] / d.scale[f]; }
    if (R === 1) out[i] = s[0] > 0 ? d.classes[1] : d.classes[0];
    else { let b = 0; for (let r = 1; r < R; r++) if (s[r] > s[b]) b = r; out[i] = d.classes[b]; }
  }
  return out;
}

/** Folds that hold out a contiguous block of every action, or of every group if given (camera.train_decoders). */
export function makeFolds(labels, k = FOLDS, groups = null) {
  const folds = new Int32Array(labels.length), by = new Map();
  labels.forEach((l, i) => { const g = groups ? groups[i] : l; if (!by.has(g)) by.set(g, []); by.get(g).push(i); });
  for (const idx of by.values()) idx.forEach((i, j) => (folds[i] = Math.min(Math.floor((j * k) / idx.length), k - 1)));
  return folds;
}

/** Accuracy of fold k (trained on the other folds): {acc} or null if the fold can't be scored. */
export function cvFold(X, labels, k, { C = C_DEFAULT, weights = null, groups = null } = {}) {
  const folds = makeFolds(labels, FOLDS, groups), tr = [], te = [];
  labels.forEach((_, i) => (folds[i] === k ? te : tr).push(i));
  const trLabels = tr.map((i) => labels[i]);
  if (new Set(trLabels).size < 2 || !te.length) return null;
  const d = fitDecoder(subRows(X, tr), trLabels, C, weights ? tr.map((i) => weights[i]) : null), p = predict(d, subRows(X, te));
  let ok = 0, tot = 0;
  te.forEach((i, j) => { const wi = weights ? weights[i] : 1; ok += wi * (p[j] === labels[i]); tot += wi; });
  return { acc: ok / tot };
}

/** Cross-validate and fit one decoder. onFit() is called after each of the FOLDS + 1 fits.
 *  weights: optional weight per row (fits and CV accuracy); groups: blocks for the folds instead of the labels. */
export function trainWithCV(X, labels, { C = C_DEFAULT, onFit = () => {}, weights = null, groups = null } = {}) {
  const folds = makeFolds(labels, FOLDS, groups), correct = [];
  for (let k = 0; k < FOLDS; k++) {
    const tr = [], te = [];
    labels.forEach((_, i) => (folds[i] === k ? te : tr).push(i));
    const trLabels = tr.map((i) => labels[i]);
    if (new Set(trLabels).size >= 2 && te.length) {
      const d = fitDecoder(subRows(X, tr), trLabels, C, weights ? tr.map((i) => weights[i]) : null), p = predict(d, subRows(X, te));
      let ok = 0, tot = 0;
      te.forEach((i, j) => { const wi = weights ? weights[i] : 1; ok += wi * (p[j] === labels[i]); tot += wi; });
      correct.push(ok / tot);
    }
    onFit();
  }
  const model = fitDecoder(X, labels, C, weights);
  onFit();
  return { model, cv_accuracy: correct.length ? correct.reduce((a, b) => a + b, 0) / correct.length : null };
}
