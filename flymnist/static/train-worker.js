// Trains camera decoders in a background thread (static version of the app; the server version
// trains on the server). In: {X: {n, F, indptr, idx, val}, labels, cols, weights?, groups?, fold?}.
// Without fold: progress after each of the 6 fits, then {type: "done", decoder, metrics}.
// With fold = k (0..4): one cross-validation fold -> {type: "done", fold: k, cv: {acc} | null};
// fold = -1: the final fit on all rows -> {type: "done", fold: -1, decoder, neurons}. (api.js runs
// the 6 fits of every decoder in a pool of these workers, in parallel.)
import { C_DEFAULT, FOLDS, cvFold, fitDecoder, trainWithCV } from "./logreg.js";

onmessage = (e) => {
  const { X, labels, cols, weights = null, groups = null, fold = null } = e.data;
  try {
    if (fold === null || fold === undefined) {
      let fits = 0;
      const { model, cv_accuracy } = trainWithCV(X, labels, { weights, groups, onFit: () => postMessage({ type: "progress", done: ++fits, total: FOLDS + 1 }) });
      postMessage({ type: "done", decoder: { cols, ...model }, metrics: { cv_accuracy, neurons: X.F } });
    } else if (fold < 0) {
      postMessage({ type: "done", fold, decoder: { cols, ...fitDecoder(X, labels, C_DEFAULT, weights) }, neurons: X.F });
    } else {
      postMessage({ type: "done", fold, cv: cvFold(X, labels, fold, { weights, groups }) });
    }
  } catch (err) {
    postMessage({ type: "error", message: err.message || String(err) });
  }
};
