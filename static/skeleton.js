// Stick figure from body pose landmarks: what the fly sees in the "skeleton" gesture mode.
// Same spec (version 2: body + magnified hands) as flymnist/skeleton.py; keep the two identical (see the spec there).
//   input  33 MediaPipe Pose landmarks [x, y, visibility], x/y in pixels of the UNMIRRORED camera image
//   output Float32Array(36 * 48), row-major, values in [0, 1], fed to the brain directly (no local contrast)

export const SPEC_VERSION = 2;
export const H = 36, W = 48;          // camera IMG_SHAPE
export const S = 9.0;                 // pixels per shoulder width
export const CX = 23.5, CY = 22.0;    // where the shoulder midpoint lands (column, row)
export const HALF = 1.5;              // line half-width, pixels (thick: thin lines barely drive the brain)
export const HEAD_R = 2.7;            // head ring radius, pixels
export const VIS = 0.5;               // landmarks below this visibility are left out
export const RATE_HZ = 1000;          // input firing rate of a white pixel for stick figures
export const HAND_MAX_DIST = 1.0;     // a hand farther than this (x shoulder width) from its pose wrist is ignored
const NOSE = 0, L_SH = 11, R_SH = 12;
export const SEGMENTS = [[11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16],
  [15, 17], [15, 19], [15, 21], [17, 19],
  [16, 18], [16, 20], [16, 22], [18, 20]];

/** MediaPipe result landmarks (normalized x, y in [0, 1]) -> [x px, y px, visibility] per the spec. */
export function toPixels(landmarks, width, height) {
  return landmarks.map((p) => [p.x * width, p.y * height, p.visibility ?? 0]);
}

/** 33 landmarks [x, y, visibility] -> (33) image positions [col, row], or null if the shoulders are missing. */
export function normalise(lm) {
  if (lm[L_SH][2] < VIS || lm[R_SH][2] < VIS) return null;
  const cx = (lm[L_SH][0] + lm[R_SH][0]) / 2, cy = (lm[L_SH][1] + lm[R_SH][1]) / 2;
  const w = Math.hypot(lm[L_SH][0] - lm[R_SH][0], lm[L_SH][1] - lm[R_SH][1]);
  if (!(w > 1e-6)) return null;
  return lm.map((p) => [CX + (p[0] - cx) / w * S, CY + (p[1] - cy) / w * S]);
}

function segDist(px, py, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1], L2 = dx * dx + dy * dy;
  const t = L2 === 0 ? 0 : Math.min(Math.max(((px - a[0]) * dx + (py - a[1]) * dy) / L2, 0), 1);
  return Math.hypot(px - (a[0] + t * dx), py - (a[1] + t * dy));
}
const clamp01 = (v) => Math.min(Math.max(v, 0), 1);

/** Detected hands (each Array(21) of [x, y], any order) -> {left, right} by the nearest pose wrist
 *  (15 = the person's left, 16 = right). Two hands: the pairing with the smaller total wrist distance
 *  (ties: first hand left). A hand farther than HAND_MAX_DIST shoulder widths from its wrist is dropped. */
export function assignHands(pose, hands) {
  const out = { left: null, right: null };
  if (!pose || !hands || !hands.length) return out;
  const sw = Math.hypot(pose[L_SH][0] - pose[R_SH][0], pose[L_SH][1] - pose[R_SH][1]);
  const d = (hand, k) => Math.hypot(hand[0][0] - pose[k][0], hand[0][1] - pose[k][1]);
  let pairs;
  if (hands.length === 1) pairs = d(hands[0], 15) <= d(hands[0], 16) ? [[hands[0], 15]] : [[hands[0], 16]];
  else {
    const [a, b] = hands;
    pairs = d(a, 15) + d(b, 16) <= d(a, 16) + d(b, 15) ? [[a, 15], [b, 16]] : [[a, 16], [b, 15]];
  }
  for (const [hand, k] of pairs) if (d(hand, k) <= HAND_MAX_DIST * sw) out[k === 15 ? "left" : "right"] = hand;
  return out;
}

// ---- hands (spec 2): each hand magnified in its own box in a bottom corner ----
export const HAND_BONES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]];
export const INSET = 12;                                  // inset box size, pixels
export const INSET_COL = { right: 0, left: W - INSET };   // the person's right hand on the image left, as the body
export const INSET_ROW = H - INSET;
export const PALM_PX = 3.5;                               // pixels per palm length (wrist 0 -> middle-finger base 9)
export const HAND_HALF = 0.5;                             // finger line half-width, pixels

const inBox = (k, i, j) => j >= INSET_COL[k] && j < INSET_COL[k] + INSET && i >= INSET_ROW;

/** 33 landmarks [x, y, visibility] (pixels, unmirrored) [+ hands {left, right}: Array(21) of [x, y] | null]
 *  -> Float32Array(36 * 48) in [0, 1]. render(pose) alone = no hands drawn (the corner boxes stay empty). */
export function render(lm, hands = null) {
  const img = new Float64Array(H * W);
  const p = lm ? normalise(lm) : null;
  if (!p) return Float32Array.from(img);
  const shapes = SEGMENTS.filter(([a, b]) => lm[a][2] >= VIS && lm[b][2] >= VIS);
  const head = lm[NOSE][2] >= VIS;
  for (let i = 0; i < H; i++) {
    for (let j = 0; j < W; j++) {
      if (inBox("left", i, j) || inBox("right", i, j)) continue;   // the corner boxes belong to the hands
      let v = 0;
      for (const [a, b] of shapes) v = Math.max(v, clamp01(HALF + 0.5 - segDist(j, i, p[a], p[b])));
      if (head) v = Math.max(v, clamp01(HALF + 0.5 - Math.abs(Math.hypot(j - p[NOSE][0], i - p[NOSE][1]) - HEAD_R)));
      img[i * W + j] = v;
    }
  }
  for (const k of ["left", "right"]) {
    const q = hands && hands[k];
    if (!q) continue;
    const palm = Math.hypot(q[9][0] - q[0][0], q[9][1] - q[0][1]);
    if (!(palm > 1e-6)) continue;
    let sx = 0, sy = 0;
    for (const pt of q) { sx += pt[0]; sy += pt[1]; }
    const cx = sx / 21, cy = sy / 21;
    const ox = INSET_COL[k] + (INSET - 1) / 2, oy = INSET_ROW + (INSET - 1) / 2;
    const pts = q.map((pt) => [ox + (pt[0] - cx) / palm * PALM_PX, oy + (pt[1] - cy) / palm * PALM_PX]);
    for (let i = INSET_ROW; i < H; i++) {
      for (let j = INSET_COL[k]; j < INSET_COL[k] + INSET; j++) {
        let v = img[i * W + j];
        for (const [a, b] of HAND_BONES) v = Math.max(v, clamp01(HAND_HALF + 0.5 - segDist(j, i, pts[a], pts[b])));
        img[i * W + j] = v;
      }
    }
  }
  return Float32Array.from(img);
}
