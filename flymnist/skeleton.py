"""Stick figure from body pose landmarks: what the fly sees in the "skeleton" gesture mode.

Instead of the camera picture, the fly's eyes get a drawing of the person's upper-body pose
(MediaPipe Pose landmarks), so the background, clothes, light and the person's size and place in
the frame no longer matter. flymnist/static/skeleton.js implements the SAME spec for the browser;
keep the two identical (tests compare them).

SPEC (version 2 = version 1 + magnified hands)
  input   33 MediaPipe Pose landmarks [x, y, visibility]; x, y in pixels of the UNMIRRORED camera image
          (equal scale on both axes: normalized x * image width, y * image height); optionally the hands
          {left, right} (the person's sides), each 21 Hand Landmarker points [x, y] in the same pixels or None.
          assign_hands gives each detected hand to the side of the nearest pose wrist (15 left, 16 right),
          the pairing with the smaller total distance, dropping hands > HAND_MAX_DIST shoulder widths away.
  frame   c = midpoint of the shoulders (11, 12), w = distance between the shoulders. A landmark at
          (x, y) is drawn at column CX + (x - c.x) / w * S and row CY + (y - c.y) / w * S
          (pixel (row i, col j) has its centre at (j, i)). The person's right side stays on the image
          left, as the brain sees camera frames. No shoulders (visibility < VIS) or w < 1e-6: blank image.
  draw    SEGMENTS: lines of half-width HALF px between landmarks whose visibilities are both >= VIS;
          the head: a ring of radius HEAD_R px around the nose (0) if visible.
          value at a pixel = max over shapes of clamp(HALF + 0.5 - d, 0, 1), where d = distance from the
          pixel centre to the segment (to the ring: |distance to the nose - HEAD_R|).
  hands   two INSET x INSET boxes in the bottom corners belong to the hands (the body is not drawn there):
          the person's right hand in the bottom-left box, the left hand bottom-right. A hand is drawn in its
          box: centre of its 21 points (sequential sums / 21) at the box centre, scale PALM_PX px per palm
          length (point 0 -> 9), rotation kept; HAND_BONES as lines of half-width HAND_HALF, clipped to the box.
  output  IMG_SHAPE (36 x 48) float32 in [0, 1], fed to the brain directly (no local-contrast step:
          the drawing already consists of edges), at RATE_HZ for a white pixel (camera pictures: 500 Hz).
          Thick lines + 1000 Hz drive the descending neurons about as strongly as camera pictures do.
"""

import numpy as np

SPEC_VERSION = 2
H, W = 36, 48                      # camera.IMG_SHAPE
S = 9.0                            # pixels per shoulder width
CX, CY = 23.5, 22.0                # where the shoulder midpoint lands (column, row)
HALF = 1.5                         # line half-width, pixels (thick: thin lines barely drive the brain)
HEAD_R = 2.7                       # head ring radius, pixels
VIS = 0.5                          # landmarks below this visibility are left out
RATE_HZ = 1000.0                   # input firing rate of a white pixel for stick figures
HAND_MAX_DIST = 1.0                # a hand farther than this (x shoulder width) from its pose wrist is ignored
NOSE, L_SH, R_SH = 0, 11, 12
SEGMENTS = [(11, 12),                                   # shoulders
            (11, 13), (13, 15), (12, 14), (14, 16),     # upper arms, forearms (MediaPipe: 11/13/15 = person's left)
            (15, 17), (15, 19), (15, 21), (17, 19),     # left hand: wrist-pinky, wrist-index, wrist-thumb, pinky-index
            (16, 18), (16, 20), (16, 22), (18, 20)]     # right hand


def normalise(lm):
    """(33, >=3) landmarks -> (33, 2) image positions per the spec, or None if the shoulders are missing."""
    lm = np.nan_to_num(np.asarray(lm, np.float64))
    if lm[L_SH, 2] < VIS or lm[R_SH, 2] < VIS:
        return None
    c = (lm[L_SH, :2] + lm[R_SH, :2]) / 2
    w = np.hypot(*(lm[L_SH, :2] - lm[R_SH, :2]))
    if not w > 1e-6:
        return None
    return np.stack([CX + (lm[:, 0] - c[0]) / w * S, CY + (lm[:, 1] - c[1]) / w * S], 1)


def _seg_dist(px, py, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    t = np.zeros_like(px) if L2 == 0 else np.clip(((px - a[0]) * dx + (py - a[1]) * dy) / L2, 0, 1)
    return np.hypot(px - (a[0] + t * dx), py - (a[1] + t * dy))


def assign_hands(pose, hands):
    """Detected hands (each (21, >=2) [x, y], any order) -> {"left", "right"} by the nearest pose wrist
    (15 = the person's left, 16 = right); same rule as skeleton.js assignHands."""
    out = {"left": None, "right": None}
    if pose is None or not hands:
        return out
    pose = np.asarray(pose, np.float64)
    sw = np.hypot(pose[L_SH, 0] - pose[R_SH, 0], pose[L_SH, 1] - pose[R_SH, 1])
    d = lambda hand, k: float(np.hypot(hand[0][0] - pose[k, 0], hand[0][1] - pose[k, 1]))
    if len(hands) == 1:
        pairs = [(hands[0], 15)] if d(hands[0], 15) <= d(hands[0], 16) else [(hands[0], 16)]
    else:
        a, b = hands[0], hands[1]
        pairs = [(a, 15), (b, 16)] if d(a, 15) + d(b, 16) <= d(a, 16) + d(b, 15) else [(a, 16), (b, 15)]
    for hand, k in pairs:
        if d(hand, k) <= HAND_MAX_DIST * sw:
            out["left" if k == 15 else "right"] = hand
    return out


# ---- hands (spec 2) ----
HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
POSE_HAND = {"left": [(15, 17), (15, 19), (15, 21), (17, 19)], "right": [(16, 18), (16, 20), (16, 22), (18, 20)]}
INSET = 12                         # inset box size, pixels (bottom corners)
INSET_COL = {"right": 0, "left": W - INSET}   # the person's right hand on the image left, as the body
INSET_ROW = H - INSET
PALM_PX = 3.5                      # inset scale: pixels per palm length (wrist 0 -> middle-finger base 9)
HAND_HALF = 0.5                    # finger line half-width in the inset, pixels
INPLACE_HALF = 0.6                 # finger line half-width when drawn in place (experiment)
LAYOUT = "inset"                   # the spec; "inplace" (fingers at body scale) / "none" only for experiments


def _lines(img, px, py, pts, bones, half, mask=None):
    for a, b in bones:
        v = np.clip(half + 0.5 - _seg_dist(px, py, pts[a], pts[b]), 0, 1)
        img = np.maximum(img, v if mask is None else v * mask)
    return img


def render(lm, hands=None, layout=None):
    """33 landmarks [x, y, visibility] (pixels, unmirrored camera image) [+ hands {"left", "right"}: (21, 2)
    pixel landmarks or None] -> (36, 48) stick figure in [0, 1]."""
    layout = layout or LAYOUT
    lm = np.nan_to_num(np.asarray(lm, np.float64))      # missing landmarks (NaN) count as invisible
    img = np.zeros((H, W))
    p = normalise(lm)
    if p is None:
        return img.astype(np.float32)
    hands = hands or {}
    has = {k: hands.get(k) is not None for k in ("left", "right")}
    py, px = np.mgrid[0:H, 0:W].astype(np.float64)
    skip = set()
    if layout == "inplace":
        skip = {seg for k in ("left", "right") if has[k] for seg in POSE_HAND[k]}
    for a, b in SEGMENTS:
        if (a, b) not in skip and lm[a, 2] >= VIS and lm[b, 2] >= VIS:
            img = np.maximum(img, np.clip(HALF + 0.5 - _seg_dist(px, py, p[a], p[b]), 0, 1))
    if lm[NOSE, 2] >= VIS:
        d = np.abs(np.hypot(px - p[NOSE, 0], py - p[NOSE, 1]) - HEAD_R)
        img = np.maximum(img, np.clip(HALF + 0.5 - d, 0, 1))
    if layout == "inplace":
        c = (lm[L_SH, :2] + lm[R_SH, :2]) / 2
        w = np.hypot(*(lm[L_SH, :2] - lm[R_SH, :2]))
        for k in ("left", "right"):
            if has[k]:
                q = np.asarray(hands[k], np.float64)[:, :2]
                img = _lines(img, px, py, np.stack([CX + (q[:, 0] - c[0]) / w * S, CY + (q[:, 1] - c[1]) / w * S], 1),
                             HAND_BONES, INPLACE_HALF)
    elif layout == "inset":
        for k in ("left", "right"):
            box = (px >= INSET_COL[k] - 0.5) & (px < INSET_COL[k] + INSET - 0.5) & (py >= INSET_ROW - 0.5)
            img = np.where(box, 0.0, img)                  # the corner boxes belong to the hands
            if not has[k]:
                continue
            q = np.asarray(hands[k], np.float64)[:, :2]
            palm = np.hypot(*(q[9] - q[0]))
            if not palm > 1e-6:
                continue
            cen = np.array([sum(float(v) for v in q[:, 0]), sum(float(v) for v in q[:, 1])]) / 21   # sequential sums, as in JS
            o = np.array([INSET_COL[k] + (INSET - 1) / 2, INSET_ROW + (INSET - 1) / 2])
            img = _lines(img, px, py, o + (q - cen) / palm * PALM_PX, HAND_BONES, HAND_HALF, box.astype(np.float64))
    return img.astype(np.float32)
