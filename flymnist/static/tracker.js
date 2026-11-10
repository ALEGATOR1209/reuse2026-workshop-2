// Body and hand tracking in the browser (MediaPipe Tasks Vision), for the stick-figure gesture mode.
//
// CONTRACT (other code builds on it; do not change):
//   const tracker = await createTracker({ hands = true })
//   tracker.detect(videoOrCanvas, tMs) -> {
//     pose:  Array(33) of [x, y, visibility] | null,
//     hands: { left: Array(21) of [x, y] | null, right: Array(21) of [x, y] | null },
//   }
//   tracker.close()
// Coordinates are pixels of the UNMIRRORED frame (as the camera delivers it; a CSS-mirrored preview
// does not matter). left / right = the person's own side. Each detected hand is given to the side whose
// pose wrist is nearest (skeleton.js assignHands, same rule as the Python training data), not by
// MediaPipe's handedness label. No body found -> pose null and no hands.
//
// Models: pinned @mediapipe/tasks-vision from jsdelivr; the official Pose (full) and Hand Landmarker
// models from storage.googleapis.com. (MediaPipe Tasks sends usage metrics to Google; the images
// themselves stay on the device.)

import { assignHands } from "./skeleton.js";

export const TASKS_VERSION = "1.1.1";
const TASKS = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VERSION}`;
const POSE_MODEL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task";
const HAND_MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

export async function createTracker({ hands = true } = {}) {
  const { FilesetResolver, PoseLandmarker, HandLandmarker } = await import(`${TASKS}/vision_bundle.mjs`);
  const fileset = await FilesetResolver.forVisionTasks(`${TASKS}/wasm`);
  const poseTask = await PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: POSE_MODEL }, runningMode: "VIDEO", numPoses: 1,
  });
  const handTask = hands ? await HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: HAND_MODEL }, runningMode: "VIDEO", numHands: 2,
  }) : null;
  let lastT = -Infinity;

  return {
    detect(src, tMs) {
      const t = Math.max(Math.round(tMs), lastT + 1);   // MediaPipe needs strictly increasing timestamps
      lastT = t;
      const w = src.videoWidth || src.width, h = src.videoHeight || src.height;
      let pose = null;
      poseTask.detectForVideo(src, t, (r) => {          // copy inside the callback (results are freed after it)
        if (r.landmarks.length) pose = r.landmarks[0].map((p) => [p.x * w, p.y * h, p.visibility ?? 0]);
      });
      let found = [];
      if (handTask && pose) found = handTask.detectForVideo(src, t).landmarks.map((hand) => hand.map((p) => [p.x * w, p.y * h]));
      return { pose, hands: assignHands(pose, found) };
    },
    close() {
      poseTask.close();
      if (handTask) handTask.close();
    },
  };
}
