"""Progress of the current long-running job (decoder training, knockout evaluation), for the web page to poll."""

import threading
import time


class Progress:
    def __init__(self):
        self._lock = threading.Lock()
        self._s = {"active": False}

    def start(self, task, stage, total, step=1, steps=1):
        """Begin a job, or the next stage of one ('step' of 'steps')."""
        with self._lock:
            now = time.time()
            started = self._s["started"] if self._s.get("active") and self._s.get("task") == task else now
            self._s = {"active": True, "task": task, "stage": stage, "step": step, "steps": steps,
                       "done": 0, "total": max(int(total), 1), "started": started, "stage_started": now}

    def advance(self, n=1):
        with self._lock:
            if self._s.get("active"):
                self._s["done"] = min(self._s["done"] + n, self._s["total"])

    def finish(self):
        with self._lock:
            self._s = {"active": False}

    def get(self):
        with self._lock:
            s = dict(self._s)
        if s["active"]:
            done = s["done"]  # fractional while a batch is half simulated
            now = time.time()
            s["elapsed"] = round(now - s["started"], 1)
            run = now - s["stage_started"]
            s["eta"] = round(run / done * (s["total"] - done), 1) if done > 0.05 * s["total"] or run > 5 and done else None
            s["fraction"] = round(done / s["total"], 4)
            s["done"] = int(done + 1e-6)
            del s["started"], s["stage_started"]
        return s
