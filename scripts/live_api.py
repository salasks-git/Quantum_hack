"""Live mode for the 3D tour (pure logic, no HTTP; routes live in scripts/qpu_server.py).

- start_train(body): retrain all models with user-chosen settings in a background thread.
- train_status(run_id) / run_json(run_id): poll progress, fetch the tour JSON when done.
- predict(body): classify a custom patient given n_qubits angles in [0, pi].

Never contacts IBM. Settings ranges mirror the Streamlit sidebar (qmed/ui/sidebar.py).
Default-run strategy: predict({slug, angles}) without run_id lazily trains the run with the
shipped static JSON's settings (STATIC_SETTINGS; 202 while training) so it is predictable too.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
for p in (str(ROOT), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import export_tour_data as etd  # noqa: E402
from qmed.config import (  # noqa: E402
    BANDWIDTH_OPTIONS, DATASETS, DEFAULT_BANDWIDTH, ENTANGLEMENT_OPTIONS, FEATURE_MAP_OPTIONS, QUBIT_OPTIONS,
)

SLUGS = {Path(p).stem: label for label, p in DATASETS.items()}
CSV = {Path(p).stem: Path(p) for p in DATASETS.values()}
# Settings of the shipped web/public/data/<slug>.json (export_tour_data defaults).
STATIC_SETTINGS = {"max_samples": etd.MAX_SAMPLES, "n_qubits": etd.N_QUBITS, "feature_map": etd.FEATURE_MAP_ZZ,
                   "reps": etd.REPS, "entanglement": etd.ENTANGLEMENT, "vqc_iter": etd.VQC_ITER,
                   "kernel_bandwidth": float(etd.KERNEL_BANDWIDTH)}
# Values used for fields missing from POST /api/live/train (Streamlit sidebar defaults; the
# sidebar's bandwidth default differs from the static files' 1.0, so clients should send it).
DEFAULTS = {**STATIC_SETTINGS, "kernel_bandwidth": float(DEFAULT_BANDWIDTH)}
# Exactly the Streamlit sidebar ranges.
RANGES = {
    "max_samples": list(range(60, 181, 20)),
    "n_qubits": list(QUBIT_OPTIONS),
    "feature_map": list(FEATURE_MAP_OPTIONS),
    "reps": [1, 2, 3],
    "entanglement": list(ENTANGLEMENT_OPTIONS),
    "vqc_iter": list(range(15, 76, 5)),
    "kernel_bandwidth": [float(b) for b in BANDWIDTH_OPTIONS],
}
ACTIVE_STATES = {"queued", "running"}
CONFIG = {"grid_size": 24, "max_runs": 8}
RUNS: dict = {}
LOCK = threading.RLock()


class LiveError(Exception):
    def __init__(self, code, message, **extra):
        super().__init__(message)
        self.code, self.message, self.extra = code, message, extra


# ------------------------------------------------------------------ settings
def validate_settings(body) -> tuple[str, dict]:
    if not isinstance(body, dict):
        raise LiveError(400, "Body must be a JSON object.")
    unknown = sorted(set(body) - {"slug", *DEFAULTS})
    if unknown:
        raise LiveError(400, f"Unknown field(s): {', '.join(unknown)}.")
    slug = body.get("slug")
    if not isinstance(slug, str) or slug not in SLUGS:
        raise LiveError(404, "Unknown dataset slug.")
    settings = {}
    for key, default in DEFAULTS.items():
        val = body.get(key, default)
        allowed = RANGES[key]
        if isinstance(default, bool) or isinstance(val, bool):
            raise LiveError(400, f"'{key}' must not be a boolean.")
        if isinstance(default, int) and not isinstance(val, int):
            raise LiveError(400, f"'{key}' must be an integer.")
        if isinstance(default, float):
            if not isinstance(val, (int, float)):
                raise LiveError(400, f"'{key}' must be a number.")
            match = [a for a in allowed if abs(a - val) < 1e-9]
            if not match:
                raise LiveError(400, f"'{key}' must be one of {allowed}.")
            val = match[0]
        elif val not in allowed:
            raise LiveError(400, f"'{key}' must be one of {allowed}.")
        settings[key] = val
    return slug, settings


def make_run_id(slug: str, settings: dict) -> str:
    """Stable 12-hex-char id for (slug, settings)."""
    canon = json.dumps({"slug": slug, **{k: settings[k] for k in DEFAULTS}}, sort_keys=True)
    return hashlib.sha1(canon.encode()).hexdigest()[:12]


def default_run_id(slug: str) -> str:
    """run_id of the run matching the shipped static JSON for this slug."""
    return make_run_id(slug, STATIC_SETTINGS)


# ------------------------------------------------------------------ training
def _active_run():
    for rid, rec in RUNS.items():
        if rec["state"] in ACTIVE_STATES:
            return rid
    return None


def _evict():
    """Keep at most CONFIG['max_runs'] finished runs (oldest finished dropped first)."""
    finished = sorted((r["finished"] or 0, rid) for rid, r in RUNS.items() if r["state"] not in ACTIVE_STATES)
    while len(RUNS) > CONFIG["max_runs"] and finished:
        RUNS.pop(finished.pop(0)[1], None)


def start_train(body) -> dict:
    slug, settings = validate_settings(body)
    rid = make_run_id(slug, settings)
    with LOCK:
        rec = RUNS.get(rid)
        if rec is not None and rec["state"] != "error":
            return {"run_id": rid, "state": rec["state"]}
        active = _active_run()
        if active is not None:
            raise LiveError(409, "Another training run is in progress; wait for it to finish.", run_id=active)
        RUNS[rid] = {"run_id": rid, "slug": slug, "settings": settings, "state": "queued", "stage": "queued",
                     "progress": 0.0, "started": time.time(), "finished": None, "error": None,
                     "bundle": None, "json": None, "lock": threading.Lock()}
        _evict()
        threading.Thread(target=_train_worker, args=(rid,), name=f"live-train-{rid}", daemon=True).start()
    return {"run_id": rid, "state": "queued"}


def _train_worker(rid: str) -> None:
    with LOCK:
        rec = RUNS[rid]
        slug, settings = rec["slug"], dict(rec["settings"])

    def progress(stage, frac):
        with LOCK:
            rec.update(state="running", stage=stage, progress=round(float(frac), 3))

    try:
        bundle = etd.train_bundle(SLUGS[slug], CSV[slug], progress=progress, **settings)
        progress("exporting", 0.98)
        out = etd.bundle_to_json(bundle, CONFIG["grid_size"])
        out["settings"]["max_samples"] = settings["max_samples"]
        out["settings"]["kernel_bandwidth"] = settings["kernel_bandwidth"]
        out["custom_run"] = True
        out["run_id"] = rid
        with LOCK:
            rec.update(state="done", stage="done", progress=1.0, bundle=bundle, json=out, finished=time.time())
    except Exception as e:  # noqa: BLE001
        with LOCK:
            rec.update(state="error", error=f"Training failed: {type(e).__name__}: {e}"[:500],
                       finished=time.time())


def _get(rid) -> dict:
    rec = RUNS.get(rid) if isinstance(rid, str) else None
    if rec is None:
        raise LiveError(404, "Unknown run id (runs only live in this server process).")
    return rec


def train_status(rid) -> dict:
    with LOCK:
        rec = _get(rid)
        end = rec["finished"] or time.time()
        out = {"run_id": rid, "state": rec["state"], "stage": rec["stage"], "progress": rec["progress"],
               "elapsed_s": round(end - rec["started"], 1)}
        if rec["error"]:
            out["error"] = rec["error"]
        return out


def run_json(rid) -> dict:
    with LOCK:
        rec = _get(rid)
        if rec["state"] != "done":
            raise LiveError(404, f"Run is not finished (state: {rec['state']}).", state=rec["state"])
        return rec["json"]


# ------------------------------------------------------------------- predict
def _angles(raw, n_qubits: int) -> tuple[np.ndarray, bool]:
    if not isinstance(raw, list) or len(raw) != n_qubits:
        raise LiveError(400, f"'angles' must be a list of {n_qubits} numbers.")
    vals = []
    for v in raw:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise LiveError(400, "'angles' must contain finite numbers.")
        vals.append(float(v))
    x = np.array(vals, dtype=float)
    clamped = np.clip(x, 0.0, np.pi)
    return clamped, bool(np.any(clamped != x))


def predict(body):
    """Returns a dict (200) or (202, dict) while the run is still training."""
    if not isinstance(body, dict):
        raise LiveError(400, "Body must be a JSON object.")
    rid = body.get("run_id")
    if rid is None:
        slug = body.get("slug")
        if not isinstance(slug, str) or slug not in SLUGS:
            raise LiveError(400 if slug is None else 404, "Send 'run_id', or 'slug' for the default-settings run.")
        rid = default_run_id(slug)
        with LOCK:
            rec = RUNS.get(rid)
            if rec is None or rec["state"] == "error":
                start_train({"slug": slug, **STATIC_SETTINGS})
    with LOCK:
        rec = _get(rid)
        state = rec["state"]
        if state in ACTIVE_STATES:
            return 202, {"state": "training", "run_id": rid, "stage": rec["stage"], "progress": rec["progress"]}
        if state == "error":
            raise LiveError(500, rec["error"] or "Training failed.", run_id=rid)
        bundle = rec["bundle"]
    x, clamped = _angles(body.get("angles"), bundle.settings["n_qubits"])
    X = x.reshape(1, -1)
    with rec["lock"]:
        models = {}
        for key, name in etd.MODEL_KEYS.items():
            m = bundle.res[name]["model"]
            pred = int(np.asarray(m.predict(X)).ravel()[0])
            prob = float(np.asarray(m.predict_proba(X))[0, 1])
            models[key] = {"pred": pred, "prob": prob}
        hp = 0.5 * models["rbf"]["prob"] + 0.5 * models["qsvc"]["prob"]   # same as qmed.models
        models["hybrid"] = {"pred": int(hp >= 0.5), "prob": hp}
        krow = np.asarray(bundle.q_kernel.evaluate(x_vec=X, y_vec=bundle.data.X_train_q))[0]
    for m in models.values():
        m["prob"] = float(np.round(m["prob"], 4))
    krow_r = etd._r(krow)
    pca = bundle.angle_scaler.inverse_transform(X)
    std = bundle.pca.inverse_transform(pca)
    raw = bundle.std_scaler.inverse_transform(std)
    return {
        "run_id": rid, "angles": etd._r(x), "clamped": clamped,
        "models": models, "kernel_row": krow_r,
        "partner": int(max(range(len(krow_r)), key=lambda i: krow_r[i])),
        "reconstructed": {"pca": etd._r(pca[0]), "standardized": etd._r(std[0]), "raw": etd._r(raw[0])},
    }
