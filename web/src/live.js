// Client for the local helper's live API (training only; never touches /api/qpu/submit or /api/qpu/backends).
const API = "/api";

async function jsonOf(res) {
  try { return await res.json(); } catch { return {}; }
}
function httpError(res, body, fallback) {
  const e = new Error((body && body.error) || fallback || `HTTP ${res.status}`);
  e.status = res.status;
  e.runId = body && body.run_id;
  return e;
}

// Resolves {up, live}; never throws (helper down => {up:false, live:false}).
export async function liveHealth(fetchFn = fetch) {
  try {
    const res = await fetchFn(`${API}/qpu/health`);
    if (!res.ok) return { up: false, live: false };
    const b = await jsonOf(res);
    return { up: true, live: b.live === true };
  } catch { return { up: false, live: false }; }
}

export async function startTrain(settings, fetchFn = fetch) {
  const res = await fetchFn(`${API}/live/train`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(settings),
  });
  const b = await jsonOf(res);
  if (!res.ok) throw httpError(res, b, "Could not start training");
  return b; // {run_id, state}
}

const sleep = (ms, signal) => new Promise((resolve, reject) => {
  if (signal && signal.aborted) return reject(abortError());
  const t = setTimeout(resolve, ms);
  if (signal) signal.addEventListener("abort", () => { clearTimeout(t); reject(abortError()); }, { once: true });
});
function abortError() { const e = new Error("aborted"); e.name = "AbortError"; return e; }

// Polls until done (resolves final status); rejects on error state, HTTP failure or abort.
export async function pollTrain(runId, { onUpdate, signal, intervalMs = 400, fetchFn = fetch } = {}) {
  for (;;) {
    if (signal && signal.aborted) throw abortError();
    const res = await fetchFn(`${API}/live/train/status?run=${encodeURIComponent(runId)}`, { signal });
    const s = await jsonOf(res);
    if (!res.ok) throw httpError(res, s, "Status request failed");
    if (signal && signal.aborted) throw abortError();
    if (onUpdate) onUpdate(s);
    if (s.state === "done") return s;
    if (s.state === "error") throw new Error(s.error || "Training failed");
    await sleep(intervalMs, signal);
  }
}

export async function loadRun(runId, fetchFn = fetch) {
  const res = await fetchFn(`${API}/live/run/${encodeURIComponent(runId)}.json`);
  const b = await jsonOf(res);
  if (!res.ok) throw httpError(res, b, "Could not load run");
  return b;
}
