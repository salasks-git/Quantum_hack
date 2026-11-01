// Browser client for the local QPU helper (scripts/qpu_server.py, proxied at /api by Vite).
// Never talks to IBM directly; the helper holds the credentials. Submitting is the only call
// that can use real QPU time, and it refuses to run unless the caller passes { confirm: true }.

export const API_BASE = "/api/qpu";
export const FINAL_STATUSES = new Set(["DONE", "ERROR", "CANCELLED"]);
const MAX_CONSECUTIVE_FAILURES = 3;

const doFetch = (url, init) => globalThis.fetch(url, init);

async function readJson(res) {
  try { return await res.json(); } catch { return null; }
}

function abortError() {
  const e = new Error("Polling aborted");
  e.name = "AbortError";
  return e;
}

/**
 * GET /api/qpu/health (local only, never contacts IBM).
 * -> {ok:true, credentials, runtime, jobs_used, jobs_max, max_execution_time_s, shots}
 *  | {ok:false, reason:"server-down"}            fetch failed, or the proxy/static host answered without helper JSON
 *  | {ok:false, reason:"error", message}         the helper answered with an error
 */
export async function health({ base = API_BASE } = {}) {
  let res;
  try {
    res = await doFetch(`${base}/health`, { cache: "no-store" });
  } catch {
    return { ok: false, reason: "server-down" };
  }
  const body = await readJson(res);
  if (!body || typeof body !== "object") return { ok: false, reason: "server-down" };
  if (res.ok && body.ok === true) {
    return {
      ok: true,
      credentials: !!body.credentials,
      runtime: !!body.runtime,
      jobs_used: Number(body.jobs_used) || 0,
      jobs_max: Number(body.jobs_max) || 0,
      max_execution_time_s: body.max_execution_time_s,
      shots: body.shots,
    };
  }
  return { ok: false, reason: "error", message: String(body.error || `Helper answered HTTP ${res.status}`) };
}

/**
 * GET /api/qpu/token: one-time click token (single use, ~30 s). Free, never contacts IBM.
 * Only call it from inside the Confirm click handler.
 * -> {ok:true, token} | {ok:false, error}
 */
export async function token({ base = API_BASE } = {}) {
  let res;
  try {
    res = await doFetch(`${base}/token`, { cache: "no-store" });
  } catch {
    return { ok: false, error: "server-down" };
  }
  const body = await readJson(res);
  if (res.ok && body && typeof body.token === "string" && body.token) return { ok: true, token: body.token };
  return { ok: false, error: String((body && body.error) || `Helper answered HTTP ${res.status}`) };
}

/**
 * POST /api/qpu/submit {slug, patient, confirm:true, run_id?} with header X-QPU-Token.
 * Uses real QPU time unless the helper has a cached result. Never retried.
 * Refuses (no network call) unless opts.confirm === true AND a click token is given.
 * runId: a live Custom run's id, so the helper uses that run's settings and patient pair.
 * -> {kind:"submitted", job_id, backend, pending_jobs, partner, ideal_fidelity, submitted_at}
 *  | {kind:"cached", job_id, hw_fidelity, ideal_fidelity, noise_gap, backend, partner, counts, ...}
 *  | {kind:"busy", job_id, error}      409: another job is in flight
 *  | {kind:"budget", error}            429: per-run job budget used
 *  | {kind:"error", status, error}     400/404/502/503/500
 *  | {kind:"server-down"}
 */
export async function submit(slug, patient, { confirm = false, runId = null, token: tok = null, base = API_BASE } = {}) {
  if (confirm !== true) throw new Error("submit() requires { confirm: true } (real QPU time)");
  if (typeof tok !== "string" || !tok) throw new Error("submit() requires a click token from token()");
  const payload = { slug, patient, confirm: true };
  if (runId) payload.run_id = runId;
  let res;
  try {
    res = await doFetch(`${base}/submit`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-QPU-Token": tok },
      body: JSON.stringify(payload),
    });
  } catch {
    return { kind: "server-down" };
  }
  const body = await readJson(res);
  if (!body || typeof body !== "object") return { kind: "server-down" };
  if (res.ok) return body.cached ? { kind: "cached", ...body } : { kind: "submitted", ...body };
  if (res.status === 409) return { kind: "busy", job_id: body.job_id, error: body.error };
  if (res.status === 429) return { kind: "budget", error: body.error };
  return { kind: "error", status: res.status, error: String(body.error || `HTTP ${res.status}`) };
}

function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal && signal.aborted) return reject(abortError());
    const id = setTimeout(() => { if (signal) signal.removeEventListener("abort", onAbort); resolve(); }, ms);
    function onAbort() { clearTimeout(id); reject(abortError()); }
    if (signal) signal.addEventListener("abort", onAbort, { once: true });
  });
}

/**
 * Poll GET /api/qpu/status?job=<id> every intervalMs until a final status (status calls are free).
 * onUpdate(json) gets every status response. Resolves with the final response:
 *   {status:"DONE", hw_fidelity, ideal_fidelity, noise_gap, ...} | {status:"ERROR"|"CANCELLED", error}
 *   | {status:"UNKNOWN", error}  (helper restarted: 404)
 *   | {status:"SERVER-DOWN", error} after 3 consecutive network/proxy failures.
 * Rejects with an AbortError as soon as `signal` aborts; no fetch happens after that.
 */
export async function pollStatus(jobId, { onUpdate, signal, intervalMs = 5000, base = API_BASE } = {}) {
  let failures = 0;
  for (;;) {
    if (signal && signal.aborted) throw abortError();
    let res = null, body = null;
    try {
      res = await doFetch(`${base}/status?job=${encodeURIComponent(jobId)}`, { cache: "no-store", signal });
      body = await readJson(res);
    } catch (e) {
      if (signal && signal.aborted) throw abortError();
      res = null;
    }
    if (signal && signal.aborted) throw abortError();
    if (res && res.status === 404 && body && body.error) {
      const fin = { job_id: jobId, status: "UNKNOWN", error: String(body.error) };
      if (onUpdate) onUpdate(fin);
      return fin;
    }
    if (res && res.ok && body && typeof body.status === "string") {
      failures = 0;
      if (onUpdate) onUpdate(body);
      if (FINAL_STATUSES.has(body.status)) return body;
    } else {
      failures += 1;
      if (failures >= MAX_CONSECUTIVE_FAILURES) {
        const down = !body || !body.error;
        const fin = {
          job_id: jobId,
          status: down ? "SERVER-DOWN" : "ERROR",
          error: down ? "Lost contact with the QPU helper." : String(body.error),
        };
        if (onUpdate) onUpdate(fin);
        return fin;
      }
    }
    await sleep(intervalMs, signal);
  }
}

/**
 * GET /api/qpu/backends: lists IBM backends and queues (contacts IBM, submits nothing, uses no QPU time).
 * Only ever called from an explicit user click.
 * -> {ok:true, rows:[{name, num_qubits, operational, pending_jobs}] sorted by queue}
 *  | {ok:false, error}
 */
export async function backends({ base = API_BASE } = {}) {
  let res;
  try {
    res = await doFetch(`${base}/backends`, { cache: "no-store" });
  } catch {
    return { ok: false, error: "Start the helper: python scripts/qpu_server.py" };
  }
  const body = await readJson(res);
  if (!res.ok || !Array.isArray(body)) {
    return { ok: false, error: String((body && body.error) || `Helper answered HTTP ${res.status}`) };
  }
  const rows = body.map((b) => ({
    name: String(b.name), num_qubits: Number(b.num_qubits) || 0, operational: !!b.operational,
    pending_jobs: Number.isFinite(Number(b.pending_jobs)) ? Number(b.pending_jobs) : null,
  })).sort((a, b) => (a.pending_jobs ?? Infinity) - (b.pending_jobs ?? Infinity) || a.name.localeCompare(b.name));
  return { ok: true, rows };
}
