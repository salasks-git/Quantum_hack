// Scene 6 live-run panel: a small state machine (testable, client injected) + an HTML renderer.
// FREE TIER: the only path to client.submit() is requestRun() -> confirm step -> confirmRun().
// activate()/deactivate() (step changes, patient/disease switches, Play all, page load) only ever
// call health() and pollStatus() (both free); they never submit.
import * as qpuClient from "./qpu.js";

export const RUN_LABEL = "🚀 Run on real IBM QPU";
export const MSG = {
  serverDown: "Start the helper: python scripts/qpu_server.py",
  noCredentials: "No IBM account found: save one with QiskitRuntimeService.save_account(channel='ibm_quantum_platform', …) or set QISKIT_IBM_TOKEN",
  noRuntime: "pip install qiskit-ibm-runtime",
  budget: (used, max) => `QPU budget for this session used (${used}/${max})`,
};
export const confirmText = (shots = 1024, capS = 60) =>
  `This sends 1 circuit × ${shots} shots to a real IBM QPU and uses free-tier time (~seconds, capped at ${capS} s). Continue?`;

const keyOf = (s) => `${s.slug}|${s.patient}`;
const argmax = (row) => {
  let best = -1, bi = -1;
  (row || []).forEach((v, i) => { if (v > best) { best = v; bi = i; } });
  return bi;
};
const isAbort = (e) => e && e.name === "AbortError";

/**
 * Phases: hidden | checking | unavailable | ready | confirm | submitting | running | done | failed
 * selection: {slug, patient, kRow} where kRow = data.qsvc.kernel_test[patient] (scene 3's K row).
 */
export function createQpuController({ client = qpuClient, onChange = () => {}, intervalMs = 5000 } = {}) {
  let active = false;
  let sel = null;
  let gen = 0; // bumps on every activate/deactivate; stale async callbacks compare against it
  let poll = null; // AbortController of the UI polling loop
  const jobs = new Map(); // key -> {job_id, backend, pending_jobs, partner, ideal_fidelity, submitted_at, slug, patient}
  const jobKey = new Map(); // job_id -> key (only jobs this page submitted)
  const results = new Map(); // key -> final result + {slug, patient, cached}
  const st = { phase: "hidden", health: null, messages: [], job: null, status: null, foreign: false, error: "" };

  const emit = () => onChange(view());

  function healthMessages(h) {
    if (!h) return [];
    if (!h.ok) return [h.reason === "server-down" ? MSG.serverDown : `QPU helper error: ${h.message || "unknown"}`];
    const m = [];
    if (!h.credentials) m.push(MSG.noCredentials);
    if (!h.runtime) m.push(MSG.noRuntime);
    if (h.jobs_used >= h.jobs_max) m.push(MSG.budget(h.jobs_used, h.jobs_max));
    return m;
  }
  const canRun = () => !!(st.health && st.health.ok && healthMessages(st.health).length === 0);

  function abortPoll() {
    if (poll) { poll.abort(); poll = null; }
  }

  function settleFromHealth() {
    st.messages = healthMessages(st.health);
    st.phase = canRun() ? "ready" : "unavailable";
  }

  async function checkHealth() {
    const g = gen;
    st.phase = "checking"; st.job = null; st.status = null; st.foreign = false; st.error = "";
    emit();
    const h = await client.health();
    if (g !== gen) return;
    st.health = h;
    settleFromHealth();
    emit();
  }

  async function refreshHealthQuietly() {
    const g = gen;
    const h = await client.health();
    if (g !== gen) return;
    st.health = h;
    emit();
  }

  function startPolling(jobId) {
    abortPoll();
    const g = gen;
    const ac = new AbortController();
    poll = ac;
    client.pollStatus(jobId, {
      signal: ac.signal,
      intervalMs,
      onUpdate: (s) => { if (g === gen && !ac.signal.aborted) { st.status = s; emit(); } },
    }).then((fin) => {
      if (poll === ac) poll = null;
      const key = jobKey.get(jobId);
      const rec = key ? jobs.get(key) : null;
      if (fin.status === "DONE" && rec) {
        results.set(key, { ...rec, ...fin, slug: rec.slug, patient: rec.patient, cached: false });
        jobs.delete(key);
      } else if (rec && fin.status !== "SERVER-DOWN") {
        jobs.delete(key); // job ended (error/cancelled/unknown): allow a new, explicitly confirmed run
      }
      if (g !== gen) return;
      if (!st.foreign && fin.status === "DONE" && rec) {
        st.phase = "done";
        refreshHealthQuietly();
        emit();
      } else if (st.foreign && fin.status !== "SERVER-DOWN") {
        checkHealth(); // the other job is out of the way; show the (still unconfirmed) Run button again
      } else {
        st.phase = "failed";
        st.error = fin.status === "SERVER-DOWN" ? `${fin.error} ${MSG.serverDown}` : (fin.error || `Job ended with status ${fin.status}.`);
        emit();
      }
    }).catch((e) => {
      if (poll === ac) poll = null;
      if (isAbort(e) || g !== gen) return;
      st.phase = "failed";
      st.error = String((e && e.message) || e);
      emit();
    });
  }

  function activate(selection) {
    if (active && sel && keyOf(sel) === keyOf(selection)) { sel = selection; emit(); return; } // e.g. Replay
    abortPoll();
    gen += 1;
    active = true;
    sel = selection;
    const key = keyOf(sel);
    st.job = null; st.status = null; st.foreign = false; st.error = "";
    if (results.has(key)) {
      st.phase = "done";
      emit();
      refreshHealthQuietly();
    } else if (jobs.has(key)) {
      st.job = jobs.get(key);
      st.phase = "running";
      emit();
      startPolling(st.job.job_id); // status only, never a new submission
      refreshHealthQuietly();
    } else {
      checkHealth();
    }
  }

  function deactivate() {
    abortPoll();
    gen += 1;
    active = false;
    st.phase = "hidden";
    emit();
  }

  function requestRun() {
    if (!active || st.phase !== "ready" || !canRun()) return false;
    st.phase = "confirm";
    emit();
    return true;
  }

  function cancel() {
    if (st.phase !== "confirm") return;
    st.phase = "ready";
    emit();
  }

  async function confirmRun() {
    if (!active || st.phase !== "confirm" || !canRun()) return;
    const g = gen;
    const s = sel;
    const key = keyOf(s);
    st.phase = "submitting";
    emit();
    let r;
    try {
      r = await client.submit(s.slug, s.patient, { confirm: true });
    } catch (e) {
      r = { kind: "error", error: String((e && e.message) || e) };
    }
    // Record outcomes by their own key even if the user has moved on, so coming back resumes/shows them.
    if (r.kind === "submitted") {
      jobs.set(key, { ...r, slug: s.slug, patient: s.patient });
      jobKey.set(r.job_id, key);
    } else if (r.kind === "cached") {
      results.set(key, { ...r, slug: s.slug, patient: s.patient, cached: true });
    }
    if (g !== gen) return;
    switch (r.kind) {
      case "submitted":
        st.job = jobs.get(key);
        st.phase = "running";
        emit();
        startPolling(r.job_id);
        refreshHealthQuietly();
        break;
      case "cached":
        st.phase = "done";
        emit();
        break;
      case "busy":
        st.foreign = !jobKey.has(r.job_id) || jobKey.get(r.job_id) !== key;
        st.job = jobKey.has(r.job_id) ? jobs.get(jobKey.get(r.job_id)) || { job_id: r.job_id } : { job_id: r.job_id };
        st.phase = "running";
        emit();
        startPolling(r.job_id);
        break;
      case "budget": {
        const h = await client.health();
        if (g !== gen) return;
        st.health = h;
        settleFromHealth();
        if (st.phase !== "unavailable") { st.phase = "unavailable"; st.messages = [r.error || "QPU budget for this session used"]; }
        emit();
        break;
      }
      case "server-down":
        st.health = { ok: false, reason: "server-down" };
        settleFromHealth();
        emit();
        break;
      default:
        st.phase = "failed";
        st.error = r.error || "Submission failed.";
        emit();
    }
  }

  function recheck() {
    if (!active || !(st.phase === "failed" || st.phase === "unavailable")) return;
    checkHealth();
  }

  function view() {
    const key = sel ? keyOf(sel) : null;
    const res = key ? results.get(key) : null;
    const result = res && sel && res.slug === sel.slug && res.patient === sel.patient ? res : null;
    const partner = result && Number.isInteger(result.partner) ? result.partner : argmax(sel && sel.kRow);
    const kPair = sel && sel.kRow && partner >= 0 ? sel.kRow[partner] : null;
    return {
      phase: st.phase, active, sel: sel ? { slug: sel.slug, patient: sel.patient } : null,
      health: st.health, messages: st.messages.slice(), canRun: canRun(),
      job: st.job, status: st.status, foreign: st.foreign, error: st.error,
      result: st.phase === "done" ? result : null, partner, kPair,
    };
  }

  return { activate, deactivate, requestRun, cancel, confirmRun, recheck, view };
}

// ------------------------------------------------------------------ rendering
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const n3 = (x) => (typeof x === "number" && isFinite(x) ? x.toFixed(3) : "–");
const fmtElapsed = (s) => {
  if (!(s >= 0)) return "–";
  s = Math.floor(s);
  const m = Math.floor(s / 60), r = s % 60;
  return m ? `${m}m ${String(r).padStart(2, "0")}s` : `${r}s`;
};
const msgHtml = (m) => {
  // put the command / env-var part in <code> without changing the message text
  const e = esc(m);
  return e
    .replace("python scripts/qpu_server.py", "<code>python scripts/qpu_server.py</code>")
    .replace("pip install qiskit-ibm-runtime", "<code>pip install qiskit-ibm-runtime</code>")
    .replace("QISKIT_IBM_TOKEN", "<code>QISKIT_IBM_TOKEN</code>");
};
const STAGE_OF = (status) => {
  if (!status) return 0;
  if (status === "RUNNING") return 1;
  if (status === "DONE") return 2;
  return 0; // INITIALIZING / VALIDATING / QUEUED
};

export function renderQpuPanel(v, nowS = Date.now() / 1000) {
  if (!v || v.phase === "hidden") return "";
  const h = v.health && v.health.ok ? v.health : null;
  const budget = h ? `<span class="qp-budget" title="Jobs this helper run / budget">jobs ${h.jobs_used}/${h.jobs_max}</span>` : "";
  const shots = (h && h.shots) || 1024, cap = (h && h.max_execution_time_s) || 60;
  let html = `<div class="qp-head"><span>Real IBM QPU · free tier</span>${budget}</div>`;
  if (v.sel) {
    const pair = v.partner >= 0
      ? `patient #${v.sel.patient} vs training #${v.partner} (most similar) · scene 3 K = ${n3(v.kPair)}`
      : `patient #${v.sel.patient}`;
    html += `<div class="qp-sub">${esc(pair)}</div>`;
  }
  switch (v.phase) {
    case "checking":
      html += `<div class="qp-msg qp-dim">Checking the local QPU helper…</div>`;
      break;
    case "unavailable":
      html += v.messages.map((m) => `<div class="qp-msg qp-warn">${msgHtml(m)}</div>`).join("");
      html += `<div class="qp-actions"><button class="qp-btn" disabled>${esc(RUN_LABEL)}</button><button class="qp-link" data-act="recheck">Check again</button></div>`;
      break;
    case "ready":
      html += `<div class="qp-actions"><button class="qp-btn qp-go" data-act="run">${esc(RUN_LABEL)}</button></div>`;
      html += `<div class="qp-note">1 circuit × ${esc(shots)} shots · capped at ${esc(cap)} s · asks before sending</div>`;
      break;
    case "confirm":
      html += `<div class="qp-confirm" role="alertdialog" aria-label="Confirm QPU run"><div>${esc(confirmText(shots, cap))}</div>`
        + `<div class="qp-actions"><button class="qp-btn" data-act="cancel">Cancel</button><button class="qp-btn qp-go" data-act="confirm">Run</button></div></div>`;
      break;
    case "submitting":
      html += `<div class="qp-msg qp-dim"><span class="qp-spin"></span>Submitting to IBM…</div>`;
      break;
    case "running": {
      const j = v.job || {};
      const s = v.status || {};
      const stage = STAGE_OF(s.status);
      const local = j.submitted_at ? nowS - j.submitted_at : NaN;
      const elapsed = Math.max(Number(s.elapsed_s) || 0, isFinite(local) ? local : 0);
      if (v.foreign) html += `<div class="qp-msg qp-warn">Another QPU job is still in flight; showing its progress.</div>`;
      html += `<dl class="qp-kv">`
        + (j.backend ? `<dt>Backend</dt><dd>${esc(j.backend)}</dd>` : "")
        + (j.pending_jobs !== undefined ? `<dt>Queue</dt><dd>${esc(j.pending_jobs)} job(s) ahead at submit</dd>` : "")
        + `<dt>Job id</dt><dd class="qp-id">${esc(j.job_id || "–")}</dd>`
        + `<dt>Elapsed</dt><dd>${fmtElapsed(elapsed)}</dd></dl>`;
      html += `<div class="qp-steps">${["queued", "running", "done"].map((n, i) =>
        `<span class="qp-step${i < stage ? " past" : ""}${i === stage ? " now" : ""}">${n}</span>`).join("<i>→</i>")}</div>`;
      html += `<div class="qp-note">${esc(s.status || "QUEUED")} · checks every 5 s · you can leave this step and come back</div>`;
      break;
    }
    case "done": {
      const r = v.result;
      if (!r) break;
      html += `<div class="qp-done">${r.cached ? `<span class="qp-tag">cached result</span>` : `<span class="qp-tag qp-new">done</span>`}`
        + `<span class="qp-dim">${esc(r.backend || "")}${r.job_id ? ` · ${esc(r.job_id)}` : ""}</span></div>`;
      html += `<dl class="qp-kv qp-res">`
        + `<dt>Hardware fidelity</dt><dd><b>${n3(r.hw_fidelity)}</b> <span class="qp-dim">P(000)</span></dd>`
        + `<dt>Ideal fidelity</dt><dd>${n3(r.ideal_fidelity)} <span class="qp-dim">simulator</span></dd>`
        + `<dt>Noise gap</dt><dd><b>${n3(r.noise_gap)}</b> <span class="qp-dim">ideal − hardware</span></dd>`
        + `<dt>Scene 3 K</dt><dd>${n3(v.kPair)} <span class="qp-dim">same pair</span></dd></dl>`;
      break;
    }
    case "failed":
      html += `<div class="qp-msg qp-err">${msgHtml(v.error || "Something went wrong.")}</div>`;
      html += `<div class="qp-actions"><button class="qp-link" data-act="recheck">Check again</button></div>`;
      break;
    default:
      break;
  }
  return html;
}

/** Wire a panel element: render on every change, delegate clicks, tick the elapsed clock while running. */
export function mountQpuPanel(el, { client = qpuClient, intervalMs = 5000 } = {}) {
  let last = null;
  const draw = (v) => {
    last = v;
    el.hidden = v.phase === "hidden";
    const html = renderQpuPanel(v);
    if (el.innerHTML !== html) el.innerHTML = html;
  };
  const ctl = createQpuController({ client, intervalMs, onChange: draw });
  el.addEventListener("click", (e) => {
    const b = e.target.closest("[data-act]");
    if (!b || b.disabled) return;
    const act = b.dataset.act;
    if (act === "run") ctl.requestRun();
    else if (act === "cancel") ctl.cancel();
    else if (act === "confirm") ctl.confirmRun();
    else if (act === "recheck") ctl.recheck();
  });
  el.addEventListener("keydown", (e) => { if (e.key === "Escape") ctl.cancel(); });
  setInterval(() => { if (last && last.phase === "running") draw(ctl.view()); }, 1000);
  draw(ctl.view());
  return ctl;
}
