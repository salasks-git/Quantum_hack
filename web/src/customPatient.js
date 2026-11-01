// Task 22c: custom patient sliders -> live predictions from the local helper -> a synthetic test patient
// injected into a working copy of the tour data, so every existing scene shows it unchanged.
import * as qpuClient from "./qpu.js";

export const PREDICT_URL = "/api/live/predict";
export const MSG = {
  serverDown: "Start the helper: python scripts/qpu_server.py",
  warming: "warming up models…",
  warmTimeout: "The helper is still training the models; move a slider to retry.",
  qpuDisabled: "Live QPU runs use real test patients only",
};
export const CUSTOM_LABEL = "🧪 Custom patient";

/** Pure: predict response -> synthetic patient (label null = unknown). */
export function buildSyntheticPatient(resp) {
  const rec = resp.reconstructed || {};
  return {
    index: "custom",
    label: null,
    angles: resp.angles.slice(),
    raw: (rec.raw || []).slice(),
    standardized: (rec.standardized || []).slice(),
    pca: (rec.pca || []).slice(),
    models: JSON.parse(JSON.stringify(resp.models)),
    reconstructed: true, // scene 1 caption: bars are approximate
  };
}

/** Pure: a working copy of `data` with the custom patient appended; `data` itself is never mutated. */
export function withCustomPatient(data, resp) {
  const patient = buildSyntheticPatient(resp);
  const test = [...data.test, patient];
  const kernel_test = [...data.qsvc.kernel_test, resp.kernel_row.slice()];
  return {
    data: { ...data, test, qsvc: { ...data.qsvc, kernel_test } },
    index: test.length - 1,
  };
}

/**
 * Debounced predictor. request(body) waits `debounceMs`, POSTs, and on 202 (training) retries every
 * `retryMs` for at most `maxWaitMs`. A newer request() supersedes (cancels) any older one.
 */
export function createPredictor({ fetchImpl = (...a) => globalThis.fetch(...a), url = PREDICT_URL, debounceMs = 250, retryMs = 1000, maxWaitMs = 30000,
  onResult = () => {}, onWarming = () => {}, onError = () => {} } = {}) {
  let gen = 0, timer = null;
  const maxTries = Math.max(1, Math.ceil(maxWaitMs / retryMs));
  async function run(g, body, tries) {
    let res, json = null;
    try {
      res = await fetchImpl(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      try { json = await res.json(); } catch { json = null; }
    } catch {
      if (g === gen) onError(MSG.serverDown, "server-down");
      return;
    }
    if (g !== gen) return;
    if (res.status === 202) {
      if (tries >= maxTries) { onError(MSG.warmTimeout, "timeout"); return; }
      onWarming(json || {});
      timer = setTimeout(() => { if (g === gen) run(g, body, tries + 1); }, retryMs);
      return;
    }
    if (res.ok && json && json.models) { onResult(json); return; }
    onError(String((json && json.error) || `Helper answered HTTP ${res.status}`), "http");
  }
  return {
    request(body) {
      gen += 1;
      clearTimeout(timer);
      const g = gen;
      timer = setTimeout(() => run(g, body, 0), debounceMs);
    },
    cancel() { gen += 1; clearTimeout(timer); },
  };
}

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/**
 * Panel controller. getCtx() -> {data (the ORIGINAL data), slug, patient (real patient index)}.
 * onPatient(resp) is called with each live prediction; onRevert() when "Use real patient" is clicked.
 */
export function mountCustomPatient({ button, panel, getCtx, onPatient, onRevert, fetchImpl, healthFn = () => qpuClient.health(), debounceMs, retryMs, maxWaitMs }) {
  let open = false, active = false, up = false, sliders = [];
  const predictor = createPredictor({
    fetchImpl, debounceMs, retryMs, maxWaitMs,
    onResult: (resp) => { active = true; setStatus(resp.clamped ? "angles clamped to 0..π" : "", false); onPatient(resp); },
    onWarming: () => setStatus(MSG.warming, false),
    onError: (m, kind) => { if (kind === "server-down") up = false; setStatus(kind === "server-down" ? MSG.serverDown : m, true); syncDisabled(); },
  });
  panel.innerHTML = `<div class="cp-head"><span>🧪 Custom patient</span><button class="cp-close" aria-label="Close">✕</button></div>`
    + `<div class="cp-note">Invent a patient: one slider per qubit angle (0 to π)</div><div class="cp-sliders"></div>`
    + `<div class="cp-actions"><button class="cp-real">Use real patient</button></div><div class="cp-status" aria-live="polite"></div>`
    + `<div class="cp-note cp-qpu">${esc(MSG.qpuDisabled)}</div>`;
  const box = panel.querySelector(".cp-sliders"), statusEl = panel.querySelector(".cp-status");
  function setStatus(msg, err) {
    statusEl.textContent = msg;
    statusEl.classList.toggle("cp-err", !!err);
    if (msg === MSG.serverDown) statusEl.innerHTML = `Start the helper: <code>python scripts/qpu_server.py</code>`;
  }
  function syncDisabled() {
    sliders.forEach((s) => { s.disabled = !up; });
    panel.querySelector(".cp-real").disabled = !active && !up;
  }
  function build() {
    const ctx = getCtx();
    const base = ctx.data.test[ctx.patient].angles;
    const n = (ctx.data.settings && ctx.data.settings.n_qubits) || base.length;
    box.innerHTML = "";
    sliders = [];
    for (let k = 0; k < n; k++) {
      const v = Math.min(Math.PI, Math.max(0, base[k] ?? 0));
      const row = document.createElement("label");
      row.className = "cp-row";
      row.innerHTML = `<span>x${k + 1}</span><input type="range" min="0" max="${Math.PI.toFixed(4)}" step="0.05" value="${v.toFixed(2)}"><output>${v.toFixed(2)}</output>`;
      const input = row.querySelector("input"), out = row.querySelector("output");
      input.addEventListener("input", () => {
        out.textContent = (+input.value).toFixed(2);
        const c = getCtx();
        const key = c.data.run_id ? { run_id: c.data.run_id } : { slug: c.slug };
        predictor.request({ ...key, angles: sliders.map((s) => +s.value) });
      });
      box.appendChild(row);
      sliders.push(input);
    }
    syncDisabled();
  }
  async function setOpen(v) {
    open = !!v;
    panel.hidden = !open;
    button.classList.toggle("on", open);
    button.setAttribute("aria-pressed", String(open));
    if (!open) return;
    build();
    setStatus("Checking the helper…", false);
    const h = await healthFn();
    up = !!(h && h.ok);
    setStatus(up ? "" : MSG.serverDown, !up);
    syncDisabled();
  }
  /** The tour left the custom patient (patient/disease changed): forget it and rebuild the sliders. */
  function reset() { predictor.cancel(); active = false; setStatus("", false); if (open) { build(); } }
  button.addEventListener("click", () => setOpen(!open));
  panel.querySelector(".cp-close").addEventListener("click", () => setOpen(false));
  panel.querySelector(".cp-real").addEventListener("click", () => { predictor.cancel(); active = false; setStatus("", false); onRevert(); build(); });
  addEventListener("keydown", (e) => {
    if (e.code !== "KeyC" || e.repeat || e.metaKey || e.ctrlKey || e.altKey) return;
    const tag = document.activeElement && document.activeElement.tagName;
    if (tag !== "SELECT" && tag !== "INPUT") setOpen(!open);
  });
  setOpen(false);
  return { setOpen, isOpen: () => open, isActive: () => active, reset };
}
