// Settings drawer: mirrors the Streamlit sidebar; retrains via the local helper's live API.
import * as liveApi from "./live.js";

export const OPTIONS = {
  max_samples: { min: 60, max: 180, step: 20 },
  n_qubits: [2, 3, 4],
  feature_map: ["ZZFeatureMap", "PauliFeatureMap (Z, YY, ZX)"],
  reps: { min: 1, max: 3, step: 1 },
  entanglement: ["linear", "full", "circular"],
  kernel_bandwidth: [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0], // qmed.config.BANDWIDTH_OPTIONS
  vqc_iter: { min: 15, max: 75, step: 5 },
};
const FALLBACK = { max_samples: 100, n_qubits: 3, feature_map: "ZZFeatureMap", reps: 2, entanglement: "linear", kernel_bandwidth: 1.0, vqc_iter: 35 };
export const HELPER_MSG = "Live retraining needs the local helper. Start it with `python scripts/qpu_server.py`, then reload. The tour keeps working on the precomputed data.";

const snapList = (list, v, d) => (list.includes(v) ? v : d);
const snapRange = (r, v, d) => (Number.isFinite(v) && v >= r.min && v <= r.max && (v - r.min) % r.step === 0 ? v : d);

// Defaults come from the loaded data's settings (never hard-coded beyond a fallback for absent keys).
export function defaultsFrom(data) {
  const s = (data && data.settings) || {};
  return {
    max_samples: snapRange(OPTIONS.max_samples, s.max_samples, FALLBACK.max_samples),
    n_qubits: snapList(OPTIONS.n_qubits, s.n_qubits, FALLBACK.n_qubits),
    feature_map: snapList(OPTIONS.feature_map, s.feature_map, FALLBACK.feature_map),
    reps: snapRange(OPTIONS.reps, s.reps, FALLBACK.reps),
    entanglement: snapList(OPTIONS.entanglement, s.entanglement, FALLBACK.entanglement),
    kernel_bandwidth: snapList(OPTIONS.kernel_bandwidth, s.kernel_bandwidth === undefined ? 1.0 : s.kernel_bandwidth, FALLBACK.kernel_bandwidth),
    vqc_iter: snapRange(OPTIONS.vqc_iter, s.vqc_iter, FALLBACK.vqc_iter),
  };
}

export function badgeText(settings) {
  const s = settings || {};
  const fm = String(s.feature_map || "").startsWith("Pauli") ? "Pauli" : "ZZ";
  return `Custom run · ${s.n_qubits} qubits · s=${s.kernel_bandwidth} · ${fm} · ${s.entanglement} · d${s.reps} · ${s.max_samples} pts · VQC ${s.vqc_iter}`;
}

function el(tag, props = {}, ...kids) {
  const e = document.createElement(tag);
  Object.assign(e, props);
  kids.forEach((k) => e.append(k));
  return e;
}

export function mountSettingsDrawer({ button, container, badge, getData, getSlug, getLabel = () => "", onApply, onReset, onToast = () => {}, live = liveApi }) {
  let vals = defaultsFrom(getData());
  let open = false, canLive = false, busy = false, custom = false, ctl = null;
  const ctrls = {};

  const msg = el("div", { className: "sd-msg" });
  const stage = el("div", { className: "sd-stage" });
  const bar = el("div", { className: "sd-bar" }, el("div", { className: "sd-fill" }));
  const prog = el("div", { className: "sd-prog", hidden: true }, stage, bar);

  function row(label, input, out) {
    return el("label", { className: "sd-row" }, el("span", { className: "sd-lab" }, label), input, out || "");
  }
  function select(key, list) {
    const s = el("select", { className: "sd-in" });
    list.forEach((o) => s.append(el("option", { value: String(o), textContent: String(o) })));
    s.addEventListener("change", () => { vals[key] = typeof list[0] === "number" ? Number(s.value) : s.value; });
    ctrls[key] = s;
    return row(key, s);
  }
  function slider(key, r) {
    const i = el("input", { type: "range", className: "sd-in", min: r.min, max: r.max, step: r.step });
    const out = el("output", { className: "sd-out" });
    i.addEventListener("input", () => { vals[key] = Number(i.value); out.textContent = i.value; });
    ctrls[key] = i; ctrls[key + "_out"] = out;
    return row(key, i, out);
  }
  function listSlider(key, list) { // select_slider over discrete options
    const i = el("input", { type: "range", className: "sd-in", min: 0, max: list.length - 1, step: 1 });
    const out = el("output", { className: "sd-out" });
    i.addEventListener("input", () => { vals[key] = list[Number(i.value)]; out.textContent = String(vals[key]); });
    ctrls[key] = i; ctrls[key + "_out"] = out;
    return row(key, i, out);
  }

  const dataset = el("div", { className: "sd-dataset" });
  const labels = {
    max_samples: "2. Cohort sample size", n_qubits: "3. Qubits (PCA features)", feature_map: "4. Quantum feature map",
    reps: "5. Circuit depth (reps)", entanglement: "6. Entanglement", kernel_bandwidth: "7. Kernel bandwidth (s)", vqc_iter: "8. VQC max iterations",
  };
  const rows = [
    slider("max_samples", OPTIONS.max_samples), select("n_qubits", OPTIONS.n_qubits), select("feature_map", OPTIONS.feature_map),
    slider("reps", OPTIONS.reps), select("entanglement", OPTIONS.entanglement), listSlider("kernel_bandwidth", OPTIONS.kernel_bandwidth),
    slider("vqc_iter", OPTIONS.vqc_iter),
  ];
  Object.keys(labels).forEach((k, i) => { rows[i].firstChild.textContent = labels[k]; });

  const goBtn = el("button", { className: "sd-go", type: "button", textContent: "Train & replay" });
  const resetBtn = el("button", { className: "sd-reset", type: "button", textContent: "Reset to precomputed" });
  const closeBtn = el("button", { className: "sd-close", type: "button", textContent: "✕", title: "Close" });
  const head = el("div", { className: "sd-head" }, el("span", { className: "sd-title" }, "⚙️ Quantum pipeline config"), closeBtn);
  container.append(head, dataset, ...rows, el("div", { className: "sd-actions" }, goBtn, resetBtn), prog, msg);

  function syncControls() {
    Object.keys(OPTIONS).forEach((k) => {
      const c = ctrls[k];
      if (k === "kernel_bandwidth") c.value = String(OPTIONS[k].indexOf(vals[k]));
      else c.value = String(vals[k]);
      if (ctrls[k + "_out"]) ctrls[k + "_out"].textContent = String(vals[k]);
    });
    dataset.textContent = `1. Dataset: ${getLabel() || getSlug() || ""} (change it in the top bar)`;
  }
  function applyEnabled() {
    const dis = !canLive || busy;
    Object.keys(OPTIONS).forEach((k) => { ctrls[k].disabled = !canLive || busy; });
    goBtn.disabled = dis;
    resetBtn.disabled = !(custom || busy);
    container.classList.toggle("sd-disabled", !canLive);
  }
  function setMsg(text, kind = "") { msg.textContent = text; msg.className = `sd-msg ${kind}`; }

  function setOpen(v) {
    open = !!v;
    container.hidden = !open;
    button.classList.toggle("on", open);
    button.setAttribute("aria-pressed", String(open));
    container.classList.toggle("sd-open", open);
    if (open) checkHealth();
  }
  async function checkHealth() {
    const h = await live.liveHealth();
    canLive = h.live;
    if (!canLive && !busy) setMsg(HELPER_MSG, "warn");
    else if (!busy && msg.classList.contains("warn")) setMsg("");
    applyEnabled();
  }

  function setBadge(text) {
    if (!badge) return;
    badge.textContent = text || "";
    badge.hidden = !text;
  }
  function clearCustom() { // called when the active data is the static set again
    if (ctl) ctl.abort();
    custom = false; busy = false; ctl = null;
    setBadge("");
    prog.hidden = true;
    vals = defaultsFrom(getData());
    syncControls(); applyEnabled();
  }

  function collect() { return { slug: getSlug(), ...vals }; }

  async function run() {
    if (busy || !canLive) return;
    const payload = collect();
    ctl = new AbortController();
    const mine = ctl;
    busy = true; applyEnabled(); setMsg("");
    prog.hidden = false; stage.textContent = "starting…"; bar.firstChild.style.width = "0%";
    try {
      const { run_id } = await live.startTrain(payload);
      const fin = await live.pollTrain(run_id, {
        signal: mine.signal,
        onUpdate: (s) => {
          const pct = Math.round((s.progress || 0) * 100);
          stage.textContent = `${s.stage || s.state} · ${pct}% · ${Number(s.elapsed_s || 0).toFixed(1)} s`;
          bar.firstChild.style.width = `${pct}%`;
        },
      });
      const data = await live.loadRun(run_id);
      if (mine.signal.aborted || getSlug() !== payload.slug) return;
      custom = true;
      const sett = { ...vals, ...(data.settings || {}) };
      stage.textContent = `done in ${Number(fin.elapsed_s || 0).toFixed(1)} s`;
      bar.firstChild.style.width = "100%";
      onApply(data);
      setBadge(badgeText(sett));
      if (Number(sett.n_qubits) !== 3) onToast("2/4-qubit runs: scenes 1–3 adapt after the next update");
    } catch (e) {
      if (e && e.name === "AbortError") return;
      prog.hidden = true;
      setMsg(e && e.status === 409 ? "Another training is running" : (e && e.message) || "Training failed", "err");
    } finally {
      if (ctl === mine) { busy = false; ctl = null; applyEnabled(); }
    }
  }

  button.addEventListener("click", () => setOpen(!open));
  closeBtn.addEventListener("click", () => setOpen(false));
  goBtn.addEventListener("click", run);
  resetBtn.addEventListener("click", () => {
    if (ctl) ctl.abort();
    Promise.resolve(onReset()).then(() => { clearCustom(); setMsg(""); });
  });

  syncControls(); applyEnabled();
  container.hidden = true;
  checkHealth();
  return {
    setOpen, isOpen: () => open, clearCustom, refresh: () => { syncControls(); applyEnabled(); },
    collect, run, isCustom: () => custom, values: () => ({ ...vals }), checkHealth, _ctrls: ctrls, _goBtn: goBtn, _resetBtn: resetBtn, _msg: msg,
  };
}
