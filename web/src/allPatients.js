import { MODEL_KEYS, MODEL_LABELS, verdictFor } from "./verdict.js";
import { modelMetrics } from "./metrics.js";

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// Pure: every model's prediction for every test patient, plus per-model summary.
export function buildPredictionTable(data, keys = MODEL_KEYS) {
  const tests = data.test || [];
  const n = tests.length;
  const summary = {};
  keys.forEach((k) => { summary[k] = { accuracy: 0, missed: 0, sick: 0 }; });
  const rows = tests.map((p, i) => {
    const cells = {};
    keys.forEach((k) => {
      const m = p.models && p.models[k];
      if (!m) return;
      const correct = m.pred === p.label;
      cells[k] = { pred: m.pred, prob: m.prob, correct, borderline: verdictFor(m).borderline };
      if (correct) summary[k].accuracy += 1;
      if (p.label === 1) { summary[k].sick += 1; if (m.pred === 0) summary[k].missed += 1; }
    });
    return { index: p.index != null ? p.index : i, label: p.label, cells };
  });
  keys.forEach((k) => { summary[k].accuracy = n ? summary[k].accuracy / n : 0; });
  return { rows, summary };
}

export function renderPredictionTable(container, table, { currentPatient = -1, onSelect = () => {}, keys = MODEL_KEYS } = {}) {
  const n = table.rows.length;
  const head = `<tr><th>#</th><th>Actual</th>${keys.map((k) => `<th>${esc(MODEL_LABELS[k] || k)}</th>`).join("")}</tr>`;
  const body = table.rows.map((r) => {
    const cells = keys.map((k) => {
      const c = r.cells[k];
      if (!c) return `<td>—</td>`;
      const sick = c.pred === 1;
      return `<td><span class="ap-chip ${sick ? "sick" : "well"}">${sick ? "DISEASE" : "HEALTHY"}</span> `
        + `<span class="ap-prob">${Math.round(c.prob * 100)}%</span> `
        + `<span class="${c.correct ? "ap-ok" : "ap-bad"}">${c.correct ? "✓" : "✗"}</span>`
        + `${c.borderline ? ` <span class="ap-warn" title="borderline">⚠</span>` : ""}</td>`;
    }).join("");
    const cls = r.index === currentPatient ? ' class="cur"' : "";
    return `<tr data-index="${esc(r.index)}"${cls}><td>${esc(r.index)}</td>`
      + `<td><span class="ap-act ${r.label === 1 ? "sick" : "well"}">${r.label === 1 ? "Disease" : "Healthy"}</span></td>${cells}</tr>`;
  }).join("");
  const foot = `<tr class="ap-foot"><td colspan="2">Per model</td>${keys.map((k) => {
    const s = table.summary[k];
    return `<td>Accuracy ${Math.round(s.accuracy * n)}/${n} (${Math.round(s.accuracy * 100)}%)<br>Missed ${s.missed}/${s.sick}</td>`;
  }).join("")}</tr>`;
  container.innerHTML = `<table class="ap-table"><thead>${head}</thead><tbody>${body}</tbody><tfoot>${foot}</tfoot></table>`;
  container.querySelectorAll("tbody tr").forEach((tr) => {
    tr.addEventListener("click", () => onSelect(+tr.dataset.index));
  });
}

// Streamlit-parity benchmark table + one 2x2 confusion matrix per model.
export function renderMetrics(container, test, keys = MODEL_KEYS) {
  const ms = keys.map((k) => ({ k, m: modelMetrics(test, k) }));
  const f3 = (v) => v.toFixed(3);
  const cols = [["F2 🎯", (m) => f3(m.f2)], ["Recall", (m) => f3(m.recall)], ["Missed", (m) => String(m.missed)],
    ["F1", (m) => f3(m.f1)], ["Precision", (m) => f3(m.precision)], ["Accuracy", (m) => `${(m.accuracy * 100).toFixed(0)}%`],
    ["ROC-AUC", (m) => (m.rocAuc == null ? "—" : f3(m.rocAuc))]];
  const head = `<tr><th>Model</th>${cols.map(([h]) => `<th>${h}</th>`).join("")}</tr>`;
  const rows = ms.map(({ k, m }) => `<tr><td>${esc(MODEL_LABELS[k] || k)}</td>${cols.map(([, f]) => `<td>${f(m)}</td>`).join("")}</tr>`).join("");
  const cell = (n, max, cls) => `<td class="${cls}" style="--t:${max ? (n / max).toFixed(2) : 0}">${n}</td>`;
  const cms = ms.map(({ k, m }) => {
    const { tp, fp, fn, tn } = m.cm, max = Math.max(tp, fp, fn, tn);
    return `<div class="ap-cm"><div class="ap-cm-t">${esc(MODEL_LABELS[k] || k)}</div><table>`
      + `<tr><th></th><th>Pred Healthy</th><th>Pred Disease</th></tr>`
      + `<tr><th>Actual Healthy</th>${cell(tn, max, "ok")}${cell(fp, max, "bad")}</tr>`
      + `<tr><th>Actual Disease</th>${cell(fn, max, "bad")}${cell(tp, max, "ok")}</tr></table></div>`;
  }).join("");
  container.innerHTML = `<div class="ap-mt">📊 Metrics · ${test.length} test patients</div>`
    + `<table class="ap-metrics"><thead>${head}</thead><tbody>${rows}</tbody></table>`
    + `<div class="ap-mt">Confusion matrices · Actual × Predicted</div><div class="ap-cms">${cms}</div>`;
}

// Overlay controller. panel contains .ap-title, .ap-close, .ap-body (built here).
export function mountAllPatients(button, panel, { getData, getPatient, onSelect }) {
  panel.innerHTML = `<div class="ap-head"><span class="ap-title"></span><button class="ap-close" title="Close (Esc)" aria-label="Close">✕</button></div><div class="ap-body"></div>`;
  const metricsBox = document.createElement("div");
  metricsBox.className = "ap-metrics-box";
  const title = panel.querySelector(".ap-title");
  const body = panel.querySelector(".ap-body");
  let open = false;
  const render = () => {
    const data = getData();
    if (!open || !data) return;
    const t = buildPredictionTable(data);
    title.textContent = `All test patients · ${data.label} · ${t.rows.length} patients the models never saw`;
    renderPredictionTable(body, t, { currentPatient: getPatient(), onSelect: (i) => { setOpen(false); onSelect(i); } });
    renderMetrics(metricsBox, data.test || []);
    body.appendChild(metricsBox);
  };
  function setOpen(v) {
    open = !!v;
    panel.hidden = !open;
    button.classList.toggle("on", open);
    button.setAttribute("aria-pressed", String(open));
    if (open) { render(); body.scrollTop = 0; }
  }
  button.addEventListener("click", () => setOpen(!open));
  panel.querySelector(".ap-close").addEventListener("click", () => setOpen(false));
  addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.code === "Escape" && open) { setOpen(false); return; }
    if (e.code === "KeyA" && !e.repeat) {
      const tag = document.activeElement && document.activeElement.tagName;
      if (tag !== "SELECT" && tag !== "INPUT") setOpen(!open);
    }
  });
  setOpen(false);
  return { setOpen, isOpen: () => open, refresh: render };
}
