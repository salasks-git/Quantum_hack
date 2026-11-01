// Dataset Explorer overlay: KPIs, table, summary stats for the selected disease.
import { modelMetrics } from "./metrics.js";
import { fmtDuration } from "./timing.js";
import { MODEL_KEYS, MODEL_LABELS } from "./verdict.js";
import { ELITE_KEYS, ELITE_NAMES, loadElite } from "./elite.js";
export const MAX_ROWS = 200;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (n, total) => (total ? ((100 * n) / total).toFixed(1) : "0.0") + "%";

export function renderKpis(d) {
  const tiles = [
    ["Total Patient Records", Number(d.n_rows).toLocaleString("en-US")],
    ["Raw Clinical Features", String(d.n_features)],
    ["Positive Diagnoses (Disease)", `${d.positives} (${pct(d.positives, d.n_rows)})`],
    ["Negative Diagnoses (Healthy)", `${d.negatives} (${pct(d.negatives, d.n_rows)})`],
  ];
  return `<div class="ds-kpis">${tiles.map(([k, v]) => `<div class="ds-kpi"><div class="ds-kv">${esc(v)}</div><div class="ds-kl">${esc(k)}</div></div>`).join("")}</div>`;
}

export function renderTable(d, showAll = false) {
  const tIdx = d.columns.indexOf("target");
  const shown = showAll ? d.rows : d.rows.slice(0, MAX_ROWS);
  const head = `<tr><th>#</th>${d.columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>`;
  const body = shown.map((r, i) => `<tr><td>${i}</td>${r.map((v, j) => j === tIdx
    ? `<td><span class="ds-chip ${v === 1 ? "pos" : "neg"}">${v === 1 ? "Disease" : "Healthy"}</span></td>`
    : `<td>${esc(v)}</td>`).join("")}</tr>`).join("");
  const truncated = shown.length < d.rows.length;
  const note = truncated
    ? `<div class="ds-note">Showing first ${shown.length} of ${d.rows.length} <button id="ds-showall">Show all</button></div>` : "";
  return `${note}<div class="ds-scroll"><table class="ds-table"><thead>${head}</thead><tbody>${body}</tbody></table></div>`;
}

export function renderStats(d) {
  const cols = ["count", "mean", "std", "min", "p25", "p50", "p75", "max"];
  const labels = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"];
  return `<details class="ds-stats"><summary>Summary statistics</summary><div class="ds-scroll"><table class="ds-table"><thead><tr><th>column</th>${labels.map((l) => `<th>${l}</th>`).join("")}</tr></thead><tbody>${
    d.stats.map((s) => `<tr><td>${esc(s.column)}</td>${cols.map((c) => `<td>${esc(s[c])}</td>`).join("")}</tr>`).join("")}</tbody></table></div></details>`;
}

// ---- Model performance (pipeline models from the active data's test[], elite models from elite.json) ----
const FAMILY = { linear: "classical", rbf: "classical", qsvc: "quantum", vqc: "quantum", hybrid: "hybrid" };
const f3 = (v) => (typeof v === "number" && Number.isFinite(v) ? v.toFixed(3) : "\u2014");
const pctAcc = (v) => (typeof v === "number" && Number.isFinite(v) ? (100 * v).toFixed(1) + "%" : "\u2014");
const dur = (v) => (typeof v === "number" && Number.isFinite(v) ? fmtDuration(v) : "\u2014");

/** Rows for the performance section. active: the data the pipeline models are evaluated on. */
export function performanceRows(active, elite) {
  const test = (active.test || []).filter((p) => p.label != null);
  const sick = test.filter((p) => p.label === 1).length;
  const rows = MODEL_KEYS.map((k) => {
    const m = modelMetrics(active.test || [], k), t = (active.timings || {})[k] || {};
    return { key: k, name: MODEL_LABELS[k], family: FAMILY[k], f2: m.f2, recall: m.recall, missed: m.missed, sick,
      precision: m.precision, f1: m.f1, accuracy: m.accuracy, auc: m.rocAuc, train: t.train_s, inf: t.inf_s };
  });
  const lb = (elite && elite.leaderboard) || [];
  for (const k of ELITE_KEYS) {
    const r = lb.find((x) => x.key === k);
    if (!r) continue;
    rows.push({ key: k, name: ELITE_NAMES[k] || r.model, family: "elite", f2: r.f2, recall: r.recall, missed: null, sick,
      precision: r.precision, f1: r.f1, accuracy: r.accuracy, auc: r.roc_auc, train: r.train_s, inf: r.inf_s, total: r.time_s });
  }
  return { rows, n: test.length };
}

export function renderPerformance(active, elite, custom = false) {
  const { rows, n } = performanceRows(active, elite);
  const s = active.settings || {};
  const bits = [s.n_qubits != null ? `${s.n_qubits} qubits` : null, s.feature_map || null,
    s.kernel_bandwidth != null ? `s=${s.kernel_bandwidth}` : null].filter(Boolean).join(", ");
  const head = `<div class="ds-note">Evaluated on ${n} held-out test patients (the models never saw them). Settings: ${esc(bits || "n/a")} (${custom ? "Custom run" : "precomputed"})</div>`;
  const ranked = rows.filter((r) => typeof r.f2 === "number").sort((a, b) => b.f2 - a.f2);
  const lead = ranked.length ? ranked[0].key : null;
  const bars = `<div class="ds-bars">${ranked.map((r) => `<div class="ds-bar${r.key === lead ? " lead" : ""}"><span class="ds-bn">${r.key === lead ? "\u{1F3C6} " : ""}${esc(r.name)}</span><span class="ds-bt"><span class="ds-bf ${r.family}" style="width:${Math.max(0, Math.min(1, r.f2)) * 100}%"></span></span><span class="ds-bv">${f3(r.f2)}</span></div>`).join("")}</div>`;
  const cols = ["Model", "Family", "F2 \u{1F3AF}", "Recall", "Missed (n/sick)", "Precision", "F1", "Accuracy", "ROC-AUC", "Train time", "Inference time"];
  const body = rows.map((r) => `<tr class="${r.key === lead ? "lead" : ""}"><td>${r.key === lead ? "\u{1F3C6} " : ""}${esc(r.name)}</td><td><span class="ds-fam ${r.family}">${r.family}</span></td><td>${f3(r.f2)}</td><td>${f3(r.recall)}</td><td>${r.missed == null ? "\u2014" : `${r.missed}/${r.sick}`}</td><td>${f3(r.precision)}</td><td>${f3(r.f1)}</td><td>${pctAcc(r.accuracy)}</td><td>${f3(r.auc)}</td><td>${dur(r.train != null ? r.train : r.total)}</td><td>${dur(r.inf)}</td></tr>`).join("");
  const eliteNote = !elite || !(elite.leaderboard || []).length
    ? `<div class="ds-note">Elite results not exported for this dataset</div>`
    : custom ? `<div class="ds-note">Elite models: precomputed (default settings), not retrained live.</div>` : "";
  return `<section class="ds-perf"><h3>\u{1F4C8} Model performance on this dataset</h3>${head}${bars}<div class="ds-scroll"><table class="ds-table"><thead><tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table></div>${eliteNote}</section>`;
}

export function renderExplorer(d, showAll = false, perf = "") {
  return `<div class="ds-head"><h2>${esc(d.label)}</h2><button id="ds-close" aria-label="Close">&#10005;</button></div>${renderKpis(d)}${perf}${renderTable(d, showAll)}${renderStats(d)}`;
}

export function renderError(slug, msg, close = true) {
  return `<div class="ds-head"><h2>Dataset</h2>${close ? '<button id="ds-close" aria-label="Close">&#10005;</button>' : ""}</div><div class="ds-err">Could not load dataset "${esc(slug)}": ${esc(msg)}</div>`;
}

// button, container: DOM elements. getSlug(): current disease slug.
// getData() (optional): the active tour data (Custom run aware); isCustom() (optional): a Custom run is active.
// Without getData the precomputed <slug>.json is fetched. eliteLoader(slug) (optional): defaults to loadElite.
export function mountDatasetExplorer({ button, container, getSlug, fetchJson, getData, isCustom, eliteLoader }) {
  const fetcher = fetchJson || ((u) => fetch(u).then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); }));
  const cache = new Map();
  let open = false, showAll = false, token = 0;

  async function render() {
    if (!open) return;
    const slug = getSlug();
    if (!slug) return;
    const my = ++token;
    let d = cache.get(slug);
    if (!d) {
      container.innerHTML = `<div class="ds-head"><h2>Loading ${esc(slug)}…</h2><button id="ds-close" aria-label="Close">&#10005;</button></div>`;
      wire();
      try { d = await fetcher(`/data/${slug}.dataset.json`); cache.set(slug, d); }
      catch (e) { if (my === token && open) { container.innerHTML = renderError(slug, e.message); wire(); } return; }
    }
    if (my !== token || !open) return;
    let perf = "";
    try {
      const custom = !!(isCustom && isCustom());
      let active = getData ? getData() : null;
      if (!active || active.slug !== slug) active = await fetcher(`/data/${slug}.json`);
      let elite = null;
      try { elite = await (eliteLoader || loadElite)(slug); } catch { elite = null; }
      perf = renderPerformance(active, elite, custom);
    } catch (e) { perf = `<section class="ds-perf"><div class="ds-note">Model performance unavailable: ${esc(e.message)}</div></section>`; }
    if (my !== token || !open) return;
    container.innerHTML = renderExplorer(d, showAll, perf);
    wire();
  }
  function wire() {
    const c = container.querySelector && container.querySelector("#ds-close");
    if (c) c.addEventListener("click", () => setOpen(false));
    const s = container.querySelector && container.querySelector("#ds-showall");
    if (s) s.addEventListener("click", () => { showAll = true; render(); });
  }
  function setOpen(v) {
    open = !!v;
    container.hidden = !open;
    button.classList.toggle("on", open);
    button.setAttribute("aria-pressed", String(open));
    if (open) { showAll = false; render(); }
  }
  button.addEventListener("click", () => setOpen(!open));
  addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const tag = document.activeElement && document.activeElement.tagName;
    if (e.code === "KeyD" && !e.repeat && tag !== "SELECT" && tag !== "INPUT") setOpen(!open);
    else if (e.code === "Escape" && open) setOpen(false);
  });
  return { setOpen, isOpen: () => open, refresh: () => { showAll = false; render(); } };
}
