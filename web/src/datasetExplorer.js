// Dataset Explorer overlay: KPIs, table, summary stats for the selected disease.
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

export function renderExplorer(d, showAll = false) {
  return `<div class="ds-head"><h2>${esc(d.label)}</h2><button id="ds-close" aria-label="Close">&#10005;</button></div>${renderKpis(d)}${renderTable(d, showAll)}${renderStats(d)}`;
}

export function renderError(slug, msg, close = true) {
  return `<div class="ds-head"><h2>Dataset</h2>${close ? '<button id="ds-close" aria-label="Close">&#10005;</button>' : ""}</div><div class="ds-err">Could not load dataset "${esc(slug)}": ${esc(msg)}</div>`;
}

// button, container: DOM elements. getSlug(): current disease slug.
export function mountDatasetExplorer({ button, container, getSlug, fetchJson }) {
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
    container.innerHTML = renderExplorer(d, showAll);
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
