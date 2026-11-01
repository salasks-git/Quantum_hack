const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const SAFE_COLOUR = /^(#[0-9a-fA-F]{3,8}|[a-zA-Z]+|rgba?\([\d\s.,%]+\))$/;

// Render legend items into container as <dl>-style rows. Text is escaped.
export function renderLegend(container, items) {
  container.innerHTML = (items || []).map((it) => {
    const dot = it.swatch && SAFE_COLOUR.test(it.swatch)
      ? `<span class="lg-dot" style="background:${it.swatch}"></span>` : "";
    return `<div class="lg-row"><div class="lg-sym">${dot}${esc(it.symbol)}</div><div class="lg-txt">${esc(it.text)}</div></div>`;
  }).join("");
}

// Wire a toggle button to a panel; open by default; state persisted (best effort).
export function mountLegendToggle(button, panel, { storageKey = "qmed.legend.open" } = {}) {
  let open = true;
  try {
    const v = localStorage.getItem(storageKey);
    if (v === "0") open = false;
  } catch (e) { /* storage unavailable */ }
  const apply = () => {
    panel.hidden = !open;
    button.classList.toggle("on", open);
    button.setAttribute("aria-pressed", String(open));
  };
  const setOpen = (v) => {
    open = !!v;
    try { localStorage.setItem(storageKey, open ? "1" : "0"); } catch (e) { /* ignore */ }
    apply();
  };
  button.addEventListener("click", () => setOpen(!open));
  apply();
  return { setOpen, isOpen: () => open };
}
