// Top-bar hamburger menu (Task 28). Items only call injected handlers (the existing buttons' click()).
export function mountMenu({ button, panel, items, shortcuts = [], getStatus = () => [] }) {
  let open = false;
  const rows = [];
  panel.setAttribute("role", "menu");
  button.setAttribute("aria-haspopup", "menu");
  button.setAttribute("aria-expanded", "false");
  items.forEach((it) => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "mn-item"; b.setAttribute("role", "menuitem"); b.tabIndex = -1;
    b.innerHTML = `<span class="mn-ic"></span><span class="mn-lab"></span><span class="mn-dot" aria-hidden="true"></span><kbd class="mn-key"></kbd>`;
    b.querySelector(".mn-ic").textContent = it.icon;
    b.querySelector(".mn-lab").textContent = it.label;
    b.querySelector(".mn-key").textContent = it.hint || "";
    b.addEventListener("click", () => { setOpen(false); it.onSelect(); });
    panel.appendChild(b); rows.push({ b, it });
  });
  const sc = document.createElement("div");
  sc.className = "mn-sc"; sc.setAttribute("role", "group"); sc.setAttribute("aria-label", "Keyboard shortcuts");
  sc.innerHTML = `<div class="mn-sc-t">⌨ Keyboard shortcuts</div>` + shortcuts.map(([k, d]) => `<div><kbd>${k}</kbd> ${d}</div>`).join("");
  const st = document.createElement("div");
  st.className = "mn-status"; st.setAttribute("role", "status");
  panel.append(sc, st);

  function refresh() {
    rows.forEach(({ b, it }) => b.classList.toggle("is-open", !!(it.isOn && it.isOn())));
    st.textContent = getStatus().join(" · ");
    st.hidden = !st.textContent;
  }
  function focusRow(i) { const n = rows.length; if (n) rows[(i + n) % n].b.focus(); }
  function setOpen(v, restoreFocus = true) {
    v = !!v;
    if (v === open) return;
    open = v;
    panel.hidden = !open;
    button.setAttribute("aria-expanded", String(open));
    if (open) refresh();
    else if (restoreFocus) button.focus();
  }
  button.addEventListener("click", () => setOpen(!open));
  button.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); setOpen(true); focusRow(0); }
  });
  panel.addEventListener("keydown", (e) => {
    const i = rows.findIndex((r) => r.b === document.activeElement);
    if (e.key === "ArrowDown") { e.preventDefault(); focusRow(i + 1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); focusRow(i < 0 ? -1 : i - 1); }
    else if (e.key === "Tab") setOpen(false, false);
  });
  document.addEventListener("keydown", (e) => {
    if (open && e.key === "Escape") { e.stopPropagation(); setOpen(false); }
  }, true);
  document.addEventListener("click", (e) => {
    if (open && !panel.contains(e.target) && !button.contains(e.target)) setOpen(false, false);
  });
  panel.hidden = true;
  return { setOpen, isOpen: () => open, refresh };
}
