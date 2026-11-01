// @vitest-environment jsdom
import { describe, it, expect, vi } from "vitest";
import { mountMenu } from "./menu.js";

function setup() {
  document.body.innerHTML = `<button id="b">menu</button><div id="p"></div><p id="out">x</p>`;
  const onSelect = vi.fn();
  const m = mountMenu({
    button: document.getElementById("b"), panel: document.getElementById("p"),
    items: [{ icon: "i", label: "Legend", hint: "L", onSelect, isOn: () => true }, { icon: "j", label: "Two", onSelect: () => {} }],
    shortcuts: [["Space", "pause"]], getStatus: () => ["Custom run active"],
  });
  return { m, onSelect, b: document.getElementById("b"), p: document.getElementById("p") };
}
describe("menu", () => {
  it("toggles with aria attributes", () => {
    const { b, p } = setup();
    expect(b.getAttribute("aria-haspopup")).toBe("menu");
    expect(b.getAttribute("aria-expanded")).toBe("false");
    expect(p.hidden).toBe(true);
    b.click();
    expect(b.getAttribute("aria-expanded")).toBe("true");
    expect(p.hidden).toBe(false);
    expect(p.getAttribute("role")).toBe("menu");
    expect(p.querySelectorAll('[role="menuitem"]').length).toBe(2);
    expect(p.querySelector(".mn-item").classList.contains("is-open")).toBe(true);
    expect(p.querySelector(".mn-status").textContent).toBe("Custom run active");
    b.click();
    expect(p.hidden).toBe(true);
  });
  it("Esc closes and refocuses the button", () => {
    const { b, p } = setup();
    b.click();
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    expect(p.hidden).toBe(true);
    expect(document.activeElement).toBe(b);
  });
  it("outside click closes, inside click does not", () => {
    const { b, p } = setup();
    b.click();
    p.querySelector(".mn-sc").click();
    expect(p.hidden).toBe(false);
    document.getElementById("out").click();
    expect(p.hidden).toBe(true);
  });
  it("item click calls handler and closes", () => {
    const { b, p, onSelect } = setup();
    b.click();
    p.querySelector(".mn-item").click();
    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(p.hidden).toBe(true);
  });
  it("arrow keys move focus", () => {
    const { b, p } = setup();
    b.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true }));
    const items = p.querySelectorAll(".mn-item");
    expect(document.activeElement).toBe(items[0]);
    items[0].dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true }));
    expect(document.activeElement).toBe(items[1]);
  });
});
