// @vitest-environment jsdom
import { describe, it, expect, vi } from "vitest";
import { readFileSync } from "node:fs";
import { buildPredictionTable, renderPredictionTable } from "./allPatients.js";
import { MODEL_KEYS } from "./verdict.js";

const m = (pred, prob) => ({ pred, prob });
const hand = {
  label: "Test",
  test: [
    { index: 0, label: 1, models: { linear: m(1, 0.9), rbf: m(0, 0.4) } },
    { index: 1, label: 1, models: { linear: m(0, 0.45), rbf: m(0, 0.2) } },
    { index: 2, label: 0, models: { linear: m(1, 0.7), rbf: m(0, 0.1) } },
    { index: 3, label: 0, models: { linear: m(0, 0.55), rbf: m(0, 0.3) } },
  ],
};

describe("buildPredictionTable", () => {
  const t = buildPredictionTable(hand, ["linear", "rbf"]);
  it("computes correct, accuracy, missed, sick", () => {
    expect(t.rows.map((r) => r.cells.linear.correct)).toEqual([true, false, false, true]);
    expect(t.summary.linear).toEqual({ accuracy: 0.5, missed: 1, sick: 2 });
    expect(t.summary.rbf).toEqual({ accuracy: 0.5, missed: 2, sick: 2 });
  });
  it("passes borderline through verdictFor", () => {
    expect(t.rows[0].cells.rbf.borderline).toBe(false);
    expect(t.rows[3].cells.linear.borderline).toBe(true); // pred 0 but prob .55
    expect(t.rows[1].cells.linear.borderline).toBe(false);
  });
  it("real data has 25 rows with all model keys", () => {
    const d = JSON.parse(readFileSync("public/data/heart_disease_statlog.json", "utf8"));
    const r = buildPredictionTable(d);
    expect(r.rows.length).toBe(25);
    r.rows.forEach((row) => MODEL_KEYS.forEach((k) => expect(row.cells[k]).toBeTruthy()));
  });
});

describe("renderPredictionTable", () => {
  it("escapes text, highlights current, calls onSelect", () => {
    const c = document.createElement("div");
    const evil = { rows: [{ index: 7, label: 1, cells: { "<b>x</b>": { pred: 1, prob: 0.9, correct: true, borderline: true } } }], summary: { "<b>x</b>": { accuracy: 1, missed: 0, sick: 1 } } };
    const onSelect = vi.fn();
    renderPredictionTable(c, evil, { currentPatient: 7, onSelect, keys: ["<b>x</b>"] });
    expect(c.querySelector("b")).toBeNull();
    expect(c.innerHTML).toContain("&lt;b&gt;x&lt;/b&gt;");
    expect(c.querySelector("tr.cur")).toBeTruthy();
    expect(c.querySelector(".ap-warn").title).toBe("borderline");
    c.querySelector("tbody tr").click();
    expect(onSelect).toHaveBeenCalledWith(7);
  });
});
