import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { modelMetrics } from "./metrics.js";
import { renderPerformance, performanceRows, renderKpis, renderTable, renderStats, renderExplorer, renderError, MAX_ROWS } from "./datasetExplorer.js";

const fixture = (n = 4) => ({
  slug: "x", label: "<b>Lab</b>", columns: ["a<i>", "target"],
  rows: Array.from({ length: n }, (_, i) => [i + 0.5, i % 4 === 0 ? 1 : 0]),
  stats: [{ column: "a<i>", count: n, mean: 1, std: 2, min: 0, p25: 1, p50: 2, p75: 3, max: 4 }],
  n_rows: n, n_features: 1, positives: 1, negatives: n - 1,
});

describe("datasetExplorer", () => {
  it("escapes text", () => {
    const h = renderExplorer(fixture());
    expect(h).not.toContain("<b>Lab");
    expect(h).toContain("&lt;b&gt;Lab&lt;/b&gt;");
    expect(h).toContain("a&lt;i&gt;");
    expect(renderError("s<x>", "bad & worse")).toContain("s&lt;x&gt;");
  });
  it("KPI percentages", () => {
    const h = renderKpis(fixture(4));
    expect(h).toContain("1 (25.0%)");
    expect(h).toContain("3 (75.0%)");
  });
  it("chips", () => {
    const h = renderTable(fixture(4));
    expect(h).toContain("Disease");
    expect(h).toContain("Healthy");
  });
  it("truncates to 200 with Show all", () => {
    const d = fixture(250);
    const h = renderTable(d);
    expect((h.match(/<tr>/g) || []).length).toBe(MAX_ROWS + 1);
    expect(h).toContain("Showing first 200 of 250");
    expect(h).toContain("ds-showall");
    const all = renderTable(d, true);
    expect((all.match(/<tr>/g) || []).length).toBe(251);
    expect(all).not.toContain("ds-showall");
  });
  it("stats table", () => {
    expect(renderStats(fixture())).toContain("Summary statistics");
  });
});

describe("model performance section", () => {
  const rd = (f) => JSON.parse(readFileSync(new URL(`../public/data/${f}`, import.meta.url), "utf8"));
  const d = rd("vertebral_column.json"), e = rd("vertebral_column.elite.json");
  it("renders 5 pipeline rows + 3 elite rows", () => {
    const h = renderPerformance(d, e);
    expect((h.match(/<tr class=/g) || []).length).toBe(8);
    expect(h).toContain("Model performance on this dataset");
    expect(h).toContain("QuHaMA");
    expect(h).toContain("precomputed");
  });
  it("pipeline metrics equal modelMetrics", () => {
    const { rows } = performanceRows(d, e);
    for (const k of ["linear", "rbf", "qsvc", "vqc", "hybrid"]) {
      const m = modelMetrics(d.test, k), r = rows.find((x) => x.key === k);
      expect([r.f2, r.recall, r.precision, r.accuracy, r.auc]).toEqual([m.f2, m.recall, m.precision, m.accuracy, m.rocAuc]);
    }
  });
  it("excludes a custom patient (label null)", () => {
    const withCustom = { ...d, test: [...d.test, { ...d.test[0], label: null }] };
    expect(performanceRows(withCustom, e).n).toBe(d.test.length);
    expect(performanceRows(withCustom, e).rows[0].f2).toBe(performanceRows(d, e).rows[0].f2);
  });
  it("missing elite.json shows the note and 5 rows", () => {
    const h = renderPerformance(d, null);
    expect(h).toContain("Elite results not exported for this dataset");
    expect((h.match(/<tr class=/g) || []).length).toBe(5);
  });
  it("highlights the F2 leader and labels Custom run", () => {
    const { rows } = performanceRows(d, e);
    const best = rows.reduce((a, b) => (b.f2 > a.f2 ? b : a));
    const h = renderPerformance(d, e, true);
    expect(h).toContain(`<span class="ds-bn">\u{1F3C6} ${best.name}</span>`);
    expect((h.match(/\u{1F3C6}/gu) || []).length).toBe(2);
    expect(h).toContain("Custom run");
    expect(h).toContain("precomputed (default settings)");
  });
});
