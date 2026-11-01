import { describe, it, expect } from "vitest";
import { renderKpis, renderTable, renderStats, renderExplorer, renderError, MAX_ROWS } from "./datasetExplorer.js";

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
