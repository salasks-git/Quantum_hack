import { describe, it, expect, vi, afterEach } from "vitest";
import { loadDisease } from "./data.js";
afterEach(() => vi.unstubAllGlobals());
describe("loadDisease", () => {
  it("returns parsed JSON", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, json: async () => ({ slug: "x" }) })));
    expect(await loadDisease("x")).toEqual({ slug: "x" });
  });
  it("rejects on 404", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, status: 404 })));
    await expect(loadDisease("nope")).rejects.toThrow("Could not load data for 'nope'");
  });
});
