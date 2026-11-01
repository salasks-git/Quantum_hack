import { describe, it, expect } from "vitest";
import { liveHealth, startTrain, pollTrain, loadRun } from "./live.js";

const ok = (body, status = 200) => ({ ok: status < 400, status, json: async () => body });

describe("live client", () => {
  it("start -> poll -> load flow", async () => {
    const calls = [];
    let n = 0;
    const f = async (url, opts) => {
      calls.push(url);
      if (url === "/api/live/train") return ok({ run_id: "abc", state: "queued" });
      if (url.startsWith("/api/live/train/status")) { n += 1; return ok(n < 3 ? { state: "running", stage: "VQC training", progress: 0.5 } : { state: "done", progress: 1 }); }
      if (url === "/api/live/run/abc.json") return ok({ custom_run: true });
      throw new Error("unexpected " + url);
    };
    const { run_id } = await startTrain({ slug: "x" }, f);
    const seen = [];
    const fin = await pollTrain(run_id, { onUpdate: (s) => seen.push(s.state), intervalMs: 0, fetchFn: f });
    expect(fin.state).toBe("done");
    expect(seen).toEqual(["running", "running", "done"]);
    expect((await loadRun(run_id, f)).custom_run).toBe(true);
    expect(calls.some((u) => /qpu\/(submit|backends)/.test(u))).toBe(false);
  });
  it("abort stops polling", async () => {
    let n = 0;
    const ctl = new AbortController();
    const f = async () => { n += 1; return ok({ state: "running" }); };
    const p = pollTrain("abc", { signal: ctl.signal, intervalMs: 5, fetchFn: f, onUpdate: () => { if (n === 2) ctl.abort(); } });
    await expect(p).rejects.toMatchObject({ name: "AbortError" });
    const at = n;
    await new Promise((r) => setTimeout(r, 30));
    expect(n).toBe(at);
  });
  it("409 carries status and message", async () => {
    const f = async () => ok({ error: "Another training run is in progress; wait for it to finish.", run_id: "zz" }, 409);
    await expect(startTrain({}, f)).rejects.toMatchObject({ status: 409, runId: "zz" });
  });
  it("error state rejects with message", async () => {
    const f = async () => ok({ state: "error", error: "boom" });
    await expect(pollTrain("a", { fetchFn: f })).rejects.toThrow("boom");
  });
  it("health: down and not-live", async () => {
    expect(await liveHealth(async () => { throw new Error("ECONNREFUSED"); })).toEqual({ up: false, live: false });
    expect(await liveHealth(async () => ok({}, 502))).toEqual({ up: false, live: false });
    expect(await liveHealth(async () => ok({ ok: true }))).toEqual({ up: true, live: false });
    expect(await liveHealth(async () => ok({ live: true }))).toEqual({ up: true, live: true });
  });
});
