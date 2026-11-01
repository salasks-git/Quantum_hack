import { describe, it, expect, vi, afterEach } from "vitest";
import { health, submit, pollStatus } from "./qpu.js";
import { createQpuController, renderQpuPanel, MSG, RUN_LABEL, confirmText } from "./qpuPanel.js";

const jsonRes = (status, body) => ({ ok: status >= 200 && status < 300, status, json: async () => body });
const textRes = (status) => ({ ok: false, status, json: async () => { throw new SyntaxError("not json"); } });
const flush = async (n = 6) => { for (let i = 0; i < n; i++) await Promise.resolve(); };

afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });

describe("qpu client: health", () => {
  it("maps a network error to server-down", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("Failed to fetch"); }));
    expect(await health()).toEqual({ ok: false, reason: "server-down" });
  });
  it("maps a proxy failure (non-JSON 500) to server-down", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => textRes(500)));
    expect(await health()).toEqual({ ok: false, reason: "server-down" });
  });
  it("passes through the helper's health fields", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => jsonRes(200, {
      ok: true, credentials: true, runtime: false, jobs_used: 1, jobs_max: 3, max_execution_time_s: 60, shots: 1024,
    })));
    const h = await health();
    expect(h).toMatchObject({ ok: true, credentials: true, runtime: false, jobs_used: 1, jobs_max: 3 });
  });
});

describe("qpu client: submit", () => {
  it("refuses without confirm and makes no request", async () => {
    const f = vi.fn();
    vi.stubGlobal("fetch", f);
    await expect(submit("heart", 0)).rejects.toThrow(/confirm/);
    expect(f).not.toHaveBeenCalled();
  });
  it("sends confirm:true and maps 409/429/cached", async () => {
    const f = vi.fn()
      .mockResolvedValueOnce(jsonRes(409, { error: "busy", job_id: "J1" }))
      .mockResolvedValueOnce(jsonRes(429, { error: "budget" }))
      .mockResolvedValueOnce(jsonRes(200, { cached: true, job_id: "J0", hw_fidelity: 0.8 }));
    vi.stubGlobal("fetch", f);
    expect(await submit("heart", 2, { confirm: true })).toMatchObject({ kind: "busy", job_id: "J1" });
    expect(JSON.parse(f.mock.calls[0][1].body)).toEqual({ slug: "heart", patient: 2, confirm: true });
    expect(await submit("heart", 2, { confirm: true })).toMatchObject({ kind: "budget" });
    expect(await submit("heart", 2, { confirm: true })).toMatchObject({ kind: "cached", hw_fidelity: 0.8 });
  });
});

describe("qpu client: pollStatus", () => {
  it("calls onUpdate until done and resolves with the final result", async () => {
    const seq = [
      jsonRes(200, { job_id: "J", status: "QUEUED", elapsed_s: 1 }),
      jsonRes(200, { job_id: "J", status: "RUNNING", elapsed_s: 2 }),
      jsonRes(200, { job_id: "J", status: "DONE", elapsed_s: 3, hw_fidelity: 0.7, ideal_fidelity: 0.9, noise_gap: 0.2 }),
    ];
    const f = vi.fn(async () => seq.shift());
    vi.stubGlobal("fetch", f);
    const onUpdate = vi.fn();
    const fin = await pollStatus("J", { onUpdate, intervalMs: 1 });
    expect(onUpdate.mock.calls.map((c) => c[0].status)).toEqual(["QUEUED", "RUNNING", "DONE"]);
    expect(fin).toMatchObject({ status: "DONE", hw_fidelity: 0.7 });
    expect(f).toHaveBeenCalledTimes(3);
    expect(f.mock.calls[0][0]).toBe("/api/qpu/status?job=J");
  });
  it("an aborted signal stops polling with no further fetches", async () => {
    vi.useFakeTimers();
    const f = vi.fn(async () => jsonRes(200, { job_id: "J", status: "QUEUED", elapsed_s: 1 }));
    vi.stubGlobal("fetch", f);
    const ac = new AbortController();
    const p = pollStatus("J", { signal: ac.signal, intervalMs: 5000 });
    const caught = p.catch((e) => e);
    await vi.advanceTimersByTimeAsync(5000);
    expect(f).toHaveBeenCalledTimes(2);
    ac.abort();
    const err = await caught;
    expect(err.name).toBe("AbortError");
    await vi.advanceTimersByTimeAsync(60000);
    expect(f).toHaveBeenCalledTimes(2);
  });
  it("gives up after repeated helper failures instead of spinning forever", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("Failed to fetch"); }));
    const fin = await pollStatus("J", { intervalMs: 1 });
    expect(fin.status).toBe("SERVER-DOWN");
  });
});

// ---------------------------------------------------------------- panel controller
const GOOD = { ok: true, credentials: true, runtime: true, jobs_used: 0, jobs_max: 3, max_execution_time_s: 60, shots: 1024 };
const kRow = [0.1, 0.95, 0.3];
const sel = (patient, slug = "heart") => ({ slug, patient, kRow });

function fakeClient(over = {}) {
  const polls = [];
  const c = {
    health: vi.fn(async () => GOOD),
    submit: vi.fn(async () => ({ kind: "submitted", job_id: "J1", backend: "ibm_fake", pending_jobs: 4, partner: 1, ideal_fidelity: 0.95, submitted_at: 1000 })),
    pollStatus: vi.fn((jobId, opts) => new Promise((resolve, reject) => {
      polls.push({ jobId, opts, resolve });
      opts.signal.addEventListener("abort", () => { const e = new Error("aborted"); e.name = "AbortError"; reject(e); });
    })),
    polls,
    ...over,
  };
  return c;
}

describe("qpu panel controller (free-tier guards)", () => {
  it("never submits on load, re-activation, patient switch or Play all", async () => {
    const client = fakeClient();
    const ctl = createQpuController({ client });
    ctl.activate(sel(0));
    await flush();
    expect(ctl.view().phase).toBe("ready");
    for (let i = 0; i < 5; i++) { ctl.activate(sel(i % 3)); ctl.deactivate(); ctl.activate(sel(1)); await flush(); }
    expect(client.health).toHaveBeenCalled();
    expect(client.submit).not.toHaveBeenCalled();
  });

  it("button opens a confirm step; submit only after Run is confirmed", async () => {
    const client = fakeClient();
    const ctl = createQpuController({ client });
    ctl.activate(sel(0));
    await flush();
    ctl.confirmRun(); // no confirm step open yet -> ignored
    await flush();
    expect(client.submit).not.toHaveBeenCalled();
    ctl.requestRun();
    expect(ctl.view().phase).toBe("confirm");
    expect(client.submit).not.toHaveBeenCalled();
    ctl.cancel();
    expect(ctl.view().phase).toBe("ready");
    ctl.requestRun();
    ctl.confirmRun();
    ctl.confirmRun(); // double click
    await flush();
    expect(client.submit).toHaveBeenCalledTimes(1);
    expect(client.submit).toHaveBeenCalledWith("heart", 0, { confirm: true });
    expect(ctl.view().phase).toBe("running");
    expect(client.pollStatus).toHaveBeenCalledTimes(1);
  });

  it("shows the result only for the matching {slug, patient}; switching aborts polling", async () => {
    const client = fakeClient();
    const ctl = createQpuController({ client });
    ctl.activate(sel(0));
    await flush();
    ctl.requestRun(); ctl.confirmRun();
    await flush();
    const first = client.polls[0];
    ctl.activate(sel(2)); // switch patient
    await flush();
    expect(first.opts.signal.aborted).toBe(true);
    expect(ctl.view().phase).toBe("ready");
    expect(ctl.view().result).toBeNull();
    ctl.activate(sel(0)); // back: resumes polling the same job, no new submit
    await flush();
    expect(client.submit).toHaveBeenCalledTimes(1);
    expect(client.polls.length).toBe(2);
    expect(client.polls[1].jobId).toBe("J1");
    client.polls[1].resolve({ job_id: "J1", status: "DONE", hw_fidelity: 0.81, ideal_fidelity: 0.95, noise_gap: 0.14, partner: 1, backend: "ibm_fake" });
    await flush();
    const v = ctl.view();
    expect(v.phase).toBe("done");
    expect(v.result).toMatchObject({ slug: "heart", patient: 0, hw_fidelity: 0.81, cached: false });
    expect(v.kPair).toBe(0.95);
    ctl.activate(sel(0, "other"));
    await flush();
    expect(ctl.view().result).toBeNull();
  });

  it("leaving step 6 aborts polling", async () => {
    const client = fakeClient();
    const ctl = createQpuController({ client });
    ctl.activate(sel(0));
    await flush();
    ctl.requestRun(); ctl.confirmRun();
    await flush();
    ctl.deactivate();
    expect(client.polls[0].opts.signal.aborted).toBe(true);
    expect(ctl.view().phase).toBe("hidden");
  });

  it("disables the button with the specific messages", async () => {
    const cases = [
      [{ ok: false, reason: "server-down" }, MSG.serverDown],
      [{ ...GOOD, credentials: false }, MSG.noCredentials],
      [{ ...GOOD, runtime: false }, MSG.noRuntime],
      [{ ...GOOD, jobs_used: 3 }, "QPU budget for this session used (3/3)"],
    ];
    for (const [h, msg] of cases) {
      const client = fakeClient({ health: vi.fn(async () => h) });
      const ctl = createQpuController({ client });
      ctl.activate(sel(0));
      await flush();
      expect(ctl.view().phase).toBe("unavailable");
      expect(ctl.view().messages).toContain(msg);
      ctl.requestRun(); ctl.confirmRun();
      await flush();
      expect(client.submit).not.toHaveBeenCalled();
      expect(renderQpuPanel(ctl.view())).not.toContain('data-act="run"');
    }
    expect(MSG.serverDown).toBe("Start the helper: python scripts/qpu_server.py");
    expect(MSG.noRuntime).toBe("pip install qiskit-ibm-runtime");
  });

  it("handles 409 (shows the running job) and 429 (budget)", async () => {
    const client = fakeClient({ submit: vi.fn(async () => ({ kind: "busy", job_id: "J9", error: "busy" })) });
    const ctl = createQpuController({ client });
    ctl.activate(sel(0));
    await flush();
    ctl.requestRun(); ctl.confirmRun();
    await flush();
    expect(ctl.view().phase).toBe("running");
    expect(ctl.view().job.job_id).toBe("J9");
    expect(client.polls[0].jobId).toBe("J9");
    // a foreign job's result is never shown as this patient's result
    client.polls[0].resolve({ job_id: "J9", status: "DONE", hw_fidelity: 0.5 });
    await flush();
    expect(ctl.view().result).toBeNull();

    const c2 = fakeClient({ submit: vi.fn(async () => ({ kind: "budget", error: "budget" })), health: vi.fn().mockResolvedValueOnce(GOOD).mockResolvedValue({ ...GOOD, jobs_used: 3 }) });
    const ctl2 = createQpuController({ client: c2 });
    ctl2.activate(sel(0));
    await flush();
    ctl2.requestRun(); ctl2.confirmRun();
    await flush();
    expect(ctl2.view().phase).toBe("unavailable");
    expect(ctl2.view().messages).toContain("QPU budget for this session used (3/3)");
  });

  it("shows cached results with a tag and renders the free-tier texts", async () => {
    const client = fakeClient({ submit: vi.fn(async () => ({ kind: "cached", job_id: "J0", hw_fidelity: 0.7, ideal_fidelity: 0.95, noise_gap: 0.25, partner: 1, backend: "ibm_x" })) });
    const ctl = createQpuController({ client });
    ctl.activate(sel(0));
    await flush();
    expect(renderQpuPanel(ctl.view())).toContain(RUN_LABEL);
    expect(renderQpuPanel(ctl.view())).toContain("0/3");
    ctl.requestRun();
    expect(renderQpuPanel(ctl.view())).toContain(confirmText(1024, 60));
    expect(confirmText(1024, 60)).toBe("This sends 1 circuit × 1024 shots to a real IBM QPU and uses free-tier time (~seconds, capped at 60 s). Continue?");
    ctl.confirmRun();
    await flush();
    const v = ctl.view();
    expect(v.phase).toBe("done");
    expect(v.result.cached).toBe(true);
    expect(renderQpuPanel(v)).toContain("cached result");
    expect(client.pollStatus).not.toHaveBeenCalled();
  });

  it("escapes server-provided strings", async () => {
    const client = fakeClient({ submit: vi.fn(async () => ({ kind: "error", status: 502, error: "<img src=x onerror=alert(1)>" })) });
    const ctl = createQpuController({ client });
    ctl.activate(sel(0));
    await flush();
    ctl.requestRun(); ctl.confirmRun();
    await flush();
    const html = renderQpuPanel(ctl.view());
    expect(html).not.toContain("<img");
    expect(html).toContain("&lt;img");
  });
});
