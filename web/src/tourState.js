// Pure-ish tour state helpers used by main.js (kept separate so they are unit-testable without WebGL).
import { MSG as CUSTOM_MSG } from "./customPatient.js";

export const FIXTURE_QPU_MSG = "Dev fixture data (?fixture=…) is not a live run: real IBM QPU runs are disabled here.";
export const CUSTOM_QPU_MSG = `${CUSTOM_MSG.qpuDisabled}: pick a real test patient to send it to IBM.`;

/**
 * Selection for the scene 6 QPU panel. Uses the ACTIVE data (static JSON or a live Custom run): its
 * settings, its kernel row and its run_id, so the helper runs exactly that config and pair.
 * Custom slider patients and dev fixtures get a `disabled` reason instead (Run button disabled).
 */
export function qpuSelection(st) {
  const p = st.data.test[st.patient] || {};
  const runId = (st.base && st.base.run_id) || null;
  let disabled = null;
  if (p.label == null) disabled = CUSTOM_QPU_MSG;
  else if (runId && String(runId).startsWith("fixture-")) disabled = FIXTURE_QPU_MSG;
  const label = ((st.index || []).find((d) => d.slug === st.slug) || {}).label || st.data.label || st.slug;
  return {
    slug: st.slug, patient: st.patient, runId: disabled ? null : runId, label, disabled,
    kRow: st.data.qsvc.kernel_test[st.patient], settings: st.data.settings || null,
  };
}

/**
 * C2: adopt new base data (disease switch, Custom run, Reset to precomputed). Resolve the REAL patient
 * index first (a custom patient's synthetic index does not exist in the new data), clamp it, refill the
 * patient select, and only THEN reset the custom-patient panel (its build() reads data.test[patient]).
 */
export function adoptData(st, data, { fillPatients = () => {}, resetCustom = () => {} } = {}) {
  const real = st.data && st.data !== st.base ? st.realPatient : st.patient;
  st.base = data;
  st.data = data;
  st.patient = st.realPatient = Math.max(0, Math.min(Number(real) || 0, data.test.length - 1));
  fillPatients();
  resetCustom();
  return st.patient;
}

/** I1: the dropdown's choice is read BEFORE leaving the custom patient (which refills the select). */
export function choosePatient(st, i, { leaveCustom = () => {} } = {}) {
  if (st.data.test[i] && st.data.test[i].label != null) leaveCustom();
  st.patient = i;
  return i;
}
