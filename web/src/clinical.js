// Clinical scoreboard over a disease's whole test set, matching the Streamlit benchmark tab:
// sklearn fbeta_score(y_test, pred, beta=2, zero_division=0); "missed" = false negatives.
// Uses each model's hard `pred`, never `prob`.

const BETA2 = 4; // beta = 2

export function clinicalScores(test, key) {
  let tp = 0, fp = 0, fn = 0, tn = 0;
  for (const p of test) {
    const sick = p.label === 1;
    const flagged = p.models[key].pred === 1;
    if (sick && flagged) tp++;
    else if (sick) fn++;
    else if (flagged) fp++;
    else tn++;
  }
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  // F2 = (1+4)·P·R / (4·P + R), 0 when the denominator is 0. Computed from the counts
  // (algebraically identical) so equal ratios compare exactly equal for tie-breaking.
  const den = (1 + BETA2) * tp + BETA2 * fn + fp;
  const f2 = tp && den ? ((1 + BETA2) * tp) / den : 0;
  return { tp, fp, fn, tn, recall, precision, f2, missed: fn };
}

/** Key with the highest F2; ties -> fewer missed diagnoses, then earlier in `keys`. */
export function clinicalLeader(test, keys) {
  let best = null, bestS = null;
  for (const k of keys) {
    const s = clinicalScores(test, k);
    if (!bestS || s.f2 > bestS.f2 + 1e-12 || (Math.abs(s.f2 - bestS.f2) <= 1e-12 && s.missed < bestS.missed)) {
      best = k;
      bestS = s;
    }
  }
  return best;
}
