// Pure helpers for scene 6's timing bars and efficiency frontier.

/** Python's f"{x:.{d}f}" (round-half-even on the exact binary value); JS toFixed rounds exact ties up. */
export function pyFixed(x, d) {
  const s = x.toFixed(d);
  const exact = Math.abs(x).toFixed(Math.min(100, d + 30));
  const tail = exact.slice(exact.indexOf(".") + 1 + d);
  if (!/^50*$/.test(tail)) return s; // not an exact tie
  const kept = exact.slice(0, exact.length - tail.length).replace(/\.$/, "");
  const lastDigit = Number(kept[kept.length - 1]);
  if (lastDigit % 2 === 0) return (x < 0 ? "-" : "") + kept; // tie, even digit: round down
  return s;
}

/** Same rule as qmed/metrics.py fmt_duration: '1.234 s' for >= 10 ms, else '4.2 ms'. */
export function fmtDuration(seconds) {
  return seconds >= 0.01 ? `${pyFixed(seconds, 3)} s` : `${pyFixed(seconds * 1000, 1)} ms`;
}

/**
 * Keys of the Pareto-optimal points, sorted by time. A point is dominated if another point has
 * time <= t AND score >= s with at least one strict. Identical points do not dominate each other.
 */
export function paretoFront(points) {
  return points
    .filter((p) => !points.some((o) => o !== p && o.time <= p.time && o.score >= p.score && (o.time < p.time || o.score > p.score)))
    .sort((a, b) => a.time - b.time || b.score - a.score)
    .map((p) => p.key);
}

/** Position along an axis of `length` for a log scale spanning [min, max] (clamped to the axis). */
export function logScale(value, min, max, length) {
  const f = Math.log(value / min) / Math.log(max / min);
  return Math.min(1, Math.max(0, f)) * length;
}

/** Axis label for a power-of-ten number of seconds: 1e-4 -> "100 µs", 0.001 -> "1 ms", 0.1 -> "100 ms", 1 -> "1 s", 100 -> "100 s". */
export function tickLabel(exp) {
  const s = 10 ** exp;
  if (s < 0.001) return `${Math.round(s * 1e6)} µs`;
  return s < 1 ? `${Math.round(s * 1000)} ms` : `${Math.round(s)} s`;
}
