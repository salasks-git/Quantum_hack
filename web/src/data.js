const base = () => (import.meta.env && import.meta.env.BASE_URL) || "/";

async function getJson(path, errMsg) {
  const res = await fetch(base() + path);
  if (!res.ok) throw new Error(errMsg);
  try {
    return await res.json();
  } catch {
    throw new Error(errMsg);
  }
}

export function loadIndex() {
  return getJson("data/index.json", "Could not load data index");
}

export function loadDisease(slug) {
  return getJson(`data/${slug}.json`, `Could not load data for '${slug}'`);
}

// Dev/test only: `?fixture=q2|q4` loads a real 2- or 4-qubit live-run export from web/test-fixtures/
// (vertebral_column, 60 samples, VQC 15) instead of the static file. Lazy chunks: never fetched otherwise.
const FIXTURES = import.meta.glob("../test-fixtures/vertebral_q*.json", { import: "default" });
export const FIXTURE_SLUG = "vertebral_column";

export async function loadFixture(name) {
  const load = /^q\d$/.test(name || "") && FIXTURES[`../test-fixtures/vertebral_${name}.json`];
  if (!load) throw new Error(`Unknown test fixture '${name}'`);
  return load();
}
