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

// Kernel Bandwidth Lab data (lazy; cached per slug, failures are not cached).
const bandwidthCache = new Map();
export function loadBandwidth(slug) {
  if (!bandwidthCache.has(slug)) {
    const p = getJson(`data/${slug}.bandwidth.json`, `Could not load Kernel Bandwidth Lab data for '${slug}'`);
    bandwidthCache.set(slug, p);
    p.catch(() => bandwidthCache.delete(slug));
  }
  return bandwidthCache.get(slug);
}
