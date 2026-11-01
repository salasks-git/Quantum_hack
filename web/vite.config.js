import { defineConfig } from 'vite';

// M7 / free-tier hardening: the helper port is configurable. Agents, smoke runs and test harnesses
// MUST set QMED_HELPER_PORT to a stub or unused port, never the presenter's real helper (8765).
const port = Number(process.env.QMED_HELPER_PORT) || 8765;
const proxy = { '/api': { target: `http://127.0.0.1:${port}`, changeOrigin: false } };

export default defineConfig({
  server: { proxy },
  preview: { proxy },
});
