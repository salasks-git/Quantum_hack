import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import "./style.css";
import { loadIndex, loadDisease, loadFixture, FIXTURE_SLUG } from "./data.js";
import { SceneManager } from "./sceneManager.js";
import { mountDatasetExplorer } from "./datasetExplorer.js";
import { renderLegend, mountLegendToggle } from "./legend.js";
import { mountAllPatients } from "./allPatients.js";
import { mountSettingsDrawer } from "./settingsDrawer.js";
import scene1 from "./scenes/scene1.js";
import scene2 from "./scenes/scene2.js";
import scene3 from "./scenes/scene3.js";
import scene4 from "./scenes/scene4.js";
import scene5 from "./scenes/scene5.js";
import scene6 from "./scenes/scene6.js";
import { mountQpuPanel } from "./qpuPanel.js";
import sceneBandwidth from "./scenes/sceneBandwidth.js";
import sceneQuhama from "./scenes/sceneQuhama.js";
import { mountCustomPatient, withCustomPatient, CUSTOM_LABEL, MSG as CUSTOM_MSG } from "./customPatient.js"; // 22c

// Step table: step n (1-based) = STEP_FACTORIES[n - 1]. Append new scenes here; the step count,
// the step dots, Prev/Next bounds, the URL ?step= range and web/scripts/smoke.sh all derive from it.
const STEP_FACTORIES = [scene1, scene2, scene3, scene4, scene5, scene6, sceneBandwidth, sceneQuhama];
const N_STEPS = STEP_FACTORIES.length;
const clampStep = (n) => Math.min(N_STEPS, Math.max(1, n));
const QPU_STEP = 6; // scene 6 shows the live-run panel

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const state = { index: [], slug: null, base: null, realPatient: 0, data: null, patient: 0, step: 1, playAll: false, advanced: false };

const stage = $("stage");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setClearColor(0x070b16);
stage.prepend(renderer.domElement);
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableZoom = false;
controls.autoRotate = true;
controls.autoRotateSpeed = 1.0; // time-based (r160): ~0.105 rad/s; SceneManager sways it within +-0.5 rad
const manager = new SceneManager(renderer, camera, controls);
const pauseAt = parseFloat(params.get("pauseAt")); // rehearsal/debug: auto-pause at scene time N
const fixture = params.get("fixture"); // dev/test: q2 | q4 = 2/4-qubit fixture for vertebral_column

function resize() {
  const w = stage.clientWidth, h = stage.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
}
addEventListener("resize", resize);
new ResizeObserver(resize).observe(stage); // top-bar wrap, error banner, window size

function showError(msg) { const e = $("error"); e.textContent = msg; e.hidden = false; }
function clearError() { $("error").hidden = true; }

// --- dataset explorer overlay ---
const datasetExplorer = mountDatasetExplorer({ button: $("dataset-btn"), container: $("dataset-overlay"), getSlug: () => state.slug });
if (params.get("dataset") === "1") datasetExplorer.setOpen(true);
const legendToggle = mountLegendToggle($("legend-btn"), $("legend"), { storageKey: "qmed.legend.open" });
function updateLegend() {
  const cur = manager.current;
  const panel = $("legend");
  panel.innerHTML = "";
  const head = document.createElement("div");
  head.className = "lg-head";
  head.textContent = `Legend \u00B7 ${cur.title}`;
  const body = document.createElement("div");
  renderLegend(body, cur.legend || []);
  panel.append(head, body);
  panel.scrollTop = 0;
}

// Live IBM QPU panel (step 6 only). It only ever runs health/status checks on its own;
// a job is submitted solely by the user clicking Run and then confirming (never by Play all or load).
const qpuPanel = mountQpuPanel($("qpu-panel"));
function updateQpuPanel() {
  const on = state.step === QPU_STEP && !!state.data;
  stage.classList.toggle("qpu-on", on);
  if (on && state.data.test[state.patient].label == null) { // 22c: custom patient -> no live QPU run
    qpuPanel.deactivate();
    $("qpu-panel").hidden = false;
    $("qpu-panel").innerHTML = `<div class="qp-head"><span>Real IBM QPU · free tier</span></div><div class="qp-msg qp-warn">${CUSTOM_MSG.qpuDisabled}</div>`;
  } else if (on) {
    qpuPanel.activate({ slug: state.slug, patient: state.patient, kRow: state.data.qsvc.kernel_test[state.patient] });
  } else {
    qpuPanel.deactivate();
  }
}

function rebuild() {
  if (!state.data) return;
  state.advanced = false;
  manager.show(STEP_FACTORIES[state.step - 1], { data: state.data, patient: state.patient });
  updateLegend();
  updateQpuPanel();
  document.querySelectorAll(".dot").forEach((d) => d.classList.toggle("active", +d.dataset.step === state.step));
  history.replaceState(null, "", `?disease=${state.slug}&patient=${state.patient}&step=${state.step}${fixture ? `&fixture=${fixture}` : ""}`);
}

function fillPatients() {
  const sel = $("patient");
  sel.innerHTML = "";
  state.data.test.forEach((p, i) => {
    const o = document.createElement("option");
    o.value = i;
    o.textContent = p.label == null ? CUSTOM_LABEL : `Test patient #${i} — actual: ${p.label === 1 ? "Disease" : "Healthy"}`;
    sel.appendChild(o);
  });
  state.patient = Math.min(state.patient, state.data.test.length - 1);
  sel.value = state.patient;
}

async function selectDisease(slug) {
  try {
    const data = fixture && slug === FIXTURE_SLUG ? await loadFixture(fixture) : await loadDisease(slug);
    clearError();
    state.slug = slug;
    state.base = data;
    state.data = data;
    state.realPatient = state.patient;
    customPatient.reset();
    $("disease").value = slug;
    fillPatients();
    rebuild();
    allPatients.refresh(); // all-patients overlay follows the disease
    datasetExplorer.refresh();
    settingsDrawer.clearCustom(); // static data for this disease: drop any Custom run state
  } catch (e) {
    showError(e.message);
  }
}

// --- Settings drawer + live retraining (Task 22a) ---
function toast(msg) {
  const t = $("toast");
  t.textContent = msg; t.hidden = false;
  clearTimeout(toast.t);
  toast.t = setTimeout(() => { t.hidden = true; }, 6000);
}
// Data-swap hook: replace the active dataset object (Custom run) and re-render the current step.
function swapData(data) {
  state.base = data; // a Custom run replaces the base data; the custom patient restarts from it
  state.data = data;
  state.realPatient = state.patient;
  customPatient.reset();
  fillPatients();
  rebuild();
  allPatients.refresh();
  datasetExplorer.refresh();
}
const settingsDrawer = mountSettingsDrawer({
  button: $("settings-btn"), container: $("settings-drawer"), badge: $("custom-badge"),
  getData: () => state.data, getSlug: () => state.slug,
  getLabel: () => (state.index.find((d) => d.slug === state.slug) || {}).label,
  onApply: swapData, onReset: () => selectDisease(state.slug), onToast: toast,
});
if (params.get("settings") === "1") settingsDrawer.setOpen(true);
// --- end Settings drawer ---

// --- All patients overlay (Task 17) ---
const allPatients = mountAllPatients($("all-btn"), $("allpatients"), {
  getData: () => state.data,
  getPatient: () => (state.data.test[state.patient] || {}).index ?? state.patient,
  onSelect: (i) => { if (!Number.isInteger(i)) return; leaveCustom(); state.patient = i; $("patient").value = i; rebuild(); },
});
// --- end All patients overlay ---

// --- Custom patient (Task 22c): sliders -> /api/live/predict -> synthetic patient appended to a copy of the data ---
function leaveCustom() {
  if (state.data === state.base) return;
  state.data = state.base;
  state.patient = state.realPatient;
  fillPatients();
  customPatient.reset();
}
const customPatient = mountCustomPatient({
  button: $("custom-btn"), panel: $("custom-panel"),
  getCtx: () => ({ data: state.base, slug: state.slug, patient: state.data === state.base ? state.patient : state.realPatient }),
  onPatient: (resp) => {
    if (state.data === state.base) state.realPatient = state.patient;
    const c = withCustomPatient(state.base, resp);
    state.data = c.data;
    state.patient = c.index;
    fillPatients();
    rebuild();
  },
  onRevert: () => { leaveCustom(); rebuild(); },
});
// --- end Custom patient ---

function gotoStep(n) {
  state.step = clampStep(n);
  rebuild();
}

$("disease").addEventListener("change", (e) => { state.patient = 0; selectDisease(e.target.value); });
$("patient").addEventListener("change", (e) => {
  if (state.data.test[+e.target.value].label != null) leaveCustom(); // 22c: picking a real patient drops the custom one
  state.patient = +e.target.value; rebuild();
});
$("prev").addEventListener("click", () => gotoStep(state.step - 1));
$("next").addEventListener("click", () => gotoStep(state.step + 1));
$("replay").addEventListener("click", rebuild);
// one dot per step in the table
$("dots").innerHTML = "";
for (let n = 1; n <= N_STEPS; n++) {
  const d = document.createElement("button");
  d.className = "dot";
  d.dataset.step = n;
  d.textContent = n;
  d.addEventListener("click", () => gotoStep(n));
  $("dots").appendChild(d);
}
$("playall").addEventListener("click", () => {
  state.playAll = !state.playAll;
  $("playall").classList.toggle("on", state.playAll);
  if (state.playAll) gotoStep(1);
});

function syncPause() {
  const p = manager.paused;
  $("pause").innerHTML = p ? "&#9654; Resume" : "&#9208; Pause";
  $("pause").classList.toggle("on", p);
}
function togglePause() { manager.togglePause(); syncPause(); }
$("pause").addEventListener("click", togglePause);
addEventListener("keydown", (e) => {
  if (e.code === "KeyL" && !e.repeat && !e.metaKey && !e.ctrlKey && !e.altKey) {
    const tag = document.activeElement && document.activeElement.tagName;
    if (tag !== "SELECT" && tag !== "INPUT") { legendToggle.setOpen(!legendToggle.isOpen()); return; }
  }
  if (e.code !== "Space" || e.repeat) return;
  const tag = document.activeElement && document.activeElement.tagName;
  if (tag === "SELECT") return;
  if (tag === "BUTTON" || tag === "INPUT") document.activeElement.blur();
  e.preventDefault();
  togglePause();
});

function updateBadge() {
  const b = $("badge");
  let txt = "";
  if (manager.paused) txt = "\u23F8 Paused";
  else if (manager.holding > 0) txt = `\u23F8 3s hold \u00B7 ${Math.ceil(manager.holding)}`;
  if (b.textContent !== txt) b.textContent = txt;
  b.hidden = !txt;
  const cap = $("caption");
  const ch = cap.innerHTML ? cap.offsetHeight + "px" : "0px";
  b.style.setProperty("--cap-h", ch);
  stage.style.setProperty("--cap-h", ch);
}

function loop() {
  requestAnimationFrame(loop);
  if (pauseAt >= 0 && !manager.paused && manager.elapsed >= pauseAt) { manager.elapsed = pauseAt; manager.setPaused(true); syncPause(); } // snap: slow (software-GL) frames can overshoot
  const caption = manager.tick();
  if ($("caption").innerHTML !== caption) $("caption").innerHTML = caption;
  updateBadge();
  if (state.playAll && !state.advanced && manager.elapsed > manager.duration) {
    if (state.step < N_STEPS) { state.advanced = true; gotoStep(state.step + 1); }
    else { state.playAll = false; $("playall").classList.remove("on"); }
  }
}

async function init() {
  resize();
  state.step = clampStep(parseInt(params.get("step")) || 1);
  state.patient = Math.max(0, parseInt(params.get("patient")) || 0);
  try {
    state.index = await loadIndex();
  } catch (e) { showError(e.message); return; }
  const sel = $("disease");
  state.index.forEach((d) => {
    const o = document.createElement("option");
    o.value = d.slug;
    o.textContent = `${d.label} (${d.n_features} features)`;
    sel.appendChild(o);
  });
  const slug = params.get("disease") || (fixture ? FIXTURE_SLUG : state.index[0].slug);
  if (![...sel.options].some((o) => o.value === slug)) {
    const o = document.createElement("option");
    o.value = slug; o.textContent = slug; sel.appendChild(o);
  }
  sel.value = slug;
  loop();
  await selectDisease(slug);
  if (params.get("all") === "1") allPatients.setOpen(true);
}
init();
