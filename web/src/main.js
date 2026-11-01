import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import "./style.css";
import { loadIndex, loadDisease } from "./data.js";
import { SceneManager } from "./sceneManager.js";
import { renderLegend, mountLegendToggle } from "./legend.js";
import { mountAllPatients } from "./allPatients.js";
import scene1 from "./scenes/scene1.js";
import scene2 from "./scenes/scene2.js";
import scene3 from "./scenes/scene3.js";
import scene4 from "./scenes/scene4.js";
import scene5 from "./scenes/scene5.js";

const STEP_FACTORIES = [scene1, scene2, scene3, scene4, scene5];

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const state = { index: [], slug: null, data: null, patient: 0, step: 1, playAll: false, advanced: false };

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

function rebuild() {
  if (!state.data) return;
  state.advanced = false;
  manager.show(STEP_FACTORIES[state.step - 1], { data: state.data, patient: state.patient });
  updateLegend();
  document.querySelectorAll(".dot").forEach((d) => d.classList.toggle("active", +d.dataset.step === state.step));
  history.replaceState(null, "", `?disease=${state.slug}&patient=${state.patient}&step=${state.step}`);
}

function fillPatients() {
  const sel = $("patient");
  sel.innerHTML = "";
  state.data.test.forEach((p, i) => {
    const o = document.createElement("option");
    o.value = i;
    o.textContent = `Test patient #${i} — actual: ${p.label === 1 ? "Disease" : "Healthy"}`;
    sel.appendChild(o);
  });
  state.patient = Math.min(state.patient, state.data.test.length - 1);
  sel.value = state.patient;
}

async function selectDisease(slug) {
  try {
    const data = await loadDisease(slug);
    clearError();
    state.slug = slug;
    state.data = data;
    $("disease").value = slug;
    fillPatients();
    rebuild();
    allPatients.refresh(); // all-patients overlay follows the disease
  } catch (e) {
    showError(e.message);
  }
}

// --- All patients overlay (Task 17) ---
const allPatients = mountAllPatients($("all-btn"), $("allpatients"), {
  getData: () => state.data,
  getPatient: () => state.patient,
  onSelect: (i) => { state.patient = i; $("patient").value = i; rebuild(); },
});
// --- end All patients overlay ---

function gotoStep(n) {
  state.step = Math.min(5, Math.max(1, n));
  rebuild();
}

$("disease").addEventListener("change", (e) => { state.patient = 0; selectDisease(e.target.value); });
$("patient").addEventListener("change", (e) => { state.patient = +e.target.value; rebuild(); });
$("prev").addEventListener("click", () => gotoStep(state.step - 1));
$("next").addEventListener("click", () => gotoStep(state.step + 1));
$("replay").addEventListener("click", rebuild);
document.querySelectorAll(".dot").forEach((d) => d.addEventListener("click", () => gotoStep(+d.dataset.step)));
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
  if (pauseAt >= 0 && !manager.paused && manager.elapsed >= pauseAt) { manager.setPaused(true); syncPause(); }
  const caption = manager.tick();
  if ($("caption").innerHTML !== caption) $("caption").innerHTML = caption;
  updateBadge();
  if (state.playAll && !state.advanced && manager.elapsed > manager.duration) {
    if (state.step < 5) { state.advanced = true; gotoStep(state.step + 1); }
    else { state.playAll = false; $("playall").classList.remove("on"); }
  }
}

async function init() {
  resize();
  state.step = Math.min(5, Math.max(1, parseInt(params.get("step")) || 1));
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
  const slug = params.get("disease") || state.index[0].slug;
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
