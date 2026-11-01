import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import "./style.css";
import { loadIndex, loadDisease } from "./data.js";
import { SceneManager } from "./sceneManager.js";
import { makePlaceholder } from "./scenes/placeholder.js";
import scene1 from "./scenes/scene1.js";
import scene3 from "./scenes/scene3.js";
import scene5 from "./scenes/scene5.js";

// Replace entries with real scene factories as they are built.
const STEP_FACTORIES = [scene1, makePlaceholder(2), scene3, makePlaceholder(4), scene5];

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
controls.autoRotateSpeed = 0.6;
const manager = new SceneManager(renderer, camera, controls);

function resize() {
  const w = stage.clientWidth, h = stage.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
}
addEventListener("resize", resize);

function showError(msg) { const e = $("error"); e.textContent = msg; e.hidden = false; }
function clearError() { $("error").hidden = true; }

function rebuild() {
  if (!state.data) return;
  state.advanced = false;
  manager.show(STEP_FACTORIES[state.step - 1], { data: state.data, patient: state.patient });
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
  } catch (e) {
    showError(e.message);
  }
}

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

function loop() {
  requestAnimationFrame(loop);
  const caption = manager.tick();
  if ($("caption").innerHTML !== caption) $("caption").innerHTML = caption;
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
}
init();
