import { app } from "../../scripts/app.js";

const NODE_NAME = "Kimodo_CurveToPoints";

const DEFAULT_CURVE = [
  { x: 0, y: 0, z: 0 },
  { x: 2, y: 2, z: 2 },
  { x: 4, y: 0, z: 4 },
];

const MIN_WIDTH = 360;
const MIN_HEIGHT = 340;

const EDITOR_HTML = `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { overflow:hidden; background:#2a2a3a; font-family:monospace; user-select:none; }
  #c { width:100%; height:100vh; display:block; }
  #info {
    position:absolute; top:6px; left:8px;
    color:#888; font-size:10px; pointer-events:none;
  }
  #hint {
    position:absolute; bottom:6px; left:8px; right:8px;
    text-align:center; color:#555; font-size:9px;
    pointer-events:none;
  }
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="info">3D Curve · <span id="ptCount">0</span> pts</div>
<div id="hint">Left-click: select/add · Drag point: move · Del: delete · Right-drag: orbit · Scroll: zoom</div>
<script type="importmap">
{"imports":{
  "three":"__THREE_BASE_URL__/three/three.module.js",
  "three/addons/":"__THREE_BASE_URL__/three/"
}}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';

// ---- 3D curve math (inlined) ----
function dist3D(a, b) {
  const dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}
function computeChordLengthParams(points) {
  const chords = [0];
  for (let i = 1; i < points.length; i++) chords.push(chords[i - 1] + dist3D(points[i - 1], points[i]));
  const total = chords[chords.length - 1];
  let t = chords.map(c => total > 1e-8 ? c / total : c);
  // Ensure strictly increasing t — nudge duplicates by epsilon
  for (let i = 1; i < t.length; i++) {
    if (t[i] <= t[i - 1]) t[i] = t[i - 1] + 1e-10;
  }
  // Clamp last value to 1.0
  if (t.length > 0) t[t.length - 1] = 1.0;
  return { t, total };
}
function computeTangents3D(points, t) {
  const n = points.length, tangents = [];
  for (let i = 0; i < n; i++) {
    let tx, ty, tz;
    if (i === 0) { tx = points[1].x - points[0].x; ty = points[1].y - points[0].y; tz = points[1].z - points[0].z; }
    else if (i === n - 1) { tx = points[n - 1].x - points[n - 2].x; ty = points[n - 1].y - points[n - 2].y; tz = points[n - 1].z - points[n - 2].z; }
    else {
      const dt = t[i + 1] - t[i - 1];
      if (dt > 1e-8) { tx = (points[i + 1].x - points[i - 1].x) / dt; ty = (points[i + 1].y - points[i - 1].y) / dt; tz = (points[i + 1].z - points[i - 1].z) / dt; }
      else { tx = points[i + 1].x - points[i - 1].x; ty = points[i + 1].y - points[i - 1].y; tz = points[i + 1].z - points[i - 1].z; }
    }
    tangents.push({ x: tx, y: ty, z: tz });
  }
  return tangents;
}
function hermiteInterpVal(v0, v1, m0, m1, t) {
  const t2 = t * t, t3 = t2 * t;
  return (2 * t3 - 3 * t2 + 1) * v0 + (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) * v1 + (t3 - t2) * m1;
}
function evalHermiteSegment3D(p0, p1, m0, m1, t01) {
  return { x: hermiteInterpVal(p0.x, p1.x, m0.x, m1.x, t01), y: hermiteInterpVal(p0.y, p1.y, m0.y, m1.y, t01), z: hermiteInterpVal(p0.z, p1.z, m0.z, m1.z, t01) };
}
function sampleCurve3D(points, tangents, t, resolution) {
  if (points.length < 2) return points.slice();
  const result = [];
  for (let i = 0; i < resolution; i++) {
    const u = i / (resolution - 1);
    let seg = 0;
    while (seg < points.length - 2 && t[seg + 1] <= u) seg++;
    const t0 = t[seg], t1 = t[seg + 1], segLen = t1 - t0, localT = segLen > 1e-8 ? (u - t0) / segLen : 0;
    const m0 = { x: tangents[seg].x * segLen, y: tangents[seg].y * segLen, z: tangents[seg].z * segLen };
    const m1 = { x: tangents[seg + 1].x * segLen, y: tangents[seg + 1].y * segLen, z: tangents[seg + 1].z * segLen };
    result.push(evalHermiteSegment3D(points[seg], points[seg + 1], m0, m1, localT));
  }
  return result;
}

// ---- state ----
let points = [];
let selectedIdx = -1;
let curveReady = false;

const curveRes = 100;
const maxPoints = 24;
const ptRadius = 0.18;

// ---- direct drag state ----
let isDraggingPoint = false;
let dragPointIdx = -1;
let dragPlane = new THREE.Plane();
let dragOffset = new THREE.Vector3();
const _dragIsect = new THREE.Vector3();

// ---- renderer ----
const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x2a2a3a);
renderer.shadowMap.enabled = false;

// ---- scene ----
const scene = new THREE.Scene();
scene.add(new THREE.AmbientLight(0xffffff, 0.5));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
dirLight.position.set(5, 10, 7);
scene.add(dirLight);
const fillLight = new THREE.DirectionalLight(0x8888ff, 0.3);
fillLight.position.set(-3, 1, -4);
scene.add(fillLight);

// ground grid
const grid = new THREE.GridHelper(20, 20, 0x4466aa, 0x334488);
scene.add(grid);

// axes
const axes = new THREE.AxesHelper(2.5);
scene.add(axes);

// invisible ground plane for click-to-add
const groundGeo = new THREE.PlaneGeometry(20, 20);
const groundMat = new THREE.MeshBasicMaterial({ visible: false, side: THREE.DoubleSide });
const groundPlane = new THREE.Mesh(groundGeo, groundMat);
groundPlane.rotation.x = -Math.PI / 2;
scene.add(groundPlane);

// ---- camera ----
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.position.set(6, 5, 8);
camera.lookAt(2, 0, 2);

// ---- controls ----
const orbitControls = new OrbitControls(camera, canvas);
orbitControls.target.set(2, 0, 2);
orbitControls.mouseButtons = {
  LEFT: THREE.MOUSE.ROTATE,
  MIDDLE: THREE.MOUSE.PAN,
  RIGHT: THREE.MOUSE.DOLLY,
};
orbitControls.touches = {
  ONE: THREE.TOUCH.PAN,
  TWO: THREE.TOUCH.DOLLY_PAN,
};
orbitControls.update();

const transformControls = new TransformControls(camera, renderer.domElement);
transformControls.setMode('translate');
transformControls.setSize(0.6);
transformControls.addEventListener('dragging-changed', (e) => {
  orbitControls.enabled = !e.value;
});
scene.add(transformControls);

transformControls.addEventListener('change', () => {
  if (selectedIdx >= 0 && selectedIdx < points.length) {
    const pos = transformControls.object?.position;
    if (pos) {
      points[selectedIdx].x = pos.x;
      points[selectedIdx].y = pos.y;
      points[selectedIdx].z = pos.z;
      rebuildCurve();
      notifyParent();
    }
  }
});

// ---- groups ----
const pointGroup = new THREE.Group();
const curveGroup = new THREE.Group();
scene.add(pointGroup);
scene.add(curveGroup);

// ---- helpers ----
function makePointMesh(color) {
  const geo = new THREE.SphereGeometry(ptRadius, 16, 16);
  const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.3, metalness: 0.1 });
  const mesh = new THREE.Mesh(geo, mat);
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(ptRadius * 1.5, ptRadius * 1.7, 24),
    new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0, side: THREE.DoubleSide })
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = -ptRadius * 0.5;
  mesh.add(ring);
  mesh.userData.ring = ring;
  return mesh;
}

function rebuildScene() {
  // clear
  while (pointGroup.children.length) pointGroup.remove(pointGroup.children[0]);
  while (curveGroup.children.length) curveGroup.remove(curveGroup.children[0]);

  if (points.length < 2) return;

  // point meshes
  points.forEach((p, i) => {
    const mesh = makePointMesh(i === selectedIdx ? 0xffaa44 : 0x44aaff);
    mesh.position.set(p.x, p.y, p.z);
    mesh.userData.idx = i;
    mesh.userData.ring.material.opacity = i === selectedIdx ? 0.4 : 0;
    pointGroup.add(mesh);
  });

  // rebuild curve geometry
  const { t } = computeChordLengthParams(points);
  const tangents = computeTangents3D(points, t);
  const samples = sampleCurve3D(points, tangents, t, curveRes);

  // curve line
  const positions = new Float32Array(samples.length * 3);
  samples.forEach((s, i) => {
    positions[i * 3] = s.x;
    positions[i * 3 + 1] = s.y;
    positions[i * 3 + 2] = s.z;
  });
  const curveGeo = new THREE.BufferGeometry();
  curveGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const curveMat = new THREE.LineBasicMaterial({ color: 0x44aaff, linewidth: 2 });
  const curveLine = new THREE.Line(curveGeo, curveMat);
  curveGroup.add(curveLine);

  // control polygon (dashed)
  const polyPositions = new Float32Array(points.length * 3);
  points.forEach((p, i) => {
    polyPositions[i * 3] = p.x;
    polyPositions[i * 3 + 1] = p.y;
    polyPositions[i * 3 + 2] = p.z;
  });
  const polyGeo = new THREE.BufferGeometry();
  polyGeo.setAttribute('position', new THREE.BufferAttribute(polyPositions, 3));
  const polyMat = new THREE.LineDashedMaterial({ color: 0x666688, dashSize: 0.08, gapSize: 0.08 });
  const polyLine = new THREE.Line(polyGeo, polyMat);
  polyLine.computeLineDistances();
  curveGroup.add(polyLine);

  document.getElementById('ptCount').textContent = points.length;
}

function rebuildCurve() {
  while (curveGroup.children.length) curveGroup.remove(curveGroup.children[0]);
  if (points.length < 2) return;

  const { t } = computeChordLengthParams(points);
  const tangents = computeTangents3D(points, t);
  const samples = sampleCurve3D(points, tangents, t, curveRes);

  const positions = new Float32Array(samples.length * 3);
  samples.forEach((s, i) => {
    positions[i * 3] = s.x;
    positions[i * 3 + 1] = s.y;
    positions[i * 3 + 2] = s.z;
  });
  const curveGeo = new THREE.BufferGeometry();
  curveGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const curveMat = new THREE.LineBasicMaterial({ color: 0x44aaff });
  curveGroup.add(new THREE.Line(curveGeo, curveMat));

  const polyPositions = new Float32Array(points.length * 3);
  points.forEach((p, i) => {
    polyPositions[i * 3] = p.x;
    polyPositions[i * 3 + 1] = p.y;
    polyPositions[i * 3 + 2] = p.z;
  });
  const polyGeo = new THREE.BufferGeometry();
  polyGeo.setAttribute('position', new THREE.BufferAttribute(polyPositions, 3));
  const polyMat = new THREE.LineDashedMaterial({ color: 0x666688, dashSize: 0.08, gapSize: 0.08 });
  const polyLine = new THREE.Line(polyGeo, polyMat);
  polyLine.computeLineDistances();
  curveGroup.add(polyLine);
}

function updatePointVisuals() {
  pointGroup.children.forEach((mesh, i) => {
    const p = points[i];
    if (p) {
      mesh.position.set(p.x, p.y, p.z);
      const isSel = i === selectedIdx;
      mesh.material.color.setHex(isSel ? 0xffaa44 : 0x44aaff);
      mesh.userData.ring.material.opacity = isSel ? 0.4 : 0;
    }
  });
}

function loadCurve(data) {
  points = data.map(p => ({ x: p.x ?? 0, y: p.y ?? 0, z: p.z ?? 0 }));
  selectedIdx = -1;
  isDraggingPoint = false;
  dragPointIdx = -1;
  pendingHitIdx = -1;
  orbitControls.enabled = true;
  if (transformControls.object) {
    transformControls.detach();
  }
  rebuildScene();
}

function selectPoint(idx) {
  if (selectedIdx === idx) return;
  if (transformControls.object) {
    transformControls.detach();
  }
  selectedIdx = idx;
  updatePointVisuals();
  if (idx >= 0 && idx < pointGroup.children.length) {
    transformControls.attach(pointGroup.children[idx]);
  }
}

// ---- raycaster picking ----
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

function getPointMeshes() {
  return pointGroup.children;
}

function hitTestPoint(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  pointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const meshes = getPointMeshes();
  const intersects = raycaster.intersectObjects(meshes);
  if (intersects.length > 0) {
    return intersects[0].object.userData.idx;
  }
  return -1;
}

function hitTestGround(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  pointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const intersects = raycaster.intersectObject(groundPlane);
  if (intersects.length > 0) {
    return intersects[0].point;
  }
  return null;
}

// ---- mouse handling ----
let mouseDownPos = { x: 0, y: 0 };
let isPointerDown = false;
let pendingHitIdx = -1;

canvas.addEventListener('pointerdown', (e) => {
  mouseDownPos.x = e.clientX;
  mouseDownPos.y = e.clientY;
  isPointerDown = true;

  // Check if we hit a point → prepare for potential direct drag
  if (e.button === 0) {
    pendingHitIdx = hitTestPoint(e.clientX, e.clientY);
  } else {
    pendingHitIdx = -1;
  }
});

canvas.addEventListener('pointermove', (e) => {
  // Continuous drag update
  if (isDraggingPoint && dragPointIdx >= 0) {
    const rect = canvas.getBoundingClientRect();
    pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    _dragIsect.set(0, 0, 0);
    if (raycaster.ray.intersectPlane(dragPlane, _dragIsect)) {
      _dragIsect.add(dragOffset);
      points[dragPointIdx].x = _dragIsect.x;
      points[dragPointIdx].y = _dragIsect.y;
      points[dragPointIdx].z = _dragIsect.z;
      if (pointGroup.children[dragPointIdx]) {
        pointGroup.children[dragPointIdx].position.copy(_dragIsect);
      }
      rebuildCurve();
    }
    return;
  }

  // Start drag from a point hit
  if (!isPointerDown || pendingHitIdx < 0 || e.buttons !== 1) return;
  const dx = e.clientX - mouseDownPos.x;
  const dy = e.clientY - mouseDownPos.y;
  if (dx * dx + dy * dy < 25) return;

  isDraggingPoint = true;
  dragPointIdx = pendingHitIdx;
  pendingHitIdx = -1;

  orbitControls.enabled = false;
  if (transformControls.object) transformControls.detach();

  selectPoint(dragPointIdx);

  const p = points[dragPointIdx];
  const pointPos = new THREE.Vector3(p.x, p.y, p.z);
  const camDir = new THREE.Vector3();
  camera.getWorldDirection(camDir);
  dragPlane.setFromNormalAndCoplanarPoint(camDir, pointPos);

  const rect = canvas.getBoundingClientRect();
  pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  _dragIsect.set(0, 0, 0);
  if (raycaster.ray.intersectPlane(dragPlane, _dragIsect)) {
    dragOffset.copy(pointPos).sub(_dragIsect);
  }
});

canvas.addEventListener('pointerup', (e) => {
  if (!isPointerDown) return;
  isPointerDown = false;

  if (isDraggingPoint) {
    isDraggingPoint = false;
    dragPointIdx = -1;
    orbitControls.enabled = true;
    notifyParent();
    return;
  }

  pendingHitIdx = -1;
  const dx = e.clientX - mouseDownPos.x;
  const dy = e.clientY - mouseDownPos.y;
  const dist = Math.sqrt(dx * dx + dy * dy);

  // Ignore drags (orbit/transform movement)
  if (dist > 5) return;
  if (e.button !== 0) return;

  // Check point hit
  const hitIdx = hitTestPoint(e.clientX, e.clientY);
  if (hitIdx >= 0) {
    selectPoint(hitIdx);
    return;
  }

  // Empty click: deselect or add point on ground
  if (points.length >= maxPoints) return;
  const hit = hitTestGround(e.clientX, e.clientY);
  if (hit) {
    const newPt = { x: parseFloat(hit.x.toFixed(4)), y: parseFloat(hit.y.toFixed(4)), z: parseFloat(hit.z.toFixed(4)) };
    points.push(newPt);
    rebuildScene();
    selectPoint(points.length - 1);
    notifyParent();
  } else {
    selectPoint(-1);
  }
});

canvas.addEventListener('pointercancel', () => {
  if (isDraggingPoint) {
    isDraggingPoint = false;
    dragPointIdx = -1;
    orbitControls.enabled = true;
  }
  isPointerDown = false;
  pendingHitIdx = -1;
});

canvas.addEventListener('contextmenu', (e) => e.preventDefault());

window.addEventListener('keydown', (e) => {
  if ((e.key === 'Delete' || e.key === 'Backspace') && selectedIdx >= 0 && points.length > 2) {
    points.splice(selectedIdx, 1);
    selectedIdx = -1;
    if (transformControls.object) transformControls.detach();
    rebuildScene();
    notifyParent();
  }
});

// ---- resize ----
function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w || canvas.height !== h) {
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
}

// ---- animate ----
function animate() {
  requestAnimationFrame(animate);
  resize();
  orbitControls.update();
  renderer.render(scene, camera);
}
requestAnimationFrame(animate);

// ---- parent communication ----
function notifyParent() {
  window.parent.postMessage({
    type: 'CURVE_UPDATED',
    points: points.map(p => ({ x: p.x, y: p.y, z: p.z })),
  }, '*');
}

window.addEventListener('message', (e) => {
  if (e.data?.type === 'CURVE_READY') {
    // parent acknowledged ready, send current curve data
  }
  if (e.data?.type === 'LOAD_CURVE') {
    loadCurve(e.data.points);
  }
  if (e.data?.type === 'RESIZE') {
    resize();
  }
});

// Signal ready
window.parent.postMessage({ type: 'VIEWER_READY' }, '*');

// Load initial data if sent before ready
const initialData = window.__initialCurveData;
if (initialData) {
  loadCurve(initialData);
}

</script>
</body>
</html>`;

// ---- ComfyUI Extension ----

// Compute local three.js base URL from this module's URL
const __threeBaseUrl = new URL('.', import.meta.url).href.replace(/\/+$/, '');

app.registerExtension({
  name: "kimodo.CurveToPoints",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name !== NODE_NAME) return;

    nodeType.prototype.onNodeCreated = function () {
      this.serialize_widgets = true;
      this.min_size = [MIN_WIDTH, MIN_HEIGHT];
      this.resizable = true;
      this.size = this.computeSize();

      this.curvePoints = DEFAULT_CURVE.map(p => ({ ...p }));
      this.editorReady = false;

      // Create iframe with 3D editor
      const iframe = document.createElement("iframe");
      iframe.style.width = "100%";
      iframe.style.height = "100%";
      iframe.style.border = "none";
      iframe.style.backgroundColor = "#2a2a3a";
      iframe.style.display = "block";

      const html = EDITOR_HTML.replace(/__THREE_BASE_URL__/g, __threeBaseUrl);
      iframe.srcdoc = html;

      const widget = this.addDOMWidget("curve3d", "KIMODO_CURVE_3D", iframe, {
        getValue() { return ""; },
        setValue(v) {},
      });
      widget.computeSize = function (width) {
        const w = width || MIN_WIDTH;
        return [w, Math.max(w * 0.75, MIN_HEIGHT)];
      };
      widget.element = iframe;
      this.curveIframe = iframe;

      // Listen for ready message
      window.addEventListener("message", (event) => {
        if (event.data?.type === "VIEWER_READY" && event.source === iframe.contentWindow) {
          this.editorReady = true;
          this._sendCurveToIframe();
        }
        if (event.data?.type === "CURVE_UPDATED" && event.source === iframe.contentWindow) {
          this.curvePoints = event.data.points.map(p => ({ x: p.x, y: p.y, z: p.z }));
          this.syncCurveToWidget();
        }
      });

      // Resize observer
      const ro = new ResizeObserver(() => {
        if (iframe.contentWindow) {
          const rect = iframe.getBoundingClientRect();
          iframe.contentWindow.postMessage({ type: "RESIZE", width: rect.width, height: rect.height }, "*");
        }
      });
      ro.observe(iframe);
      const origRemoved = this.onRemoved;
      this.onRemoved = function () {
        ro.disconnect();
        origRemoved?.apply(this, arguments);
      };

      // Hidden widget for sync
      this.jsonWidget = this.addWidget("text", "curve_json", JSON.stringify(this.curvePoints), () => {});
      this.jsonWidget.hidden = true;
      this.properties.curve_json = JSON.stringify(this.curvePoints);

      this.setDirtyCanvas(true, true);
    };

    nodeType.prototype.computeSize = function () {
      return [MIN_WIDTH, MIN_HEIGHT];
    };

    nodeType.prototype._sendCurveToIframe = function () {
      const iframe = this.curveIframe;
      if (!iframe || !iframe.contentWindow) return;
      iframe.contentWindow.postMessage({
        type: "LOAD_CURVE",
        points: this.curvePoints.map(p => ({ x: p.x, y: p.y, z: p.z })),
      }, "*");
    };

    const origOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (o) {
      if (origOnConfigure) origOnConfigure.call(this, o);
      try {
        const raw = this.properties?.curve_json;
        if (raw) {
          const data = typeof raw === "string" ? JSON.parse(raw) : raw;
          if (Array.isArray(data) && data.length >= 2) {
            this.curvePoints = data.map(p => ({ x: p.x ?? 0, y: p.y ?? 0, z: p.z ?? 0 }));
            this._sendCurveToIframe();
          }
        }
      } catch (e) {
        console.warn("[Kimodo_CurveToPoints] Failed to parse curve_json:", e);
      }
    };

    nodeType.prototype.syncCurveToWidget = function () {
      const json = JSON.stringify(this.curvePoints);
      this.jsonWidget.value = json;
      this.properties.curve_json = json;
    };
  },
});
