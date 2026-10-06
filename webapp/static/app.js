"use strict";
const COLORS = ["#2b83ba", "#fdae61", "#d7191c"];
const METRICS = [["accuracy", "Accuracy"], ["f1_macro", "F1-macro"], ["auc_ovr", "AUC (OvR)"], ["mcc", "MCC"], ["cv_f1", "F1 validación cruzada"]];
const $ = (id) => document.getElementById(id);
let models = [];
const state = { A: null, B: null, mapA: null, mapB: null, layerA: null, layerB: null, marker: null, dataA: null, dataB: null };

// ---------- util ----------
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v) => (typeof v === "number" ? v.toFixed(3) : "–");
async function api(url, opts) {
  const r = await fetch(url, opts);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || j.message || r.statusText);
  return j;
}
function className(meta, k) { return meta.class_names[k] ?? `clase ${k}`; }
function rangeText(meta, k) { const r = meta.class_ranges[k]; return r ? `${r[0].toFixed(1)} – ${r[1].toFixed(1)}` : ""; }
function chip(meta, k) { return `<span class="chip c${k}">${esc(className(meta, k))}</span>`; }

// ---------- tabs ----------
document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
  ["compare", "models"].forEach((t) => ($("tab-" + t).hidden = t !== b.dataset.tab));
  if (b.dataset.tab === "compare") { state.mapA.invalidateSize(); state.mapB.invalidateSize(); }
}));

// ---------- maps ----------
function makeMap(id) {
  const m = L.map(id, { preferCanvas: true }).setView([1.5, -77.9], 8);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 18, attribution: "© OpenStreetMap" }).addTo(m);
  return m;
}
function syncMaps(a, b) {
  let busy = false;
  const link = (src, dst) => src.on("move", () => {
    if (busy) return; busy = true; dst.setView(src.getCenter(), src.getZoom(), { animate: false }); busy = false;
  });
  link(a, b); link(b, a);
}
function cellColor(layer, d, i) {
  if (layer === "err") return d.pred[i] === d.real[i] ? "#1a7f4b" : "#d7191c";
  return COLORS[d[layer][i]] || "#888";
}
function drawLayer(side) {
  const map = state["map" + side], d = state["data" + side], layer = $("layer").value;
  if (state["layer" + side]) map.removeLayer(state["layer" + side]);
  if (!d) return;
  const g = L.layerGroup();
  d.bounds.forEach((b, i) => {
    L.rectangle([[b[0], b[1]], [b[2], b[3]]], { stroke: false, fillColor: cellColor(layer, d, i), fillOpacity: 0.62, interactive: false }).addTo(g);
  });
  g.addTo(map); state["layer" + side] = g;
  const meta = d.meta;
  $("legend" + side).innerHTML = layer === "err"
    ? `<span><i style="background:#1a7f4b"></i>acierto</span><span><i style="background:#d7191c"></i>error</span>`
    : meta.class_names.map((n, k) => `<span><i style="background:${COLORS[k]}"></i>${esc(n)} (${rangeText(meta, k)})</span>`).join("");
}
async function loadSide(side, id) {
  state[side] = id;
  state["data" + side] = null; drawLayer(side);
  if (!id) return;
  const d = await api(`/api/models/${encodeURIComponent(id)}/map`);
  if (state[side] !== id) return;                    // el usuario cambió de modelo mientras cargaba
  state["data" + side] = d;
  $("title" + side).textContent = `Modelo ${side}: ${d.meta.name}`;
  drawLayer(side);
  state["map" + side].setView(d.center, state["map" + side].getZoom() < 8 ? 8 : state["map" + side].getZoom());
}

// ---------- métricas y matrices ----------
function renderMetrics() {
  const a = models.find((m) => m.id === state.A), b = models.find((m) => m.id === state.B);
  if (!a || !b) { $("metrics").innerHTML = "<p class='hint'>Selecciona dos modelos.</p>"; return; }
  const rows = METRICS.map(([k, label]) => {
    const va = k === "cv_f1" ? a.cv_f1 : a.metrics[k], vb = k === "cv_f1" ? b.cv_f1 : b.metrics[k];
    const ca = va > vb ? "best" : "", cb = vb > va ? "best" : "";
    return `<tr><th>${label}</th><td class="num ${ca}">${fmt(va)}</td><td class="num ${cb}">${fmt(vb)}</td><td class="num">${fmt(va - vb)}</td></tr>`;
  }).join("");
  $("metrics").innerHTML = `<table><tr><th></th><th>A: ${esc(a.name)}</th><th>B: ${esc(b.name)}</th><th>A − B</th></tr>${rows}</table>
    <p class="hint">En negrita, el mejor valor de cada métrica. Con ~90 píxeles de prueba, diferencias pequeñas (≈ 0.03) no son concluyentes.</p>`;
  cm("A", a); cm("B", b);
  const p = (m) => Object.entries(m.params || {}).map(([k, v]) => `${esc(k)}=${typeof v === "number" ? +v.toPrecision(4) : esc(JSON.stringify(v))}`).join(", ");
  $("params").innerHTML = `<table><tr><th></th><th>Escalador</th><th>Discretizador</th><th>Reductor</th><th>Modelo</th><th>Kernel</th><th>Hiperparámetros</th></tr>
    ${[["A", a], ["B", b]].map(([s, m]) => `<tr><th>${s}</th><td>${esc(m.scaler)}</td><td>${esc(m.discretizer)}</td><td>${esc(m.reducer)}</td><td>${esc(m.model)}</td><td>${esc(m.kernel)}</td><td>${p(m)}</td></tr>`).join("")}</table>`;
}
function cm(side, m) {
  $("cmTitle" + side).textContent = `Matriz de confusión – ${side}: ${m.name}`;
  const M = m.confusion_matrix, max = Math.max(...M.flat(), 1);
  let h = `<table class="cm"><tr><th>real ↓ / pred →</th>${m.class_names.map((n) => `<th>${esc(n)}</th>`).join("")}</tr>`;
  M.forEach((row, i) => {
    h += `<tr><th>${esc(m.class_names[i])}</th>` + row.map((v, j) => {
      const a = (v / max) * 0.75; const bg = `rgba(43,108,176,${a.toFixed(2)})`; const col = a > 0.45 ? "#fff" : "inherit";
      return `<td class="${i === j ? "diag" : ""}" style="background:${bg};color:${col}">${v}</td>`;
    }).join("") + "</tr>";
  });
  $("cm" + side).innerHTML = h + "</table>";
}

// ---------- clic en el mapa ----------
async function onClick(e) {
  const { lat, lng } = e.latlng;
  if (state.marker) { state.marker.remove(); }
  state.marker = L.circleMarker(e.latlng, { radius: 6, color: "#000", weight: 2, fillColor: "#fff", fillOpacity: 1 });
  state.marker.addTo(e.target);
  const ids = [state.A, state.B].filter(Boolean);
  const box = $("clickResult"); box.hidden = false;
  box.innerHTML = "Consultando…";
  try {
    const r = await api(`/api/predict?lat=${lat}&lon=${lng}&ids=${ids.map(encodeURIComponent).join(",")}`);
    const col = (side, id) => {
      const m = models.find((x) => x.id === id), x = r.results[id];
      if (!x) return "";
      if (!x.found) return `<div><h3>${side}: ${esc(x.name)}</h3><p>Fuera de la zona de estudio: el píxel ${esc(x.dataset)} más cercano está a ${x.nearest_km} km.</p></div>`;
      const note = x.exact ? "" : `<p class="hint">El punto no cae en un píxel con datos: se usan las bandas del píxel ${esc(x.dataset)} más cercano (a ${x.distance_km} km), por lo que es una estimación.</p>`;
      return `<div><h3>${side}: ${esc(x.name)}</h3>${note}
        <p>Clasificación: ${chip(m, x.predicted.index)} <span class="hint">irradiancia ${rangeText(m, x.predicted.index)}</span></p>
        <p>Clase real del píxel (${esc(x.dataset)}, value=${x.value.toFixed(1)}): ${chip(m, x.real.index)}
           <span class="${x.correct ? "ok" : "bad"}">${x.correct ? "✓ acierta" : "✗ falla"}</span></p></div>`;
    };
    box.innerHTML = `<h3>Punto: ${lat.toFixed(4)}°, ${lng.toFixed(4)}°</h3><div class="grid">${col("A", state.A)}${state.B === state.A ? "" : col("B", state.B)}</div>`;
  } catch (err) { box.textContent = "Error: " + err.message; }
}

// ---------- tabla de modelos ----------
function renderModelTable() {
  const rows = models.map((m) => `<tr><td>${esc(m.name)}</td><td>${esc(m.dataset)}</td><td>${esc(m.model)}-${esc(m.kernel)}</td>
    <td class="num">${fmt(m.metrics.accuracy)}</td><td class="num">${fmt(m.metrics.f1_macro)}</td><td class="num">${fmt(m.metrics.auc_ovr)}</td><td class="num">${fmt(m.metrics.mcc)}</td>
    <td><a href="/api/models/${encodeURIComponent(m.id)}/download"><button>Descargar</button></a> <button class="danger" data-del="${esc(m.id)}">Eliminar</button></td></tr>`).join("");
  $("modelTable").innerHTML = `<tr><th>Modelo</th><th>Dataset</th><th>Algoritmo-kernel</th><th>Acc</th><th>F1</th><th>AUC</th><th>MCC</th><th></th></tr>${rows}`;
  document.querySelectorAll("[data-del]").forEach((b) => b.addEventListener("click", async () => {
    if (!confirm("¿Eliminar el modelo " + b.dataset.del + "?")) return;
    await api("/api/models/" + encodeURIComponent(b.dataset.del), { method: "DELETE" });
    await refreshModels();
  }));
}
$("uploadForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(); fd.append("file", $("file").files[0]);
  try { const m = await api("/api/models", { method: "POST", body: fd }); $("uploadMsg").textContent = "Modelo guardado: " + m.name; $("file").value = ""; await refreshModels(); }
  catch (err) { $("uploadMsg").textContent = "Error: " + err.message; }
});

// ---------- inicio ----------
function fillSelect(sel, current) {
  sel.innerHTML = models.map((m) => `<option value="${esc(m.id)}">${esc(m.name)}</option>`).join("");
  if (models.some((m) => m.id === current)) sel.value = current;
}
async function refreshModels() {
  models = await api("/api/models");
  models.sort((a, b) => b.metrics.mcc - a.metrics.mcc);
  fillSelect($("selA"), state.A); fillSelect($("selB"), state.B);
  renderModelTable();
  if (models.length) {
    if (!models.some((m) => m.id === state.A)) $("selA").value = models[0].id;
    if (!models.some((m) => m.id === state.B)) $("selB").value = models[Math.min(1, models.length - 1)].id;
  }
  await updateSelection();
}
async function updateSelection() {
  const a = $("selA").value || null, b = $("selB").value || null;
  await Promise.all([a !== state.A || !state.dataA ? loadSide("A", a) : 0, b !== state.B || !state.dataB ? loadSide("B", b) : 0]);
  renderMetrics();
}
window.addEventListener("DOMContentLoaded", async () => {
  state.mapA = makeMap("mapA"); state.mapB = makeMap("mapB");
  syncMaps(state.mapA, state.mapB);
  state.mapA.on("click", onClick); state.mapB.on("click", onClick);
  $("selA").addEventListener("change", updateSelection);
  $("selB").addEventListener("change", updateSelection);
  $("layer").addEventListener("change", () => { drawLayer("A"); drawLayer("B"); });
  try { await refreshModels(); } catch (e) { $("metrics").textContent = "Error cargando modelos: " + e.message; }
});
