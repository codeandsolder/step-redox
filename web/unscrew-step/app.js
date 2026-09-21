import init, { optimize_step, resize_linear_pattern } from "./pkg/step_redox_wasm.js";

let wasmReady = null;
let sourceFile = null;
let currentBytes = null;
let currentName = "optimized.step";
let currentReport = null;

const $ = (id) => document.getElementById(id);
const drop = $("drop");
const fileInput = $("file");
const run = $("run");
const result = $("result");
const status = $("status");

function humanBytes(n) {
  const units = ["B", "kB", "MB", "GB"];
  let i = 0, v = Number(n);
  while (v >= 1000 && i < units.length - 1) { v /= 1000; i++; }
  return `${v.toFixed(i ? 2 : 0)} ${units[i]}`;
}

function setStatus(text, error = false) {
  status.hidden = !text;
  status.textContent = text || "";
  status.classList.toggle("error", error);
}

function setFile(file) {
  sourceFile = file;
  currentBytes = null;
  currentReport = null;
  result.hidden = true;
  run.disabled = !file;
  $("fileMeta").textContent = file ? `${file.name} · ${humanBytes(file.size)}` : "No file selected";
}

fileInput.addEventListener("change", () => setFile(fileInput.files?.[0] || null));
for (const name of ["dragenter", "dragover"]) {
  drop.addEventListener(name, (e) => { e.preventDefault(); drop.classList.add("drag"); });
}
for (const name of ["dragleave", "drop"]) {
  drop.addEventListener(name, (e) => { e.preventDefault(); drop.classList.remove("drag"); });
}
drop.addEventListener("drop", (e) => setFile(e.dataTransfer.files?.[0] || null));

async function ensureWasm() {
  if (!wasmReady) wasmReady = init();
  return wasmReady;
}

function recoveryText(s) {
  const bits = [];
  if (s.straight_bspline_lines_recovered) bits.push(`${s.straight_bspline_lines_recovered} straight splines → lines`);
  if (s.exact_bezier_curves_recovered) bits.push(`${s.exact_bezier_curves_recovered} exact Beziers`);
  if (s.v_extrusion_surfaces_recovered) bits.push(`${s.v_extrusion_surfaces_recovered} extrusion surfaces`);
  if (s.partition_components_recovered) bits.push(`${s.partition_components_recovered} partitioned bodies recovered`);
  if (s.face_coalesce_faces_removed) bits.push(`${s.face_coalesce_faces_removed} redundant faces removed`);
  if (s.instanced_solids) bits.push(`${s.instanced_solids} solids instanced`);
  return bits.length ? bits.join(" · ") : "Canonicalized STEP structure";
}

function renderCompatibility(c) {
  const box = $("compatibility");
  box.className = "compatibility " + (c.conservative_structure ? "ok" : "warn");
  if (c.conservative_structure) {
    box.textContent = "No audited structural compatibility risks remain.";
  } else {
    const detail = Object.entries(c.structural_risk_entities)
      .map(([k,v]) => `${k}: ${v}`).join(" · ");
    box.textContent = `Uses compact/less-universal constructs: ${detail}`;
  }
}

function renderPatterns(patterns) {
  const panel = $("patternsPanel");
  const host = $("patterns");
  host.replaceChildren();
  panel.hidden = !patterns.length;
  patterns.forEach((p, index) => {
    const row = document.createElement("div");
    row.className = "pattern";
    const safeLinear = p.dimension === 1 && p.fill_ratio === 1 && p.grid_shape?.[0] === p.item_ids.length;
    row.innerHTML = `
      <div>
        <strong>Pattern ${index + 1}: ${p.item_ids.length} instances</strong>
        <div class="meta">
          <span>${p.dimension}D</span>
          <span>pitch ${p.pitch.map(x => Number(x).toFixed(6)).join(" × ")} mm</span>
          <span>grid ${p.grid_shape.join(" × ")}</span>
          <span>residual ${Number(p.max_residual_mm).toExponential(2)} mm</span>
        </div>
      </div>
      <div class="edit">
        ${safeLinear ? `
          <input class="count" type="number" min="1" step="1" value="${p.item_ids.length}" aria-label="instance count">
          <select class="anchor" aria-label="anchor">
            <option value="center">center</option>
            <option value="start">start</option>
            <option value="end">end</option>
          </select>
          <button class="apply">Apply</button>
        ` : "<span>read-only pattern</span>"}
      </div>`;
    if (safeLinear) {
      row.querySelector(".apply").addEventListener("click", () => editPattern(index, row));
    }
    host.append(row);
  });
}

function renderReport(report, inputSize, outputSize) {
  currentReport = report;
  $("inputBytes").textContent = humanBytes(inputSize);
  $("outputBytes").textContent = humanBytes(outputSize);
  $("ratio").textContent = `${(100 * outputSize / inputSize).toFixed(2)}%`;
  const s = report.stats;
  $("entities").textContent = s ? `${s.input_entities.toLocaleString()} → ${s.output_entities.toLocaleString()}` : "—";
  $("recoverySummary").textContent = s ? recoveryText(s) : "Pattern edit";
  renderCompatibility(report.compatibility);
  renderPatterns(report.patterns || []);
  $("report").textContent = JSON.stringify(report, null, 2);
  result.hidden = false;
}

run.addEventListener("click", async () => {
  if (!sourceFile) return;
  run.disabled = true;
  setStatus("Loading optimizer…");
  try {
    await ensureWasm();
    const profile = document.querySelector('input[name="profile"]:checked').value;
    const input = new Uint8Array(await sourceFile.arrayBuffer());
    setStatus(`Recovering CAD structure locally (${profile})…`);
    await new Promise(requestAnimationFrame);
    const out = optimize_step(input, profile);
    currentBytes = out.bytes;
    currentName = sourceFile.name.replace(/\.(stp|step)$/i, "") + `.${profile}.step`;
    const report = JSON.parse(out.reportJson);
    renderReport(report, input.length, currentBytes.length);
    setStatus("");
  } catch (e) {
    console.error(e);
    setStatus(String(e), true);
  } finally {
    run.disabled = false;
  }
});

async function editPattern(index, row) {
  if (!currentBytes) return;
  const count = Number(row.querySelector(".count").value);
  const anchor = row.querySelector(".anchor").value;
  if (!Number.isInteger(count) || count < 1) return;
  setStatus(`Regenerating pattern ${index + 1} locally…`);
  try {
    await ensureWasm();
    const beforeSize = currentBytes.length;
    const out = resize_linear_pattern(currentBytes, index, count, anchor);
    currentBytes = out.bytes;
    const report = JSON.parse(out.reportJson);
    // Pattern edits operate on the current normalized file, not the original.
    renderReport(report, beforeSize, currentBytes.length);
    setStatus("Pattern updated. Surrounding non-pattern bodies are unchanged.");
  } catch (e) {
    console.error(e);
    setStatus(String(e), true);
  }
}

$("download").addEventListener("click", () => {
  if (!currentBytes) return;
  const blob = new Blob([currentBytes], { type: "model/step" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = currentName;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
