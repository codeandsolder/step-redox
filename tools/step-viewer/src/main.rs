use anyhow::{Context, Result, bail};
use clap::Parser;
use ruststep::ast::{EntityInstance, Name, Parameter, Record};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use monstertruck_meshing::prelude::*;
use monstertruck_io::step::load::Table;

#[derive(Parser, Debug)]
#[command(about = "Generate an interactive before/after STEP body/face viewer")]
struct Cli {
    before: PathBuf,
    after: PathBuf,
    output: PathBuf,

    #[arg(long = "face")]
    faces: Vec<u64>,

    #[arg(long = "surface")]
    surfaces: Vec<u64>,

    #[arg(long = "shell")]
    shells: Vec<u64>,

    #[arg(long, default_value_t = 0.01)]
    tolerance: f64,
}

#[derive(Clone)]
struct StepDoc {
    exchange: ruststep::ast::Exchange,
    entity_types: HashMap<u64, String>,
    inbound: HashMap<u64, Vec<u64>>,
    shell_faces: HashMap<u64, Vec<u64>>,
    shell_types: HashMap<u64, String>,
}

#[derive(Debug, Clone, Serialize)]
struct TargetMeshes {
    label: String,
    requested_kind: String,
    requested_id: u64,
    face_id: Option<u64>,
    surface_id: Option<u64>,
    shell_id: u64,
    shell_kind: String,
    solid_ids: Vec<u64>,
    before_body_obj: String,
    before_face_obj: Option<String>,
    after_body_obj: String,
    after_face_obj: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.tolerance <= 0.0 || !cli.tolerance.is_finite() {
        bail!("--tolerance must be finite and positive");
    }
    if cli.faces.is_empty() && cli.surfaces.is_empty() && cli.shells.is_empty() {
        bail!("select at least one --face, --surface, or --shell");
    }

    let before_text = std::fs::read_to_string(&cli.before)
        .with_context(|| format!("read {}", cli.before.display()))?;
    let after_text = std::fs::read_to_string(&cli.after)
        .with_context(|| format!("read {}", cli.after.display()))?;

    let before = StepDoc::parse(&before_text).context("parse before STEP")?;
    let after = StepDoc::parse(&after_text).context("parse after STEP")?;
    let before_table = Table::from_step(&before_text).context("build before Monstertruck table")?;
    let after_table = Table::from_step(&after_text).context("build after Monstertruck table")?;

    let mut targets = Vec::new();
    let mut seen = HashSet::new();

    for &surface in &cli.surfaces {
        let faces = before.faces_for_surface(surface);
        if faces.is_empty() {
            bail!("surface #{surface} is not used directly by an ADVANCED_FACE");
        }
        for face in faces {
            let key = ("surface", surface, face);
            if seen.insert(key) {
                targets.push(build_face_target(
                    &before,
                    &after,
                    &before_table,
                    &after_table,
                    face,
                    Some(surface),
                    "surface",
                    surface,
                    cli.tolerance,
                )?);
            }
        }
    }

    for &face in &cli.faces {
        let key = ("face", face, face);
        if seen.insert(key) {
            targets.push(build_face_target(
                &before,
                &after,
                &before_table,
                &after_table,
                face,
                before.surface_for_face(face),
                "face",
                face,
                cli.tolerance,
            )?);
        }
    }

    for &shell in &cli.shells {
        let key = ("shell", shell, shell);
        if seen.insert(key) {
            targets.push(build_shell_target(
                &before,
                &after,
                &before_table,
                &after_table,
                shell,
                cli.tolerance,
            )?);
        }
    }

    let html = render_html(&cli.before, &cli.after, &targets)?;
    std::fs::write(&cli.output, html).with_context(|| format!("write {}", cli.output.display()))?;
    eprintln!(
        "wrote {} target(s) to {}",
        targets.len(),
        cli.output.display()
    );
    Ok(())
}

impl StepDoc {
    fn parse(text: &str) -> Result<Self> {
        let exchange = ruststep::parser::parse(text)?;
        let mut entity_types = HashMap::new();
        let mut inbound = HashMap::<u64, Vec<u64>>::new();
        let mut shell_faces = HashMap::new();
        let mut shell_types = HashMap::new();

        for section in &exchange.data {
            for entity in &section.entities {
                let id = entity_id(entity);
                let ty = entity_type(entity);
                let children = entity_refs(entity);
                for &child in &children {
                    inbound.entry(child).or_default().push(id);
                }
                if matches!(ty.as_str(), "OPEN_SHELL" | "CLOSED_SHELL") {
                    if let Some(record) = simple_record(entity) {
                        if let Some(faces) = shell_face_refs(record) {
                            shell_faces.insert(id, faces);
                            shell_types.insert(id, ty.clone());
                        }
                    }
                }
                entity_types.insert(id, ty);
            }
        }

        Ok(Self {
            exchange,
            entity_types,
            inbound,
            shell_faces,
            shell_types,
        })
    }

    fn surface_for_face(&self, face: u64) -> Option<u64> {
        let entity = self.find_entity(face)?;
        let record = simple_record(entity)?;
        if record.name != "ADVANCED_FACE" {
            return None;
        }
        let Parameter::List(params) = &record.parameter else {
            return None;
        };
        entity_ref_value(params.get(2)?)
    }

    fn faces_for_surface(&self, surface: u64) -> Vec<u64> {
        self.inbound
            .get(&surface)
            .into_iter()
            .flatten()
            .copied()
            .filter(|id| {
                self.entity_types
                    .get(id)
                    .is_some_and(|t| t == "ADVANCED_FACE")
            })
            .filter(|id| self.surface_for_face(*id) == Some(surface))
            .collect()
    }

    fn containing_shells(&self, face: u64) -> Vec<u64> {
        self.shell_faces
            .iter()
            .filter_map(|(&shell, faces)| faces.contains(&face).then_some(shell))
            .collect()
    }

    fn preferred_shell(&self, face: u64) -> Result<u64> {
        let mut shells = self.containing_shells(face);
        if shells.is_empty() {
            bail!("face #{face} is not contained in an OPEN_SHELL/CLOSED_SHELL");
        }
        shells.sort_by_key(|id| {
            let closed_penalty =
                usize::from(self.shell_types.get(id).map(String::as_str) != Some("CLOSED_SHELL"));
            let face_count = self.shell_faces.get(id).map_or(0, Vec::len);
            (closed_penalty, std::cmp::Reverse(face_count), *id)
        });
        Ok(shells[0])
    }

    fn solid_ancestors(&self, shell: u64) -> Vec<u64> {
        self.inbound
            .get(&shell)
            .into_iter()
            .flatten()
            .copied()
            .filter(|id| {
                matches!(
                    self.entity_types.get(id).map(String::as_str),
                    Some("MANIFOLD_SOLID_BREP" | "BREP_WITH_VOIDS" | "FACETED_BREP")
                )
            })
            .collect()
    }

    fn find_entity(&self, id: u64) -> Option<&EntityInstance> {
        self.exchange
            .data
            .iter()
            .flat_map(|section| section.entities.iter())
            .find(|entity| entity_id(entity) == id)
    }
}

#[allow(clippy::too_many_arguments)]
fn build_face_target(
    before: &StepDoc,
    after: &StepDoc,
    before_table: &Table,
    after_table: &Table,
    face: u64,
    surface: Option<u64>,
    requested_kind: &str,
    requested_id: u64,
    tolerance: f64,
) -> Result<TargetMeshes> {
    let shell = before.preferred_shell(face)?;
    let after_shell = if after.shell_faces.contains_key(&shell) {
        shell
    } else {
        after.preferred_shell(face)?
    };

    let (before_body_obj, before_face_obj) =
        mesh_shell_and_face(before, before_table, shell, Some(face), tolerance)?;
    let (after_body_obj, after_face_obj) =
        mesh_shell_and_face(after, after_table, after_shell, Some(face), tolerance)?;

    let solids = before.solid_ancestors(shell);
    let shell_kind = before
        .shell_types
        .get(&shell)
        .cloned()
        .unwrap_or_else(|| "SHELL".into());
    let surface_suffix = if requested_kind == "surface" {
        String::new()
    } else {
        surface.map_or_else(String::new, |id| format!(" · surface #{id}"))
    };
    let label = format!(
        "{} #{} · face #{}{} · {} #{}{}",
        requested_kind,
        requested_id,
        face,
        surface_suffix,
        shell_kind,
        shell,
        if solids.is_empty() {
            String::new()
        } else {
            format!(
                " · solid {}",
                solids
                    .iter()
                    .map(|id| format!("#{id}"))
                    .collect::<Vec<_>>()
                    .join(",")
            )
        }
    );

    Ok(TargetMeshes {
        label,
        requested_kind: requested_kind.to_string(),
        requested_id,
        face_id: Some(face),
        surface_id: surface,
        shell_id: shell,
        shell_kind,
        solid_ids: solids,
        before_body_obj,
        before_face_obj,
        after_body_obj,
        after_face_obj,
    })
}

fn build_shell_target(
    before: &StepDoc,
    after: &StepDoc,
    before_table: &Table,
    after_table: &Table,
    shell: u64,
    tolerance: f64,
) -> Result<TargetMeshes> {
    let (before_body_obj, _) = mesh_shell_and_face(before, before_table, shell, None, tolerance)?;
    let (after_body_obj, _) = mesh_shell_and_face(after, after_table, shell, None, tolerance)?;
    let solids = before.solid_ancestors(shell);
    let shell_kind = before
        .shell_types
        .get(&shell)
        .cloned()
        .unwrap_or_else(|| "SHELL".into());

    Ok(TargetMeshes {
        label: format!("{shell_kind} #{shell}"),
        requested_kind: "shell".into(),
        requested_id: shell,
        face_id: None,
        surface_id: None,
        shell_id: shell,
        shell_kind,
        solid_ids: solids,
        before_body_obj,
        before_face_obj: None,
        after_body_obj,
        after_face_obj: None,
    })
}

fn mesh_shell_and_face(
    doc: &StepDoc,
    table: &Table,
    shell_id: u64,
    face_id: Option<u64>,
    tolerance: f64,
) -> Result<(String, Option<String>)> {
    let step_shell = table
        .shell
        .get(&shell_id)
        .with_context(|| format!("Monstertruck did not parse shell #{shell_id}"))?;
    let compressed = table
        .to_compressed_shell(step_shell)
        .map_err(|e| anyhow::anyhow!("convert shell #{shell_id}: {e}"))?;
    let meshed = compressed.robust_triangulation(tolerance);
    let body_obj = polygon_to_obj(&meshed.to_polygon())?;

    let Some(face_id) = face_id else {
        return Ok((body_obj, None));
    };
    let source_faces = doc
        .shell_faces
        .get(&shell_id)
        .with_context(|| format!("missing source face list for shell #{shell_id}"))?;
    if meshed.faces.len() != source_faces.len() {
        bail!(
            "cannot reliably map face IDs in shell #{shell_id}: STEP has {} faces, Monstertruck converted {}",
            source_faces.len(),
            meshed.faces.len()
        );
    }
    let face_index = source_faces
        .iter()
        .position(|&id| id == face_id)
        .with_context(|| format!("face #{face_id} not in shell #{shell_id}"))?;

    // Tessellate in full-shell context first: constrained face meshing can depend on
    // shared edge polylines. Isolating the CompressedFace before tessellation caused
    // valid trimmed faces to produce no polygon at all.
    let one_face = monstertruck_topology::compress::CompressedShell {
        vertices: meshed.vertices.clone(),
        edges: meshed.edges.clone(),
        faces: vec![meshed.faces[face_index].clone()],
        vertex_stable_ids: meshed.vertex_stable_ids.clone(),
        edge_stable_ids: meshed.edge_stable_ids.clone(),
        face_stable_ids: meshed
            .face_stable_ids
            .as_ref()
            .map(|ids| vec![ids[face_index]]),
    };
    let face_obj = polygon_to_obj(&one_face.to_polygon())?;
    Ok((body_obj, Some(face_obj)))
}

fn polygon_to_obj(polygon: &monstertruck_mesh::PolygonMesh) -> Result<String> {
    let mut bytes = Vec::new();
    monstertruck_mesh::obj::write(polygon, &mut bytes)?;
    Ok(String::from_utf8(bytes)?)
}

fn render_html(before: &Path, after: &Path, targets: &[TargetMeshes]) -> Result<String> {
    let targets_json = serde_json::to_string(targets)?;
    let before_name = serde_json::to_string(&before.display().to_string())?;
    let after_name = serde_json::to_string(&after.display().to_string())?;

    let mut html = String::new();
    write!(
        html,
        r###"<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>step-redox body compare</title>
<style>
html,body{{margin:0;height:100%;overflow:hidden;background:#101214;color:#e8eaed;font:13px system-ui,sans-serif}}
#ui{{position:absolute;z-index:20;left:12px;top:12px;right:12px;display:flex;gap:10px;align-items:flex-start;pointer-events:none}}
.panel{{pointer-events:auto;background:#171a1eea;border:1px solid #41464d;border-radius:8px;padding:9px 11px;max-width:min(760px,calc(100vw - 46px))}}
label{{display:inline-flex;align-items:center;gap:5px;margin-right:10px}} select{{max-width:520px;background:#24282e;color:#eee;border:1px solid #555;border-radius:4px;padding:4px}}
button,input{{accent-color:#8ab4f8}} button{{background:#282d33;color:#eee;border:1px solid #555;border-radius:4px;padding:4px 8px}}
#info{{font-family:ui-monospace,monospace;color:#b8c0ca;margin-top:6px;white-space:pre-wrap}}
#views{{display:grid;grid-template-columns:1fr 1fr;width:100%;height:100%}}
.view{{position:relative;min-width:0;overflow:hidden}}
.view+.view{{border-left:1px solid #333}}
.badge{{position:absolute;left:12px;bottom:12px;z-index:10;background:#171a1edc;border:1px solid #41464d;border-radius:6px;padding:6px 8px;max-width:calc(100% - 42px);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
canvas{{display:block;width:100%;height:100%}}
@media(max-width:800px){{#views{{grid-template-columns:1fr;grid-template-rows:1fr 1fr}}.view+.view{{border-left:0;border-top:1px solid #333}}}}
</style>
</head>
<body>
<div id="ui"><div class="panel">
<div>
<select id="target"></select>
<button id="fit">fit</button>
<label><input id="sync" type="checkbox" checked> sync cameras</label>
<label><input id="body" type="checkbox" checked> body</label>
<label><input id="highlight" type="checkbox" checked> highlight face</label>
<label><input id="wire" type="checkbox"> wireframe</label>
</div>
<div id="info"></div>
</div></div>
<div id="views">
<div class="view" id="beforeView"><div class="badge">before · <span id="beforeName"></span></div></div>
<div class="view" id="afterView"><div class="badge">after · <span id="afterName"></span></div></div>
</div>
<script type="module">
const THREE=await import("https://cdn.jsdelivr.net/npm/three@0.180.0/+esm");
const {{OrbitControls}}=await import("https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/controls/OrbitControls.js/+esm");
const {{OBJLoader}}=await import("https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/loaders/OBJLoader.js/+esm");
const TARGETS={targets_json};
const BEFORE={before_name};
const AFTER={after_name};
document.querySelector("#beforeName").textContent=BEFORE;
document.querySelector("#afterName").textContent=AFTER;
const loader=new OBJLoader();

function makeView(rootId,faceColor){{
 const root=document.querySelector(rootId);
 const scene=new THREE.Scene();scene.background=new THREE.Color(0x101214);
 const camera=new THREE.PerspectiveCamera(38,1,0.001,1e7);camera.position.set(5,5,5);
 const renderer=new THREE.WebGLRenderer({{antialias:true}});renderer.setPixelRatio(Math.min(devicePixelRatio,2));root.prepend(renderer.domElement);
 const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
 scene.add(new THREE.HemisphereLight(0xffffff,0x26303a,2.2));
 const key=new THREE.DirectionalLight(0xffffff,2.4);key.position.set(4,7,8);scene.add(key);
 const fill=new THREE.DirectionalLight(0xffffff,1.0);fill.position.set(-6,-3,2);scene.add(fill);
 const group=new THREE.Group();scene.add(group);
 const bodyMat=new THREE.MeshStandardMaterial({{color:0x9aa1a8,metalness:.05,roughness:.72,transparent:true,opacity:.50,side:THREE.DoubleSide}});
 const faceMat=new THREE.MeshStandardMaterial({{color:faceColor,metalness:.05,roughness:.55,side:THREE.DoubleSide,polygonOffset:true,polygonOffsetFactor:-1,polygonOffsetUnits:-1}});
 function clear(){{while(group.children.length){{const o=group.children.pop();o.traverse?.(x=>{{x.geometry?.dispose();}});}}}}
 function parse(objText,mat){{const o=loader.parse(objText);o.traverse(x=>{{if(x.isMesh){{x.material=mat;x.geometry.computeVertexNormals();}}}});return o;}}
 function load(body,face){{clear();const bo=parse(body,bodyMat);bo.userData.kind="body";group.add(bo);if(face){{const fo=parse(face,faceMat);fo.userData.kind="face";group.add(fo);}}updateVisibility();fit();}}
 function updateVisibility(){{const showBody=document.querySelector("#body").checked,showFace=document.querySelector("#highlight").checked,wire=document.querySelector("#wire").checked;bodyMat.wireframe=wire;faceMat.wireframe=wire;for(const o of group.children)o.visible=o.userData.kind==="body"?showBody:showFace;}}
 function fit(){{const box=new THREE.Box3().setFromObject(group);if(box.isEmpty())return;const c=box.getCenter(new THREE.Vector3()),sz=box.getSize(new THREE.Vector3()),r=Math.max(sz.length(),1e-6);controls.target.copy(c);camera.near=Math.max(r/10000,1e-6);camera.far=Math.max(r*100,100);camera.updateProjectionMatrix();camera.position.set(c.x+r*.72,c.y+r*.72,c.z+r*.72);controls.update();}}
 function resize(){{const w=Math.max(root.clientWidth,1),h=Math.max(root.clientHeight,1);camera.aspect=w/h;camera.updateProjectionMatrix();renderer.setSize(w,h,false);}}
 return {{root,scene,camera,renderer,controls,group,load,fit,resize,updateVisibility}};
}}
const left=makeView("#beforeView",0xffa447),right=makeView("#afterView",0x46d4ff);
let syncing=false;
function copyCamera(a,b){{if(syncing||!document.querySelector("#sync").checked)return;syncing=true;b.camera.position.copy(a.camera.position);b.camera.quaternion.copy(a.camera.quaternion);b.controls.target.copy(a.controls.target);b.controls.update();syncing=false;}}
left.controls.addEventListener("change",()=>copyCamera(left,right));right.controls.addEventListener("change",()=>copyCamera(right,left));
const sel=document.querySelector("#target");
TARGETS.forEach((t,i)=>{{const o=document.createElement("option");o.value=i;o.textContent=t.label;sel.appendChild(o);}});
function show(i){{const t=TARGETS[i];left.load(t.before_body_obj,t.before_face_obj);right.load(t.after_body_obj,t.after_face_obj);document.querySelector("#info").textContent=["target: "+t.requested_kind+" #"+t.requested_id,t.face_id!=null?"face: #"+t.face_id:null,t.surface_id!=null?"surface: #"+t.surface_id:null,"context: "+t.shell_kind+" #"+t.shell_id+(t.solid_ids.length?" · solid "+t.solid_ids.map(x=>"#"+x).join(","):"")].filter(Boolean).join("\n");copyCamera(left,right);}}
sel.addEventListener("change",()=>show(Number(sel.value)));
document.querySelector("#fit").addEventListener("click",()=>{{left.fit();copyCamera(left,right);}});
for(const id of ["body","highlight","wire"])document.querySelector("#"+id).addEventListener("change",()=>{{left.updateVisibility();right.updateVisibility();}});
function resize(){{left.resize();right.resize();}}window.addEventListener("resize",resize);resize();show(0);
(function loop(){{requestAnimationFrame(loop);left.controls.update();right.controls.update();left.renderer.render(left.scene,left.camera);right.renderer.render(right.scene,right.camera);}})();
</script>
</body>
</html>"###
    )?;
    Ok(html)
}

fn entity_id(entity: &EntityInstance) -> u64 {
    match entity {
        EntityInstance::Simple { id, .. } | EntityInstance::Complex { id, .. } => *id,
    }
}

fn entity_type(entity: &EntityInstance) -> String {
    match entity {
        EntityInstance::Simple { record, .. } => record.name.clone(),
        EntityInstance::Complex { .. } => "COMPLEX".to_string(),
    }
}

fn simple_record(entity: &EntityInstance) -> Option<&Record> {
    match entity {
        EntityInstance::Simple { record, .. } => Some(record),
        EntityInstance::Complex { .. } => None,
    }
}

fn shell_face_refs(record: &Record) -> Option<Vec<u64>> {
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let Parameter::List(faces) = params.get(1)? else {
        return None;
    };
    faces.iter().map(entity_ref_value).collect()
}

fn entity_refs(entity: &EntityInstance) -> Vec<u64> {
    let mut out = Vec::new();
    match entity {
        EntityInstance::Simple { record, .. } => visit_parameter_refs(&record.parameter, &mut out),
        EntityInstance::Complex { subsuper, .. } => {
            for record in &subsuper.0 {
                visit_parameter_refs(&record.parameter, &mut out);
            }
        }
    }
    out
}

fn visit_parameter_refs(parameter: &Parameter, out: &mut Vec<u64>) {
    match parameter {
        Parameter::Ref(Name::Entity(id)) => out.push(*id),
        Parameter::Typed { parameter, .. } => visit_parameter_refs(parameter, out),
        Parameter::List(items) => {
            for item in items {
                visit_parameter_refs(item, out);
            }
        }
        _ => {}
    }
}

fn entity_ref_value(parameter: &Parameter) -> Option<u64> {
    match parameter {
        Parameter::Ref(Name::Entity(id)) => Some(*id),
        _ => None,
    }
}
