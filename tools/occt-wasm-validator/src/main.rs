use occt_wasm::OcctKernel;
use sha2::{Digest, Sha256};
use std::{env, fs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).expect("STEP path");
    let bytes = fs::read(&path)?;
    let text = String::from_utf8(bytes)?;
    let mut k = OcctKernel::new()?;
    let shape = k.import_step(&text)?;
    let bbox = k.get_bounding_box(shape, true)?;
    let volume = k.get_volume(shape)?;
    let area = k.get_surface_area(shape)?;
    let faces = k.get_sub_shapes(shape, "Face")?.len();
    let edges = k.get_sub_shapes(shape, "Edge")?.len();
    let vertices = k.get_sub_shapes(shape, "Vertex")?.len();
    let mesh = k.tessellate(shape, 0.002, 0.2)?;
    let mut tris: Vec<[[i64; 3]; 3]> = Vec::with_capacity(mesh.indices.len() / 3);
    let q = |v: f32| (v as f64 * 1e9).round() as i64;
    for tri in mesh.indices.chunks_exact(3) {
        let mut pts = [[0i64;3];3];
        for j in 0..3 {
            let vi = tri[j] as usize * 3;
            pts[j] = [q(mesh.positions[vi]), q(mesh.positions[vi+1]), q(mesh.positions[vi+2])];
        }
        pts.sort();
        tris.push(pts);
    }
    tris.sort();
    let mut h = Sha256::new();
    for tri in &tris {
        for p in tri {
            for v in p {
                h.update(v.to_le_bytes());
            }
        }
    }
    println!("path={path}");
    println!("bbox={:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
        bbox.min.x,bbox.min.y,bbox.min.z,bbox.max.x,bbox.max.y,bbox.max.z);
    println!("volume={volume:.15}");
    println!("area={area:.15}");
    println!("faces={faces} edges={edges} vertices={vertices}");
    println!("mesh_vertices={} triangles={}", mesh.positions.len()/3, mesh.indices.len()/3);
    println!("mesh_hash={:x}", h.finalize());
    Ok(())
}
