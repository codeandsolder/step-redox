use occt_wasm::OcctKernel;
use sha2::{Digest, Sha256};
use std::{env, fmt::Write as _, fs, io};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args()
        .nth(1)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "missing STEP path argument"))?;
    let bytes = fs::read(&path)?;
    let text = String::from_utf8(bytes)?;
    let mut kernel = OcctKernel::new()?;
    let shape = kernel.import_step(&text)?;
    let bbox = kernel.get_bounding_box(shape, true)?;
    let volume = kernel.get_volume(shape)?;
    let area = kernel.get_surface_area(shape)?;
    let faces = kernel.get_sub_shapes(shape, "Face")?.len();
    let edges = kernel.get_sub_shapes(shape, "Edge")?.len();
    let vertices = kernel.get_sub_shapes(shape, "Vertex")?.len();
    let mesh = kernel.tessellate(shape, 0.002, 0.2)?;
    let mut tris: Vec<[[i64; 3]; 3]> = Vec::with_capacity(mesh.indices.len() / 3);
    let quantize = |value: f32| (f64::from(value) * 1e9).round() as i64;
    for tri in mesh.indices.chunks_exact(3) {
        let mut points = [[0_i64; 3]; 3];
        for index in 0..3 {
            let vertex_index = tri[index] as usize * 3;
            points[index] = [
                quantize(mesh.positions[vertex_index]),
                quantize(mesh.positions[vertex_index + 1]),
                quantize(mesh.positions[vertex_index + 2]),
            ];
        }
        points.sort();
        tris.push(points);
    }
    tris.sort();
    let mut hasher = Sha256::new();
    for tri in &tris {
        for point in tri {
            for value in point {
                hasher.update(value.to_le_bytes());
            }
        }
    }
    let digest = hasher.finalize();
    let mut digest_hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut digest_hex, "{byte:02x}")?;
    }

    println!("path={path}");
    println!(
        "bbox={:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
        bbox.min.x, bbox.min.y, bbox.min.z, bbox.max.x, bbox.max.y, bbox.max.z
    );
    println!("volume={volume:.15}");
    println!("area={area:.15}");
    println!("faces={faces} edges={edges} vertices={vertices}");
    println!(
        "mesh_vertices={} triangles={}",
        mesh.positions.len() / 3,
        mesh.indices.len() / 3
    );
    println!("mesh_hash={digest_hex}");
    Ok(())
}
