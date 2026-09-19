# step-redox STEP viewer

Topology-aware before/after viewer for investigating STEP rewrites without loading native OCCT.

The generator parses STEP with `ruststep`, converts the selected owning shell through `truck-stepio`, tessellates with Truck, and emits one HTML file with the meshes embedded as OBJ text. The browser view uses Three.js for display.

## Build

Reuse the repository target directory so Truck is not rebuilt into a second target tree:

```sh
CARGO_TARGET_DIR=target cargo build --release --manifest-path tools/step-viewer/Cargo.toml
```

## Use

Select one or more STEP entity IDs. A surface selection resolves its direct `ADVANCED_FACE`, then resolves the face's containing shell. When a face belongs to more than one shell, the viewer prefers a closed shell and then the larger shell, which avoids choosing temporary one-face extraction shells.

```sh
target/release/step-redox-viewer before.step after.step comparison.html --surface 19400
```

Selectors can be repeated and mixed:

```sh
target/release/step-redox-viewer before.step after.step comparison.html --surface 4721 --face 23934 --shell 11769
```

The HTML provides:
- synchronized side-by-side orbit cameras;
- owning body/shell mesh before and after;
- selected face highlighted independently;
- body/highlight visibility toggles;
- wireframe toggle;
- face/surface/shell/solid entity IDs in the selector and detail panel.

Tessellation tolerance defaults to `0.01` STEP model units and can be changed with `--tolerance`.

## Current scope

The viewer operates on directly defined `OPEN_SHELL` / `CLOSED_SHELL` source geometry. It deliberately does not expand `MAPPED_ITEM` assembly placements yet. This is sufficient for inspecting step-redox transformations on source faces and bodies; mapped-instance world-space visualization can be added separately if needed.

The emitted HTML imports Three.js modules from jsDelivr when opened in a browser; the STEP parsing and tessellation path itself is pure Rust/Truck.
