# Unscrew your STEP browser prototype

This is intentionally a static site. STEP parsing, semantic recovery, compact/compat
output and editing run in WebAssembly in the user's browser.

When step-redox proves that a periodic body and one or more instance rows share a
single count parameter, the UI exposes one atomic count control. Regeneration
updates the body and every coupled row together; those rows are then read-only in
the lower-level pattern panel so the UI cannot create a half-resized component.

Build the bindings:

```sh
cd ../step-redox-wasm
cargo build --release --target wasm32-unknown-unknown
wasm-bindgen \
  --target web \
  --out-dir ../unscrew-step/pkg \
  target/wasm32-unknown-unknown/release/step_redox_wasm.wasm
```

Then serve this directory over HTTP. No application backend is required.
