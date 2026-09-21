# Unscrew your STEP browser prototype

This is intentionally a static site. STEP parsing, semantic recovery, compact/compat
output and safe instance-pattern edits run in WebAssembly in the user's browser.

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
