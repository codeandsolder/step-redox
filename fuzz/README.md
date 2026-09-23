# step-redox fuzz targets

These targets exercise hostile STEP input at the two highest-value boundaries:

- `clean_idempotent`: if safe cleanup accepts arbitrary input, its output must be accepted again and the second cleanup must be byte-identical.
- `solid_extrusions`: arbitrary input must never panic while probing whole-solid extrusion recovery.
- `solid_revolutions`: arbitrary input must never panic while probing whole-solid revolution recovery.

Run with nightly Rust and `cargo-fuzz`:

```sh
cargo +nightly fuzz run clean_idempotent fuzz/corpus/clean_idempotent -- -dict=fuzz/step.dict
cargo +nightly fuzz run solid_extrusions fuzz/corpus/solid_extrusions -- -dict=fuzz/step.dict
cargo +nightly fuzz run solid_revolutions -- -dict=fuzz/step.dict
```

Crash artifacts remain local under `fuzz/artifacts/` and are not committed automatically.
