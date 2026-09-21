#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRATE="$ROOT/tools/occt-wasm-validator"
BIN="$CRATE/target/release/step-redox-occt-wasm-validator"

if [[ ! -x "$BIN" ]]; then
    cargo build --release --manifest-path "$CRATE/Cargo.toml"
fi

# Keep malformed input from turning parser bugs into host-wide resource exhaustion.
# WASM already contains memory corruption; these caps contain CPU/RSS failure modes.
VM_KIB="${STEP_REDOX_OCCT_VM_KIB:-6291456}"
TIMEOUT="${STEP_REDOX_OCCT_TIMEOUT:-180}"
ulimit -v "$VM_KIB"
exec timeout --signal=KILL "$TIMEOUT" "$BIN" "$@"
