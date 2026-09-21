# KCL + Truck architecture spike

This experiment tests two deliberately separate choices:

1. **KCL is the publishable/parametric CAD language.**
2. **Truck is an interchangeable local B-rep evaluation backend.**

Neither is the step-redox recovery IR itself. Recovery keeps a small backend-neutral
constructive DAG with source/provenance links and exact B-rep fallbacks. The spike now
proves this separation mechanically: one `ExtrudeZ` node lowers independently to KCL text
and to Truck B-rep; there is no hand-maintained duplicate model description.

## Boundaries

- Existing step-redox STEP ingestion remains authoritative for arbitrary vendor input.
  Truck's STEP loader is not used as an ingestion gate.
- KCL is emitted from the recovered constructive DAG and parsed/recast in optional conformance tests.
- Unrecovered exact B-rep can remain a sidecar STEP/foreign-import leaf while recovery proceeds; KCL intentionally treats foreign geometry as read-only.
- Truck evaluates supported constructive nodes and exports clean B-rep/STEP.
- No recovery detector stores Truck topology handles.
- The preferred interchange with the kernel is flattened/compressed topology where
  practical, not long-lived Arc/Mutex topology.
- Kernel output is never accepted merely because the operation returned Ok: it must
  pass the existing independent geometry/dimension/mass/render validation harness.

## Dependency posture

Use upstream Truck first. Keep all calls behind a narrow backend module and pin a
known-good git revision so replacing the source with our fork or another backend is mechanical.

Fork triggers include:

- hard-coded tolerance policy becomes materially wrong for our corpus and cannot be
  fixed upstream quickly;
- supported operations panic rather than return a typed failure;
- an approximation silently replaces analytic geometry we need to preserve;
- required STEP output forms or topology invariants are blocked upstream;
- a required exact operation cannot be implemented cleanly without carrying a downstream patch.

Kernel tolerances stay behind the adapter rather than becoming recovery-IR semantics.
Our proof tolerances remain explicit and independent.

## KCL dependency

`kcl-syntax` is currently a lossless lexer / future parser. The complete parser AST and
recasting API still live in `kcl-lib`. This spike therefore makes `kcl-lib` an **optional**
`kcl-conformance` feature with default features disabled. The normal backend path does
not compile it at all; it simply emits KCL. Production recovery should not depend on the
KCL runtime or Zoo engine protocol.

Parse/recast conformance belongs in optional tooling/CI until the standalone syntax
crate exposes the parser/typed AST.

## Current oracle

The initial 10 x 6 x 2 mm polygon extrusion produced a 7,112-byte STEP with the pinned
Truck revision. Independent OpenCascade import reported one valid solid/shell, six faces,
120.0 mm^3 volume, 184.0 mm^2 area, and the expected 10 x 6 x 2 mm bounding extent
(the OCP bounding box includes its normal 1e-7 mm tolerance).
