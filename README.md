# step-redox

step-redox rewrites ISO 10303-21 STEP files into smaller, semantically equivalent
Part 21 files.

The project now has three output policies:

- `--profile manual`: only explicitly selected transformations
- `--profile compat`: broadly compatible plain-BREP cleanup
- `--profile compact`: proven semantic recovery, instancing, and aggressive compact output

The current semantic layer also recovers regular instance patterns, coupled count
parameters, and periodic body grammars. On a validated 2.00 mm dual-row header,
step-redox independently recovers two contact rows plus a 24-face-per-site housing
grammar. Resizing the recovered 24-site model to 36 sites reproduces an
independently exported 72-contact sibling, and shrinking the 36-site sibling back
to 24 sites reproduces the independent 48-contact model. Both directions preserve
the expected solid/face/edge/vertex counts and dimensions to numerical noise. The
direct graph rewrite produces a valid closed B-rep without booleans or tessellation.

The high-level count editor composes the recovered body grammar and every coupled
instance row atomically:

    cargo run --release --bin step-count-resize -- input.step output.step --sites 36

Raw input is normalized into the editable semantic representation first. The
count editor supports start-, end-, and center-anchored placement. Center anchoring
composes equal edits at both physical ends and currently requires an even site-count
delta. The editor re-runs semantic detection after mutation and fails rather than
returning a model unless the requested count and the same coupled body/instance
grammar are recovered again.

The validated 24↔36-site header regressions now pass with center anchoring under
identity alignment in both directions: no compensating translation is permitted by
the comparison harness.

A separate read-only periodic-chain analyzer recovers repeated structure inside
fused solids even when no editable `MAPPED_ITEM` row exists. On three independent
first-party TE Connectivity siblings it proves the same 1.500 mm family grammar:
14/15/18 sites, 70 interior site faces + 3 gap faces per added position, six
stretch faces, and symmetric 33-face end regions. This matches the independently
observed +73 faces per position across the family. The result is intentionally
read-only; it does not authorize TE count mutation yet.

A browser/WASM prototype lives under `web/`, including an experimental
"Unscrew your STEP" front end. The Rust library exposes detected patterns,
periodic bodies, and higher-level recovered count parameters; bidirectional
periodic-body growth/shrink are implemented in `periodic_resize.rs`.

## Validation and CI

The regression harness under `validation/` downloads public upstream CAD fixtures
by URL, verifies SHA-256, runs step-redox, and validates results independently with
OpenCascade. Fixtures are cached in CI and are not redistributed in this repo.

Checks include:

- B-rep validity
- solid/shell/face/edge/vertex counts
- bounding-box center and dimensions
- volume, surface area, total edge length
- center of mass and inertia tensor
- software-rendered ±X/±Y/±Z and isometric views with image/silhouette diffs

The initial CI corpus includes public JLC/EasyEDA models and first-party TE
Connectivity STEP models. GitHub Actions uploads the numerical report and render
contact sheets for inspection.

Run locally with:

    cargo build --release --bin step-redox
    python -m pip install -r validation/requirements.txt
    python validation/harness.py run \
      --manifest validation/fixtures.json \
      --cache .cache/step-redox-validation \
      --out validation/out \
      --step-redox target/release/step-redox

The current default/manual pipeline is deliberately conservative unless a profile
or experimental pass is selected.

## Safe/default passes

- parse the Part 21 exchange structure with ruststep
- accept UTF-8 and legacy GBK STEP text
- emit non-ASCII strings as standard X2 Unicode escape sequences
- emit compact numeric/text serialization and dense entity IDs
- structurally intern equal value/support entities:
  - points, directions, vectors, placements
  - common curves and surfaces
  - colours and presentation style values
  - units, uncertainty values, and representation contexts
- consolidate otherwise-unreferenced one-item presentation representations and
  layer assignments
- reparse/idempotence tests cover the writer and reference rewriting

Topological entities are intentionally excluded from interning.

## CLI

cargo run --release -- input.step output.step

Useful switches:

    --no-intern
    --no-presentation-consolidation
    --experimental-recover-straight-bspline-lines
    --experimental-instance-z90
    --experimental-instance-spherical-caps
    --minify-placeholder-names
    --no-dense-ids
    --json

--json writes cleanup statistics to stderr.

## EasyEDA corpus results

On a random 100-model sample from the JLCPCB/EasyEDA archive:

- input: 237,872,181 bytes
- output: 79,914,474 bytes
- weighted output ratio: 33.60%
- median per-file ratio: 33.11%
- successful parses/rewrites: 100 / 100

Independent B-rep reconstruction with truck-stepio matched the safe output
against lexical-only output for 15 sampled models.

Representative files:

| model | original | safe output | ratio |
|---|---:|---:|---:|
| QFN-14 | 1,776,783 | 576,652 | 32.45% |
| BGA-636 | 5,628,709 | 1,652,726 | 29.36% |
| SOP-4 | 1,750,016 | 604,036 | 34.52% |
| connector | 1,582,856 | 558,029 | 35.25% |
| THT capacitor | 1,070,051 | 334,831 | 31.29% |

The 132,381,990-byte largest connector in the current corpus cleans to
56,269,340 bytes (42.51%) with the same safe passes.

A later random 1,000-model scan parsed and cleaned **1000 / 1000** files after adding compatibility for legal empty STEP aggregates such as `SHAPE_REPRESENTATION('',(),#ctx)`. The parser shim exists because ruststep 0.4's aggregate grammar currently requires one-or-more parameters despite the Part 21 grammar allowing zero.

With `--experimental-instance-z90`, 263 / 1000 models (26.3%) contained at least one strictly proven repeated-solid group. Those triggered models shrank from 361,325,878 safe-tier bytes to 265,374,780 bytes: an additional **26.6%** reduction after the safe cleanup, saving 95,951,098 bytes in the sample.

## Experimental mapped-item work

`--experimental-instance-z90` instances repeated top-level `MANIFOLD_SOLID_BREP` objects using standard STEP `REPRESENTATION_MAP` + `MAPPED_ITEM`.

It is deliberately conservative. A candidate group must have matching topological counts, matching vertex geometry under a proven Z-axis quarter-turn + translation, matching shell/face/bound/loop/edge connectivity, matching recursively normalized supporting curve/surface geometry, and one uniform explicit face style per solid.

Each source-to-target transform is proven directly; it is not inferred from canonical-orientation bookkeeping. The map origin is global zero and each target placement encodes the direct rigid transform `p' = R*p + d`.

Current corpus examples:

| model | original | safe output | aggressive output | aggressive ratio |
|---|---:|---:|---:|---:|
| QFN-14 | 1,776,783 | 576,652 | 496,998 | 27.97% |
| SOP-4 | 1,750,016 | 604,036 | 429,156 | 24.52% |

The current strict detector instances 12 solids across 4 proven groups in the QFN and 6 solids across 2 groups in the SOP sample. Some geometrically equivalent objects are intentionally left expanded when the STEP-level proof is ambiguous.

The aggressive output is byte-idempotent under a second cleanup pass.

### OpenCascade validation

The mapped-item path is independently tested with OpenCascade 8 via the scripts under `tools/`.

For both QFN and SOP examples, STEP import succeeds; solid count and topology counts are preserved; individual solid volume, surface area, center of mass, and bounding box match; matched-solid boolean overlap is complete to floating-point noise; and bidirectional boolean cuts leave no measurable residual geometry.

Worst observed missing intersection volume is about `3.5e-18 mm^3` for QFN and `1.7e-16 mm^3` for SOP.

This validation caught two real implementation errors during development: accidentally putting mapping-target placements in the parent representation, and incorrect mapping-origin/target transform construction. It remains part of the acceptance criteria for expanding the aggressive rewrite set.

## Experimental straight B-spline recovery

`--experimental-recover-straight-bspline-lines` replaces exporter-generated
`B_SPLINE_CURVE_WITH_KNOTS` edge supports with native STEP `LINE` geometry
when the equivalence can be proved from the STEP topology and control polygon.

The pass is intentionally stricter than a collinearity test. A candidate must
be a finite, clamped, non-closed, non-self-intersecting B-spline; its control
points must be collinear and monotonic without endpoint overshoot; it must have
exactly one inbound use, as the support curve of one `EDGE_CURVE`; and that
edge's two `VERTEX_POINT` coordinates must match the first/last spline poles
within `1e-12 mm` (in either orientation). Direction support is shared only
after a second endpoint-residual proof against the representative direction.
Only former control-point `CARTESIAN_POINT` records that become completely
unreferenced are garbage-collected.

This extra endpoint proof is necessary in real SolidWorks output. On
`CONN-SMD_ASP-184330-01-1`, a simpler prototype found 83,489 collinear
splines, but 35 of them were trimmed by `EDGE_CURVE` vertices that did not
match the control-polygon endpoints. Rewriting those 35 changed reconstructed
surface area by about `2.84 mm^2` and was rejected.

The strict pass recovers **83,454** curves, removes **166,793** orphan control
points, and uses 132 shared signed direction groups. The already-cleaned model
shrinks from **64,532,069** to **42,884,762 bytes** (another **33.55%**).
The output is byte-idempotent on a second cleanup. OpenCascade 8 reconstructs
the same 35,491 faces, 200,978 edges, 401,956 vertices, and 89 shells; total
edge length is exactly unchanged, surface-area delta is about
`-6.9e-11 mm^2`, and bounding-box deltas are at floating-point noise.

## Experimental planar positive-feature arrays

`--experimental-instance-planar-positive-features` factors repeated positive
features that are fused into a shared closed shell through a planar host face.
It is a geometry-changing aggressive pass: one canonical feature is closed on
the exact host plane and reused through `REPRESENTATION_MAP` /
`MAPPED_ITEM`, while the expanded feature faces and matching host holes are
removed.

Detection is deliberately strict. The source must be a closed shell, the host
must be planar, every extracted feature edge must be exactly two-manifold with
the feature or host, the complete feature/host interface must match exactly one
host `FACE_BOUND`, all feature vertices must lie on the host's outward side,
and normalized B-rep topology/support geometry must match under a Z
quarter-turn. Mixed or ambiguous styling is rejected. Families smaller than
eight are left expanded.

Validated examples:

- PGA1331: **23,536,653 -> 11,865,565 bytes**, 1 array / 1 family /
  **1,331 instances**, 234,093 entities removed. OpenCascade reports both
  B-reps valid, volume delta about `6.47e-10 mm^3`, and exact
  `source - compact = 0.0` / `compact - source = 0.0` boolean residuals.
- CONN-SMD_ASP-184330-01-1 after straight-line recovery:
  **42,884,762 -> 8,732,968 bytes**, 2 arrays / 6 families /
  **1,108 instances**, 625,853 entities removed. Independent OpenCascade
  validation finds the source, residual body, and all mapped feature solids
  valid, with both whole-body boolean residuals exactly `0.0`.
- BGA-636: **1,652,726 -> 534,060 bytes**, **636 instances**. Both
  whole-shape boolean residuals are exactly `0.0`; this is 70 bytes smaller
  than the older dedicated spherical-cap pass.

On the 40 largest raw corpus models, the strict pass triggers on only **5/40**
models but saves **58,722,511 bytes** incrementally on those five. Combined
with the other current aggressive passes, the top-40 processed total falls from
**528,783,798** to **442,214,381 bytes**.

The output is byte-idempotent after all applicable aggressive passes have run.

## Experimental planar spherical-cap arrays

`--experimental-instance-spherical-caps` is the older specialized implementation for large arrays of identical spherical-cap features fused into one substrate solid. The generic planar positive-feature pass now subsumes the validated BGA-636 case and produces a slightly smaller result; this flag is retained while broader corpus overlap is audited.

The pass is intentionally narrow. It requires two matching spherical faces per feature, an exact two-edge circular interface to one shared planar face, a one-to-one matching `FACE_BOUND` hole in that plane, identical normalized B-rep topology and style across the feature array, and identical sphere centers modulo translation.

It removes the circular holes from the substrate plane, keeps one feature as a canonical closed solid by adding a planar interface disk, and maps that solid to all feature locations with `REPRESENTATION_MAP` / `MAPPED_ITEM`. This changes solid decomposition and introduces hidden coincident interface faces, so it is kept separate from the topology-preserving safe tier.

On the BGA-636 corpus example:

- original EasyEDA STEP: **5,628,709 bytes**
- safe cleanup: **1,652,726 bytes**
- spherical-cap instancing: **569,223 bytes**
- final size: **10.11% of the original**
- 636 spherical-cap instances
- 24,133 additional entities removed
- byte-idempotent under a second cleanup

OpenCascade validates both source and compact shapes. Their bounding boxes are identical and their whole-shape volumes differ only by about `5e-12 mm^3`. More decisively, boolean `source - compact` and `compact - source` both produce exactly `0.0` residual volume. Surface area intentionally increases because the compact representation contains 636 hidden closure disks at the former fused interfaces.

A lexical survey of 891 downloaded BGA/FCBGA/UFBGA/NFBGA/CSPBGA/WLCSP-family models found 484 with at least 100 spherical-surface records, 160 with at least 500, and 47 with at least 1000. The worst sampled model contains 4,346 spherical surfaces, so this optimization is not specific to one BGA.

## Optional placeholder-name minification

`--minify-placeholder-names` changes exact `NONE` placeholder names to empty strings only on an explicit allowlist of geometry/topology/presentation entity types. It does not touch product IDs, descriptions, or arbitrary string fields.

This is metadata-minifying rather than metadata-preserving, so it is not enabled by default. On representative cleaned files it saves another roughly 6-7%:

- QFN safe: 576,652 -> 534,484 bytes
- SOP safe: 604,036 -> 561,636 bytes
- BGA-636 after spherical-cap instancing: 569,223 -> 534,130 bytes

The minified output is byte-idempotent. OpenCascade geometry comparison remains exact within the established tolerances, and XDE import of the SOP preserves all 484 face colors with the same color histogram.

## Validation philosophy

The default tier should be boring:

1. parse source
2. rewrite only identities STEP permits to be shared
3. preserve topology identity
4. parse output again
5. require byte-idempotence on a second cleanup
6. compare reconstructed B-rep geometry in an independent implementation on
   corpus samples

More invasive transformations belong behind explicit experimental/aggressive
modes until separately validated.
