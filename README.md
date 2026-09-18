# step-redox

step-redox rewrites ISO 10303-21 STEP files into smaller, semantically equivalent
Part 21 files.

The current default pipeline is deliberately conservative. It does not merge
topological identity objects (vertices, edges, loops, faces, shells, or solids).

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
