# STEP validation harness

This directory is the regression suite for step-redox.

Upstream CAD files are not stored in git. The fixture manifest records public
download URLs, SHA-256 checksums, and source notes. CI downloads a fixture only
when it is absent from its private cache and rejects it if the checksum changes.

Each case runs step-redox or the semantic count-generator CLI and validates
the result with several independent signals. The same manifest therefore covers
ordinary optimizer roundtrips and cross-model generation. The strongest
pair now exercises the same recovered header grammar in both directions:
48 contacts → 72 contacts and 72 contacts → 48 contacts, each compared with an
independently published sibling model.

- OCCT import and BRepCheck validity
- solid, shell, face, edge, and vertex counts
- bounding-box center and extents
- compound-level volume, surface area, total edge length
- per-solid topology, position, extents, volume, and area
- summed per-solid material volume/area for multi-solid assemblies
- center of mass and inertia tensor
- deterministic software renders from +X, -X, +Y, -Y, +Z, -Z, and two isometric views
- render mean absolute error, changed-pixel fraction, and silhouette XOR

For mapped multi-solid assemblies, OpenCascade's one-shot compound volume can differ by several ppm even when every imported solid matches independently to numerical noise. The harness therefore keeps the compound integral visible but uses the summed per-solid volume as the stricter occupied-material invariant.

OCCT performs tessellation, but rendering is done in software with Pillow.
There is no OpenGL/GPU/display-server dependency, so CI images should be
deterministic and easy to inspect as artifacts.

Thresholds are explicit per test case in fixtures.json. Exact topology can be
required independently from geometric tolerances. A case comparing two
independent vendor models may use bbox-center alignment to permit a known rigid
translation without permitting scaling or shape changes.

Local invocation:

    python -m pip install -r validation/requirements.txt
    cargo build --release --bin step-redox --bin step-count-resize
    python validation/harness.py run \
      --manifest validation/fixtures.json \
      --cache .cache/step-redox-validation \
      --out validation/out \
      --step-redox target/release/step-redox \
      --step-count-resize target/release/step-count-resize

CI uploads validation/out as an artifact, including report.json, summary.md, and
the reference/candidate/difference image triplets for every camera.

Fixture policy:

1. Prefer public first-party manufacturer CAD URLs when practical.
2. Otherwise use stable public first-party distribution/model endpoints.
3. Never commit the downloaded STEP files.
4. Pin every fixture by SHA-256.
5. Keep enough provenance in source_note to locate/review the source later.
