# Release provenance

## Source snapshot

This clean public package was assembled from the evidence-bearing state of the
private working repository at source commit:

```text
68f8076ccf5335b57d96f79cb0a29807b96b67b6
```

The public repository intentionally starts with a new history and does not carry
development logs, internal working notes, submission correspondence, or temporary
files from that workspace.

## Frozen transport source

The minimum JAX-PAD production source required by the released transport scripts
is vendored under `src/pad/`. The evidence artifacts bind the two decisive files
to these SHA-256 values:

```text
src/pad/mc_forward.py
9e1236fa90da223d56788ed2bccdd1e32eccb787788f03a4f0e5ebca47bf4b0f

src/pad/physics_engine_v4.py
4616e22226a015645f15c0dfa8582fab73a3c18d4bb6b1ff26bb7d54bae77f68
```

`scripts/verify_public_release.py` recomputes both hashes and refuses a mismatch.

## Production environment

The production AD pool records:

- 32 independent streams;
- 200,000 photons per wavelength per stream;
- four wavelengths and 30 annular bins per wavelength;
- JAX 0.9.1, GPU backend;
- NVIDIA H100 execution partition;
- CUDA 12.9 and `ptxas` 12.9.86.

The replay-pMC pool uses the same frozen photon streams and transport kernel. Full
per-stream metadata and provenance are stored with the NPZ and JSON artifacts.

## Evidence boundary

The archive supports exact inspection of the reported numerical evidence and
CPU regeneration of the pooled analysis, leave-one-stream-out analysis, and
primary figures. Manuscript and submission files are intentionally outside this
public release. Re-executing the production transport is optional and requires
compatible GPU resources. Agreement between the released AD and replay-pMC
constructions is an estimator-robustness result, not independent validation of
either derivative.
