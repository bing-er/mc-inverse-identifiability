# Auditing local identifiability in layered diffuse-reflectance inversion with differentiable Monte Carlo

This repository is the public reproducibility package for the manuscript
*Auditing local identifiability in layered diffuse-reflectance inversion with
differentiable Monte Carlo*.

It contains the frozen production configuration, machine-readable artifacts,
analysis code, exact transport-kernel snapshot, and executable checks underlying
the reported results. The archive is intentionally limited to scientific and
reproducibility materials; manuscript files, development logs, internal working
notes, submission correspondence, and temporary files are not included.

## Scope

The package supports two levels of reproduction:

1. **Evidence verification (CPU, minutes).** Validate the frozen stream pool,
   artifact hashes, released JSON, and numerical summaries without rerunning
   Monte Carlo transport.
2. **Transport regeneration (GPU, optional).** Recreate the production transport
   streams using the frozen JAX-PAD kernel and the recorded CUDA/H100 configuration.
   This is substantially more expensive and is not required to inspect or reproduce
   the reported analyses from the archived stream artifacts.

The Candidate 3 grid run is a non-confirmatory design pilot. Its artifacts do not
claim coverage and do not emit a Candidate 3 verdict.

## Repository layout

```text
spec/                 frozen numerical and execution configurations
src/                  reusable analysis code and frozen transport kernel
scripts/              analysis, validation, figure, and cluster entry points
data/derived/         frozen production AD and replay-pMC stream artifacts
data/pilot/           small pilot input
results/              machine-readable evidence used by the manuscript
```

The manuscript sources, compiled PDFs, translations, submission material, and
working protocols are deliberately maintained outside this public repository.

## Quick verification

Python 3.11 or newer is recommended. The archived local analysis environment used
Python 3.14.6 with the versions in `requirements-lock.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
./reproduce.sh verify
```

The verification command checks the public-package boundary, validates every JSON
artifact, verifies the frozen transport source hashes, and confirms the complete
32-stream AD and replay-pMC pools.

To rebuild the primary pooled analysis and leave-one-stream-out sensitivity into a
separate `reproduced/` directory:

```bash
./reproduce.sh analysis
```

To regenerate the three primary figures from the frozen JSON artifacts into the
ignored local directory `reproduced/figures/`:

```bash
./reproduce.sh figures
```

## Frozen transport environment

The production AD pool used JAX 0.9.1 on an NVIDIA H100 partition with CUDA 12.9
and `ptxas` 12.9.86. The per-stream metadata and provenance records preserve the
exact production settings. The two transport files that determine the photon path
and production aggregation are included under `src/pad/` and are checked against
the hashes embedded in the archived evidence.

The replay-pMC and AD pools each contain 32 streams of 200,000 photons per
wavelength. Analysis-only reproduction uses these archived streams and does not
require a GPU.

## External data

The homogeneous liquid-phantom observations discussed in the manuscript were
reported by Lan, McClarren, and Vishwanath (2023), DOI
`10.1364/BOE.490164`. They are not redistributed in this package. The archived
legacy summaries are used only for the qualitative comparisons stated in the
manuscript.

## Integrity

`MANIFEST.sha256` records the SHA-256 digest of every released file except the
manifest itself. Run `./reproduce.sh manifest` to verify it.

## Citation

Citation metadata are provided in `CITATION.cff`. A version-specific Zenodo DOI
will be added before journal submission.

## Licenses

Software is licensed under the MIT License (`LICENSE`). Data, results,
specifications, and release metadata are licensed under CC BY 4.0
(`LICENSE-DATA`).
