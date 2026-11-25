# Reproducibility guide

Run every command from the repository root.

## Requirements

- CPython 3.11 on `PATH` as `python3.11`, or pass `PYTHON=/path/to/python3.11`. numpy 1.24.4 has no wheels for 3.12. The committed outputs were produced with CPython 3.11.4. Any other 3.11.x gives the same numbers, but `protocol.json` files record `sys.version`, so their bytes will differ.
- `make`, and network access for `make setup` (PyPI) and `make data` (public data hosts).
- About 6 GB of free disk for the source objects, plus about 1 GB per reproduction run.
- Peak memory: about 2 GiB, reached while building the TLC truth vectors.

## Two environments

`make setup` builds two virtual environments from the same interpreter.

| Environment | Lock file | numpy | Used for |
| --- | --- | --- | --- |
| `.venv` | `requirements.lock` | 1.24.4 | real-data experiments, tooling |
| `.venv-synthetic` | `requirements-synthetic.lock` | 2.1.3 | synthetic experiments |

The synthetic ledgers store raw floating-point certificate values, and a few of them sit within rounding distance of τ. Their exact bits depend on the numpy summation routines, so the synthetic experiments are pinned to numpy 2.1.3. Two tests in `tests/test_canonical_consistency.py` check those raw values; they are skipped under numpy 1.x and run in `.venv-synthetic`. `make test` runs the full suite in both environments.

## Commands

| Command | What it checks | Needs data |
| --- | --- | --- |
| `make test` | Mathematical, unit, integration, and result-regression tests | No |
| `make verify` | Environment matches the lock; every import resolves inside the repository or to a pinned package; every committed experiment output matches `checksums/experiment_outputs.sha256`; hash locks, protocol hashes, and panel shapes hold | No |
| `make check-claims` | Each number in `manifests/REGRESSION_TARGETS.json` is recomputed from ledger rows and from the summary file, and the README quotes it at the stated rounding | No |
| `make data` | Downloads the 54 source objects and verifies their SHA-256 | — |
| `make reproduce` | Rebuilds the panel from the source objects and reruns all experiments in a fresh directory, then compares every output with the committed file | Yes |

## What `make reproduce` does

`scripts/reproduce.py` never writes under `experiments/`. It creates `reproductions/<run id>/` containing:

- `workspace/`, a copy of `src/` and every experiment runner in the repository layout, with the source objects linked under `data/source_objects/`;
- `logs/<stage>.{stdout,stderr}.log` and `logs/<stage>.metrics.json` (exit code, wall time, peak memory);
- `comparison.json`, which compares every output file with the committed file of the same path;
- `run_manifest.json`, which is deterministic for fixed inputs: source hashes, runner hashes, commands, exit codes, verdicts, and any upstream pinning;
- `stage_metrics.json`, with timings and memory;
- `claim_check.json`, which recomputes the registered numbers from the replayed ledgers.

Each output file is classified as one of:

| Class | Meaning |
| --- | --- |
| `BITWISE_IDENTICAL` | Same bytes |
| `SEMANTIC_EQUAL` | Same parsed content (JSON, JSONL, parquet, npy) after excluding volatile fields: wall-clock timings, interpreter strings, and hashes of input files that themselves hold timings |
| `NUMERIC_EQUAL` | Floats differ by at most 1e-12 |
| `VOLATILE_ONLY` | A timing file whose content differs |
| `DIFFERENT`, `NEW_FILE`, `NOT_PRODUCED` | Failure |

An experiment is `VERIFIED_BITWISE` when every file is byte-identical, and `VERIFIED_NUMERICALLY` when every file is in one of the first four classes.

The order-statistic, matched-rate, and time-forward runners refuse to start unless their upstream files carry the SHA-256 written in their source. The box-event and order-statistic ledgers store per-cell wall-clock runtimes, so a fresh ledger never carries the committed hash. For such a stage the orchestrator temporarily places the committed copy of each mismatching upstream file in the workspace, runs the stage, and restores the replayed file. Each substitution is recorded in `run_manifest.json` as `UPSTREAM_PINNED_TO_COMMITTED`. The replayed upstream file is compared with the committed one separately, so every number still has to agree.

## Run times

Measured by `make reproduce` on an Apple-silicon laptop (about 39 min in total):

| Stage | Wall time |
| --- | --- |
| Synthetic experiments (5 runners) | 2.5 min |
| Real-data panel (build, independent check, smoke test) | 17 min |
| Box-event evaluation | 3.7 min |
| Order-statistic evaluation | 7.7 min |
| Matched-rate utility | 4.4 min |
| Time-forward utility | 2.6 min |
| Consistency audit | 10 s |

## Running a single experiment

Each runner writes into its own directory, so `python experiments/<name>/run.py` replaces the committed outputs in place. `git diff` and `make verify` then show what changed. Timing files and the box-event and order-statistic ledgers change on every run because they record wall-clock runtimes; after rerunning one of those, update the hash locks of the downstream runners before running them. Use `make reproduce` to rerun everything without touching the committed files.
