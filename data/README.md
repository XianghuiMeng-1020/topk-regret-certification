# Data

All real-data inputs are public, dated objects. Each one is pinned by URL, byte length, and SHA-256 in [`../manifests/SOURCE_OBJECTS.json`](../manifests/SOURCE_OBJECTS.json) and [`../checksums/source_objects.sha256`](../checksums/source_objects.sha256).

| Directory | Contents | In Git |
| --- | --- | --- |
| `source_objects/tlc/` | 4 NYC TLC yellow-taxi monthly parquet files: 2024-01 and 2024-02 (calibration), 2024-04 and 2024-05 (evaluation) | No, run `make data` |
| `source_objects/gharchive/` | 50 GH Archive hourly files at 15:00 UTC: 14 calibration days (2024-01-02 to 2024-01-15) and 36 evaluation days from 2024-02-01 | No, run `make data` |

`make data` downloads every missing object and discards any byte stream whose length or SHA-256 differs from the pin. About 5.9 GB is needed (TLC 0.22 GB, GH Archive 5.69 GB). `make verify` re-hashes whatever is present.

## Derived inputs

[`../experiments/real_data_panel/build_truths.py`](../experiments/real_data_panel/build_truths.py) turns the source objects into the inputs of every real-data experiment, following [`panel_spec.json`](../experiments/real_data_panel/panel_spec.json):

- `candidate_universes/`: the 261 TLC pickup zones and the 2,000 most active GitHub repositories over the calibration days;
- `truth_vectors/`: 81 count vectors, one per evaluation day (45 TLC days, 36 GitHub days), each with a SHA-256 in `manifests/`;
- `integrity/`: independent recounts and shape checks written by `check_integrity.py`.

These derived files are committed, so every experiment after the panel runs without downloading anything.

## Terms

NYC TLC trip records are published by the NYC Taxi and Limousine Commission as open data. GH Archive files are public event logs. This repository redistributes none of the raw files. It records where to fetch them and the exact bytes expected, and commits only derived per-day count vectors.
