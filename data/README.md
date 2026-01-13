# Data

This repository intentionally does **not** version the full CasinoLimit-derived CSV used in the lab
(to avoid licensing / redistribution issues).

## Expected raw file

Look the raw CSV here:

- `data/raw/114_commands_double_labeled_37_classes_counted.csv`

The CSV is expected to contain (at least) these columns:

- `proctitle` (string) — command-line / process title
- `technique` (string) — original MITRE ATT&CK technique label (fine-grained)
- `technique_grouped` (string) — grouped label used in the lab (37 classes)
- `count` (int) — frequency / weight

A tiny example is provided in `data/sample/commands_sample.csv` to show the format.

## Download

See the project README for pointers to the CasinoLimit dataset and associated paper/tooling.
