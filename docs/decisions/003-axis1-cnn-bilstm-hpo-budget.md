# ADR 003 - CNN-BiLSTM HPO Budget on Axis 1

**Status:** Accepted
**Date:** 2026-05-06

## Context

Axis 1 evaluates CICIoT2023 under a static protocol with four model families. For
`cnn_bilstm`, the full dataset loaded for the run holds 46,776,700 flows, split
into approximately 28,066,020 training samples, 9,355,340 validation samples and
9,355,340 test samples.

Unlike tabular models, `cnn_bilstm` operates on temporal windows. This makes each
Optuna trial substantially more expensive, since every epoch traverses millions of
training and validation windows. In the run observed on 2026-05-05, a complete
trial took approximately 1h52. An HPO with 50 trials for this family would imply
several days of hyperparameter search alone, before the five final per-seed
repetitions.

A Windows multiprocessing error was also observed during validation:
`OSError: [Errno 22] Invalid argument`, followed by `pickle data was truncated`.
The cause was the use of `DataLoader` with workers spawned via `spawn`,
serializing large datasets. For this reason, Axis 1 must run PyTorch models with
`--num-workers 0` on Windows.

## Decision

For the paper, the production result of `cnn_bilstm` on Axis 1 will be generated
with the following command:

```bash
uv run python scripts/run_axis1.py --output-dir data/results/axis1_final --n-trials 2 --models cnn_bilstm --num-workers 0
```

The `--n-trials 2` parameter defines a limited and explicitly documented
computational budget for the `cnn_bilstm` HPO. The `--num-workers 0` parameter is
part of the official Windows protocol, adopted to avoid `DataLoader`
serialization failures.

Since Optuna persists the study in
`data/results/axis1_final/cnn_bilstm/optuna.db`, re-runs in that same directory
are resumable. That SQLite database and the `summary.json` must be preserved
locally as traceability evidence of the run; note that `data/results/*` is
covered by `.gitignore`, so neither file ships with the published artifact (see
Traceability below). Should it be necessary to reproduce exactly two fresh HPO
attempts, the previous `optuna.db` must be archived before the new run.

## Rationale

### Computational feasibility

The observed per-trial cost makes `--n-trials 50` impractical on local hardware
for `cnn_bilstm`. Reducing it to 2 trials yields a sequential baseline that is
executable on the full dataset, without replacing the protocol with artificial
subsampling.

### Preservation of the main protocol

The reduction applies only to the HPO budget, not to the dataset, the split, the
final seeds or the metrics. The result still uses the complete CICIoT2023 and the
five final seeds defined by Axis 1.

### Traceability for the paper

The command explicitly records:

- dataset and official output: `data/results/axis1_final`
- evaluated family: `cnn_bilstm`
- search budget: `--n-trials 2`
- Windows-safe operational decision: `--num-workers 0`

This choice must be reported in the paper text as a computational budget
limitation, precluding any interpretation that `cnn_bilstm` received an
exhaustive search equivalent to 50 trials.

What a third party can verify in the published artifact is
`results/paper_artifacts/axis1/cnn_bilstm/results.json`, which records
`n_trials` = 2, `best_hpo_val_f1` = 0.9888064587070832 and the selected
`best_params`, alongside the `git_hash` and `timestamp` of the run that produced
it. The `optuna.db` itself is local run output under `data/results/`, excluded by
`.gitignore` and therefore not distributed: per-trial durations and the pruning
history are available to the authors only.

## Consequences

- The `cnn_bilstm` results on Axis 1 are valid as a baseline under a limited
  computational budget.
- The paper must declare `n_trials=2` for the `cnn_bilstm` HPO.
- Performance comparisons must focus on the final model and its test metrics, not
  on equality of HPO effort across families.
- `--num-workers 0` may reduce loading parallelism, but it avoids the Windows
  multiprocessing failure and increases run robustness.
- Should additional computational resources become available, a larger search may
  be run as a complementary analysis, without replacing this official protocol.

## Alternatives considered

|Alternative|Reason for rejection|
|---|---|
|Keep `--n-trials 50`|Estimated cost of several days for the `cnn_bilstm` HPO alone, unfeasible for the paper schedule|
|Use `--sample-frac`|Would cut cost but change the effective dataset, weakening the comparison with the other Axis 1 models|
|Use `num_workers > 0`|Caused serialization/multiprocessing failure on Windows with large datasets|
|Drop `cnn_bilstm` from Axis 1|Would reduce sequential baseline coverage and impoverish the experimental comparison|
