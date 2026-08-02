# ADR 005 - HPO Budget on Axis 2 (MAWIFlow)

**Status:** Accepted
**Date:** 2026-05-07

## Context

Axis 2 applies HPO once on the anchor year (2007) and reuses the best parameters
across every horizon k and seed. This differs from Axis 1, where HPO is performed
over the complete dataset. Here the anchor year is split temporally into
60/20/20 fit / validation / internal-test subsets, the three parts summing to the
full year (`temporal_train_val_test_split`, exercised by
`tests/test_temporal_protocol_integration.py`).

The published manifest records 2,718,818 flows for 2007 across 5 capture days
(`results/paper_artifacts/data_manifest.json`), which yields approximately:

- fit: ~1,631,291 samples
- validation: ~543,764 samples
- internal test: ~543,764 samples

Earlier revisions of this ADR reported ~978,775 fit and ~326,258 validation
samples. Those figures date from the pilot data build and are inconsistent with
the published manifest: they describe an anchor year of about 1.63M flows rather
than 2.72M, and were never recomputed after the final dataset was assembled —
2007 is the only year that deviates from the one-day-per-quarter sampling policy,
carrying a fifth capture day. The figures above supersede them. The budget
decision below does not depend on the exact split sizes; they are reported only
to characterize the scale at which each HPO trial runs.

The pilot (`axis2_pilot_seed42_clean`) completed the following HPO budgets:

|Model|Pilot n_trials|Observed cost per trial|
|---|---|---|
|XGBoost|50|~seconds (CPU)|
|MLP|5|not measured (fast)|
|CNN-BiLSTM|5|not measured|
|Transformer|5|not measured|

The default of the `run_axis2.py` script is `--n-trials 50`. Running production
without stating `--n-trials 5` for MLP, CNN-BiLSTM and Transformer would cause 45
extra trials per model (since Optuna counts the trials already present in the
database and completes up to the target), needlessly increasing total HPO time
ahead of the 5 forward-chaining seeds.

## Decision

The Axis 2 production run uses the following HPO budgets:

|Model|Production n_trials|Flag stated|
|---|---|---|
|XGBoost|50|`--n-trials 50` (same as the default)|
|MLP|5|`--n-trials 5`|
|CNN-BiLSTM|5|`--n-trials 5`|
|Transformer|5|`--n-trials 5`|

In the original run, the pilot `optuna.db` databases were copied into the
production directory beforehand, so that the script recognized the already
completed trials and skipped HPO entirely
(`remaining_trials = max(0, n_trials - finished) = 0`).

That step is **not reproducible by third parties**: the pilot directory
(`data/results/axis2_pilot_seed42_clean/`) was local run output, covered by
`data/results/*` in `.gitignore`, and does not ship with the published artifact.
It has therefore been removed from the commands below, which start from a clean
clone. With an empty directory the count of completed trials is zero, and the
explicit `--n-trials` makes HPO run from scratch on the anchor year with the same
budget declared in the table above.

Canonical commands (run in sequence, one model at a time):

```bash
uv run python scripts/run_axis2.py \
  --output-dir data/results/axis2_final \
  --num-workers 0 --models xgboost --n-trials 50 --resume

uv run python scripts/run_axis2.py \
  --output-dir data/results/axis2_final \
  --num-workers 0 --models mlp --n-trials 5 --resume

uv run python scripts/run_axis2.py \
  --output-dir data/results/axis2_final \
  --num-workers 0 --models cnn_bilstm --n-trials 5 --resume

uv run python scripts/run_axis2.py \
  --output-dir data/results/axis2_final \
  --num-workers 0 --models transformer --n-trials 5 --resume
```

`--resume` preserves the resumption of already completed seeds; on an empty
directory it has no effect. These are the same commands given under Option B of
Claim #2 in the artifact README.

## Rationale

### Axis 2 HPO is ancillary, not central

The goal of Axis 2 is to assess temporal robustness via forward-chaining with R=5
seeds. HPO quality affects the absolute performance level but does not change the
relative ordering of the models — which is the paper's central argument. An
HPO with 5 trials on the anchor year suffices to rule out bad configurations; the
additional refinement from 45 further trials would not alter the conclusions.

### Consistency with the validated pilot

The pilot with seed=42 and 5 HPO trials already reproduced the ordering that
sustains the paper's argument: LightTransformer ahead of CNN-BiLSTM, XGBoost and
MLP on nAUT_1. Raising the HPO budget in production only would create an
asymmetry between pilot and production with no clear scientific benefit.

The absolute levels observed in the pilot must not be cited as results: the pilot
used a single seed, whereas Table 6 of the paper reports the mean over R=5 seeds
with a bootstrap confidence interval. The published nAUT_1 values for the
cumulative window are LightTransformer 0.568, CNN-BiLSTM 0.439, XGBoost 0.377 and
MLP 0.282, verifiable in
`results/paper_artifacts/axis2/<model>/k_cumulative/results.json`. What carried
over from the pilot to production was the ordering, not the magnitude.

### Disproportionately high computational cost for sequential models

CNN-BiLSTM and Transformer require processing sliding windows in sequence. Even
over the 2007 subset (~1.3M samples), each trial of these models may take tens of
minutes. Adding 45 extra trials would double or triple the total Axis 2 time
without changing the conclusions.

### XGBoost already completed 50 trials in the pilot

XGBoost is fast enough to complete 50 trials in a few minutes on the anchor year.
The re-HPO observed in the production run (2026-05-07) finished in ~1 s,
confirming that the cost is negligible. Keeping 50 trials for XGBoost preserves
maximal rigor for the strongest baseline model.

### Traceability

The pilot `optuna.db` databases record the 5 completed trials with their seeds,
parameters and validation metrics. As stated in the Decision section, they are
local run output under `data/results/`, excluded by `.gitignore`, and do **not**
ship with the published artifact — they are traceability evidence available to
the authors, not to a third party.

What a third party can verify in the artifact is
`results/paper_artifacts/axis2/<model>/k_cumulative/results.json`, which records
`n_trials` (50 for XGBoost, 5 for the other three families) and the `best_params`
reused across every horizon and seed. Re-deriving the search itself requires
re-running HPO with the commands above, which is the reproducible path.

## Consequences

- The paper declares `n_trials=5` for MLP, CNN-BiLSTM and Transformer in the Axis 2
  HPO (anchor year 2007), and `n_trials=50` for XGBoost.
- The budget difference across families is justified by the observed computational
  cost, consistent with the criterion already adopted on Axis 1 (ADR 003, ADR 004).
- `--num-workers 0` is mandatory on Windows to avoid `DataLoader` serialization
  failure (see ADR 003).

## Alternatives considered

|Alternative|Reason for rejection|
|---|---|
|Keep `--n-trials 50` for all|45 extra trials per sequential model: high cost with no gain in the conclusions|
|Use `--sample-frac` to speed up HPO|Would change the effective anchor-year dataset and bias the partition point|
|Run HPO from scratch (without copying the DBs)|Discards valid pilot work; XGBoost re-HPO is fine, but the sequential models would pay avoidable cost|
