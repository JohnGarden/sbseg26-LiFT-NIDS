# ADR 004 - Transformer HPO Budget on Axis 1

**Status:** Accepted
**Date:** 2026-05-06

## Context

Axis 1 evaluates CICIoT2023 under a static protocol with four model families. For
`transformer`, the full dataset holds 46,776,700 flows, split into approximately
28,066,020 training samples, 9,355,340 validation samples and 9,355,340 test
samples.

The `transformer` operates on temporal windows (flow sequences), making each
Optuna trial substantially more expensive than for tabular models. In the run
observed on 2026-05-06:

|Trial|State|Validation F1|Duration|
|---|---|---|---|
|0|COMPLETE|0.9892|2h33m54s|
|1|FAIL (Ctrl+C)|—|3m13s|

The observed cost of ~9,235 s/trial implies that an HPO with 50 trials would take
approximately **128 hours (5.3 days)** for hyperparameter search alone, before the
five final per-seed repetitions. That cost is unfeasible within the paper's
schedule.

The hyperparameters found in trial 0 were:

```text
lr           = 5.612e-4
weight_decay = 9.507e-4
dropout      = 0.233
head_dropout = 0.200
batch_size   = 512
```

## Decision

The production result of `transformer` on Axis 1 will be generated with the
following command:

```bash
uv run python scripts/run_axis1.py \
  --output-dir data/results/axis1_final \
  --n-trials 1 \
  --models transformer \
  --num-workers 0
```

The `--n-trials 1` parameter instructs the script to accept the already completed
trial 0 as the full HPO budget. The resume mechanism in `run_optuna_search`
detects that `finished >= n_trials` (2 >= 1) and returns immediately with the best
parameters from the existing database, without running new trials.

## Rationale

### Architecture fixed by the literature (main argument)

The Transformer architecture is fixed by the compact FlowTransformer
configuration identified by Manocchio et al. (2024): 2 layers, 2 heads,
d_model=128, record-level projection, Last Token head. The hyperparameters
searched by Optuna are exclusively training parameters (learning rate, weight
decay, dropout, batch size), which have marginal impact on a model whose
expressiveness is already determined by the architecture. An exhaustive HPO would
be redundant given that the architectural search space is collapsed to a single
point.

### Computational feasibility

The cost of ~2h34min per trial, multiplied by 50 trials, exceeds 128 hours of
compute for HPO alone — not counting the five final repetitions with distinct
seeds. That cost is disproportionately high for an axis that serves as a sanity
check (CICIoT2023) ahead of the main axis (MAWIFlow).

### The trial 0 result is competitive

The validation F1 obtained in trial 0 (0.9892) is comparable to the best MLP
result (0.99097, obtained in trial 14, the last of the 15 in that budget), whose
HPO budget limitation had already been decided by the same per-trial cost
criterion.

### Acknowledged limitation: no estimate of the forgone gain

With a single completed trial there is no basis for estimating how much a larger
budget would yield: there is neither observed dispersion nor a trajectory of the
running best value. The only empirical evidence available on Axis 1 indicates
that the gain is not negligible — for the MLP, the best `val_F1` rose from
0.98894 in trial 0 to 0.99097 in trial 14, roughly 0.002 across 15 trials
(ADR 009). It is reasonable to assume that a larger budget would also find a
slightly higher `val_F1` for the Transformer.

Therefore **the budget was closed by cost, not by demonstrated convergence**, and
the Transformer results on Axis 1 must not be read as an estimate of the family's
performance ceiling. This caveat is the same one recorded in ADR 009 and applies
equally to ADR 003. What sustains the decision is not the assumption that further
search would be useless, but the three arguments above: the architecture is fixed
by the literature, the cost is prohibitive, and Axis 1 is a sanity check, not the
axis on which the paper's conclusions rest.

### Consistency with earlier decisions

CNN-BiLSTM (ADR 003) was already limited to 2 trials for the same reason. The MLP
was limited to 15 trials, by the same per-trial cost criterion. XGBoost completed
50 trials by being fast enough per trial. The HPO budget difference across
families is justified by the observed computational cost, not by a different
quality criterion.

### Traceability

The database `data/results/axis1_final/transformer/optuna.db` records the
complete trial 0 with its seed, parameters and metric, and a re-assessment run by
the authors may inspect or extend the study from it. That database is local run
output under `data/results/`, excluded by `.gitignore`, and does **not** ship
with the published artifact.

What a third party can verify is
`results/paper_artifacts/axis1/transformer/results.json`, which records
`n_trials` = 1, `best_hpo_val_f1` = 0.9892027091277759 and the `best_params`
reproduced in the Context section above, alongside the `git_hash` and
`timestamp` of the run.

## Consequences

- The `transformer` results on Axis 1 are valid as a sanity check under a limited
  computational budget, not as an estimate of the family's performance ceiling.
- The paper must declare `n_trials=1` for the `transformer` HPO on Axis 1, citing
  this ADR as justification.
- The five final repetitions with distinct seeds (42, 123, 456, 789, 1024) use the
  complete dataset in full and the hyperparameters from trial 0.
- `--num-workers 0` is mandatory on Windows to avoid `DataLoader` serialization
  failure (see ADR 003).

## Alternatives considered

|Alternative|Reason for rejection|
|---|---|
|Keep `--n-trials 50`|Estimated cost of ~128 hours, unfeasible for the schedule|
|Use `--sample-frac` for HPO|Would change the effective dataset, weakening the comparison with the other models|
|Search only LR and batch_size (2 parameters)|Would not reduce per-trial cost; the bottleneck is epoch time, not parameter sampling|
|Drop `transformer` from Axis 1|Would remove the paper's central model from the sanity check, harming experimental coverage|
