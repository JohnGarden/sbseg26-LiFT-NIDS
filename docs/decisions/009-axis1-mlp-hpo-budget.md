# ADR 009 - MLP HPO Budget on Axis 1

**Status:** Accepted
**Date:** 2026-07-26 (formal record of a decision executed on 2026-05-04)

## Context

Axis 1 evaluates CICIoT2023 under a static protocol with four model families. The
project's default HPO budget is 50 Optuna trials per model, met in full only by
XGBoost, whose per-trial cost is about 1 minute.

The `mlp` operates over the complete dataset loaded for the run: 46,776,700
flows, split into approximately 28,066,020 training samples, 9,355,340 validation
samples and 9,355,340 test samples. Each trial traverses that volume per epoch, so
the per-trial cost sits an order of magnitude above XGBoost.

The `axis1_mlp` study, executed on 2026-05-04 between 09:41 and 21:23, spent
**11.7 wall-clock hours** on 15 trials with an assigned value. Per-trial duration
ranged from 14.3 to 100.2 minutes, averaging roughly 44 minutes; the spread comes
from the search space, which varies hidden-layer widths and `batch_size`, and from
median pruning, which ends bad trials early. Extrapolating the observed average,
50 trials would cost about 37 hours of search alone, before the five final
per-seed repetitions.

This ADR was written after ADRs 003, 004 and 005 because the MLP decision had been
recorded only in the project's internal notes, which are not published with the
artifact. Since the paper states that the trial budgets in Table 3 are documented
as ADRs, the absence of this record left the MLP row without public support.

## Decision

The production result of `mlp` on Axis 1 was generated with the following command:

```bash
uv run python scripts/run_axis1.py --output-dir data/results/axis1_final --n-trials 15 --models mlp --num-workers 0
```

The `--n-trials 15` parameter defines a reduced and explicitly documented HPO
budget. The `--num-workers 0` is part of the official Windows protocol, adopted in
ADR 003 to avoid `DataLoader` serialization failures.

The study persisted in `data/results/axis1_final/mlp/optuna.db` records 16 trial
entries: 12 completed, 3 pruned by `MedianPruner` (trials 7, 12 and 13) and 1
failed (trial 15). The first 15 produced a value and constitute the declared
effective budget.

## Rationale

### Computational feasibility

The observed cost makes `--n-trials 50` disproportionate for a family serving as
the tabular baseline. The 35 additional trials would cost about 26 hours, which
would compete directly with the budget of the sequential models and the
LightTransformer — more expensive per trial and more central to the paper's
claims. For comparison, XGBoost's 50 trials totalled 0.82 wall-clock hours.

### The MLP is a baseline, not the central model

The paper's contribution lies in the temporal forward-chaining evaluation and in
the comparative behaviour of the families under drift. The MLP enters as the
neural tabular baseline; exhaustively maximizing its HPO does not change the
comparative conclusions, whose Axis 1 ordering is dominated by XGBoost by a wide
margin (macro F1 0.928 against 0.864 for the MLP).

### The marginal gain is small in absolute terms

The best `val_F1` value went from 0.98894 in trial 0 to 0.99097 in trial 14: a
gain of roughly 0.002 across the whole budget. Under the 97.7% attack prevalence
of CICIoT2023, positive-class F1 is weakly discriminative in that range — a point
the paper itself raises when treating static scores as insufficient evidence of
temporal robustness.

### Acknowledged limitation: no convergence was demonstrated

The best value occurred in the **last** trial of the budget (trial 14), and the
running best was still rising at the end: 0.98910 over the first 5 trials, 0.98996
over the first 10 and 0.99097 over all 15. Therefore **it cannot be claimed that
the search had converged**; the budget was closed by cost, not by stabilization of
the optimizer. A larger budget would probably find a slightly higher `val_F1`.
This caveat applies equally to ADRs 003 and 004, whose budgets are smaller still,
and it is the reason the paper describes these budgets as limited rather than as
exhaustive searches.

### Traceability

The two numbers underpinning this decision are verifiable in the published
artifact, in `results/paper_artifacts/axis1/mlp/results.json`:

- `n_trials` = 15
- `best_hpo_val_f1` = 0.990974775493164, alongside the selected `best_params`

Per-trial durations live in the study's `optuna.db`, which is not versioned as it
is local run output (see `.gitignore`).

## Consequences

- The `mlp` results on Axis 1 are valid as a baseline under a limited
  computational budget, not as an estimate of the family's performance ceiling.
- The paper declares `n_trials=15` for the MLP on Axis 1 (Table 3), consistent
  with the published `results.json`.
- Comparisons across families must focus on the final model's test metrics, not on
  equality of HPO effort: the budgets differ by observed per-trial cost, not by a
  distinct quality criterion.
- A larger search may be run as a complementary analysis by resuming the persisted
  study, without replacing this official protocol.

## Alternatives considered

|Alternative|Reason for rejection|
|---|---|
|Keep `--n-trials 50`|About 37 hours of HPO alone for a tabular baseline, competing with the budget of the central models|
|Stop at 5 trials|The running best `val_F1` would be 0.98910 against 0.99097 at 15; the gain was still cheap in that budget range|
|Use `--sample-frac` to cheapen the trials|Would change the effective HPO dataset and weaken the comparison with the other Axis 1 models|
|Level everyone at the smallest budget|Would penalize XGBoost, whose 50 trials cost 0.82 h in total, with no real gain in comparability|
