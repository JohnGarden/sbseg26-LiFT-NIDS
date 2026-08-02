# LiFT-NIDS — SBSeg 2026 paper artifacts

Artifacts for the paper **"Evaluating Lightweight Transformers for Network Intrusion
Detection under Temporal Drift: A Comparative Study with MAWIFlow"** (SBSeg 2026).

**Goal of the artifact:** to let a reviewer verify, in minutes and without a GPU, every
number published in the paper, and re-run the full pipeline if the datasets and the
hardware are available. The repository contains the complete code, the configurations,
the automated tests and the result artifacts (versioned JSON) that back every table in
the paper. The project compares a lightweight Transformer (the FlowTransformer
configuration proposed by Manocchio et al., 2024, reimplemented in PyTorch) against three
baselines (XGBoost, MLP, CNN-BiLSTM) along two axes: static evaluation on CICIoT2023 and
temporal forward-chaining evaluation on the MAWIFlow-subset (2007–2024).

**Paper abstract.** Static train/test splits remain the dominant evaluation protocol for
network intrusion detection systems (NIDS), yet real-world traffic evolves continuously
and deployed models degrade under temporal drift. We present a reproducible comparative
evaluation of four model families—XGBoost, MLP, CNN-BiLSTM, and a parameter-efficient
causal Transformer (LightTransformer, a compact FlowTransformer-derived
configuration)—on two complementary protocols: a static split on CICIoT2023 (sanity
check) and a temporal forward-chaining evaluation on MAWIFlow (2007–2024). On the static
benchmark all models achieve F1⁺>0.98 on the attack class, so near-ceiling static scores
are insufficient evidence of temporal robustness. Under temporal drift (cumulative
training window, R=5 seeds, all models) the LightTransformer obtains nAUT₁=0.57
(95% CI [0.52, 0.62]) versus 0.44 (CNN-BiLSTM), 0.38 (XGBoost), and 0.28 (MLP). Bootstrap
confidence intervals are consistent with this ordering, which holds across all five
seeds; a Friedman test (χ²=15.0, p=0.002) rejects equal rankings globally; one-sided
pairwise Wilcoxon signed-rank tests provide uncorrected directional evidence (p=0.031 for
all six pairs, n=5); and no pairwise comparison survives Bonferroni correction
(α<sub>c</sub>=0.0083). Our results indicate that forward-chaining temporal evaluation should be a
standard component of deployment-oriented NIDS benchmarking.

## readme.md Structure (Estrutura do readme.md)

- [Badges Considered (Selos Considerados)](#badges-considered-selos-considerados)
- [Basic Information (Informações básicas)](#basic-information-informações-básicas)
- [Dependencies (Dependências)](#dependencies-dependências)
- [Security Concerns (Preocupações com segurança)](#security-concerns-preocupações-com-segurança)
- [Installation (Instalação)](#installation-instalação)
  - [Obtaining and Organizing the Datasets](#obtaining-and-organizing-the-datasets)
- [Minimal Test (Teste mínimo)](#minimal-test-teste-mínimo)
- [Experiments (Experimentos)](#experiments-experimentos)
  - [Claim #1 — Axis 1 (CICIoT2023, static)](#claim-1--axis-1-ciciot2023-static)
  - [Claim #2 — Axis 2 (MAWIFlow, temporal)](#claim-2--axis-2-mawiflow-temporal)
  - [Supplementary Axis 2 Plots](#supplementary-axis-2-plots)
  - [Temporal Protocol and Development (Smoke) Runs](#temporal-protocol-and-development-smoke-runs)
- [Determinism and Reproducibility Policy](#determinism-and-reproducibility-policy)
- [LICENSE](#license)

Repository structure:

```text
sbseg26-LiFT-NIDS/
  configs/                  # Dataset, model and experiment YAMLs (see note below)
  src/lift_nids/            # Library: data, models, training, evaluation, interpretability
  scripts/                  # Data acquisition, canonical runners, statistics, figures
    run_axis1.py            #   canonical Axis 1 runner (CICIoT2023)
    run_axis2.py            #   canonical Axis 2 runner (MAWIFlow)
    compute_axis2_stats.py  #   reproduces the statistical tests of §4.2
  tests/                    # pytest suite (runs without the real datasets)
  results/paper_artifacts/  # Versioned JSONs backing every number in the paper
  data/{raw,processed}/     # Empty skeleton (.gitkeep); destination of the downloads
  docs/decisions/           # ADRs for the HPO budgets (003, 004, 005, 009)
  pyproject.toml + uv.lock  # Dependencies with pinned versions
  .python-version           # Pins Python 3.11 for uv
  Makefile                  # Shortcuts (make bootstrap / make test / make validate)
  LICENSE                   # MIT
```

**Note on `configs/`.** The canonical runners `run_axis1.py` and `run_axis2.py` are
**self-contained**: their protocol configuration (splits, windows, seeds, batch size,
early stopping) is embedded in code (`_CICIOT_CFG`, `_MAWIFLOW_CFG`) and adjusted through
**command-line flags**, not through the YAMLs. The `configs/**` files feed exclusively the
auxiliary runner `run_experiment.py` (one model / one horizon, driven by `--config`),
`validate_config.py` and the test suite. Therefore **no YAML needs to be edited** to
reproduce the claims below.

## Badges Considered (Selos Considerados)

The authors of this work consider applying to the following badges: "Artefatos
Disponíveis (SeloD)", "Artefatos Funcionais (SeloF)", "Artefatos Sustentáveis
(SeloS)", and "Experimentos Reprodutíveis (SeloR)".

|Badge|Evidence in this repository|
|---|---|
|**SeloD**|Public repository under the MIT license ([LICENSE](LICENSE)); all code, configuration and result artifacts are versioned.|
|**SeloF**|[Minimal test](#minimal-test-teste-mínimo) runs in ~15 s without datasets or GPU; 594-test full suite; `uv.lock` pins every version.|
|**SeloS**|Modular library in `src/lift_nids/` with type hints and docstrings; determinism and dependency policies in [Determinism and Reproducibility Policy](#determinism-and-reproducibility-policy); 4 HPO-budget ADRs in `docs/decisions/`.|
|**SeloR**|Every claim in [Experiments](#experiments-experimentos) has a verification procedure that takes minutes (real recomputation of the §4.2 statistics from the per-seed data) and a full re-run procedure; claim → artifact map in the table opening the section.|

## Basic Information (Informações básicas)

**Original execution environment of the experiments:**

- Intel i7-11800H CPU, 32 GB RAM, NVIDIA RTX 3070 Laptop GPU (8 GB VRAM), Windows 11.
- Python ≥ 3.11 and < 3.13; PyTorch ≥ 2.1 with CUDA; `uv` as environment manager.

**Shell.** The commands in this README use POSIX syntax (`bash`/`zsh`). On Windows, use
Git Bash or WSL — under native PowerShell, `cat`/`mkdir -p`/`cp -r` must be replaced by
their equivalents, and paths appear in the output with `\` instead of `/`.

**Minimum requirements for evaluation:**

- *Minimal test and claim verification:* any machine with Python 3.11/3.12 and `uv`
  (CPU only; no datasets needed — the test suite uses synthetic data and the paper's
  numbers are verified against `results/paper_artifacts/`).
- *Full re-run of the experiments:* NVIDIA GPU with ≥ 8 GB VRAM, ≥ 32 GB RAM, and Docker
  Desktop (CICFlowMeter, only for the MAWIFlow build pipeline). Estimated total time:
  tens of GPU hours (detailed in the [Experiments](#experiments-experimentos) section).

**Resources per step.** Disk values are sizes measured in the environment above; times
marked with † depend on the network/hardware used and are estimates.

|Step|Disk|RAM (peak)|GPU|Time|
|---|---|---|---|---|
|Repository clone|~10 MB (of which 5 MB are `results/paper_artifacts/`)|—|—|< 1 min †|
|`uv sync --all-groups`|**~3.8 GB** `.venv/` (PyTorch's CUDA wheel is downloaded even without a GPU — see [Dependencies](#dependencies-dependências))|< 2 GB|—|3–10 min †|
|Minimal test (3 steps)|—|< 1 GB|—|~20 s|
|Full suite|—|~2 GB|—|~25 s|
|Verification of Claim #1 (Option A)|—|< 1 GB|—|< 10 s|
|Verification of Claim #2 (Option A)|—|< 1 GB|—|< 30 s|
|Regeneration of the 3 supplementary plots|~5 MB (6 files)|< 1 GB|—|~1 min|
|CICIoT2023 download + preparation|8.4 GB (CSVs) + 0.8 GB (Parquet)|~8 GB during conversion|—|1–3 h †|
|MAWIFlow-subset download + preparation|~1.2 GB (DVC clone 0.45 GB + pcaps 0.66 GB + Parquets 77 MB)|~8 GB|—|3–8 h †|
|Full re-run (Option B)|~15 GB free recommended|~32 GB|8 GB VRAM|~90 h|

## Dependencies (Dependências)

All Python dependencies have pinned versions in `uv.lock` (single source:
`pyproject.toml`). Main components:

|Component|Version|
|---|---|
|Python|≥ 3.11, < 3.13|
|PyTorch|≥ 2.1 (CUDA cu130 on Linux/Windows via an explicit index)|
|XGBoost|≥ 2.0|
|scikit-learn|≥ 1.4|
|Optuna|≥ 3.6|
|Pandas / PyArrow|≥ 2.2 / ≥ 15.0|
|SciPy|≥ 1.13|
|Matplotlib / Seaborn|≥ 3.8 / ≥ 0.13|
|pytest|≥ 8.0|

**Mind the environment size.** `pyproject.toml` declares an explicit index
(`pytorch-cu130`) for PyTorch on Linux and Windows. Practical consequence: even on a
machine **without a GPU**, `uv sync` downloads the CUDA wheel and the resulting `.venv/`
takes up **~3.8 GB**. This does not prevent CPU-only evaluation (PyTorch with CUDA works
on CPU when no device is available), but reserve bandwidth and disk space.

**External resources (needed only for the full re-run):**

- **CICIoT2023** — public CSVs from CIC Research (UNB); requires manual download and
  conversion to Parquet.
- **MAWIFlow / MAWILab** — public captures from the MAWI backbone; download and
  construction of the labeled Parquet are automated by a script (git + DVC + Docker).

Where to download from, what to download and where to place each dataset: section
[Obtaining and Organizing the Datasets](#obtaining-and-organizing-the-datasets), inside
[Installation](#installation-instalação).

## Security Concerns (Preocupações com segurança)

Running the artifacts **poses no risk** to reviewers:

- The code only trains and evaluates classification models over tabular network-flow
  features; it does not capture traffic, does not generate malicious traffic and does not
  execute payloads.
- The datasets are public, widely used by the community and anonymized at the source
  (MAWI applies address anonymization to the published captures).
- No step requires administrator privileges, except the optional use of Docker
  (CICFlowMeter container for building MAWIFlow, needed only for the full re-run of the
  data pipeline).

## Installation (Instalação)

Single prerequisite: `uv` installed (<https://docs.astral.sh/uv/>). Python does not need
to be installed beforehand — the `.python-version` file (versioned at the repository
root) pins the version to 3.11 and `uv` provisions it automatically.

```bash
# 1. Get the repository
git clone https://github.com/JohnGarden/sbseg26-LiFT-NIDS.git
cd sbseg26-LiFT-NIDS

# 2. Create the environment with the exact versions from uv.lock (~3.8 GB — see Dependencies)
uv sync --all-groups        # equivalent shortcut: make bootstrap
```

Expected output: creation of the `.venv/` environment with every dependency from
`uv.lock` resolved without error. Quick check:

```bash
uv run python --version     # Python 3.11.x
uv run python -c "import lift_nids; print('ok')"
```

### Obtaining and Organizing the Datasets

Optional — needed only for **Option B (full pipeline run)** of the claims in
[Experiments](#experiments-experimentos). Anyone who will only run the [minimal test](#minimal-test-teste-mínimo)
or verify the [claims](#experiments-experimentos) through **Option A (inspection)** can skip this
subsection (it uses only the artifacts versioned in `results/paper_artifacts/`).

**CICIoT2023.** Page <https://www.unb.ca/cic/datasets/iotdataset-2023.html>, button
"Download the dataset" (open directory of CIC Research). Download **only the `CSV/`
folder** (not `PCAP/`, `MERGED_CSV/` nor `Supplementary material/` — they are not used by
the pipeline) and place it, preserving its per-attack-type subfolders, in:

```text
data/raw/CICIoT2023/CSV/<AttackType>/<file>.pcap.csv
# e.g.: data/raw/CICIoT2023/CSV/DDoS-ACK_Fragmentation/DDoS-ACK_Fragmentation1.pcap.csv
#       data/raw/CICIoT2023/CSV/Benign_Final/BenignTraffic.pcap.csv
```

This is the layout expected by `scripts/convert_ciciot_csv_to_parquet.py`. With `uv sync`
already done above, convert it to Parquet:

```bash
uv run python scripts/convert_ciciot_csv_to_parquet.py
```

Output: `data/raw/CICIoT2023/ciciot2023.parquet`.

**MAWIFlow / MAWILab.** No manual download: `scripts/download_mawiflow.py` clones
<https://github.com/TheLurps/MAWIFlow>, downloads the raw files via DVC and runs the
whole pipeline (bridge → CICFlowMeter → labeled Parquet) in a single command.
Prerequisites beyond `uv sync`: **git ≥ 2.25**, **DVC ≥ 3.0** (`pip install dvc` — not a
project dependency, it must be available on the `PATH` separately) and **Docker Desktop
running** (the image `thelurps/gintsengelen-cicflowmeter:4dd5319` is pulled
automatically):

```bash
# Downloads + processes the 18 years (2007-2024, the script default), 1 day/quarter
# — the same sampling used in the paper
uv run python scripts/download_mawiflow.py --months 01 04 07 10 --max-days 1
```

**Warning:** keep Docker Desktop running throughout the command above — the script runs
`docker run` once per processed day (up to 74 times, within the paper's scope) to execute
CICFlowMeter, which generates each day's flow CSV (an intermediate step; the final
per-year Parquet is only assembled afterwards, by a DuckDB join that does not use
Docker). Without Docker running, the script fails immediately with `[error] Docker daemon
is not reachable`, before processing any day.

Day substitution (when the preferred day has no capture or no published label in the
original sources — mawi.wide.ad.jp and fukuda-lab.org) is resolved at download time; the
list of days actually used in the paper — and the resulting flow counts and class balance
— is the table in `results/paper_artifacts/data_manifest.json`.

The script writes to three directories, in pipeline order:

```text
data/raw/MAWIFlow-repo/                 # MAWIFlow clone (DVC metadata)
data/raw/MAWIFlow/<YYYYMMDD>/           # pcap + raw annotations for each day
    <YYYYMMDD>1400.pcap.gz
    <YYYYMMDD>_anomalous_suspicious.csv  (+ .xml)
    <YYYYMMDD>_notice.csv                (+ .xml, optional)
data/processed/mawiflow/<year>.parquet  # final output: 2007.parquet … 2024.parquet
```

`data/processed/mawiflow/` is the only path that `configs/datasets/mawiflow.yaml` and
`scripts/run_axis2.py` (flag `--data-path`, same default) read from for training — the two
preceding directories are intermediate and are not used by training. Without access to
git/DVC, `scripts/build_mawiflow_dataset.py` is an alternative route that downloads the
same days directly from mawi.wide.ad.jp / fukuda-lab.org and produces the same final
output.

## Minimal Test (Teste mínimo)

Three checks, none of which **needs the datasets or a GPU**: the canonical runner
executes, the artifact reproduces a number published in the paper, and the test suite
passes. Resources: < 1 GB of RAM, no additional disk. Time: ~20 s for all three.

### Step 1 — The canonical runner executes

`--dry-run` assembles the full Axis 2 execution matrix and returns before any data is
loaded, so it needs no dataset:

```bash
uv run python scripts/run_axis2.py --dry-run
```

Expected output (each line is prefixed with a timestamp and logger name, omitted here;
the header is wrapped for width):

```text
=== Axis 2: MAWIFlow  models=['xgboost', 'mlp', 'cnn_bilstm', 'transformer']
    horizons=[1, 2, 3, 'cumulative']  seeds=[42, 123, 456, 789, 1024]
    combinations=16  is_full=True  run_mode=production  device=cuda ===
=== DRY RUN — no training will be performed ===
Execution matrix (16 combinations):
  xgboost × k=1 → data/results/axis2/xgboost/k_1/results.json
  ...
  transformer × k=cumulative → data/results/axis2/transformer/k_cumulative/results.json
is_full_temporal_matrix: True
is_full_seed_set: True
run_mode: production
```

`device=cuda` on a GPU machine and `device=cpu` otherwise; neither affects a dry run.

### Step 2 — The artifact reproduces a published number

This recomputes the §4.2 statistics from the versioned per-seed artifacts. Writing to a
temporary directory keeps the repository clean:

```bash
mkdir -p /tmp/axis2-mintest && cp -r results/paper_artifacts/axis2/* /tmp/axis2-mintest/
uv run python scripts/compute_axis2_stats.py --output-dir /tmp/axis2-mintest
```

Expected output — the same values published in §4.2 of the paper:

```text
Friedman: chi2=15  p=0.001817
Avg ranks: {'xgboost': 3.0, 'mlp': 4.0, 'cnn_bilstm': 2.0, 'transformer': 1.0}
Primary Wilcoxon (transformer > cnn_bilstm): p=0.03125
Saved: /tmp/axis2-mintest/stats_axis2.json
```

The last line echoes the `--output-dir` given, resolved to an absolute path (native
Windows shells print it with `\`).

### Step 3 — The test suite passes

Config validation plus the light methodological suite (config consistency and the
forward-chaining/nAUT arithmetic on synthetic data), ~15 s:

```bash
uv run python scripts/validate_config.py --config configs/experiments/exp_001_ciciot_static_xgb.yaml
uv run python -m pytest tests/test_config_consistency.py tests/test_temporal_protocol_integration.py -q
```

Expected output:

```text
[ok] Config naming validated: configs/experiments/exp_001_ciciot_static_xgb.yaml
...
SKIPPED [2] tests/test_config_consistency.py:109: Not a sequential experiment.
80 passed, 2 skipped in 14.47s
```

The **2 `skipped` are expected**, not failures: `exp_001` is a tabular experiment, and two
tests in that file apply only to sequential experiments. Any `failed` or `error` indicates
an installation problem.

Full suite (optional, ~25 s, ~2 GB of RAM):

```bash
uv run python -m pytest -q       # equivalent: make test
# Expected output: 594 passed, 2 skipped, 2 warnings in ~25s
```

## Experiments (Experimentos)

Every number published in the paper is backed by versioned JSON artifacts in
`results/paper_artifacts/`. The **complete claim → artifact map** is the table below, with
paths relative to that directory. The provenance of each run — `timestamp`, seeds,
hyperparameters — is embedded in every `results.json`/`seed_*.json`.

|Paper element|Artifact|
|---|---|
|Table 5 (Axis 1, CICIoT2023 static)|`axis1/<model>/results.json` (aggregate over R=5 seeds + per-seed values)|
|Table 6 (Axis 2, nAUT, cumulative window)|`axis2/<model>/k_cumulative/results.json` (means + bootstrap CIs) and `seed_<s>.json`|
|Statistics of §4.2 (Friedman χ²=15.0, p≈0.002; one-sided Wilcoxon LT > CNN-BiLSTM, p=0.03125)|`axis2/stats_axis2.json`|
|Per-seed ordering statements (§4.2)|`axis2/<model>/k_cumulative/seed_<s>.json`|
|Run completeness (16/16 model × horizon combinations, 5 seeds)|`axis2/summary.json`|
|Exact data scope of the MAWIFlow-subset (capture days, flow counts, class balance)|`data_manifest.json`|
|Evidence for fixed windows k ∈ {1, 2, 3}|`axis2/<model>/k_{1,2,3}/seed_<s>.json`|
|Supplementary Axis 2 plots (not in the paper)|`axis2/<model>/k_cumulative/results.json`|

No configuration file needs to be edited: the canonical runners are parameterized by
command-line flags (see the note on `configs/` under
[readme.md Structure](#readmemd-structure-estrutura-do-readmemd)); `--help` lists every flag.

Each claim below offers two routes: **Option A** — inspection of the versioned artifacts
(~1–2 min, CPU, no datasets), sufficient to verify every published number; and
**Option B** — a full pipeline run (GPU, tens of hours, requires the original datasets and
Docker for the MAWIFlow data pipeline), which is **not necessary** for verification. The
steps common to Option B — obtaining and organizing the datasets
([Obtaining and Organizing the Datasets](#obtaining-and-organizing-the-datasets)) and the
per-model HPO budgets (ADRs 003/004/005/009 in `docs/decisions/`) — are detailed in each
block. Option B's reproducibility is **statistical, not bit for bit** (AMP/FP16 and
non-deterministic CUDA kernels — disclosed in §5 of the paper and detailed in
[Determinism and Reproducibility Policy](#determinism-and-reproducibility-policy)).
Known limitations: the "Limitations" section of the paper.

### Claim #1 — Axis 1 (CICIoT2023, static)

**Claim:** in the binary static evaluation on CICIoT2023 (60/20/20 split, R=5 seeds),
every model reaches F1⁺ > 0.98 and XGBoost attains the highest macro F1 (0.928), with the
neural models 6–9 percentage points below (Table 5 of the paper).

#### Option A — Inspecting the Axis 1 metrics

Resources: < 1 GB of RAM, no additional disk, no GPU and no datasets. The R=5 seed
aggregates (means and 95% bootstrap CIs, B=1000) are already versioned in
`results/paper_artifacts/axis1/<model>/results.json`.

```bash
# Extract the Table 5 cells for the four models
uv run python -c "
import json
for m in ['xgboost','mlp','cnn_bilstm','transformer']:
    d = json.load(open(f'results/paper_artifacts/axis1/{m}/results.json'))
    for k in ('f1_macro','f1_1'):
        s = d['metrics'][k]
        print(f\"{m:12s} {k:8s} mean={s['mean']:.4f}  CI95=[{s['ci_lower']:.4f}, {s['ci_upper']:.4f}]\")
"
```

Expected output:

```text
xgboost      f1_macro mean=0.9281  CI95=[0.9280, 0.9283]
xgboost      f1_1     mean=0.9967  CI95=[0.9967, 0.9967]
mlp          f1_macro mean=0.8638  CI95=[0.8631, 0.8644]
mlp          f1_1     mean=0.9913  CI95=[0.9913, 0.9914]
cnn_bilstm   f1_macro mean=0.8404  CI95=[0.8372, 0.8438]
cnn_bilstm   f1_1     mean=0.9892  CI95=[0.9889, 0.9895]
transformer  f1_macro mean=0.8413  CI95=[0.8361, 0.8498]
transformer  f1_1     mean=0.9894  CI95=[0.9888, 0.9903]
```

This confirms the claim: the lowest F1⁺ is 0.9892 > 0.98; XGBoost leads on macro F1
(0.928), with the neural models 6.4–8.8 pp below.

**Structure of `results.json`** (for manual inspection with `cat`/`jq`): the aggregate
over the R=5 seeds is under **`metrics.<metric>`** — an object with `mean`, `std`,
`ci_lower`, `ci_upper` — and the per-seed values under **`per_seed_metrics`** (a list of
5 entries). `best_params`, `best_hpo_val_f1`, `seeds` and `timestamp` document the run
that produced the file (the versioned artifacts also carry a `git_hash` — see
[Provenance of the versioned artifacts](#provenance-of-the-versioned-artifacts)). The
`metrics.f1_macro`/`metrics.f1_1` fields correspond cell by cell to Table 5 of the paper.

#### Option B — Full pipeline run of Axis 1

Re-runs Axis 1 end to end (Optuna HPO → R=5 seeds → bootstrap CI) and rewrites the
`results.json`/`seed_<s>.json` files under `data/results/axis1_final/`.
Prerequisites: CICIoT2023 converted to Parquet — section [Obtaining and Organizing the
Datasets](#obtaining-and-organizing-the-datasets) — and an NVIDIA GPU (~8 GB VRAM).

XGBoost runs the full 50-trial default; the reduced budgets follow their own ADRs
(MLP=15, CNN-BiLSTM=2, Transformer=1 trials — ADRs 009/003/004 in
`docs/decisions/`):

```bash
uv run python scripts/run_axis1.py --output-dir data/results/axis1_final --models xgboost    --n-trials 50 --num-workers 0
uv run python scripts/run_axis1.py --output-dir data/results/axis1_final --models mlp         --n-trials 15 --num-workers 0
uv run python scripts/run_axis1.py --output-dir data/results/axis1_final --models cnn_bilstm  --n-trials 2  --num-workers 0
uv run python scripts/run_axis1.py --output-dir data/results/axis1_final --models transformer --n-trials 1  --num-workers 0
```

Resources and time on the hardware from [Basic Information](#basic-information-informações-básicas):
~9 GB of disk (CICIoT2023 CSVs + Parquet), ~32 GB of RAM, 8 GB of VRAM;
XGB ≈1.0 h; MLP ≈2.5 h; CNN-BiLSTM ≈14.7 h; LT ≈5.1 h (≈23 h in total).
At the end, the `metrics.f1_macro`/`metrics.f1_1` fields reproduce Table 5 within the
across-seed distribution (statistical reproducibility — see the note under
[Experiments](#experiments-experimentos)).

### Claim #2 — Axis 2 (MAWIFlow, temporal)

**Claim:** under the annual forward-chaining protocol on the MAWIFlow-subset (cumulative
window, R=5 seeds), the lightweight Transformer attains the highest nAUT at all three
horizons (nAUT₁, nAUT₃, nAUT₅ — Table 6 of the paper). On **nAUT₁**, the Friedman test
indicates a global difference among the four models (χ²=15.0, p≈0.002) and the
pre-specified one-sided Wilcoxon LT > CNN-BiLSTM yields p=0.03125 (§4.2 of the paper).

**Scope caveat (identical to the paper's):** with n=5 seeds, p=0.03125 is the smallest
p-value attainable by the one-sided Wilcoxon — all six pairs reach exactly that value, and
**no pairwise comparison survives Bonferroni correction** (α<sub>c</sub>=0.0083;
`p_bonferroni_6=0.1875` in `stats_axis2.json`). Only the LT > CNN-BiLSTM comparison is
confirmatory (pre-specified); the others are exploratory, with the direction chosen
*post hoc*. The main supporting evidence is the consistent ordering of the means, of the
bootstrap CIs and of the 5/5 seeds, not pairwise significance.

#### Option A — Inspecting the Axis 2 metrics

Resources: < 1 GB of RAM, no additional disk, no GPU and no datasets. The command below
**recomputes** Friedman/Wilcoxon/Bonferroni from the versioned per-seed artifacts and
shows that the values are identical to the published ones:

```bash
uv run python scripts/compute_axis2_stats.py --output-dir results/paper_artifacts/axis2
```

Expected output (values identical to those published in the paper):

```text
Friedman: chi2=15  p=0.001817
Avg ranks: {'xgboost': 3.0, 'mlp': 4.0, 'cnn_bilstm': 2.0, 'transformer': 1.0}
Primary Wilcoxon (transformer > cnn_bilstm): p=0.03125
Saved: results/paper_artifacts/axis2/stats_axis2.json
```

**Warning:** this command **overwrites** the versioned file
`results/paper_artifacts/axis2/stats_axis2.json`. **Every statistical value is rewritten
identically** — that is precisely the reproducibility evidence. Only provenance metadata
differs, in three fields: `timestamp` is refreshed, `source` records the `--output-dir`
you passed, and `git_hash` disappears — the script writes only the first two, so
regenerating drops the hash recorded by the original production run. To check and
restore:

```bash
git diff results/paper_artifacts/axis2/stats_axis2.json   # timestamp, source, git_hash
git checkout -- results/paper_artifacts/axis2/stats_axis2.json   # restores the original
```

To avoid writing to the versioned file, point the output to a temporary directory:

```bash
mkdir -p /tmp/axis2 && cp -r results/paper_artifacts/axis2/* /tmp/axis2/
uv run python scripts/compute_axis2_stats.py --output-dir /tmp/axis2
```

The nAUT values of Table 6 (means and CIs) are in
`results/paper_artifacts/axis2/<model>/k_cumulative/results.json` (field
`metrics.nAUT_{1,3,5}`, with `mean`/`ci_lower`/`ci_upper`), and the per-seed values (which
back the 5/5-seed ordering statements of §4.2) in
`results/paper_artifacts/axis2/<model>/k_cumulative/seed_<s>.json`.
The completeness of the matrix (16/16 model×horizon combinations) is recorded in
`results/paper_artifacts/axis2/summary.json`. The fixed-window results for
k ∈ {1, 2, 3} — reported descriptively in the paper (the multi-window table and the
training-regime sensitivity paragraph of §4.2), without pre-specified inferential tests —
are in `results/paper_artifacts/axis2/<model>/k_{1,2,3}/seed_<s>.json`.

#### Option B — Full pipeline run of Axis 2

Re-runs Axis 2 end to end (HPO once on the 2007 anchor year → forward-chaining protocol
over all horizons k ∈ {1, 2, 3, cumulative} → R=5 seeds → bootstrap CI) and rewrites the
`results.json`/`seed_<s>.json` files under `data/results/axis2_final/`.
Prerequisites: MAWIFlow Parquets in `data/processed/mawiflow/` — section [Obtaining and
Organizing the Datasets](#obtaining-and-organizing-the-datasets) — and an NVIDIA GPU
(~8 GB VRAM).

Each model is run with the HPO budget from ADR-005 (XGBoost=50;
MLP/CNN-BiLSTM/Transformer=5 trials — `docs/decisions/005-axis2-hpo-budget.md`):

```bash
uv run python scripts/run_axis2.py --output-dir data/results/axis2_final --models xgboost    --n-trials 50 --num-workers 0 --resume
uv run python scripts/run_axis2.py --output-dir data/results/axis2_final --models mlp         --n-trials 5  --num-workers 0 --resume
uv run python scripts/run_axis2.py --output-dir data/results/axis2_final --models cnn_bilstm  --n-trials 5  --num-workers 0 --resume
uv run python scripts/run_axis2.py --output-dir data/results/axis2_final --models transformer --n-trials 5  --num-workers 0 --resume
```

At the end, aggregate the temporal matrix and recompute the statistical tests of §4.2:

```bash
uv run python scripts/aggregate_axis2_summaries.py --output-dir data/results/axis2_final
uv run python scripts/compute_axis2_stats.py --output-dir data/results/axis2_final
```

Resources and time on the hardware from [Basic Information](#basic-information-informações-básicas):
~1.2 GB of disk for the MAWIFlow-subset (plus ~2 GB of outputs in
`data/results/axis2_final/`), ~32 GB of RAM, 8 GB of VRAM; XGB ≈4.6 h; MLP ≈15.5 h;
CNN-BiLSTM ≈16.6 h; LT ≈28.6 h (R=5, all k — ≈65 h in total). At the end, the nAUT values
reproduce Table 6 and the tests reproduce χ²=15.0, p≈0.002 and Wilcoxon p=0.03125 within
the across-seed distribution (statistical reproducibility — see the note under
[Experiments](#experiments-experimentos)).

### Supplementary Axis 2 Plots

Three scripts generate plots from the versioned artifacts: the median F1⁺(Δt) trajectory,
the nAUT comparison and the critical-difference diagram. **None of them appears in the
final paper** — the only published figure is the Axis 2 protocol diagram
(`docs/paper/figures/pipeline-axis2.png`). All three visualize the same per-seed data that
backs Table 6 and the tests of §4.2, and serve as a quick check that the versioned JSONs
are consumable end to end. Resources: ~1 minute, CPU, < 1 GB of RAM, ~5 MB of disk for the
6 outputs.

```bash
# The scripts read from data/results/axis2_final/ — point it at the versioned artifacts
mkdir -p data/results
cp -r results/paper_artifacts/axis2 data/results/axis2_final

uv run python scripts/plot_trajectory_axis2.py
uv run python scripts/plot_naut_comparison.py
uv run python scripts/plot_cd_diagram_axis2.py

# Outputs: docs/paper/figures/{trajectory-axis2,naut-comparison,cd-diagram-axis2}.{pdf,png}
```

**Note expected in the output of `plot_trajectory_axis2.py`:**

```text
  transformer: 4 seeds with trajectory data
  cnn_bilstm: 4 seeds with trajectory data
  xgboost: 5 seeds with trajectory data
  mlp: 2 seeds with trajectory data
```

This is **not an error**. The per-Δt curve requires the `median_trajectory` field (F1⁺ per
window), which is not present in every entry of `per_seed_results`: seeds reused by
`--resume` mode from `seed_<s>.json` files that did not contain that field enter the
aggregation with it empty (`_loaded.setdefault("median_trajectory", {})` in
`scripts/run_axis2.py`) and are skipped by the plot. The **scalar metrics
(nAUT₁/₃/₅, F1, precision, recall, AUC) are complete for the 5 seeds of all 4 models** —
they are what back Table 6 and the statistical tests of §4.2, which do not depend on the
trajectory. The trajectory plot illustrates the shape of the degradation and uses the
seeds available; regenerating it through Option B (full re-run, without `--resume`)
produces all 5 seeds for every model.

### Temporal Protocol and Development (Smoke) Runs

Details of the Axis 2 protocol implemented by the canonical runner `scripts/run_axis2.py`:

- Internal split of each training window: temporal 60/20/20 (`fit`, `val`,
  `internal_test`), ordered by `year` + `timestamp`, without shuffling.
- `summary.json` (produced by `scripts/aggregate_axis2_summaries.py`) records
  `is_full_temporal_matrix: true` when all 16 results
  (4 models × 4 horizons) were found without error.
- The configs `configs/experiments/exp_003`/`exp_004` are auxiliary smoke/debug tools
  (one model, one horizon at a time) and do not cover the full temporal matrix.

Reduced smoke run to validate the pipeline end to end at low cost
(requires the MAWIFlow Parquets; never use as a final result):

```bash
uv run python scripts/run_axis2.py \
    --sample-frac 0.05 --n-trials 10 \
    --models transformer xgboost --horizons 1 cumulative \
    --num-workers 0 --output-dir data/results/smoke_axis2

# Inspect the run matrix without training
uv run python scripts/run_axis2.py --dry-run
```

## Determinism and Reproducibility Policy

### What is guaranteed, and what is not

The results are reproducible **statistically** — means and bootstrap confidence intervals
over the distribution of R=5 seeds {42, 123, 456, 789, 1024} — but **not bit for bit**
across distinct runs or hardware. The causes are disclosed in §5 of the paper:

- **AMP/FP16 enabled by default** in GPU training (`src/lift_nids/training/trainer.py`,
  `use_amp=True`);
- **non-deterministic CUDA kernels**: `torch.use_deterministic_algorithms` is not enforced
  and `CUBLAS_WORKSPACE_CONFIG` is not set;
- the `deterministic` parameter of `set_global_seed` is currently a **no-op**
  (`src/lift_nids/utils/reproducibility.py` does `del deterministic`); the function fixes
  the seeds of `random`, NumPy and PyTorch and sets `cudnn.deterministic=True`, which
  covers neither AMP nor every kernel.

What **is** guaranteed: the same seeds produce the same data partition and the same
hyperparameters, and the metrics reproduce the paper's tables within the across-seed
distribution. Table 6 reproduces exactly from the versioned per-seed artifacts, without a
GPU — that is the **Option A** route of each claim. Enforcing strict determinism
(disabling AMP, enabling `use_deterministic_algorithms`) would require re-running the
experiments and remains future work.

### Experiment naming convention

The files in `configs/experiments/` follow the mandatory format below, validated by
`scripts/validate_config.py`:

```text
exp_<id>_<dataset>_<split_protocol>_<model>
```

Examples: `exp_001_ciciot_static_xgb`, `exp_003_mawiflow_temporal_cnn_bilstm`.

Each run records a unique `run_id` and `registered_at`, besides `timestamp`, seeds and
hyperparameters embedded in the corresponding `results.json`/`seed_<s>.json`.

### Provenance of the versioned artifacts

The files under `results/paper_artifacts/` carry a `git_hash` field (for example
`"git_hash": "ac59b35"`). Those hashes reference the **development repository** in which
the experiments were executed, between 2026-05-03 and 2026-05-10 — not this artifact
repository. They therefore **do not resolve** against a clone of `sbseg26-LiFT-NIDS`
(`git show ac59b35` fails). They are kept verbatim as the audit trail of the original
production runs; no number in the paper depends on resolving them, and every claim is
verifiable through Option A from the artifacts alone.

Re-runs do **not** reproduce that field: `provenance()`
(`src/lift_nids/experiments/results.py`) stamps only `timestamp`. Consequently the
`results.json`/`seed_<s>.json` written by Option B — and the `stats_axis2.json`
regenerated by `scripts/compute_axis2_stats.py`, as noted under
[Option A — Inspecting the Axis 2 metrics](#option-a--inspecting-the-axis-2-metrics) —
record `timestamp`, seeds and hyperparameters, but carry no `git_hash`. This is the
expected behaviour, not a sign of a failed run.

### Dependency policy

The environment is pinned by `pyproject.toml` + `uv.lock`, both versioned. To change
dependencies, always use `uv` — never edit `uv.lock` by hand, and commit both files
together:

```bash
uv add <package>
uv add --group dev <package>
uv add --group test <package>
uv lock
```

### Artifacts not tracked in Git

Raw and processed datasets, checkpoints, bulky logs, predictions and locally generated
tables are **not** versioned: see `.gitignore`. The output directories exist in the
repository only as a skeleton (`.gitkeep` files) and are filled in by execution.

## LICENSE

This project is distributed under the **MIT** license — see the [`LICENSE`](LICENSE) file.
