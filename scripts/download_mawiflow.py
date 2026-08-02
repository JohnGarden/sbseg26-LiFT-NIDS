#!/usr/bin/env python3
"""Download the MAWI dataset via TheLurps/MAWIFlow DVC, without the private pcap-filter.

Pipeline:
  1. Clone (or update) the TheLurps/MAWIFlow repository
  2. `dvc update` per year/month -> fetches .dump.gz + annotations from MAWI/MAWILab
  3. Bridge: rename .dump.gz -> .pcap.gz and reorganize the directory tree into the
     layout expected by process_mawiflow_dataset.py
  4. Call process_mawiflow_dataset.py (CICFlowMeter via Docker -> labeled Parquets)

Why this bypasses pcap-filter:
  MAWIFlow's `dvc repro` fails because `pcap-filter` lives in a private registry that
  is not reachable externally. Here `dvc repro` is dropped entirely: only `dvc update`
  is used (which requires nothing beyond `dvc` itself) to fetch the raw files, and
  processing is delegated to our own process_mawiflow_dataset.py.

MAWIFlow DVC layout (after dvc update):
  <repo>/data/raw/v1.1/year=YYYY/month=MM/day=DD/
      YYYYMMDD1400.dump.gz                 # gzip-compressed pcap
      YYYYMMDD_anomalous_suspicious.csv
      YYYYMMDD_anomalous_suspicious.xml
      YYYYMMDD_notice.csv                  # optional
      YYYYMMDD_notice.xml                  # optional

Output layout (expected by process_mawiflow_dataset.py):
  <raw-dir>/YYYYMMDD/
      YYYYMMDD1400.pcap.gz                 # renamed from .dump.gz (same content)
      YYYYMMDD_anomalous_suspicious.csv
      YYYYMMDD_anomalous_suspicious.xml
      YYYYMMDD_notice.csv
      YYYYMMDD_notice.xml

Usage:
  # Full pipeline (download + bridge + CICFlowMeter -> Parquets)
  uv run python scripts/download_mawiflow.py --years 2020 2021 2022

  # Download + bridge only (no CICFlowMeter)
  uv run python scripts/download_mawiflow.py --raw-only

  # Specific years and months
  uv run python scripts/download_mawiflow.py --years 2017 2018 --months 01 06

  # Full 2007-2024 dataset (every month — large)
  uv run python scripts/download_mawiflow.py --months 01 02 03 04 05 06 07 08 09 10 11 12

Requirements:
  - git >= 2.25 (supports --filter=blob:none)
  - dvc >= 3.0  (pip install dvc)
  - docker      (CICFlowMeter — only needed without --raw-only)
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MAWIFLOW_REPO_URL = "https://github.com/TheLurps/MAWIFlow.git"
YEARS_AVAILABLE   = list(range(2007, 2025))
DEFAULT_MONTHS    = ["01", "06"]


# ---------------------------------------------------------------------------
# Pré-requisitos
# ---------------------------------------------------------------------------

def check_prerequisites() -> None:
    missing = []
    hints = {
        "git": "Install git >= 2.25  https://git-scm.com/",
        "dvc": "pip install dvc",
    }
    for cmd, hint in hints.items():
        if shutil.which(cmd) is None:
            logger.error(f"'{cmd}' not found on PATH — {hint}")
            missing.append(cmd)
    if missing:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Repositório
# ---------------------------------------------------------------------------

def clone_or_update_repo(repo_dir: Path) -> None:
    """Clona MAWIFlow com blobs lazy, ou faz pull se já existir.

    Sparse checkout é evitado: o repo tem ~2 MiB de código e metadados DVC;
    --filter=blob:none já adia downloads de objetos grandes. Sparse checkout
    deixaria o .dvc/ fora da árvore de trabalho, quebrando o dvc.
    """
    if (repo_dir / ".git").exists():
        logger.info(f"Repo already exists at {repo_dir} — updating …")
        _run(["git", "fetch", "--prune"],                cwd=repo_dir)
        _run(["git", "pull", "--rebase", "--autostash"],  cwd=repo_dir)
        return

    logger.info(f"Cloning {MAWIFLOW_REPO_URL} -> {repo_dir}")
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--filter=blob:none", MAWIFLOW_REPO_URL, str(repo_dir)])
    logger.info("Clone complete.")


# ---------------------------------------------------------------------------
# DVC update (único passo que usamos do pipeline deles)
# ---------------------------------------------------------------------------

def dvc_update(repo_dir: Path, year: int, months: list[str], max_days: int | None = None) -> bool:
    """Baixa arquivos brutos via dvc update. Não requer pcap-filter.

    Quando max_days é definido, enumera os subdiretórios day=DD no clone
    (os .dvc de metadados já estão presentes) e faz dvc update dia a dia,
    parando ao atingir o limite. Isso evita baixar meses inteiros na validação.
    """
    logger.info(f"[{year}] dvc update …")
    any_ok = False

    for month in months:
        if max_days is None:
            # Baixa o mês inteiro de uma vez.
            dvc_path = f"data/raw/v1.1/year={year}/month={month}"
            result = _run(["dvc", "update", "--recursive", dvc_path], cwd=repo_dir, check=False)
            if result.returncode == 0:
                any_ok = True
            else:
                logger.warning(f"  dvc update failed for year={year}/month={month} — skipping")
        else:
            # Baixa apenas os primeiros max_days dias do mês.
            month_meta = repo_dir / "data" / "raw" / "v1.1" / f"year={year}" / f"month={month}"
            if not month_meta.exists():
                logger.warning(f"  DVC metadata not found: {month_meta}")
                continue
            day_dirs = sorted(
                d for d in month_meta.iterdir()
                if d.is_dir() and d.name.startswith("day=")
            )[:max_days]
            for day_dir in day_dirs:
                dvc_path = f"data/raw/v1.1/year={year}/month={month}/{day_dir.name}"
                result = _run(["dvc", "update", "--recursive", dvc_path], cwd=repo_dir, check=False)
                if result.returncode == 0:
                    any_ok = True
                else:
                    logger.warning(f"  dvc update failed for {dvc_path} — skipping")

    return any_ok


# ---------------------------------------------------------------------------
# Bridge: estrutura DVC → estrutura process_mawiflow_dataset.py
# ---------------------------------------------------------------------------

def bridge_dvc_to_raw(
    repo_dir: Path,
    year: int,
    months: list[str],
    raw_dir: Path,
) -> list[Path]:
    """Reorganiza os arquivos baixados pelo DVC para o layout esperado por
    process_mawiflow_dataset.py.

    MAWIFlow usa .dump.gz (pcap comprimido com gzip); process_mawiflow_dataset.py
    espera .pcap.gz — mesmo conteúdo binário, apenas extensão diferente.
    Usa hardlink quando possível (mesmo filesystem), evitando duplicação de disco.
    """
    dvc_base = repo_dir / "data" / "raw" / "v1.1"
    staged: list[Path] = []

    for month in months:
        month_dir = dvc_base / f"year={year}" / f"month={month}"
        if not month_dir.exists():
            logger.warning(f"  DVC path not found: {month_dir}")
            continue

        for day_entry in sorted(month_dir.iterdir()):
            if not day_entry.is_dir() or not day_entry.name.startswith("day="):
                continue

            day_val = day_entry.name.split("=", 1)[1].zfill(2)
            date_str = f"{year}{month}{day_val}"

            dump_gz = day_entry / f"{date_str}1400.dump.gz"
            if not dump_gz.exists():
                logger.debug(f"  {date_str}: dump.gz missing — skipping")
                continue

            ann_csv = day_entry / f"{date_str}_anomalous_suspicious.csv"
            ann_xml = day_entry / f"{date_str}_anomalous_suspicious.xml"
            if not ann_csv.exists() or not ann_xml.exists():
                logger.warning(f"  {date_str}: required annotations missing — skipping")
                continue

            target_dir = raw_dir / date_str
            target_dir.mkdir(parents=True, exist_ok=True)

            # .dump.gz → .pcap.gz (mesmo conteúdo binário)
            target_pcap_gz = target_dir / f"{date_str}1400.pcap.gz"
            if not target_pcap_gz.exists():
                _link_or_copy(dump_gz, target_pcap_gz)
                logger.info(f"  {date_str}: {dump_gz.name} → {target_pcap_gz.name}")

            # Copiar anotações
            for fname in [
                f"{date_str}_anomalous_suspicious.csv",
                f"{date_str}_anomalous_suspicious.xml",
                f"{date_str}_notice.csv",
                f"{date_str}_notice.xml",
            ]:
                src = day_entry / fname
                dst = target_dir / fname
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)

            staged.append(target_dir)
            logger.info(f"  {date_str}: ready at {target_dir}")

    return staged


def _link_or_copy(src: Path, dst: Path) -> None:
    """Cria hardlink de src para dst; usa cópia como fallback (filesystems diferentes)."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Processamento (delega ao nosso pipeline)
# ---------------------------------------------------------------------------

def run_process_script(
    raw_dir: Path,
    output_dir: Path,
    years: list[int],
    memory_gb: int,
    max_heap_gb: int | None,
) -> int:
    """Chama process_mawiflow_dataset.py para executar CICFlowMeter e gerar Parquets."""
    process_script = Path(__file__).parent / "process_mawiflow_dataset.py"
    if not process_script.exists():
        logger.error(f"Script not found: {process_script}")
        return 1

    cmd = [
        sys.executable, str(process_script),
        "--input-dir",  str(raw_dir),
        "--output-dir", str(output_dir),
        "--memory-gb",  str(memory_gb),
        "--years",      *[str(y) for y in years],
    ]
    if max_heap_gb is not None:
        cmd += ["--max-heap-gb", str(max_heap_gb)]

    logger.info("Starting process_mawiflow_dataset.py …")
    result = subprocess.run(cmd)
    return result.returncode


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------

def _run(
    cmd: list[str],
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    logger.debug("  $ %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd)
    if check and result.returncode != 0:
        logger.error("Command failed (rc=%d): %s", result.returncode, " ".join(cmd))
        sys.exit(result.returncode)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    def _abs(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else (repo_root / q).resolve()

    parser = argparse.ArgumentParser(
        description=(
            "Download the MAWI dataset via TheLurps/MAWIFlow DVC, "
            "bypassing the private pcap-filter dependency."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=YEARS_AVAILABLE, metavar="YYYY",
        help=f"Years to process (default: {YEARS_AVAILABLE[0]}–{YEARS_AVAILABLE[-1]})",
    )
    parser.add_argument(
        "--months", nargs="+", default=DEFAULT_MONTHS, metavar="MM",
        help=f"Months per year, zero-padded (default: {' '.join(DEFAULT_MONTHS)})",
    )
    parser.add_argument(
        "--repo-dir", default="data/raw/MAWIFlow-repo", metavar="PATH",
        help="Clone destination (default: data/raw/MAWIFlow-repo)",
    )
    parser.add_argument(
        "--raw-dir", default="data/raw/MAWIFlow", metavar="PATH",
        help="Staging directory for process_mawiflow_dataset.py (default: data/raw/MAWIFlow)",
    )
    parser.add_argument(
        "--output-dir", default="data/processed/mawiflow", metavar="PATH",
        help="Destination of the final Parquets (default: data/processed/mawiflow)",
    )
    parser.add_argument(
        "--max-days", type=int, default=None, metavar="N",
        help=(
            "Maximum days to download per (year, month). "
            "Use 1 or 2 for a quick pipeline validation (default: all)."
        ),
    )
    parser.add_argument(
        "--raw-only", action="store_true",
        help="Stop after dvc update + bridge; do not run CICFlowMeter.",
    )
    parser.add_argument(
        "--memory-gb", type=int, default=8, metavar="GB",
        help="Initial JVM heap for CICFlowMeter (default: 8)",
    )
    parser.add_argument(
        "--max-heap-gb", type=int, default=None, metavar="GB",
        help="Heap ceiling for CICFlowMeter retries",
    )
    args = parser.parse_args()

    check_prerequisites()

    repo_dir   = _abs(args.repo_dir)
    raw_dir    = _abs(args.raw_dir)
    output_dir = _abs(args.output_dir)
    months     = [m.zfill(2) for m in args.months]
    years      = sorted(set(args.years))

    logger.info("repo-dir   : %s", repo_dir)
    logger.info("raw-dir    : %s", raw_dir)
    logger.info("output-dir : %s", output_dir)
    logger.info("years      : %s", years)
    logger.info("months     : %s", months)

    # 1. Clonar / atualizar o repositório MAWIFlow.
    clone_or_update_repo(repo_dir)

    failed_dl     : list[int] = []
    failed_bridge : list[int] = []
    staged_years  : list[int] = []

    for year in years:
        # 2. Baixar arquivos brutos via dvc update (não precisa de pcap-filter).
        if not dvc_update(repo_dir, year, months, args.max_days):
            failed_dl.append(year)
            continue

        # 3. Bridge: .dump.gz → .pcap.gz + reorganizar estrutura de diretórios.
        staged_days = bridge_dvc_to_raw(repo_dir, year, months, raw_dir)
        if not staged_days:
            logger.warning(f"[{year}] No valid day after bridge.")
            failed_bridge.append(year)
            continue

        staged_years.append(year)
        logger.info(f"[{year}] {len(staged_days)} day(s) ready at {raw_dir}")

    # Sumário do download + bridge.
    done_dl = [y for y in years if y not in failed_dl + failed_bridge]
    print(f"\n[bridge] {len(done_dl)}/{len(years)} years ready at {raw_dir}")
    if failed_dl:
        print(f"[warn] dvc update failed : {failed_dl}")
    if failed_bridge:
        print(f"[warn] bridge failed     : {failed_bridge}")

    if args.raw_only or not staged_years:
        if staged_years:
            print(f"\nFiles in {raw_dir} ready for manual processing:")
            print(f"  uv run python scripts/process_mawiflow_dataset.py \\")
            print(f"      --input-dir {raw_dir} \\")
            print(f"      --output-dir {output_dir}")
        return

    # 4. Processar via process_mawiflow_dataset.py (CICFlowMeter → Parquets).
    rc = run_process_script(raw_dir, output_dir, staged_years, args.memory_gb, args.max_heap_gb)
    if rc != 0:
        logger.error("process_mawiflow_dataset.py exited with rc=%d", rc)
        sys.exit(rc)


if __name__ == "__main__":
    main()
