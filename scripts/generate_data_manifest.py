#!/usr/bin/env python3
"""Generate the MAWIFlow-subset data manifest (README "Checks obrigatorios" item).

Records the exact realized data scope behind the paper's "18 annual snapshots":
which capture days were sampled per year, per-year flow counts and class balance,
and the audit verdict — so the dataset is reconstructible from the repository.

Sources (all local):
* ``data/raw/mawiflow/<YYYYMMDD>/`` directory names -> realized capture days;
* ``results/registry/mawiflow_audit_summary.json``  -> per-year counts/balance
  (audit of the processed Parquets actually used by the experiments);
* ``data/processed/mawiflow/<year>.parquet`` metadata -> row-count cross-check.

Usage:
    uv run python scripts/generate_data_manifest.py
    uv run python scripts/generate_data_manifest.py --output-dir results/paper_artifacts
"""

from __future__ import annotations

import argparse
import datetime
import json
from collections import defaultdict
from pathlib import Path

PREFERRED_MONTHS = ("01", "04", "07", "10")

POLICY = (
    "Systematic temporal sampling of the TheLurps/MAWIFlow manifest: one capture day "
    "per quarter per year (preferred months 01/04/07/10; nearest available day when a "
    "preferred one is missing upstream). Results are evaluations on MAWIFlow-subset, "
    "not a full MAWIFlow/IM28 reproduction (see the README 'MAWIFlow' section)."
)

DOWNLOAD_COMMAND = (
    "uv run python scripts/download_mawiflow.py "
    "--years 2007 ... 2024 --months 01 04 07 10 --max-days 1"
)


def collect_raw_days(raw_dir: Path) -> dict[int, list[str]]:
    """Group ``YYYYMMDD`` capture-day directory names by year."""
    days: dict[int, list[str]] = defaultdict(list)
    for entry in sorted(raw_dir.iterdir()):
        name = entry.name
        if entry.is_dir() and len(name) == 8 and name.isdigit():
            days[int(name[:4])].append(name)
    return dict(days)


def parquet_row_count(path: Path) -> int | None:
    """Row count from Parquet metadata only (no data load)."""
    try:
        import pyarrow.parquet as pq

        return pq.ParquetFile(path).metadata.num_rows
    except Exception:  # noqa: BLE001 — cross-check is best-effort
        return None


def build_manifest(raw_dir: Path, audit_path: Path, processed_dir: Path) -> dict:
    """Assemble the manifest dict from the three local evidence sources."""
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    raw_days = collect_raw_days(raw_dir)

    years = []
    mismatches = []
    for stats in audit["key_stats_by_year"]:
        year = stats["year"]
        days = raw_days.get(year, [])
        months = sorted({d[4:6] for d in days})
        quarters = sorted({(int(d[4:6]) - 1) // 3 + 1 for d in days})
        pq_rows = parquet_row_count(processed_dir / f"{year}.parquet")
        if pq_rows is not None and pq_rows != stats["n_rows"]:
            mismatches.append({"year": year, "audit": stats["n_rows"], "parquet": pq_rows})
        years.append(
            {
                "year": year,
                "capture_days": days,
                "n_capture_days": len(days),
                "months_covered": months,
                "quarters_covered": quarters,
                "deviates_from_policy": len(days) != 4 or quarters != [1, 2, 3, 4],
                "n_flows": stats["n_rows"],
                "n_benign": stats["n_benign"],
                "n_anomalous": stats["n_anomalous"],
                "attack_prevalence": round(stats["attack_prevalence"], 4),
                "timestamp_min": stats["timestamp_min"],
                "timestamp_max": stats["timestamp_max"],
                "parquet_rows_crosscheck": pq_rows,
            }
        )

    return {
        "dataset": "MAWIFlow-subset",
        "sampling_policy": POLICY,
        "download_command": DOWNLOAD_COMMAND,
        "years_covered": [y["year"] for y in years],
        "n_years": len(years),
        "n_capture_days_total": sum(y["n_capture_days"] for y in years),
        "n_flows_total": sum(y["n_flows"] for y in years),
        "n_numeric_features": len(audit["columns"]["numeric_features"]),
        "per_year": years,
        "row_count_mismatches": mismatches,
        "audit_provenance": {
            "source": str(audit_path).replace("\\", "/"),
            "audit_started_at_utc": audit["audit_started_at_utc"],
            "audit_finished_at_utc": audit["audit_finished_at_utc"],
            "overall_status": audit["overall_status"],
            "approved_for_final_experiments": audit["approved_for_final_experiments"],
        },
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "generator": Path(__file__).name,
    }


def render_markdown(m: dict) -> str:
    """Human-readable companion for the JSON manifest."""
    lines = [
        "<!-- markdownlint-disable -->",
        "# MAWIFlow-subset — data manifest",
        "",
        f"**Generated:** {m['generated_at']} by `scripts/{m['generator']}`. "
        f"**Audit verdict:** {m['audit_provenance']['overall_status']} "
        f"(approved_for_final_experiments={m['audit_provenance']['approved_for_final_experiments']}, "
        f"audited {m['audit_provenance']['audit_started_at_utc'][:10]}).",
        "",
        f"**Sampling policy.** {m['sampling_policy']}",
        "",
        f"**Scope:** {m['n_years']} years ({m['years_covered'][0]}–{m['years_covered'][-1]}), "
        f"{m['n_capture_days_total']} capture days, {m['n_flows_total']:,} flows, "
        f"{m['n_numeric_features']} numeric features.",
        "",
        "| Year | Capture days | Flows | Benign | Anomalous | Attack prev. |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for y in m["per_year"]:
        days = ", ".join(y["capture_days"]) or "(raw dirs absent)"
        lines.append(
            f"| {y['year']} | {days} | {y['n_flows']:,} | {y['n_benign']:,} "
            f"| {y['n_anomalous']:,} | {y['attack_prevalence']:.1%} |"
        )
    deviations = [y for y in m["per_year"] if y["deviates_from_policy"]]
    if deviations:
        lines += ["", "**Deviations from the 1-day-per-quarter policy (realized scope):**", ""]
        for y in deviations:
            q = "".join(f"Q{i}" for i in y["quarters_covered"])
            lines.append(
                f"- **{y['year']}**: {y['n_capture_days']} days, quarters covered: {q}"
            )
    lines += [
        "",
        "Row counts cross-checked against `data/processed/mawiflow/<year>.parquet` "
        + (
            "metadata: **all match**."
            if not m["row_count_mismatches"]
            else f"metadata: **MISMATCHES** {m['row_count_mismatches']}"
        ),
        "",
        f"Reconstruction: `{m['download_command']}` — note that day substitution is "
        "resolved against upstream availability at download time; the authoritative "
        "day list is the table above.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/mawiflow"))
    parser.add_argument(
        "--audit-summary",
        type=Path,
        default=Path("results/registry/mawiflow_audit_summary.json"),
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=Path("data/processed/mawiflow")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/paper_artifacts")
    )
    args = parser.parse_args()

    manifest = build_manifest(args.raw_dir, args.audit_summary, args.processed_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "data_manifest.json"
    md_path = args.output_dir / "data_manifest.md"
    json_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(manifest), encoding="utf-8")

    print(f"Years: {manifest['n_years']}  days: {manifest['n_capture_days_total']}  "
          f"flows: {manifest['n_flows_total']:,}")
    print(f"Row-count mismatches: {manifest['row_count_mismatches'] or 'none'}")
    print(f"Saved: {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
