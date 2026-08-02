#!/usr/bin/env python3
"""Build labeled MAWIFlow parquets from local raw files.

Expects the following layout in --input-dir (default: data/raw/MAWIFlow):

  {YYYYMMDD}/
    {YYYYMMDD}1400.pcap          # or .pcap.gz — decompressed on the fly if needed
    {YYYYMMDD}_anomalous_suspicious.csv
    {YYYYMMDD}_anomalous_suspicious.xml
    {YYYYMMDD}_notice.csv        # optional
    {YYYYMMDD}_notice.xml        # optional

Pipeline per day:
  1. Run CICFlowMeter via Docker -> flow CSV
  2. Process annotations (XML + CSV) -> annotations Parquet
  3. Join flows + annotations via DuckDB -> labeled Parquet
  4. Merge per-day outputs into per-year Parquet

Output:
  {output-dir}/{year}.parquet
  Columns: CICFlowMeter features + year (int) + label (int) + label_multiclass (str)

Usage:
  # Start Docker Desktop first, then:
  uv run python scripts/process_mawiflow_dataset.py
  uv run python scripts/process_mawiflow_dataset.py --years 2020 2021 2022
  uv run python scripts/process_mawiflow_dataset.py --input-dir data/raw/MAWIFlow --memory-gb 16

Requirements:
  - Docker Desktop running with image thelurps/gintsengelen-cicflowmeter:4dd5319
  - uv dependencies: duckdb, lxml, pandas, tqdm
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import BinaryIO

import duckdb
import pandas as pd
from lxml import etree

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CICFLOWMETER_IMAGE = "thelurps/gintsengelen-cicflowmeter:4dd5319"
DATE_DIR_RE = re.compile(r"^\d{8}$")

# PCap binary format constants (RFC 1761)
_PCAP_MAGIC_LE    = 0xA1B2C3D4
_PCAP_MAGIC_BE    = 0xD4C3B2A1
_PCAP_MAGIC_NS_LE = 0xA1B23C4D
_PCAP_MAGIC_NS_BE = 0x4D3CB2A1
_PCAP_GLOBAL_HDR  = 24
_PCAP_PKT_HDR     = 16


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------

def check_docker() -> bool:
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
        return result.returncode == 0
    except Exception:
        return False


def pull_image() -> bool:
    result = subprocess.run(
        ["docker", "images", "-q", CICFLOWMETER_IMAGE],
        capture_output=True, text=True,
    )
    if result.stdout.strip():
        logger.info(f"  Image already present: {CICFLOWMETER_IMAGE}")
        return True
    logger.info(f"  Pulling {CICFLOWMETER_IMAGE} ...")
    result = subprocess.run(["docker", "pull", CICFLOWMETER_IMAGE])
    return result.returncode == 0


def _build_cfm_opts(memory_gb: int) -> str:
    return (
        f"-Xms4g -Xmx{memory_gb}g "
        f"-XX:MaxDirectMemorySize={memory_gb}g "
        "-XX:-UseGCOverheadLimit -XX:+UseG1GC -XX:MaxGCPauseMillis=200 "
        f"-Dnio.mx={memory_gb}gb -Dnio.ms={memory_gb}gb -Dnio.blocksize=64kb"
    )


def _available_memory_gb() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        return max(1, int(page_size * avail_pages / (1024 ** 3)))
    except (AttributeError, OSError, ValueError):
        return None


def _build_retry_memory_ladder(memory_gb: int, max_heap_gb: int | None) -> list[int]:
    attempts = list(dict.fromkeys([memory_gb] + [c for c in (12, 16, 24) if c > memory_gb]))
    if max_heap_gb is None:
        return attempts
    return [s for s in attempts if s <= max_heap_gb]


def _run_cicflowmeter_once(pcap_path: Path, output_dir: Path, memory_gb: int) -> tuple[bool, str]:
    cfm_opts = _build_cfm_opts(memory_gb)
    pcap_dir_abs = str(pcap_path.parent.resolve()).replace("\\", "/")
    output_dir_abs = str(output_dir.resolve()).replace("\\", "/")

    cmd = [
        "docker", "run", "--rm",
        "-e", f"CFM_OPTS={cfm_opts}",
        "-e", f"JAVA_TOOL_OPTIONS={cfm_opts}",
        "-e", f"_JAVA_OPTIONS={cfm_opts}",
        "-v", f"{pcap_dir_abs}:/pcap",
        "-v", f"{output_dir_abs}:/output",
        "--workdir", "/CICFlowMeter/bin",
        CICFLOWMETER_IMAGE,
        "/pcap", "/output",
    ]

    logger.info(f"  Running CICFlowMeter on {pcap_path.name} with {memory_gb}GB heap ...")
    result = subprocess.run(cmd, timeout=3600, capture_output=True, text=True)
    output = (result.stdout or "") + (result.stderr or "")

    # CICFlowMeter exits non-zero even on success — treat as success if Flow CSVs were created.
    flow_csvs = list(output_dir.glob("*_Flow.csv"))
    success = result.returncode == 0 or bool(flow_csvs)

    if not success:
        logger.warning(f"  CICFlowMeter failed (rc={result.returncode}, no output CSVs)")
        if output:
            logger.warning(output.strip())
    elif result.returncode != 0:
        logger.info(
            "  CICFlowMeter exited with rc=%d but produced %d CSV(s) — treating as success",
            result.returncode, len(flow_csvs),
        )
    return success, output


def _split_pcap_by_packets(
    pcap_path: Path,
    split_dir: Path,
    packets_per_chunk: int,
) -> list[Path]:
    """Split a PCap file into chunks of packets_per_chunk packets each.

    Reads the raw binary format without external dependencies. Each chunk
    receives a copy of the original global header so CICFlowMeter can parse it.
    Returns the list of chunk paths in order.
    """
    import struct

    split_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[Path] = []

    with open(pcap_path, "rb") as src:
        global_hdr = src.read(_PCAP_GLOBAL_HDR)
        if len(global_hdr) < _PCAP_GLOBAL_HDR:
            logger.warning("  PCap file too small to parse")
            return []

        magic = struct.unpack_from("<I", global_hdr)[0]
        if magic in (_PCAP_MAGIC_LE, _PCAP_MAGIC_NS_LE):
            endian = "<"
        elif magic in (_PCAP_MAGIC_BE, _PCAP_MAGIC_NS_BE):
            endian = ">"
        else:
            logger.warning(f"  Unknown PCap magic: {magic:#010x}")
            return []

        chunk_idx = 0
        pkt_count = 0
        chunk_fh = None
        chunk_path: Path | None = None

        def _open_chunk() -> tuple[Path, BinaryIO]:
            p = split_dir / f"chunk_{chunk_idx:04d}.pcap"
            fh = open(p, "wb")
            fh.write(global_hdr)
            chunks.append(p)
            return p, fh

        chunk_path, chunk_fh = _open_chunk()

        while True:
            pkt_hdr = src.read(_PCAP_PKT_HDR)
            if not pkt_hdr:
                break
            if len(pkt_hdr) < _PCAP_PKT_HDR:
                break
            incl_len = struct.unpack_from(f"{endian}I", pkt_hdr, 8)[0]
            pkt_data = src.read(incl_len)
            if len(pkt_data) < incl_len:
                break

            if pkt_count > 0 and pkt_count % packets_per_chunk == 0:
                chunk_fh.close()
                chunk_idx += 1
                chunk_path, chunk_fh = _open_chunk()

            chunk_fh.write(pkt_hdr + pkt_data)
            pkt_count += 1

        if chunk_fh:
            chunk_fh.close()

    logger.info(f"  Split {pkt_count:,} packets into {len(chunks)} chunk(s)")
    return chunks


def _merge_flow_csvs(flow_csvs: list[Path], merged_path: Path) -> bool:
    """Concatenate per-chunk *_Flow.csv files into a single merged CSV."""
    if not flow_csvs:
        return False

    with open(merged_path, "wb") as out:
        for i, csv_path in enumerate(sorted(flow_csvs)):
            with open(csv_path, "rb") as inp:
                first_line = inp.readline()
                if i == 0:
                    out.write(first_line)
                out.write(inp.read())

    logger.info(f"  Merged {len(flow_csvs)} flow CSV(s) -> {merged_path.name}")
    return True


def _run_cicflowmeter_chunked(
    pcap_path: Path,
    output_dir: Path,
    memory_gb: int,
    packets_per_chunk: int,
) -> bool:
    """Fallback: split PCap into chunks, run CICFlowMeter on each, merge CSVs."""
    split_dir = output_dir / "_chunks" / "pcap"
    chunks = _split_pcap_by_packets(pcap_path, split_dir, packets_per_chunk)
    if not chunks:
        return False

    all_flow_csvs: list[Path] = []
    for i, chunk in enumerate(chunks):
        chunk_out = output_dir / "_chunks" / f"flows_{i:04d}"
        chunk_out.mkdir(parents=True, exist_ok=True)
        success, _ = _run_cicflowmeter_once(chunk, chunk_out, memory_gb)
        if success:
            all_flow_csvs.extend(chunk_out.glob("*_Flow.csv"))
        else:
            logger.warning(f"  Chunk {i} failed — skipping")

    if not all_flow_csvs:
        shutil.rmtree(output_dir / "_chunks", ignore_errors=True)
        return False

    merged_csv = output_dir / "merged_Flow.csv"
    ok = _merge_flow_csvs(all_flow_csvs, merged_csv)
    shutil.rmtree(output_dir / "_chunks", ignore_errors=True)
    return ok


def run_cicflowmeter(
    pcap_path: Path,
    output_dir: Path,
    memory_gb: int = 8,
    max_heap_gb: int | None = None,
    packets_per_chunk: int = 200_000,
) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = list(output_dir.glob("*_Flow.csv"))
    if existing:
        logger.info(f"  Flow CSV already exists: {existing[0].name}")
        return True

    available_memory = _available_memory_gb()
    safe_cap_gb: int | None = None
    if available_memory is not None:
        safe_cap_gb = max(4, available_memory - 6)
    if max_heap_gb is not None:
        safe_cap_gb = min(safe_cap_gb, max_heap_gb) if safe_cap_gb is not None else max_heap_gb

    attempt_sizes = _build_retry_memory_ladder(memory_gb, safe_cap_gb)
    if not attempt_sizes:
        logger.warning(
            f"  Skipping CICFlowMeter: no valid heap size "
            f"(available={available_memory}GB, max-heap={max_heap_gb}GB)"
        )
        return False

    if safe_cap_gb is not None:
        logger.info(
            f"  CICFlowMeter heap attempts capped at {safe_cap_gb}GB "
            f"(available memory estimate: {available_memory}GB)"
        )

    last_failure_was_oom = False
    for i, attempt_memory in enumerate(attempt_sizes):
        success, output = _run_cicflowmeter_once(pcap_path, output_dir, attempt_memory)
        if success:
            return True
        oom = "GC overhead limit exceeded" in output or "OutOfMemoryError" in output
        last_failure_was_oom = oom
        if not oom:
            break
        if i + 1 < len(attempt_sizes):
            logger.info(f"  Retrying CICFlowMeter with {attempt_sizes[i + 1]}GB heap ...")

    if last_failure_was_oom and packets_per_chunk > 0:
        logger.info(
            f"  All heap attempts failed with OOM. "
            f"Falling back to chunked PCap processing ({packets_per_chunk:,} pkts/chunk) ..."
        )
        return _run_cicflowmeter_chunked(pcap_path, output_dir, attempt_sizes[0], packets_per_chunk)

    return False


# ---------------------------------------------------------------------------
# PCap preparation
# ---------------------------------------------------------------------------

def resolve_pcap(day_input_dir: Path, date_str: str) -> Path | None:
    """Return path to decompressed .pcap, decompressing .pcap.gz if needed."""
    pcap = day_input_dir / f"{date_str}1400.pcap"
    gz = day_input_dir / f"{date_str}1400.pcap.gz"

    if pcap.exists():
        return pcap
    if gz.exists():
        logger.info(f"  Decompressing {gz.name} ...")
        with gzip.open(gz, "rb") as f_in, open(pcap, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        return pcap

    logger.warning(f"  PCap not found in {day_input_dir} (expected {pcap.name} or {gz.name})")
    return None


# ---------------------------------------------------------------------------
# Annotation processing
# ---------------------------------------------------------------------------

def _safe_float(v: str | None) -> float | None:
    try:
        return float(v) if v else None
    except (ValueError, TypeError):
        return None


def _safe_int(v: str | None) -> int | None:
    try:
        return int(v) if v else None
    except (ValueError, TypeError):
        return None


def _safe_bool(v: str | None) -> bool | None:
    if v is None:
        return None
    return v.strip().lower() in ("1", "true", "yes")


def _zero_to_none(v: int | None) -> int | None:
    return None if v in (0, 2147483645, None) else v


def _read_annotation_csv(path: Path, kind: str) -> pd.DataFrame | None:
    try:
        if kind == "anomalous_suspicious":
            cols = [
                "anomaly_id", "src_ip", "src_port", "dst_ip", "dst_port",
                "taxonomy", "distance", "num_detectors", "label",
            ]
        else:
            cols = [
                "src_ip", "src_port", "dst_ip", "dst_port",
                "taxonomy", "heuristic", "distance", "num_detectors", "label",
            ]
        df = pd.read_csv(path, skiprows=1, header=None, names=cols)
        if kind == "notice":
            df.insert(0, "anomaly_id", range(len(df)))
        return df
    except Exception as exc:
        logger.warning(f"  Failed to parse CSV {path.name}: {exc}")
        return None


def _read_annotation_xml(path: Path) -> pd.DataFrame | None:
    try:
        tree = etree.parse(str(path))
        root = tree.getroot()
        nsmap = {
            "admd": "http://www.nict.go.jp/admd",
            "xsi": "http://www.w3.org/2001/XMLSchema-instance",
        }
        rows = []
        for idx, anomaly in enumerate(root.findall(".//anomaly", namespaces=nsmap)):
            anomaly_type = anomaly.get("type")
            values = anomaly.get("value", ",,,,").split(",")
            detectors = values[3].split(" ") if len(values) > 3 else [""] * 12

            start_el = anomaly.find(".//from")
            start = start_el.get("sec") if start_el is not None else None
            stop_el = anomaly.find(".//to")
            stop = stop_el.get("sec") if stop_el is not None else None

            for f in anomaly.findall(".//slice/filter", namespaces=nsmap):
                rows.append({
                    "anomaly_id": idx,
                    "label": anomaly_type,
                    "distance_normal": _safe_float(values[0] if values else None),
                    "distance_anomalous": _safe_float(values[1] if len(values) > 1 else None),
                    "heuristic": values[2] if len(values) > 2 else None,
                    "hough_sensitive": _safe_bool(detectors[0] if len(detectors) > 0 else None),
                    "hough_optimal": _safe_bool(detectors[1] if len(detectors) > 1 else None),
                    "hough_conservative": _safe_bool(detectors[2] if len(detectors) > 2 else None),
                    "gamma_sensitive": _safe_bool(detectors[3] if len(detectors) > 3 else None),
                    "gamma_optimal": _safe_bool(detectors[4] if len(detectors) > 4 else None),
                    "gamma_conservative": _safe_bool(detectors[5] if len(detectors) > 5 else None),
                    "kl_sensitive": _safe_bool(detectors[6] if len(detectors) > 6 else None),
                    "kl_optimal": _safe_bool(detectors[7] if len(detectors) > 7 else None),
                    "kl_conservative": _safe_bool(detectors[8] if len(detectors) > 8 else None),
                    "pca_sensitive": _safe_bool(detectors[9] if len(detectors) > 9 else None),
                    "pca_optimal": _safe_bool(detectors[10] if len(detectors) > 10 else None),
                    "pca_conservative": _safe_bool(detectors[11] if len(detectors) > 11 else None),
                    "taxonomy": values[4] if len(values) > 4 else None,
                    "start": _zero_to_none(_safe_int(start)),
                    "stop":  _zero_to_none(_safe_int(stop)),
                    **f.attrib,
                })
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception as exc:
        logger.warning(f"  Failed to parse XML {path.name}: {exc}")
        return None


def build_annotations_parquet(day_input_dir: Path, output_path: Path) -> bool:
    date_str = day_input_dir.name

    as_csv = day_input_dir / f"{date_str}_anomalous_suspicious.csv"
    as_xml = day_input_dir / f"{date_str}_anomalous_suspicious.xml"

    if not as_csv.exists() or not as_xml.exists():
        logger.warning(f"  Missing required annotation files in {day_input_dir}")
        return False

    xml_df = _read_annotation_xml(as_xml)
    csv_df = _read_annotation_csv(as_csv, "anomalous_suspicious")
    if xml_df is None or csv_df is None or xml_df.empty:
        return False

    xml_df = xml_df.reset_index(drop=True)
    csv_df = csv_df.reset_index(drop=True)

    if "distance" in csv_df.columns and "num_detectors" in csv_df.columns:
        xml_df["distance"] = csv_df["distance"].reindex(range(len(xml_df))).values
        xml_df["num_detectors"] = csv_df["num_detectors"].reindex(range(len(xml_df))).values

    combined = xml_df.copy()

    nc_csv = day_input_dir / f"{date_str}_notice.csv"
    nc_xml = day_input_dir / f"{date_str}_notice.xml"
    if nc_csv.exists() and nc_xml.exists():
        n_xml = _read_annotation_xml(nc_xml)
        n_csv = _read_annotation_csv(nc_csv, "notice")
        if n_xml is not None and n_csv is not None and not n_xml.empty:
            n_xml = n_xml.reset_index(drop=True)
            n_csv = n_csv.reset_index(drop=True)
            if "distance" in n_csv.columns:
                n_xml["distance"] = n_csv["distance"].reindex(range(len(n_xml))).values
            if "num_detectors" in n_csv.columns:
                n_xml["num_detectors"] = n_csv["num_detectors"].reindex(range(len(n_xml))).values
            combined = pd.concat([combined, n_xml], ignore_index=True)

    # Ensure optional filter columns always exist so the downstream DuckDB
    # query never hits a BinderException.  proto/src_ip/etc. come from
    # **f.attrib in the XML and are absent when no filter in a day specifies
    # that field (e.g. 20101001 has no protocol-level filters).
    for _col in ("proto", "src_ip", "src_port", "dst_ip", "dst_port"):
        if _col not in combined.columns:
            combined[_col] = None

    combined.insert(0, "rule_id", range(len(combined)))
    combined.to_parquet(output_path, index=False, compression="snappy")
    logger.info(f"  Annotations: {len(combined)} rules -> {output_path.name}")
    return True


# ---------------------------------------------------------------------------
# Flow + annotation join via DuckDB
# ---------------------------------------------------------------------------

def combine_flows_annotations(
    flows_dir: Path,
    annotations_parquet: Path,
    output_path: Path,
    year: int,
) -> bool:
    if not list(flows_dir.glob("*_Flow.csv")):
        logger.warning(f"  No *_Flow.csv found in {flows_dir}")
        return False

    flows_glob = str(flows_dir / "*_Flow.csv").replace("\\", "/")
    ann_path   = str(annotations_parquet).replace("\\", "/")
    out_path   = str(output_path).replace("\\", "/")

    conn = duckdb.connect(":memory:")

    # Defence-in-depth: if an annotation parquet was written before the
    # optional-column fix, inject NULL columns via a view so the downstream
    # query never raises a BinderException.  build_annotations_parquet now
    # always writes these columns, but old files may still be in the work dir.
    _ann_cols = {r[0] for r in conn.execute(f"DESCRIBE SELECT * FROM '{ann_path}'").fetchall()}
    _missing  = [c for c in ("proto", "src_ip", "src_port", "dst_ip", "dst_port") if c not in _ann_cols]
    if _missing:
        _null_cols = ", ".join(f"NULL::VARCHAR AS {c}" for c in _missing)
        conn.execute(f"""
            CREATE OR REPLACE VIEW v_ann_raw AS
            SELECT *, {_null_cols} FROM '{ann_path}';
        """)
        _ann_source = "v_ann_raw"
    else:
        _ann_source = f"'{ann_path}'"

    conn.execute(f"""
        CREATE OR REPLACE VIEW v_annotations AS
        SELECT *,
            (CASE WHEN start    IS NOT NULL THEN 1 ELSE 0 END +
             CASE WHEN stop     IS NOT NULL THEN 1 ELSE 0 END +
             CASE WHEN src_ip   IS NOT NULL THEN 1 ELSE 0 END +
             CASE WHEN src_port IS NOT NULL THEN 1 ELSE 0 END +
             CASE WHEN dst_ip   IS NOT NULL THEN 1 ELSE 0 END +
             CASE WHEN dst_port IS NOT NULL THEN 1 ELSE 0 END +
             CASE WHEN proto    IS NOT NULL THEN 1 ELSE 0 END) AS feature_count,
            CASE WHEN start IS NOT NULL AND stop IS NOT NULL THEN stop - start
                 ELSE NULL END AS duration
        FROM {_ann_source}
        ORDER BY feature_count DESC, duration ASC NULLS LAST;
    """)

    conn.execute(f"""
        CREATE OR REPLACE VIEW v_flows AS
        SELECT *,
            CASE
                WHEN Protocol = 6   THEN 'tcp'
                WHEN Protocol = 17  THEN 'udp'
                WHEN Protocol = 1   THEN 'icmp'
                WHEN Protocol = 2   THEN 'igmp'
                WHEN Protocol = 4   THEN 'ip'
                WHEN Protocol = 41  THEN 'ipv6'
                WHEN Protocol = 58  THEN 'icmpv6'
                WHEN Protocol = 89  THEN 'ospf'
                WHEN Protocol = 132 THEN 'sctp'
                ELSE 'unknown_' || CAST(Protocol AS VARCHAR)
            END AS protocol_name,
            ROW_NUMBER() OVER (
                ORDER BY "Timestamp", "Flow ID", "Src IP", "Src Port",
                         "Dst IP", "Dst Port", "Protocol"
            ) AS _flow_row_id
        FROM read_csv_auto('{flows_glob}', ignore_errors=true);
    """)

    row_count = conn.execute("SELECT COUNT(*) FROM v_flows").fetchone()[0]
    logger.info(f"  Flows loaded: {row_count:,}")

    conn.execute(f"""
        COPY (
            WITH matches AS (
                SELECT
                    f.*,
                    a.label AS annotation_label,
                    a.taxonomy,
                    ROW_NUMBER() OVER (
                        PARTITION BY f._flow_row_id
                        ORDER BY a.feature_count DESC, a.duration ASC NULLS LAST
                    ) AS ann_rank
                FROM v_flows f
                LEFT JOIN v_annotations a ON
                    (a.start    IS NULL OR epoch(f."Timestamp") >= a.start)   AND
                    (a.stop     IS NULL OR epoch(f."Timestamp") <= a.stop)    AND
                    (a.src_ip   IS NULL OR f."Src IP"   = a.src_ip)           AND
                    (a.src_port IS NULL OR f."Src Port" = a.src_port)         AND
                    (a.dst_ip   IS NULL OR f."Dst IP"   = a.dst_ip)           AND
                    (a.dst_port IS NULL OR f."Dst Port" = a.dst_port)         AND
                    (a.proto    IS NULL OR f.protocol_name = a.proto)
            )
            SELECT
                * EXCLUDE (ann_rank, annotation_label, protocol_name, _flow_row_id, "Label"),
                COALESCE(annotation_label, 'benign') AS label_multiclass,
                CASE WHEN annotation_label IS NULL OR annotation_label = 'benign'
                     THEN 0 ELSE 1 END AS label,
                {year} AS year
            FROM matches
            WHERE ann_rank = 1
        )
        TO '{out_path}' (FORMAT PARQUET, COMPRESSION SNAPPY);
    """)

    n_out = conn.execute(f"SELECT COUNT(*) FROM '{out_path}'").fetchone()[0]
    if n_out != row_count:
        logger.warning(
            f"  CARDINALITY MISMATCH: {row_count:,} flows loaded but {n_out:,} written "
            f"({row_count - n_out:,} rows lost). Check join logic."
        )
    logger.info(f"  Labeled flows: {n_out:,} -> {output_path.name}")
    conn.close()
    return True


# ---------------------------------------------------------------------------
# Per-day orchestration
# ---------------------------------------------------------------------------

def process_day(
    day_input_dir: Path,
    work_dir: Path,
    memory_gb: int,
    max_heap_gb: int | None,
    packets_per_chunk: int = 200_000,
) -> Path | None:
    date_str = day_input_dir.name
    try:
        day = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
    except ValueError:
        logger.warning(f"  Cannot parse date from directory name: {date_str}")
        return None

    day_work_dir = work_dir / date_str
    day_work_dir.mkdir(parents=True, exist_ok=True)

    labeled_parquet = day_work_dir / "labeled.parquet"
    if labeled_parquet.exists():
        logger.info(f"  Day {date_str} already processed.")
        return labeled_parquet

    logger.info(f"--- Day {date_str} ---")

    # Step 1: Resolve pcap (decompress if .pcap.gz)
    pcap = resolve_pcap(day_input_dir, date_str)
    if pcap is None:
        return None

    # Step 2: Run CICFlowMeter
    flows_dir = day_work_dir / "flows"
    if not run_cicflowmeter(pcap, flows_dir, memory_gb, max_heap_gb, packets_per_chunk):
        logger.warning(f"  CICFlowMeter failed for {date_str}")
        return None

    # Step 3: Build annotations Parquet
    ann_parquet = day_work_dir / "annotations.parquet"
    if not build_annotations_parquet(day_input_dir, ann_parquet):
        logger.warning(f"  Annotation processing failed for {date_str}")
        return None

    # Step 4: Join flows + annotations
    if not combine_flows_annotations(flows_dir, ann_parquet, labeled_parquet, day.year):
        logger.warning(f"  Flow+annotation join failed for {date_str}")
        return None

    shutil.rmtree(flows_dir, ignore_errors=True)
    return labeled_parquet


# ---------------------------------------------------------------------------
# Per-year orchestration
# ---------------------------------------------------------------------------

def discover_day_dirs(input_dir: Path, years: list[int] | None) -> dict[int, list[Path]]:
    """Scan input_dir for YYYYMMDD subdirectories, grouped by year."""
    result: dict[int, list[Path]] = {}
    for entry in sorted(input_dir.iterdir()):
        if not entry.is_dir() or not DATE_DIR_RE.match(entry.name):
            continue
        year = int(entry.name[:4])
        if years and year not in years:
            continue
        result.setdefault(year, []).append(entry)
    return result


def build_year_parquet(
    year: int,
    day_dirs: list[Path],
    output_dir: Path,
    work_dir: Path,
    memory_gb: int,
    max_heap_gb: int | None,
    days_per_year: int | None,
    packets_per_chunk: int = 200_000,
) -> bool:
    year_parquet = output_dir / f"{year}.parquet"
    if year_parquet.exists():
        logger.info(f"[{year}] Already done: {year_parquet}")
        return True

    logger.info(f"\n{'=' * 60}")
    logger.info(f"[{year}] Processing {len(day_dirs)} available day(s)")
    logger.info(f"{'=' * 60}")

    candidates = day_dirs[:days_per_year] if days_per_year else day_dirs

    day_parquets: list[Path] = []
    for day_dir in candidates:
        result = process_day(
            day_dir, work_dir / str(year), memory_gb, max_heap_gb, packets_per_chunk
        )
        if result is not None:
            day_parquets.append(result)

    if not day_parquets:
        logger.error(f"[{year}] No days processed successfully")
        return False

    logger.info(f"[{year}] Merging {len(day_parquets)} day(s) -> {year_parquet.name}")
    merged = pd.concat([pd.read_parquet(p) for p in day_parquets], ignore_index=True)
    merged.to_parquet(year_parquet, index=False, compression="snappy")
    logger.info(f"[{year}] Done: {len(merged):,} rows -> {year_parquet}")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build labeled MAWIFlow parquets from local raw files."
    )
    parser.add_argument(
        "--input-dir",
        default="data/raw/MAWIFlow",
        help="Directory containing YYYYMMDD subdirectories (default: data/raw/MAWIFlow)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/mawiflow",
        help="Output directory for per-year Parquet files (default: data/processed/mawiflow)",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=None,
        help="Years to process (default: all years found in --input-dir)",
    )
    parser.add_argument(
        "--days-per-year",
        type=int,
        default=None,
        metavar="N",
        help="Max days to process per year (default: all available)",
    )
    parser.add_argument(
        "--memory-gb",
        type=int,
        default=8,
        metavar="GB",
        help="Initial JVM heap for CICFlowMeter in GB (default: 8)",
    )
    parser.add_argument(
        "--max-heap-gb",
        type=int,
        default=None,
        metavar="GB",
        help="Hard cap for CICFlowMeter heap retries in GB",
    )
    parser.add_argument(
        "--pcap-chunk-packets",
        type=int,
        default=200_000,
        metavar="N",
        help=(
            "Packets per chunk when chunked PCap fallback is triggered by OOM "
            "(default: 200000; set to 0 to disable chunking)"
        ),
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help="Directory for intermediate files (default: system temp)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    input_dir  = (repo_root / args.input_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve()

    if not input_dir.exists():
        print(f"[error] Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not check_docker():
        print(
            "\n[error] Docker daemon is not reachable.\n"
            "Please start Docker Desktop and try again.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    if not pull_image():
        print(f"[error] Failed to pull {CICFLOWMETER_IMAGE}", file=sys.stderr)
        sys.exit(1)

    day_dirs_by_year = discover_day_dirs(input_dir, args.years)
    if not day_dirs_by_year:
        print(
            f"[error] No YYYYMMDD directories found in {input_dir}"
            + (f" for years {args.years}" if args.years else ""),
            file=sys.stderr,
        )
        sys.exit(1)

    if args.workdir:
        work_dir = Path(args.workdir)
        work_dir.mkdir(parents=True, exist_ok=True)
        use_temp = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="mawiflow_process_"))
        use_temp = True

    logger.info(f"Input dir:  {input_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Work dir:   {work_dir}")
    logger.info(f"Years:      {sorted(day_dirs_by_year)}")
    logger.info(f"Memory:     {args.memory_gb} GB")
    if args.max_heap_gb is not None:
        logger.info(f"Max heap:   {args.max_heap_gb} GB")

    failed: list[int] = []
    for year in sorted(day_dirs_by_year):
        ok = build_year_parquet(
            year=year,
            day_dirs=day_dirs_by_year[year],
            output_dir=output_dir,
            work_dir=work_dir,
            memory_gb=args.memory_gb,
            max_heap_gb=args.max_heap_gb,
            days_per_year=args.days_per_year,
            packets_per_chunk=args.pcap_chunk_packets,
        )
        if not ok:
            failed.append(year)

    if use_temp:
        shutil.rmtree(work_dir, ignore_errors=True)

    done = [y for y in sorted(day_dirs_by_year) if y not in failed]
    print(f"\n[done] {len(done)}/{len(day_dirs_by_year)} years processed successfully.")
    if failed:
        print(f"[warn] Failed years: {failed}")
    if done:
        print("Output files:")
        for y in done:
            p = output_dir / f"{y}.parquet"
            print(f"  {p}  ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
