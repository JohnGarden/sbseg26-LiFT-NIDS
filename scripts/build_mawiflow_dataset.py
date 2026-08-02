#!/usr/bin/env python3
"""Build the MAWIFlow dataset from public MAWI and MAWILab sources.

Pipeline per day:
  1. Download PCap from mawi.wide.ad.jp
  2. Download annotation XML/CSV from fukuda-lab.org/mawilab/v1.1
  3. Run CICFlowMeter via Docker -> flow CSV
  4. Process annotations (XML + CSV) -> annotations Parquet
  5. Join flows + annotations via DuckDB -> labeled Parquet
  6. Append year column and merge into per-year output file

Output:
  data/raw/mawiflow/{year}.parquet
  Columns: CICFlowMeter features + year (int) + label (int) + label_multiclass (str)

Usage:
  # Start Docker Desktop first, then:
  uv run python scripts/build_mawiflow_dataset.py
  uv run python scripts/build_mawiflow_dataset.py --years 2020 2021 2022
  uv run python scripts/build_mawiflow_dataset.py --days-per-year 3 --memory-gb 12

Requirements:
  - Docker Desktop running with image thelurps/gintsengelen-cicflowmeter:4dd5319
  - uv dependencies: duckdb, lxml, requests, tqdm, polars
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import requests
from lxml import etree
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAWI_PCAP_URL = "http://mawi.nezu.wide.ad.jp/mawi/samplepoint-F/{year}/{date}1400.pcap.gz"
MAWILAB_BASE_URL = "https://www.fukuda-lab.org/mawilab/v1.1/{year}/{month}/{day}/{date}_{kind}"
CICFLOWMETER_IMAGE = "thelurps/gintsengelen-cicflowmeter:4dd5319"

# PCap binary format constants (RFC 1761)
_PCAP_MAGIC_LE    = 0xA1B2C3D4
_PCAP_MAGIC_BE    = 0xD4C3B2A1
_PCAP_MAGIC_NS_LE = 0xA1B23C4D
_PCAP_MAGIC_NS_BE = 0x4D3CB2A1
_PCAP_GLOBAL_HDR  = 24
_PCAP_PKT_HDR     = 16

YEARS_AVAILABLE = list(range(2007, 2025))
# Candidate days within January tried in order until one succeeds
CANDIDATE_DAYS = [15, 10, 20, 5, 25, 1]

# CICFlowMeter feature names (84 numerical + 3 categorical)
NUMERICAL_FEATURES = [
    "Flow Duration", "Total Fwd Packet", "Total Bwd packets",
    "Total Length of Fwd Packet", "Total Length of Bwd Packet",
    "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd RST Flags", "Bwd RST Flags",
    "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s",
    "Packet Length Min", "Packet Length Max", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWR Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size",
    "Fwd Segment Size Avg", "Bwd Segment Size Avg",
    "Fwd Bytes/Bulk Avg", "Fwd Packet/Bulk Avg", "Fwd Bulk Rate Avg",
    "Bwd Bytes/Bulk Avg", "Bwd Packet/Bulk Avg", "Bwd Bulk Rate Avg",
    "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "FWD Init Win Bytes", "Bwd Init Win Bytes",
    "Fwd Act Data Pkts", "Bwd Act Data Pkts",
    "Fwd Seg Size Min", "Bwd Seg Size Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
    "Fwd TCP Retrans. Count", "Bwd TCP Retrans. Count",
    "Total TCP Retrans. Count", "Total Connection Flow Time",
]
CATEGORICAL_FEATURES = ["Protocol", "ICMP Code", "ICMP Type"]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path, desc: str = "") -> bool:
    """Download url to dest with progress bar. Returns True on success."""
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=desc or dest.name, leave=False
        ) as bar:
            for chunk in resp.iter_content(chunk_size=65536):
                fh.write(chunk)
                bar.update(len(chunk))
        return True
    except requests.HTTPError as exc:
        logger.warning(f"HTTP {exc.response.status_code} for {url}")
        return False
    except Exception as exc:
        logger.warning(f"Download failed {url}: {exc}")
        return False


def download_pcap(day: date, dest_dir: Path) -> Path | None:
    """Download and decompress the MAWI PCap for the given day."""
    date_str = day.strftime("%Y%m%d")
    url = MAWI_PCAP_URL.format(year=day.year, date=date_str)
    gz_path = dest_dir / f"{date_str}1400.pcap.gz"
    pcap_path = dest_dir / f"{date_str}1400.pcap"

    if pcap_path.exists():
        logger.info(f"  PCap already exists: {pcap_path.name}")
        return pcap_path

    logger.info(f"  Downloading PCap {date_str} ...")
    if not _download(url, gz_path, desc=f"PCap {date_str}"):
        return None

    logger.info(f"  Decompressing {gz_path.name} ...")
    with gzip.open(gz_path, "rb") as f_in, open(pcap_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()
    return pcap_path


def download_annotations(day: date, dest_dir: Path) -> bool:
    """Download MAWILAB annotation files for the given day."""
    date_str = day.strftime("%Y%m%d")
    year_s = day.strftime("%Y")
    month_s = day.strftime("%m")
    day_s = day.strftime("%d")

    base = MAWILAB_BASE_URL.format(
        year=year_s, month=month_s, day=day_s, date=date_str, kind=""
    ).rstrip("_")

    required_kinds = ["anomalous_suspicious.csv", "anomalous_suspicious.xml"]
    optional_kinds = ["notice.csv", "notice.xml"]

    for kind in required_kinds:
        url = f"{base}_{kind}"
        dest = dest_dir / f"{date_str}_{kind}"
        if dest.exists():
            continue
        if not _download(url, dest, desc=kind):
            logger.warning(f"  Required annotation missing: {kind} for {date_str}")
            return False

    for kind in optional_kinds:
        url = f"{base}_{kind}"
        dest = dest_dir / f"{date_str}_{kind}"
        if dest.exists():
            continue
        _download(url, dest, desc=kind)  # failure is OK

    return True


def _copy_if_exists(src: Path, dest: Path) -> bool:
    """Copy a file from src to dest if it exists."""
    if not src.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dest.resolve():
        return True
    shutil.copy2(src, dest)
    return True


def _populate_day_inputs_from_local(
    day: date,
    raw_source_dir: Path,
    pcap_dir: Path,
    ann_date_dir: Path,
) -> None:
    """Copy local raw MAWIFlow day files into the working day directories."""
    date_str = day.strftime("%Y%m%d")
    local_day_dir = raw_source_dir / date_str
    if not local_day_dir.exists():
        return

    logger.info(f"  Found local raw files in {local_day_dir}")
    pcap_dir.mkdir(parents=True, exist_ok=True)
    ann_date_dir.mkdir(parents=True, exist_ok=True)

    # Copy PCap if available locally.
    pcap_path = pcap_dir / f"{date_str}1400.pcap"
    if not _copy_if_exists(local_day_dir / f"{date_str}1400.pcap", pcap_path):
        gz_src = local_day_dir / f"{date_str}1400.pcap.gz"
        if gz_src.exists():
            with gzip.open(gz_src, "rb") as f_in, open(pcap_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

    # Copy annotation files from local source to working directory.
    for kind in [
        "anomalous_suspicious.csv",
        "anomalous_suspicious.xml",
        "notice.csv",
        "notice.xml",
    ]:
        _copy_if_exists(
            local_day_dir / f"{date_str}_{kind}",
            ann_date_dir / f"{date_str}_{kind}",
        )


# ---------------------------------------------------------------------------
# CICFlowMeter via Docker
# ---------------------------------------------------------------------------

def check_docker() -> bool:
    """Return True if Docker daemon is reachable."""
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def pull_image() -> bool:
    """Pull CICFlowMeter image if not present."""
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
    """Build CICFlowMeter Java options for the container."""
    return (
        f"-Xms4g -Xmx{memory_gb}g "
        f"-XX:MaxDirectMemorySize={memory_gb}g "
        "-XX:-UseGCOverheadLimit -XX:+UseG1GC -XX:MaxGCPauseMillis=200 "
        f"-Dnio.mx={memory_gb}gb -Dnio.ms={memory_gb}gb -Dnio.blocksize=64kb"
    )


def _available_memory_gb() -> int | None:
    """Return currently available host memory in GiB when detectable."""
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        available_bytes = page_size * avail_pages
        return max(1, int(available_bytes / (1024 ** 3)))
    except (AttributeError, OSError, ValueError):
        return None


def _build_retry_memory_ladder(memory_gb: int, max_heap_gb: int | None) -> list[int]:
    """Build retry heap sizes bounded by max_heap_gb."""
    base_attempts = [memory_gb]
    for candidate in (12, 16, 24):
        if memory_gb < candidate:
            base_attempts.append(candidate)
    attempts = list(dict.fromkeys(base_attempts))
    if max_heap_gb is None:
        return attempts
    return [size for size in attempts if size <= max_heap_gb]


def _run_cicflowmeter_once(pcap_path: Path, output_dir: Path, memory_gb: int) -> tuple[bool, str]:
    """Run CICFlowMeter once and return success plus combined log output."""
    cfm_opts = _build_cfm_opts(memory_gb)

    pcap_dir = pcap_path.parent
    pcap_dir_abs = str(pcap_dir.resolve()).replace("\\", "/")
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

    logger.info(f"  Running CICFlowMeter on {pcap_path.name} with {memory_gb}GB heap...")
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
    """Split a PCap file into chunks of packets_per_chunk packets each."""
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

        def _open_chunk():
            p = split_dir / f"chunk_{chunk_idx:04d}.pcap"
            fh = open(p, "wb")
            fh.write(global_hdr)
            chunks.append(p)
            return p, fh

        _, chunk_fh = _open_chunk()

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
                _, chunk_fh = _open_chunk()

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
    """Run CICFlowMeter container on pcap_path, write CSVs to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pcap_dir = pcap_path.parent  # noqa: F841 — kept for clarity

    # Check if output already exists
    existing = list(output_dir.glob("*_Flow.csv"))
    if existing:
        logger.info(f"  Flow CSV already exists: {existing[0].name}")
        return True

    available_memory = _available_memory_gb()
    safe_cap_gb = None
    if available_memory is not None:
        # Keep headroom for OS/other processes, especially on notebooks.
        safe_cap_gb = max(4, available_memory - 6)
    if max_heap_gb is not None:
        safe_cap_gb = min(safe_cap_gb, max_heap_gb) if safe_cap_gb is not None else max_heap_gb

    attempt_sizes = _build_retry_memory_ladder(memory_gb, safe_cap_gb)
    if not attempt_sizes:
        logger.warning(
            "  Skipping CICFlowMeter: no valid heap size after applying memory cap "
            f"(available={available_memory}GB, max-heap={max_heap_gb})."
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
            next_memory = attempt_sizes[i + 1]
            logger.info(f"  Retrying CICFlowMeter with {next_memory}GB heap...")

    if last_failure_was_oom and packets_per_chunk > 0:
        logger.info(
            f"  All heap attempts failed with OOM. "
            f"Falling back to chunked PCap processing ({packets_per_chunk:,} pkts/chunk) ..."
        )
        return _run_cicflowmeter_chunked(pcap_path, output_dir, attempt_sizes[0], packets_per_chunk)

    return False


# ---------------------------------------------------------------------------
# Annotation processing (ported from TheLurps/MAWIFlow annotations module)
# ---------------------------------------------------------------------------

def _read_annotation_csv(path: Path, kind: str) -> pd.DataFrame | None:
    """Parse MAWILAB annotation CSV (anomalous_suspicious or notice)."""
    try:
        if kind == "anomalous_suspicious":
            cols = [
                "anomaly_id", "src_ip", "src_port", "dst_ip", "dst_port",
                "taxonomy", "distance", "num_detectors", "label",
            ]
        else:  # notice
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
    """Parse MAWILAB anomaly XML into a DataFrame."""
    try:
        tree = etree.parse(str(path))
        root = tree.getroot()
        nsmap = {
            "admd": "http://www.nict.go.jp/admd",
            "xsi": "http://www.w3.org/2001/XMLSchema-instance",
        }
        anomalies = root.findall(".//anomaly", namespaces=nsmap)
        rows = []
        for idx, anomaly in enumerate(anomalies):
            anomaly_type = anomaly.get("type")
            values = anomaly.get("value", ",,,,").split(",")
            detectors = values[3].split(" ") if len(values) > 3 else [""] * 12
            start = anomaly.findtext(".//from", default=None)
            if start:
                start = anomaly.find(".//from").get("sec")
            stop_el = anomaly.find(".//to")
            stop = stop_el.get("sec") if stop_el is not None else None

            for f in anomaly.findall(".//slice/filter", namespaces=nsmap):
                rows.append({
                    "anomaly_id": idx,
                    "label": anomaly_type,
                    "distance_normal": _safe_float(values[0] if values else None),
                    "distance_anomalous": _safe_float(values[1] if len(values) > 1 else None),
                    "heuristic": values[2] if len(values) > 2 else None,
                    "hough_sensitive": _safe_bool(detectors[0] if detectors else None),
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
                    "stop": _zero_to_none(_safe_int(stop)),
                    **f.attrib,
                })
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception as exc:
        logger.warning(f"  Failed to parse XML {path.name}: {exc}")
        return None


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


def build_annotations_parquet(raw_dir: Path, output_path: Path) -> bool:
    """Combine anomalous_suspicious + notice annotation files -> annotations Parquet."""
    date_str = raw_dir.name  # e.g. "20070115"

    as_csv = raw_dir / f"{date_str}_anomalous_suspicious.csv"
    as_xml = raw_dir / f"{date_str}_anomalous_suspicious.xml"

    if not as_csv.exists() or not as_xml.exists():
        logger.warning(f"  Missing required annotation files in {raw_dir}")
        return False

    xml_df = _read_annotation_xml(as_xml)
    csv_df = _read_annotation_csv(as_csv, "anomalous_suspicious")
    if xml_df is None or csv_df is None or xml_df.empty:
        return False

    xml_df = xml_df.reset_index(drop=True)
    csv_df = csv_df.reset_index(drop=True)

    # Merge on row index (same as MAWIFlow: join on rn)
    if "distance" in csv_df.columns and "num_detectors" in csv_df.columns:
        xml_df["distance"] = csv_df["distance"].reindex(range(len(xml_df))).values
        xml_df["num_detectors"] = csv_df["num_detectors"].reindex(range(len(xml_df))).values

    combined = xml_df.copy()

    # Optional notice annotations
    nc_csv = raw_dir / f"{date_str}_notice.csv"
    nc_xml = raw_dir / f"{date_str}_notice.xml"
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
    """Join CICFlowMeter CSV with annotations via DuckDB, write labeled Parquet."""
    flow_csvs = list(flows_dir.glob("*_Flow.csv"))
    if not flow_csvs:
        logger.warning(f"  No *_Flow.csv found in {flows_dir}")
        return False

    flows_glob = str(flows_dir / "*_Flow.csv").replace("\\", "/")
    ann_path = str(annotations_parquet).replace("\\", "/")
    out_path = str(output_path).replace("\\", "/")

    conn = duckdb.connect(":memory:")

    conn.execute(f"""
        CREATE OR REPLACE VIEW v_annotations AS
        SELECT
            *,
            (CASE WHEN start IS NOT NULL THEN 1 ELSE 0 END +
             CASE WHEN stop IS NOT NULL THEN 1 ELSE 0 END +
             CASE WHEN src_ip IS NOT NULL THEN 1 ELSE 0 END +
             CASE WHEN src_port IS NOT NULL THEN 1 ELSE 0 END +
             CASE WHEN dst_ip IS NOT NULL THEN 1 ELSE 0 END +
             CASE WHEN dst_port IS NOT NULL THEN 1 ELSE 0 END +
             CASE WHEN proto IS NOT NULL THEN 1 ELSE 0 END) AS feature_count,
            CASE
                WHEN start IS NOT NULL AND stop IS NOT NULL THEN stop - start
                ELSE NULL
            END AS duration
        FROM '{ann_path}'
        ORDER BY feature_count DESC, duration ASC NULLS LAST;
    """)

    conn.execute(f"""
        CREATE OR REPLACE VIEW v_flows AS
        SELECT *,
            CASE
                WHEN Protocol = 6 THEN 'tcp'
                WHEN Protocol = 17 THEN 'udp'
                WHEN Protocol = 1 THEN 'icmp'
                WHEN Protocol = 2 THEN 'igmp'
                WHEN Protocol = 4 THEN 'ip'
                WHEN Protocol = 41 THEN 'ipv6'
                WHEN Protocol = 58 THEN 'icmpv6'
                WHEN Protocol = 89 THEN 'ospf'
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
                    (a.start IS NULL OR epoch(f."Timestamp") >= a.start) AND
                    (a.stop  IS NULL OR epoch(f."Timestamp") <= a.stop)  AND
                    (a.src_ip   IS NULL OR f."Src IP"   = a.src_ip)      AND
                    (a.src_port IS NULL OR f."Src Port" = a.src_port)    AND
                    (a.dst_ip   IS NULL OR f."Dst IP"   = a.dst_ip)      AND
                    (a.dst_port IS NULL OR f."Dst Port" = a.dst_port)    AND
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
# Per-day and per-year orchestration
# ---------------------------------------------------------------------------

def process_day(
    day: date,
    work_dir: Path,
    memory_gb: int,
    max_heap_gb: int | None,
    raw_source_dir: Path,
    packets_per_chunk: int = 200_000,
) -> Path | None:
    """Full pipeline for a single day. Returns path to day Parquet or None."""
    date_str = day.strftime("%Y%m%d")
    day_dir = work_dir / date_str
    day_dir.mkdir(parents=True, exist_ok=True)

    labeled_parquet = day_dir / "labeled.parquet"
    if labeled_parquet.exists():
        logger.info(f"  Day {date_str} already processed.")
        return labeled_parquet

    logger.info(f"--- Day {date_str} ---")

    # Step 1: Prepare local raw files if available
    pcap_dir = day_dir / "pcap"
    ann_raw_dir = day_dir / "annotations_raw"
    ann_date_dir = ann_raw_dir / date_str
    _populate_day_inputs_from_local(day, raw_source_dir, pcap_dir, ann_date_dir)

    # Step 2: Download PCap if missing
    pcap = download_pcap(day, pcap_dir)
    if pcap is None:
        logger.warning(f"  PCap download failed for {date_str}")
        return None

    # Step 3: Download annotations if missing
    ann_raw_dir.mkdir(exist_ok=True)
    ann_date_dir.mkdir(exist_ok=True)
    if not download_annotations(day, ann_date_dir):
        logger.warning(f"  Annotation download failed for {date_str}")
        return None

    # Step 3: Run CICFlowMeter
    flows_dir = day_dir / "flows"
    flows_dir.mkdir(exist_ok=True)
    if not run_cicflowmeter(pcap, flows_dir, memory_gb, max_heap_gb, packets_per_chunk):
        logger.warning(f"  CICFlowMeter failed for {date_str}")
        return None

    # Step 4: Build annotations Parquet
    ann_parquet = day_dir / "annotations.parquet"
    if not build_annotations_parquet(ann_date_dir, ann_parquet):
        logger.warning(f"  Annotation processing failed for {date_str}")
        return None

    # Step 5: Join flows + annotations
    if not combine_flows_annotations(flows_dir, ann_parquet, labeled_parquet, day.year):
        logger.warning(f"  Flow+annotation join failed for {date_str}")
        return None

    # Cleanup large intermediate files to save space
    shutil.rmtree(pcap_dir, ignore_errors=True)
    shutil.rmtree(flows_dir, ignore_errors=True)

    return labeled_parquet


def find_working_day(year: int, month: int = 1) -> list[date]:
    """Return candidate days in given month."""
    candidates = []
    for d in CANDIDATE_DAYS:
        try:
            candidates.append(date(year, month, d))
        except ValueError:
            pass
    return candidates


def _has_local_day(day: date, raw_source_dir: Path) -> bool:
    """Return True if the raw source dir contains a local folder for this day."""
    return raw_source_dir.exists() and (raw_source_dir / day.strftime("%Y%m%d")).is_dir()


def _prioritize_local_candidates(
    candidates: list[date], raw_source_dir: Path
) -> list[date]:
    """Return candidates ordered with local raw days first."""
    if not raw_source_dir.exists():
        return candidates

    local_days = [day for day in candidates if _has_local_day(day, raw_source_dir)]
    remote_days = [day for day in candidates if day not in local_days]
    if local_days:
        logger.info(
            f"  Prioritizing {len(local_days)} local candidate day(s) from {raw_source_dir}"
        )
    return local_days + remote_days


def build_year_parquet(
    year: int,
    output_dir: Path,
    work_dir: Path,
    days_per_year: int,
    memory_gb: int,
    max_heap_gb: int | None,
    raw_source_dir: Path,
    packets_per_chunk: int = 200_000,
) -> bool:
    """Process up to days_per_year days for the given year and merge into one Parquet."""
    year_parquet = output_dir / f"{year}.parquet"
    if year_parquet.exists():
        logger.info(f"[{year}] Already done: {year_parquet}")
        return True

    logger.info(f"\n{'=' * 60}")
    logger.info(f"[{year}] Processing up to {days_per_year} day(s)")
    logger.info(f"{'=' * 60}")

    # Gather candidate dates across multiple months if needed
    candidates: list[date] = []
    for month in range(1, 13):
        candidates.extend(find_working_day(year, month))
        if len(candidates) >= days_per_year * 3:
            break

    candidates = _prioritize_local_candidates(candidates, raw_source_dir)

    day_parquets: list[Path] = []
    for candidate in candidates:
        if len(day_parquets) >= days_per_year:
            break
        result = process_day(
            candidate, work_dir / str(year), memory_gb, max_heap_gb, raw_source_dir,
            packets_per_chunk,
        )
        if result is not None:
            day_parquets.append(result)

    if not day_parquets:
        logger.error(f"[{year}] No days processed successfully")
        return False

    # Merge daily Parquets into annual file
    logger.info(f"[{year}] Merging {len(day_parquets)} day(s) -> {year_parquet.name}")
    frames = [pd.read_parquet(p) for p in day_parquets]
    merged = pd.concat(frames, ignore_index=True)
    merged.to_parquet(year_parquet, index=False, compression="snappy")
    logger.info(f"[{year}] Done: {len(merged):,} rows -> {year_parquet}")
    return True


# ---------------------------------------------------------------------------
# Download-only helpers (pipeline phase 1)
# ---------------------------------------------------------------------------

def download_day_raw(day: date, raw_dir: Path) -> bool:
    """Download PCap + annotations for one day into raw_dir/{YYYYMMDD}/.

    Saves files in the layout expected by process_mawiflow_dataset.py:
      {YYYYMMDD}/{YYYYMMDD}1400.pcap
      {YYYYMMDD}/{YYYYMMDD}_anomalous_suspicious.csv
      {YYYYMMDD}/{YYYYMMDD}_anomalous_suspicious.xml
      {YYYYMMDD}/{YYYYMMDD}_notice.csv   (optional)
      {YYYYMMDD}/{YYYYMMDD}_notice.xml   (optional)
    """
    date_str = day.strftime("%Y%m%d")
    day_dir  = raw_dir / date_str

    pcap_ok = (day_dir / f"{date_str}1400.pcap").exists() or \
              (day_dir / f"{date_str}1400.pcap.gz").exists()
    ann_ok  = (day_dir / f"{date_str}_anomalous_suspicious.csv").exists() and \
              (day_dir / f"{date_str}_anomalous_suspicious.xml").exists()

    if pcap_ok and ann_ok:
        logger.info(f"  Day {date_str} already downloaded.")
        return True

    day_dir.mkdir(parents=True, exist_ok=True)

    if not pcap_ok:
        if download_pcap(day, day_dir) is None:
            return False

    if not ann_ok:
        if not download_annotations(day, day_dir):
            return False

    return True


def _scan_year_in_raw(raw_dir: Path, year: int) -> tuple[list[date], list[date]]:
    """Scan raw_dir for day folders of the given year.

    Returns (complete, partial) where:
      complete — folders with both PCap and required annotations
      partial  — folders with annotations but missing PCap (need PCap download only)
    """
    complete: list[date] = []
    partial:  list[date] = []
    if not raw_dir.exists():
        return complete, partial
    year_prefix = str(year)
    for entry in sorted(raw_dir.iterdir()):
        if not (entry.is_dir() and entry.name.startswith(year_prefix) and len(entry.name) == 8):
            continue
        try:
            d = date(int(entry.name[:4]), int(entry.name[4:6]), int(entry.name[6:8]))
        except ValueError:
            continue
        ds = entry.name
        pcap_ok = (entry / f"{ds}1400.pcap").exists() or (entry / f"{ds}1400.pcap.gz").exists()
        ann_ok  = (entry / f"{ds}_anomalous_suspicious.csv").exists() and \
                  (entry / f"{ds}_anomalous_suspicious.xml").exists()
        if pcap_ok and ann_ok:
            complete.append(d)
        elif ann_ok:
            partial.append(d)
    return complete, partial


def build_year_downloads(year: int, raw_dir: Path, days_per_year: int) -> bool:
    """Download raw files for up to days_per_year days of the given year.

    First checks for already-complete day folders in raw_dir. Only downloads
    if the existing count is below days_per_year. Tries CANDIDATE_DAYS across
    ALL 12 months (fixing the original bug where the month loop broke based on
    candidate count instead of successful downloads).
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"[{year}] Checking / downloading up to {days_per_year} day(s)")
    logger.info(f"{'=' * 60}")

    complete, partial = _scan_year_in_raw(raw_dir, year)

    if len(complete) >= days_per_year:
        logger.info(f"[{year}] {len(complete)} day(s) already complete — skipping download.")
        return True

    if complete:
        logger.info(f"[{year}] {len(complete)} complete; need {days_per_year - len(complete)} more.")

    # Complete partial folders first (annotations present, PCap missing).
    downloaded = len(complete)
    for day in partial:
        if downloaded >= days_per_year:
            break
        date_str = day.strftime("%Y%m%d")
        day_dir  = raw_dir / date_str
        pcap_ok  = (day_dir / f"{date_str}1400.pcap").exists() or \
                   (day_dir / f"{date_str}1400.pcap.gz").exists()
        if not pcap_ok:
            logger.info(f"  Completing partial day {date_str}: downloading PCap only ...")
            if download_pcap(day, day_dir) is not None:
                downloaded += 1
            else:
                logger.warning(f"  PCap download failed for {date_str}")
        else:
            downloaded += 1

    # Then try fresh CANDIDATE_DAYS across all 12 months for remaining slots.
    if downloaded < days_per_year:
        candidates: list[date] = []
        for month in range(1, 13):
            candidates.extend(find_working_day(year, month))
        candidates = _prioritize_local_candidates(candidates, raw_dir)

        for candidate in candidates:
            if downloaded >= days_per_year:
                break
            if download_day_raw(candidate, raw_dir):
                downloaded += 1

    if downloaded == 0:
        logger.error(f"[{year}] No days available")
        return False

    logger.info(f"[{year}] {downloaded} day(s) ready in {raw_dir}")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 1: download MAWIFlow raw files into --raw-dir/{YYYYMMDD}/. "
            "Phase 2: delegate processing to process_mawiflow_dataset.py. "
            "Use --download-only to stop after phase 1."
        )
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=YEARS_AVAILABLE,
        help="Years to download (default: 2007-2024)",
    )
    parser.add_argument(
        "--days-per-year",
        type=int,
        default=1,
        metavar="N",
        help="Representative days to download per year (default: 1)",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw/MAWIFlow",
        help="Directory for downloaded raw day folders (default: data/raw/MAWIFlow)",
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed/mawiflow",
        help="Output directory for processed per-year Parquets (default: data/processed/mawiflow)",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Stop after downloading raw files; skip the processing step.",
    )
    parser.add_argument(
        "--memory-gb",
        type=int,
        default=8,
        metavar="GB",
        help="JVM heap for CICFlowMeter in GB, passed to process_mawiflow_dataset.py (default: 8)",
    )
    parser.add_argument(
        "--max-heap-gb",
        type=int,
        default=None,
        metavar="GB",
        help="Hard cap for CICFlowMeter heap retries in GB.",
    )
    parser.add_argument(
        "--pcap-chunk-packets",
        type=int,
        default=200_000,
        metavar="N",
        help="Packets per chunk for OOM fallback, passed to process_mawiflow_dataset.py (default: 200000)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent

    def _abs(p: str) -> Path:
        return Path(p) if Path(p).is_absolute() else (repo_root / p).resolve()

    raw_dir       = _abs(args.raw_dir)
    processed_dir = _abs(args.processed_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Auto-discover years from existing raw_dir folders when --years not given.
    years = args.years
    if years == YEARS_AVAILABLE and raw_dir.exists():
        discovered = sorted({
            int(e.name[:4])
            for e in raw_dir.iterdir()
            if e.is_dir() and len(e.name) == 8 and e.name.isdigit()
        })
        if discovered:
            years = discovered
            logger.info(f"Auto-discovered years from {raw_dir}: {years}")

    logger.info(f"Raw dir:    {raw_dir}")
    logger.info(f"Years:      {years}")
    logger.info(f"Days/year:  {args.days_per_year}")

    # ------------------------------------------------------------------
    # Phase 1: Download raw files into raw_dir/{YYYYMMDD}/
    # ------------------------------------------------------------------
    failed_dl: list[int] = []
    for year in sorted(years):
        ok = build_year_downloads(
            year=year,
            raw_dir=raw_dir,
            days_per_year=args.days_per_year,
        )
        if not ok:
            failed_dl.append(year)

    done_dl = [y for y in years if y not in failed_dl]
    print(f"\n[download] {len(done_dl)}/{len(years)} years ready.")
    if failed_dl:
        print(f"[warn] Failed years: {failed_dl}")

    if args.download_only or not done_dl:
        return

    # ------------------------------------------------------------------
    # Phase 2: Process downloaded files via process_mawiflow_dataset.py
    # ------------------------------------------------------------------
    process_script = Path(__file__).parent / "process_mawiflow_dataset.py"
    processed_dir.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        sys.executable, str(process_script),
        "--input-dir",   str(raw_dir),
        "--output-dir",  str(processed_dir),
        "--memory-gb",   str(args.memory_gb),
    ]
    if args.max_heap_gb is not None:
        cmd += ["--max-heap-gb", str(args.max_heap_gb)]
    if sorted(years) != sorted(YEARS_AVAILABLE):
        cmd += ["--years"] + [str(y) for y in sorted(years)]
    if args.pcap_chunk_packets != 200_000:
        cmd += ["--pcap-chunk-packets", str(args.pcap_chunk_packets)]

    logger.info(f"Processing: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
