"""Regression tests for MAWIFlow join cardinality preservation.

Demonstrates that the old PARTITION BY "Flow ID" approach silently discards flows
when two distinct flows share the same Flow ID (same 5-tuple, different times),
and that the fixed PARTITION BY _flow_row_id approach preserves all flows.
"""

from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_db(conn):
    """Create v_annotations and v_flows views in the given DuckDB connection."""
    conn.execute("""
        CREATE TABLE raw_annotations (
            label       VARCHAR,
            taxonomy    VARCHAR,
            feature_count INT,
            duration    DOUBLE,
            start       DOUBLE,
            stop        DOUBLE,
            src_ip      VARCHAR,
            src_port    INTEGER,
            dst_ip      VARCHAR,
            dst_port    INTEGER,
            proto       VARCHAR
        );
    """)
    conn.execute("""
        CREATE VIEW v_annotations AS SELECT * FROM raw_annotations;
    """)

    # Two flows with identical 5-tuple (same "Flow ID") but different timestamps
    conn.execute("""
        CREATE TABLE raw_flows (
            "Timestamp"  DOUBLE,
            "Flow ID"    VARCHAR,
            "Src IP"     VARCHAR,
            "Src Port"   INTEGER,
            "Dst IP"     VARCHAR,
            "Dst Port"   INTEGER,
            "Protocol"   INTEGER,
            "Label"      VARCHAR
        );
    """)
    conn.execute("""
        INSERT INTO raw_flows VALUES
            (1000.0, 'A->B:80', '1.1.1.1', 12345, '2.2.2.2', 80, 6, 'BENIGN'),
            (2000.0, 'A->B:80', '1.1.1.1', 12345, '2.2.2.2', 80, 6, 'BENIGN');
    """)
    conn.execute("""
        CREATE VIEW v_flows AS
        SELECT *,
            CASE
                WHEN "Protocol" = 6  THEN 'tcp'
                WHEN "Protocol" = 17 THEN 'udp'
                ELSE 'unknown_' || CAST("Protocol" AS VARCHAR)
            END AS protocol_name,
            ROW_NUMBER() OVER (
                ORDER BY "Timestamp", "Flow ID", "Src IP", "Src Port",
                         "Dst IP", "Dst Port", "Protocol"
            ) AS _flow_row_id
        FROM raw_flows;
    """)


# ---------------------------------------------------------------------------
# Regression: old PARTITION BY "Flow ID" drops duplicate-ID flows
# ---------------------------------------------------------------------------

class TestOldPartitionDropsRows:
    """Document that PARTITION BY "Flow ID" is lossy when the 5-tuple repeats."""

    def test_old_partition_loses_second_flow(self):
        conn = duckdb.connect()
        _setup_db(conn)

        # Simulate the old (buggy) query with PARTITION BY "Flow ID"
        result = conn.execute("""
            WITH matches AS (
                SELECT
                    f.*,
                    a.label AS annotation_label,
                    a.taxonomy,
                    ROW_NUMBER() OVER (
                        PARTITION BY f."Flow ID"
                        ORDER BY a.feature_count DESC, a.duration ASC NULLS LAST
                    ) AS ann_rank
                FROM v_flows f
                LEFT JOIN v_annotations a ON
                    (a.start IS NULL OR f."Timestamp" >= a.start) AND
                    (a.stop  IS NULL OR f."Timestamp" <= a.stop)
            )
            SELECT COUNT(*) FROM matches WHERE ann_rank = 1 OR ann_rank IS NULL
        """).fetchone()[0]

        # With no annotations, a LEFT JOIN produces NULLs for ann_rank on every row.
        # ROW_NUMBER() never returns NULL, so "ann_rank IS NULL" is dead code.
        # PARTITION BY "Flow ID" groups both rows → only row with ann_rank=1 survives.
        # The second row (ann_rank=2) is silently dropped.
        assert result == 1, (
            "Old PARTITION BY 'Flow ID' should drop the second flow — "
            f"got {result} rows instead of 1"
        )

    def test_old_partition_dead_code_null_check(self):
        """ROW_NUMBER() never returns NULL, so 'ann_rank IS NULL' never fires."""
        conn = duckdb.connect()
        _setup_db(conn)

        null_count = conn.execute("""
            WITH matches AS (
                SELECT
                    ROW_NUMBER() OVER (PARTITION BY "Flow ID") AS ann_rank
                FROM v_flows
            )
            SELECT COUNT(*) FROM matches WHERE ann_rank IS NULL
        """).fetchone()[0]

        assert null_count == 0, "ROW_NUMBER() must never return NULL"


# ---------------------------------------------------------------------------
# Fix: PARTITION BY _flow_row_id preserves all flows
# ---------------------------------------------------------------------------

class TestFixedPartitionPreservesRows:
    """The fixed query uses _flow_row_id (unique per row) as the partition key."""

    def test_all_flows_preserved_no_annotations(self):
        conn = duckdb.connect()
        _setup_db(conn)

        result = conn.execute("""
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
                    (a.start IS NULL OR f."Timestamp" >= a.start) AND
                    (a.stop  IS NULL OR f."Timestamp" <= a.stop)
            )
            SELECT COUNT(*) FROM matches WHERE ann_rank = 1
        """).fetchone()[0]

        assert result == 2, (
            f"Fixed partition should preserve both flows; got {result}"
        )

    def test_all_flows_preserved_with_matching_annotation(self):
        conn = duckdb.connect()
        _setup_db(conn)

        # Add a wildcard annotation that matches all flows
        conn.execute("""
            INSERT INTO raw_annotations VALUES
                ('attack', 'scan', 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
        """)

        result = conn.execute("""
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
                    (a.start IS NULL OR f."Timestamp" >= a.start) AND
                    (a.stop  IS NULL OR f."Timestamp" <= a.stop)  AND
                    (a.src_ip   IS NULL OR f."Src IP"   = a.src_ip)  AND
                    (a.src_port IS NULL OR f."Src Port" = a.src_port) AND
                    (a.dst_ip   IS NULL OR f."Dst IP"   = a.dst_ip)  AND
                    (a.dst_port IS NULL OR f."Dst Port" = a.dst_port) AND
                    (a.proto    IS NULL OR f.protocol_name = a.proto)
            )
            SELECT COUNT(*) FROM matches WHERE ann_rank = 1
        """).fetchone()[0]

        assert result == 2, (
            f"Fixed partition should preserve both flows even with annotations; got {result}"
        )

    def test_cardinality_equality(self):
        """n_out must equal row_count for any valid run."""
        conn = duckdb.connect()
        _setup_db(conn)

        row_count = conn.execute("SELECT COUNT(*) FROM v_flows").fetchone()[0]

        n_out = conn.execute("""
            WITH matches AS (
                SELECT
                    f.*,
                    a.label AS annotation_label,
                    ROW_NUMBER() OVER (
                        PARTITION BY f._flow_row_id
                        ORDER BY a.feature_count DESC, a.duration ASC NULLS LAST
                    ) AS ann_rank
                FROM v_flows f
                LEFT JOIN v_annotations a ON (a.start IS NULL)
            )
            SELECT COUNT(*) FROM matches WHERE ann_rank = 1
        """).fetchone()[0]

        assert n_out == row_count, (
            f"Cardinality mismatch: {row_count} loaded, {n_out} written"
        )

    def test_flow_row_id_is_unique(self):
        """_flow_row_id must assign a distinct value to every row."""
        conn = duckdb.connect()
        _setup_db(conn)

        total = conn.execute("SELECT COUNT(*) FROM v_flows").fetchone()[0]
        distinct = conn.execute("SELECT COUNT(DISTINCT _flow_row_id) FROM v_flows").fetchone()[0]

        assert total == distinct == 2
