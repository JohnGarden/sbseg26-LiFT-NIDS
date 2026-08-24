<!-- markdownlint-disable -->
# MAWIFlow-subset — data manifest

**Generated:** 2026-06-10T20:10:41 by `scripts/generate_data_manifest.py`. **Audit verdict:** aprovado_com_ressalvas (approved_for_final_experiments=True, audited 2026-05-01).

**Sampling policy.** Systematic temporal sampling of the TheLurps/MAWIFlow manifest: one capture day per quarter per year (preferred months 01/04/07/10; nearest available day when a preferred one is missing upstream). Results are evaluations on MAWIFlow-subset, not a full MAWIFlow/IM28 reproduction (see the README 'MAWIFlow' section).

**Scope:** 18 years (2007–2024), 74 capture days, 9,946,112 flows, 89 numeric features.

| Year | Capture days | Flows | Benign | Anomalous | Attack prev. |
|---|---|---:|---:|---:|---:|
| 2007 | 20070101, 20070102, 20070401, 20070701, 20071001 | 2,718,818 | 1,702,996 | 1,015,822 | 37.4% |
| 2008 | 20080101, 20080401, 20080701, 20081001 | 512,856 | 337,951 | 174,905 | 34.1% |
| 2009 | 20090101, 20090401, 20090701, 20091001 | 2,178,006 | 1,331,458 | 846,548 | 38.9% |
| 2010 | 20100101, 20100401, 20100701, 20101001 | 208,334 | 97,453 | 110,881 | 53.2% |
| 2011 | 20110101, 20110401, 20110701, 20111001 | 1,408,353 | 357,499 | 1,050,854 | 74.6% |
| 2012 | 20120101, 20120401, 20120701, 20121001 | 785,775 | 214,521 | 571,254 | 72.7% |
| 2013 | 20130102, 20130402, 20130708, 20131008 | 947,552 | 620,870 | 326,682 | 34.5% |
| 2014 | 20140108, 20140408, 20140701, 20141001 | 29,442 | 16,502 | 12,940 | 44.0% |
| 2015 | 20150101, 20150401, 20150701, 20151002 | 51,972 | 8,129 | 43,843 | 84.4% |
| 2016 | 20160103, 20160402, 20160701, 20161001 | 12,026 | 11,400 | 626 | 5.2% |
| 2017 | 20170101, 20170403, 20170720, 20171001 | 6,826 | 6,686 | 140 | 2.1% |
| 2018 | 20180122, 20180401, 20180615, 20180701, 20181001 | 12,102 | 10,366 | 1,736 | 14.3% |
| 2019 | 20190101, 20190401, 20190701, 20191001 | 11,313 | 9,826 | 1,487 | 13.1% |
| 2020 | 20200102, 20200401, 20200701, 20201001 | 13,676 | 4,262 | 9,414 | 68.8% |
| 2021 | 20210103, 20210401, 20210701, 20211001 | 31,871 | 26,316 | 5,555 | 17.4% |
| 2022 | 20220101, 20220102, 20220104, 20220105 | 28,397 | 23,553 | 4,844 | 17.1% |
| 2023 | 20230108, 20230115, 20230401, 20230701 | 970,809 | 132,612 | 838,197 | 86.3% |
| 2024 | 20240101, 20240421, 20240702, 20241119 | 17,984 | 9,136 | 8,848 | 49.2% |

**Deviations from the 1-day-per-quarter policy (realized scope):**

- **2007**: 5 days, quarters covered: Q1Q2Q3Q4
- **2018**: 5 days, quarters covered: Q1Q2Q3Q4
- **2022**: 4 days, quarters covered: Q1
- **2023**: 4 days, quarters covered: Q1Q2Q3

Row counts cross-checked against `data/processed/mawiflow/<year>.parquet` metadata: **all match**.

Reconstruction: `uv run python scripts/download_mawiflow.py --years 2007 ... 2024 --months 01 04 07 10 --max-days 1` — note that day substitution is resolved against upstream availability at download time; the authoritative day list is the table above.
