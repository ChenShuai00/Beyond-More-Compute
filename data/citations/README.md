# Citation measurements

The frozen OpenAlex collection was made on 12 September 2026, 07:32:17--08:10:48 UTC. The release contains 29,591 paper measurements, 295,910 annual records for 2017--2026, and 2,239 publication-year by primary-topic reference cells.

Cumulative citations sum counts from the current OpenAlex publication year through 2026, including the publication year. Current cited_by_count determines the three top-cited indicators. The reference population is the OpenAlex core corpus of articles, conference papers and preprints; ties at the nearest-rank cutoff qualify. Self-citations are included.

reference_histograms.parquet contains the citation-count bins needed to recompute all cutoffs. counts_by_year_outside_window.parquet retains the preceding annual tail used in reconciliation. asset_sources.json records relative source paths and SHA-256 hashes.

Run `python code/citations.py` to verify the annual cumulative values, reference thresholds, top-cited indicators and baseline outcomes without network access.
