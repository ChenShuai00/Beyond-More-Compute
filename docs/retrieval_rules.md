# Candidate retrieval rules

`configs/measurement/retrieval/` records the frozen candidate-retrieval configuration from the active `random_3000_2017_2025` annotation-sampling source snapshot. These supplementary measurement materials describe how resource and model-use mentions were recalled for review. The main reproduction command operates on the frozen analysis and evaluation inputs.

| File | Rows | Contents |
|---|---:|---|
| `candidate_recall_terms.parquet` | 32,706 | Candidate terms, candidate labels, recall tiers, matching flags and provenance |
| `candidate_recall_patterns.csv` | 35 | Candidate regular expressions with matching flags and provenance |
| `negative_patterns.csv` | 10 | Expressions for identifying mathematical wording, generic hardware descriptions or model mentions requiring review |
| `label_rules.csv` | 9 | Evidence context, label precedence, suppression and fallback rules |
| `label_definitions.csv` | 7 | Definitions, positive evidence and exclusion boundaries for L1–L7 |
| `source_manifest.json` | 5 source files | Source-relative filenames, source and output SHA-256 hashes, schemas, row counts and value-preserving field projection |

All 32,706 term-rule rows are retained. There are 32,696 distinct term strings and 17,639 distinct normalized term strings. Rule IDs distinguish entries with different provenance or matching semantics; term normalization does not collapse these entries.

## Candidate terms and patterns

The term table contains 541 L1 internal-resource rules, 323 L2 cloud-resource rules, 574 L3 public-HPC rules, 700 L4 external-model/API rules and 30,568 L5 open-model rules. Its tiers comprise 818 high-precision rules, 31,794 broad-recall rules and 94 weak-context rules. `recall_tier` and `confidence` express candidate-retrieval categories and review priority.

The 35 patterns contain seven L1 hardware patterns, 25 L6 ambiguous-compute patterns and three supplementary L5 reuse-context patterns. Their tiers comprise 22 high-precision, three broad-recall and ten weak-context patterns. The L5 additions recall use of released or pretrained models, actions such as loading or initializing pretrained artifacts, and reuse of components such as backbones, encoders, detectors and feature representations. The three rule IDs are `RECALL_L5_CONTEXT_000033`, `RECALL_L5_CONTEXT_000034` and `RECALL_L5_CONTEXT_000035`.

`term`, `normalized_term`, `pattern` and `normalized_pattern` retain the source strings exactly. `case_sensitive` and `word_boundary` retain the source matching flags. In the upstream term matcher, case-insensitive matching lowercases both text and term; a word-boundary check treats alphanumeric characters and underscores as word characters. The upstream regex matcher compiles the stored `pattern` string directly, with case-insensitive matching when requested. The exported strings preserve their existing backslash representation and are ready for that same direct compilation.

`source_group` and `source_table_name` identify the originating logical table. They replace the source `source_table` field, which combined a group and filename. The source manifest records this reversible mapping. Other scientific fields and notes are retained, including imported model identifiers and original evidence descriptions.

## Evidence and label boundaries

L1 denotes internal or directly reported concrete hardware; L2 denotes commercial cloud resources; L3 denotes public, national, university or shared academic HPC resources; L4 denotes external hosted model/API access. A concrete hardware mention maps to L1 unless cloud or public-HPC evidence identifies that same resource span. Cloud and public-HPC provenance take precedence over a generic internal or ambiguous resource interpretation. Organization or university affiliation alone does not establish resource access.

L5 denotes reuse of released, open-source or pretrained models or checkpoints. Its evidence includes using, fine-tuning, initializing from, loading or evaluating a reused artifact. A model name in related work or a comparison description requires use-context review. L5 can co-occur with resource-access labels. The source rule set records L5 to support candidate recall and measurement context; the released resource-access evaluation scores L1–L4.

L6 denotes a reported use of computational resources whose concrete hardware or provenance remains unclear. L7 is the exclusive fallback when none of L1–L6 is emitted. The label rules specify sentence-level evidence contexts and required action terms for positive matches.

Negative patterns flag computational-complexity statements, mathematical uses of “compute,” isolated FLOPs or memory metrics, hardware-aware descriptions, definitions of API, comparisons with named models, and contradictory pretrained/from-scratch wording. `suppresses_labels`, `confidence` and `notes` state the intended review boundaries. The rule order applies negative-pattern suppression before the no-positive-label fallback and retains suppressed candidate evidence for audit.

## Re-export the frozen rules

From the repository root, run:

```powershell
python code/prepare_retrieval.py --source-root <upstream-source-root>
```

The source root contains `data/annotation_gold/annotation_sampling/random_3000_2017_2025/rule_data/active/`. The script verifies each of the five frozen source hashes and row counts, writes Zstandard-compressed Parquet for the term table and UTF-8 CSV for the remaining tables, and checks every exported value after reading it back. `--output-dir` selects another export directory. `source_manifest.json` records the resulting file hashes and distributions by label, tier, confidence and matching type.
