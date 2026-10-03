# Measurement materials

## Resource access

Four binary labels describe reported access to computational resources: internal resources (L1), commercial cloud resources (L2), public high-performance computing resources (L3), and external model/API resources (L4). The compact label schema, classification instruction and few-shot examples are in `configs/measurement/classification/`. Prompts also retain open-model (L5) and no-specific-resource (L0) labels. Candidate matching terms, positive patterns and exclusion rules are in `configs/measurement/retrieval/`; their specification is in [retrieval_rules.md](retrieval_rules.md).

Evaluation inputs retain paper/window IDs, binary labels, predicted labels, scores and thresholds. `code/measurement/run.py` joins labels and predictions by the saved IDs and computes confusion counts, precision, recall, F1, micro scores and macro F1 over L1–L4. It also verifies paper-level separation among training, candidate, calibration and test memberships. Model specifications and the production-model weight hash are stored with the configurations.

## GPU extraction and capacity

The extraction evaluation contains 2,698 annotated records and saved predictions for thinking-enabled and thinking-disabled modes. The scorer verifies prediction schema, record detection, hardware model, quantity and joint extraction, using both surface and normalized model matching. Prompts, JSON schema and reviewed normalization rules are included in `configs/measurement/gpu/`.

The released normalization mapping contains 5,752 observed GPU-name entries and a 191-row hardware-specification catalog. Model-level performance supplementation contains 132 models and 11 generations with observed donors. The rule uses the unweighted arithmetic mean of originally observed model capabilities in the exact generation, before the log transform. The donor table and model-level flags preserve this computation.

Paper-level capacity is the theoretical capacity of the selected reported GPU configuration. The selection policy, tie rules, per-GPU capability choice and VRAM rules are in `configuration_selection.json`. Regression dimensions are centered log2 GPU quantity, centered log2 per-GPU capability, and centered log2 per-GPU memory.

## Knowledge-space complexity

KSC measures semantic dispersion in the focal primary topic during the preceding five publication years. The frozen contract, focal-paper mapping, topic-year summaries and threshold robustness tables are released. The primary historical reference threshold is 100, with 50 and 200 as additional thresholds. The checker validates historical-span sums and means, historical-pool counts, focal self-exclusion, missingness and coverage.

The formal release covers 29,591 focal papers, 2,167 topic-year cells and 731 primary topics. Primary KSC is available for 29,322 focal papers. Its standardized regression value uses the mean and population standard deviation of those available focal values.

## Citation outcomes

The citation snapshot was collected on 12 September 2026. Citation counts and reference distributions are in `data/citations/`. The citation contract specifies the annual cumulative window and publication-year/primary-topic reference population. The offline checker regenerates percentile thresholds from count histograms, gives papers tied at each cutoff equal credit, and joins all four outcome variables to the analysis frame.

The four model outcomes are log(1 + cumulative citations), top-10%, top-5% and top-1% citation indicators. The reference thresholds, counts and annual reconciliation fields provide the measurement trail for each paper.
