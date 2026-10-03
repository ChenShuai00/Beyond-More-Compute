# Samples and membership

The unit is a conference paper identified by `paper_id`. The paper index preserves venue, publication year, track and OpenAlex linkage. `data/corpus/sample_membership.parquet` joins the initial index to the corpus, eligible resource sample, hardware measurements and analysis memberships by paper ID.

| Stage | Papers |
|---|---:|
| Initial records | 114,618 |
| Collected corpus | 113,822 |
| Metadata/text eligible | 107,872 |
| Internal compute access (L1 positive) | 54,948 |
| Hardware-eligible: L1=1, L2=0, L3=0 | 49,090 |
| Hardware-eligible with normalized GPU | 40,956 |
| Complete hardware configuration | 34,483 |
| Complete configurations with generation-mean performance | 3,524 |
| Frozen citation analysis frame | 29,591 |
| Baseline and hardware-by-API sample | 29,487 |
| Resource-by-KSC sample | 29,239 |

L4 external API access can coexist with an eligible hardware configuration. The resource labels are nonexclusive. Complete hardware membership follows the frozen strict configuration-selection flag. The reference values and the ID joins are verified by `code/verify.py`.

The collected corpus includes 9,917 Findings papers. Conference counts aggregate the collected paper IDs. The resource sample contains 107,817 distinct linked OpenAlex Work IDs across 107,872 paper records.

`baseline_membership.parquet` and `ksc_membership.parquet` preserve the model-specific inclusion flags. Reproduction verifies the ordered sample-ID hashes, fixed-effect ranks and cluster counts. The baseline/API sample hash is `d6807e4979873472ffa7565585fbeba4d5951f0ac229c12a9e95f5a10513ffb7`; the KSC sample hash is `1b6b84deb2b845abca82df6d98abdae1d2e8d682005decb2759a2810247007cc`.
