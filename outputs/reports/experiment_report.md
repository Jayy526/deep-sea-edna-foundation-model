# Experiment Report — Stage 1

## Can raw deep-sea eDNA short reads be processed through a genomic
## foundation-model pipeline to obtain useful genomic embeddings?

**A TaxDistill-inspired adaptation for short deep-sea eDNA reads.**
This is not a reproduction of TaxDistill. See `docs/METHODOLOGY.md` §2.

> **Stage 2 followed this work.** Because the supervised branch is blocked (§I), a separate unsupervised community-structure analysis was carried out over the full dataset — dereplication, diversity, and abundance-weighted clustering. See [`community_report.md`](community_report.md).

Seed `42` · Python 3.13.14 · torch 2.11.0+cu128 · transformers 5.15.0 · NVIDIA GeForce RTX 4050 Laptop GPU

---

## A. Dataset statistics

Measured over the **full dataset**, not a sample.

| Metric | Value |
|---|---|
| Source file | `SRR26872904.fasta/SRR26872904.fasta` |
| File size | 648.87 MB |
| Total sequence records | 3,026,920 |
| Unique sequence IDs | 1,513,460 |
| Records per ID | exactly 2, max multiplicity 2 |
| Total sequenced bases | 457,064,920 |
| Length — min / max | 151 / 151 bp |
| Length — mean / median | 151.0 / 151 bp |
| Length — standard deviation | 0.0 |
| Distinct read lengths | 1 |
| GC content — mean | 51.313% |
| GC content — median | 50.993% |
| GC content — standard deviation | 0.04028 |
| GC content — 5th–95th percentile | 43.71% – 56.29% |
| Reads with ambiguous bases | 5,804 (0.192%) |
| Total ambiguous bases | 6,263 |
| Reads with invalid characters | 0 |
| Reads passing QC | 3,018,522 (99.723%) |
| Reads removed by QC | 8,398 |

**Nucleotide composition (all sequenced bases):**

| Base | Count | Share |
|---|---|---|
| G | 120,921,443 | 26.45608% |
| A | 118,717,693 | 25.97392% |
| C | 114,204,531 | 24.98650% |
| T | 103,214,990 | 22.58213% |
| N | 6,263 | 0.00137% |

**Why every sequence ID occurs twice — measured, not assumed:**

| Evidence | Measured |
|---|---|
| Repeats adjacent in file order | 100.00% |
| Repeats on the same flowcell tile | 100.00% |
| Repeats with identical GC content | 1.85% |

Same cluster coordinate, adjacent in the file, but different sequence content: these are the two mates (R1/R2) of one paired-end cluster written interleaved. The repeated IDs are **not duplicated data** — both records are distinct, real sequence.

## B. Preprocessing decisions

Full dataset streamed in **78.9 s** (38,353 records/s, 8.2 MB/s). At most one record is held in memory at a time; streaming the entire 649 MB file grew resident memory by only **58.85 MB**.

Absolute RSS figures are dominated by imported libraries (the CLI imports torch and transformers to record the environment), which is why the growth attributable to streaming is the number quoted. The whole stage peaks at 1490.5 MB because a second, deliberate pass reads the metadata columns back in to compute *exact* order statistics (median, percentiles) rather than approximating them on the fly.

| Filter | Setting |
|---|---|
| `min_length` | `100` |
| `max_length` | `None` |
| `valid_alphabet` | `ACGT` |
| `ambiguity_policy` | `longest_unambiguous_run` |
| `max_ambiguous_fraction` | `0.1` |
| `min_effective_length` | `100` |
| `max_single_base_fraction` | `0.9` |
| `min_gc` | `None` |
| `max_gc` | `None` |

| QC outcome | Reads |
|---|---|
| pass | 3,018,522 |
| below min effective length | 131 |
| low complexity single base | 8,267 |

**Ambiguity policy — `longest_unambiguous_run`.** GenomeOcean's BPE vocabulary covers ACGT only, so an `N` would become `[UNK]`. This policy keeps the longest contiguous ACGT stretch of each read. It never substitutes, imputes, or pads. The cost is measured: mean effective length 150.9913 bp of 151.

## C. Short reads vs long contigs — the central methodological difference

| | TaxDistill (paper) | This work |
|---|---|---|
| Input | assembled contigs | raw sequencing reads |
| Length filter | **≥ 2,000 bp**, strictly applied | **none** |
| Typical length | ≥ 2,000 bp | 151 bp, uniformly |
| 4-mers per sequence | ≥ 1,997 | **148** |
| Assembly available | yes | no |
| Abundance across samples | yes | no (single sample) |

**The paper's ≥ 2,000 bp filter was deliberately NOT applied.** At that threshold, 0 of 3,026,920 reads survive and the pipeline has no input.

**Reads were NOT padded to 2,000 bp.** Padding a 151 bp read to 2,000 bp would make 92.5% of the resulting sequence non-biological; whatever the model computed would be largely a function of the padding scheme. No padding is applied at any stage of this pipeline — only within-batch padding for tensor shape, which the attention mask excludes from pooling entirely.

**Reads were not treated as contigs and were never concatenated.** No genomic context is invented.

The paper's own justification for its filter is that it 'ensures deep representation learning can capture sufficient contextual semantic information.' Our reads have an order of magnitude less context. Whether a pretrained genomic language model degrades more gracefully than tetranucleotide frequency in that regime is exactly what this experiment measures.

## D. TaxDistill architecture vs our adaptation

| Component | TaxDistill | This work | Label |
|---|---|---|---|
| Teacher backbone | GenomeOcean, frozen | GenomeOcean-500M, frozen — same checkpoint | PAPER-DERIVED |
| Teacher head | learnable classification head | none (no labels to train it on) | ADAPTATION |
| Student | Taxometer MLP: TNF + K abundances + total | canonical 4-mer/TNF vector, no abundance | ADAPTATION |
| Knowledge distillation | KD loss, soft labels, temperature | **not implemented** — requires labels | OUT OF SCOPE |
| Hierarchical loss | deep hierarchical loss over taxonomy tree | **not implemented** — requires a taxonomy | OUT OF SCOPE |
| Pooling | unspecified in the paper | attention-masked mean over final layer | IMPLEMENTATION DECISION |
| Evaluation | species-level F1 vs CAMI2 ground truth | label-free mate-pair retrieval + structure | ADAPTATION |

Only the **upper/representation stage** is in scope: raw reads → deep genomic embeddings, plus the analysis needed to judge whether those embeddings are usable.

## E. Foundation model / encoder used

**`pGenomeOcean/GenomeOcean-500M`** — the actual model from the paper, not a substitute.

| Property | Value |
|---|---|
| Architecture | MistralForCausalLM |
| Hidden layers | 14 |
| Vocabulary | 4,096 BPE tokens over ACGT |
| Total parameters | 541,109,760 |
| Trainable parameters | 0 |
| Frozen parameters | 541,109,760 |
| Precision | torch.bfloat16 |
| Attention implementation | sdpa |
| Device | cuda |
| Pooling | mean over the final hidden layer, attention-masked |

**Compatibility with 151 bp reads — measured on real reads before bulk inference:**

| Check | Measured |
|---|---|
| Reads probed | 1,000 |
| Tokens per read (min / mean / max) | 28 / 33.642 / 37 |
| Base pairs per token | 4.488 |
| Model token limit | 1,024 |
| Fraction of limit used | 3.61% |
| Fits without truncation | **True** |
| `[UNK]` tokens produced | 0 |
| Padding to a target length | False |

The model accepts our reads directly. No substitute encoder was needed and none is claimed to be GenomeOcean.

**Caveat stated plainly:** 151 bp is well within spec for *input length* but out of distribution with respect to the *context* GenomeOcean was pretrained on (assembled genomic sequence). Length compatibility is not representational suitability. Testing that gap is the experiment.

## F. Embedding dimensions and extraction

| | Foundation model | Baseline |
|---|---|---|
| Representation | GenomeOcean-500M | canonical 4-mer / TNF |
| Embedding dimension | **1,536** | **137** |
| Reads requested | 200,000 | 200,000 |
| Reads processed | 200,000 | 200,000 |
| Reads successful | 200,000 | 200,000 |
| Reads failed | 0 | 0 |
| Batch size | 64 | 50,000 |
| Processing time | 460.7 s | 13.0 s |
| Throughput | 434.1 reads/s | 15,364.7 reads/s |
| Time per read | 2.3034 ms | 0.0651 ms |
| Storage | 1,172.5 MB | 104.9 MB |
| Bytes per embedding | 6,148 | 550 |
| Peak GPU memory | 1,178 MB of 6,140 MB | n/a (CPU) |

**Embedding health — verified, not assumed:**

| Check | Foundation model | Baseline |
|---|---|---|
| Shape | [200000, 1536] | [200000, 137] |
| All values finite | True | True |
| NaN / Inf count | 0 / 0 | 0 / 0 |
| Dead (zero-variance) dimensions | 0 | 0 |
| Value range | [-2.984375, 2.640625] | [0.0, 0.92053] |
| Mean L2 norm | 25.079506 | 0.528198 |

**Small-subset test run first.** 2,000 reads were pushed through the complete pipeline (parse → QC → tokenise → encode → store → PCA) before any full-scale run. Verdict: *All test-run checks passed; safe to scale up.*

## G. Baseline representation

Canonical 4-mer frequency vector (136 reverse-complement-collapsed k-mers + GC content = 137 dimensions), 0 learned parameters.

Two deliberate departures from the paper's student input, both stated as adaptations rather than omissions:

1. **No VAMB 103-d projection.** That projection decorrelates features for VAMB's variational autoencoder. Here the vector feeds PCA and a linear probe, for which raw canonical frequencies are the more faithful baseline.
2. **No abundance features.** Abundance requires multiple samples mapped against a shared assembly. We have one sample of unassembled reads, so these are **not computable** — not omitted for convenience.

A 151 bp read yields 148 4-mers spread across 136 canonical bins — roughly one observation per bin. This sparsity is intrinsic to short reads and is the central reason the comparison is informative.

## H. Embedding analysis

| Metric | Foundation model | Baseline |
|---|---|---|
| Reads encoded | 200,000 | 200,000 |
| Nominal dimension | 1,536 | 137 |
| Variance in PC1 | 20.50% | 26.55% |
| Variance in PC1–PC2 | 38.50% | 44.76% |
| Variance in first 10 PCs | 78.11% | 79.41% |
| PCs for 90% variance | 31 | 26 |
| PCs for 95% variance | > 50 (not reached) | 45 |
| Participation ratio | 8.0611 | 7.0001 |

**Cluster tendency vs a dimension-shuffled null.** Silhouette on k-means labels is self-referential on its own, so each value is paired with the same statistic on a null that preserves every marginal distribution while destroying joint structure. Only the gap is interpretable.

| Representation | Best k | Real silhouette | Null silhouette | Gap |
|---|---|---|---|---|
| Foundation model | 20 | 0.6087 | -0.0004 | **0.6091** |
| Baseline | 15 | 0.6367 | 0.0073 | **0.6294** |

These clusters have **no verified taxonomic meaning**. This is a clusterability statistic, not a classification result.

UMAP visualization sample: **20,000 reads** (n_neighbors=30, min_dist=0.1, seed=42). This is the visualization sample size, distinct from the 200,000-read encoding subset and from the 3,026,920-record full dataset.

## I. Classification results

**Not available with the current dataset/experimental setup.**

This is a finding, not a gap left unexplored. Three independent label sources were checked:

| Source checked | Result |
|---|---|
| Configured label file | not configured |
| FASTA headers | 20,000 inspected, 0 contain any taxonomic field |
| Sibling annotation files | 2 candidate(s), none a taxonomy table |

Every header has exactly 3 tokens — run accession, Illumina cluster coordinate, `length=151`. Example: `SRR26872904.1 LH00271:11:225VV3LT3:5:1101:23320:1032 length=151`

Consequently the following are **not computable** and are **not estimated, simulated, or filled in**: accuracy, macro precision, macro recall, macro F1, weighted F1, per-class precision/recall/F1, confusion matrix, training/validation loss and accuracy curves, confidence histogram, reliability diagram, Brier score, Expected Calibration Error, and supervised silhouette / intra-class / inter-class distances.

The classifier (`models/classifier.py`) and its full metric suite are **implemented and runnable**; they were **not executed** because there is no valid input. Supply `classification.labels_path` and the supervised branch runs unchanged.

**What would be needed:**

- A per-read taxonomic assignment table (read_id -> taxon) produced by an independent reference-based classifier such as Kraken2, MMseqs2 or Metabuli -- the same tools TaxDistill corrects the output of.
- OR a mock-community / CAMI2-style dataset where the true source genome of every read is known by construction.
- OR curated reference amplicon sequences (e.g. SILVA/PR2 for 18S) with assignments at a stated confidence threshold.
- In every case the label source, its version, and its own error rate must be recorded, because TaxDistill's premise is that these labels are noisy.

## J. Confidence and calibration results

**Not available with the current dataset/experimental setup.** — calibration is a property of a trained classifier's predicted probabilities. With no classifier (§I), there are no probabilities to calibrate.

Noted for the record: low classification confidence would **not** constitute evidence of a novel species. Novel-taxa discovery is outside the scope of this implementation and no such inference is drawn anywhere in this work.

## K. Baseline vs foundation model

**Fairness protocol.** Both representations were computed from the same reads (`subset_main.parquet`), after the same QC, with the same seed (42), and scored with the same metrics on the same samples. Reads compared: 200,000; paired-end reads: 50,000.

### K.1 Supervised metrics

| Metric | Baseline | Foundation model |
|---|---|---|
| Accuracy | Not available with the current dataset/experimental setup. | Not available with the current dataset/experimental setup. |
| Macro precision | Not available with the current dataset/experimental setup. | Not available with the current dataset/experimental setup. |
| Macro recall | Not available with the current dataset/experimental setup. | Not available with the current dataset/experimental setup. |
| Macro F1 | Not available with the current dataset/experimental setup. | Not available with the current dataset/experimental setup. |
| Weighted F1 | Not available with the current dataset/experimental setup. | Not available with the current dataset/experimental setup. |

### K.2 Label-free retrieval benchmark

The two mates of a paired-end cluster are non-overlapping reads from the **same source DNA fragment**, hence the same source organism. Given R1's embedding, is R2 ranked above random distractors? This is measured structure we did not invent.

Pairs: 22,560 · distractors per query: 99 · chance top-1: 1.0%

| Metric | Baseline | Foundation model | Better |
|---|---|---|---|
| Top-1 retrieval accuracy | 2.9344% | 2.1587% | baseline |
| Top-5 retrieval accuracy | 9.5124% | 8.2181% | baseline |
| Top-10 retrieval accuracy | 13.4353% | 15.8245% | foundation model |
| Mean reciprocal rank | 0.0793 | 0.0774 | baseline |
| AUROC (mate vs random) | 0.5167 | 0.6200 | foundation model |
| Effect size (Cohen's d) | 0.1744 | 0.4505 | foundation model |

*Caveat carried through from the methodology:* Measures same-fragment / same-locus signal, not taxonomic accuracy. Mates of one fragment may share locus context; this is necessary but not sufficient evidence of species-level discriminability.

#### Two confounds we found and controlled for

**1. Amplicon-end confound.** Prefix analysis of the paired subset shows this is amplicon (metabarcoding) data, not shotgun: 90.0% of reads begin with one of just two conserved primer sequences (`AAGGGCACCACAAGAACG…`, `CGGTCACGTTCGTTGCCT…`), and the two mates of a cluster are **always** the two opposite ends. A uniformly-drawn distractor is therefore ~50% likely to share the *query's* end and its conserved primer sequence, making it spuriously more similar than the true mate.

The effect is large enough to invert the benchmark:

| Variant | Baseline AUROC | Foundation AUROC |
|---|---|---|
| Uncontrolled distractors | 0.3043 | 0.5696 |
| Distractors matched to the mate's amplicon end | 0.5167 | 0.6200 |

Uncontrolled, the baseline falls **below** chance (0.5) — for that representation the confound does not merely dilute the signal, it reverses it. Both representations lose ground when the control is removed.

The uncontrolled numbers are reported rather than quietly dropped: they are a genuine property of the data, and they are what exposed the confound. The end-group assignment is a technical covariate derived from primer sequence, like the flowcell tile — it is not a taxonomic label and is not used as one.

**2. Strand-orientation confound.** The two mates are sequenced from opposite strands. The canonical k-mer baseline is reverse-complement invariant by construction; a causal genomic language model is not. Reverse-complementing mate 2 puts both mates on the same strand — an exact, information-preserving operation.

| Representation | Orientation | Top-1 | AUROC |
|---|---|---|---|
| Baseline | as sequenced | 2.934% | 0.5167 |
| Baseline | mate 2 reverse-complemented | 2.934% | 0.5167 |
| Foundation model | as sequenced | 1.113% | 0.5642 |
| Foundation model | mate 2 reverse-complemented | 2.159% | 0.6200 |

The two baseline rows are **bit-for-bit identical** (max absolute difference 0.0e+00 across 25,000 transformed rows), empirically confirming its reverse-complement invariance rather than asserting it. For the foundation model the same transformation nearly doubles top-1 accuracy (1.113% → 2.159%) and raises AUROC from 0.5642 to 0.6200.

**Finding:** a substantial part of GenomeOcean's apparent weakness on paired short reads is strand orientation, not absence of fragment-level signal. Because the transformation is provably a no-op for the baseline, applying it to both keeps the comparison fair, and the strand-corrected numbers are used as the primary result above.

### K.3 Cost and capacity

| Metric | Baseline | Foundation model |
|---|---|---|
| Feature dimension | 137 | 1,536 |
| Model parameters | 0 | 541,109,760 |
| Trainable parameters | 0 | 0 |
| Training time | 0 s (not trained) | 0 s (frozen) |
| Inference throughput | 15,365 reads/s | 434 reads/s |
| Inference time per read | 0.0651 ms | 2.3034 ms |
| Storage per embedding | 550 bytes | 6,148 bytes |
| Compute device | cpu | cuda |

The baseline is **35× faster** per read. Throughput is realised cost on this machine (foundation model on GPU, baseline on CPU); it is not a claim about equivalent hardware.

**Neighbourhood agreement.** Mean Jaccard overlap of 20-nearest-neighbour sets between the two representations: **0.2558** (median 0.2121, n = 5,000). 1.0 would mean the two representations induce identical local neighbourhoods; 0.0 means they disagree completely about which reads are similar.

## L. Computational performance

Measured on this machine, warm-up excluded and CUDA-synchronised.

| Batch size | Reads/s | ms per read | Peak GPU allocated (MB) |
|---|---|---|---|
| 8 | 336.6 | 2.9705 | 1,055.9 |
| 16 | 431.8 | 2.3161 | 1,070.8 |
| 32 | 475.6 | 2.1025 | 1,101.9 |
| 64 | 481.3 | 2.0776 | 1,160.4 |
| 128 | 436.3 | 2.2921 | 1,281.5 |
| 256 | 426.5 | 2.3445 | 1,520.8 |

Best batch size: **64** at **481.3 reads/s**.

GPU: NVIDIA GeForce RTX 4050 Laptop GPU (6,140 MB). Peak usage never exceeded 1,521 MB, so this workload is compute-bound, not memory-bound, on 6 GB of VRAM.

**Full-dataset projection** (derived from measured throughput; **not executed**):

| Quantity | Projected |
|---|---|
| Reads | 3,018,522 |
| Wall-clock time | 104.5 min (1.74 h) |
| Embedding storage | 17.27 GB |
| Basis | linear extrapolation of measured steady-state throughput at the chosen batch size |

## M. Figures generated

All figures are computed from real measurements. None is illustrative.

| File | Content | Scope |
|---|---|---|
| `ambiguity.png` | Figure 3a — ambiguous-base distribution | Full dataset |
| `amplicon_architecture.png` | amplicon_architecture.png | - |
| `architecture.png` | Implemented architecture diagram | Documentation |
| `cluster_tendency.png` | Figure 6b — cluster tendency vs null | Analysis sample |
| `computational_performance.png` | Figure 12 — computational performance | Measured on this machine |
| `gc_content.png` | Figure 2 — GC-content distribution | Full dataset |
| `model_comparison.png` | Figure 8 — baseline vs foundation model | Encoding subset |
| `nucleotide_composition.png` | Figure 3b — nucleotide composition | Full dataset |
| `pca.png` | Figure 6 — PCA embedding projection | Visualization sample |
| `pca_explained_variance.png` | Figure 5 — PCA explained variance | Encoding subset |
| `qc_filtering.png` | Figure 3 — QC filtering outcome | Full dataset |
| `read_id_analysis.png` | Figure 4 — read-ID duplication evidence | Full dataset |
| `sequence_length.png` | Figure 1 — sequence length distribution | Full dataset |
| `umap.png` | Figure 7 — UMAP embedding projection | Visualization sample |
| `variable_region.png` | variable_region.png | - |

Stage 2 produced five further figures, documented in [`community_report.md`](community_report.md): `dereplication.png`, `rank_abundance.png`, `rarefaction.png`, `community_clusters.png`, `community_map.png`.

**Not generated, because the underlying data does not exist:**

| Figure | Reason |
|---|---|
| Figure 9 — confusion matrix | Not available with the current dataset/experimental setup. (no labels) |
| Figure 10 — training/validation curves | Not available with the current dataset/experimental setup. (no classifier trained) |
| Figure 11 — confidence / calibration | Not available with the current dataset/experimental setup. (no predicted probabilities) |

## N. Limitations

1. **No ground-truth taxonomy.** The dominant limitation. No supervised metric is computable, so 'useful for downstream taxonomic analysis' is assessed by proxy (mate-pair retrieval, cluster structure) rather than directly.
2. **Single sample.** No abundance features, so the paper's student input cannot be reproduced even in principle.
3. **Unassembled reads.** The paper's premise is that assembly supplies context; we deliberately test the regime where it does not.
4. **Subset, not full-dataset, inference.** Foundation-model embeddings were extracted for the encoding subset. The full-dataset cost is a labelled projection.
5. **Out-of-distribution context length.** GenomeOcean was pretrained on assembled sequence. 151 bp is in-spec for input length, not for training context.
6. **Mate-pair retrieval is a proxy**, with a stated locus-sharing confound. It demonstrates same-fragment signal, which is necessary but not sufficient for species-level discriminability.
7. **One foundation model tested.** No claim is made about genomic foundation models in general.
8. **One sequencing run, one site.** No claim is made about deep-sea eDNA in general.

## O. Is the first-stage architecture technically feasible?

**Yes, on the evidence collected here.** Specifically:

1. **The data can be processed at scale.** All 3,026,920 records stream through parsing and QC in 79 s (38,353 records/s), growing resident memory by 58.85 MB while reading a 649 MB file, with 99.72% passing QC.

2. **The paper's own foundation model accepts our short reads directly.** 151 bp → ~34 BPE tokens, 3.6% of the model's 1,024-token limit, 0 `[UNK]` tokens, 0 truncations, 0 padding. No substitute model was required.

3. **Embeddings are well-formed.** 200,000 of 200,000 reads encoded successfully (0 failures), all values finite, 0 dead dimensions out of 1,536.

4. **It is computationally feasible on modest hardware.** 481 reads/s on a 6 GB laptop GPU, peak usage under 1,521 MB. The full 3,018,522-read dataset projects to 1.7 h and 17.3 GB.

5. **The embeddings carry real, measurable genomic signal.** On the label-free mate-pair retrieval benchmark the foundation model reaches 2.16% top-1 against a 1.0% chance baseline (2.2× chance), and does not outperform the k-mer baseline (2.93%).

**What this does NOT establish.** Feasibility of the representation stage is not the same as taxonomic utility. Without ground-truth labels this experiment cannot measure whether these embeddings support accurate taxonomic assignment — only that they are producible, well-formed, computationally tractable, and carry non-trivial sequence-level structure. Any claim beyond that would not be supported by these measurements.

## P. What additional data is required for the next stage

| Requirement | Why | Enables |
|---|---|---|
| Per-read taxonomic labels (Kraken2 / MMseqs2 / Metabuli output, or a mock community with known source genomes) | Nothing supervised is measurable without them | Accuracy, macro/weighted F1, per-class metrics, confusion matrix, training curves |
| The label source's own error rate | TaxDistill's premise is that retrieval labels are noisy | Meaningful evaluation of label correction |
| A taxonomy tree for the target taxa | The paper's hierarchical loss operates over it | Deep hierarchical loss, hierarchy-aware evaluation |
| Multiple samples from the same site, mapped to a shared reference | Abundance is a cross-sample quantity | The paper's full student input (TNF + K abundances + total) |
| Assembled contigs from these reads, if assembly is viable | Enables a like-for-like comparison with the paper's ≥ 2,000 bp regime | Direct measurement of how much the 151 bp constraint actually costs |
| A held-out validation set with independent labels | Test-set integrity | Calibration, confidence analysis, honest generalisation estimates |

---

*Generated by `python main.py report`. Every value above is read from a metrics file produced by an executed run; nothing is hand-entered.*