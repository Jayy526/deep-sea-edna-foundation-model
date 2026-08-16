# Stage 2 Report — Unsupervised Community Structure

**Question:** without any taxonomic labels, what can be said about the composition and structure of this deep-sea eDNA community, and do foundation-model embeddings organise it differently from k-mer features?

Stage 1 established that no ground-truth taxonomy exists for this dataset, so no supervised metric is computable. This stage takes the honest alternative: characterise the community from sequence structure alone, and state clearly what that can and cannot establish.

---

## A. Why the unit of analysis changed

Stage 1 encoded randomly sampled **reads**. For community structure that is the wrong unit: 3 million reads are not 3 million molecules, they are a re-sampling of a much smaller set of distinct sequences at very uneven depth. Stage 2 therefore dereplicates first, making the unit the **unique sequence variant**, with read count carried as an explicit abundance weight.

| | Stage 1 | Stage 2 |
|---|---|---|
| Unit | read | unique sequence variant |
| Selection | systematic sample | full-dataset dereplication |
| Abundance | uncontrolled sampling bias | explicit weight on every statistic |

**What a variant is not.** These are exact-sequence variants: no denoising error model (no DADA2/UNOISE), no similarity clustering, no taxonomy. Sequencing error inflates the count, predominantly as singletons. That is measured below, not corrected for silently.

## B. Dereplication

| Metric | Value |
|---|---|
| Reads dereplicated (full dataset) | 3,018,522 |
| **Unique sequence variants** | **246,001** |
| Mean reads per variant | 12.27 |
| Singletons (seen once) | 198,919 (80.9% of variants) |
| Doubletons | 22,801 |
| Most abundant single variant | 397,610 reads |
| Variants with sequence retained | 47,082 |
| Reads covered by retained variants | 93.41% |
| Reverse-complement collapsing | False |

Streamed in two passes (24.9 s + 25.3 s) at 625.54 MB peak. Counting is done on 8-byte BLAKE2b digests rather than sequence strings to bound memory; the resulting collision probability is 1.64e-09 and is recorded rather than assumed away.

**Headline:** 3,018,522 reads collapse to 246,001 distinct molecules — a 12.3x redundancy factor. Encoding reads rather than variants would have spent most of the compute re-embedding identical sequences.

### B.1 Sensitivity check: reverse-complement collapsing

Dereplication was re-run collapsing each sequence with its reverse complement, to test whether the primary (orientation-sensitive) result depends on read orientation.

| | Exact (primary) | RC-collapsed |
|---|---|---|
| Unique variants | 246,001 | 245,280 |
| Singletons | 198,919 | 198,427 |
| Most abundant variant | 397,610 | 397,610 |
| Shannon H' | 5.6664 | 5.6652 |
| Pielou J | 0.4565 | 0.4565 |
| Inverse Simpson | 20.6615 | 20.6614 |
| Hill q1 | 288.99 | 288.65 |
| Chao1 | 1,113,657 | 1,113,638 |

**Result: the choice does not matter here.** Collapsing merges only 721 variants (0.29% of the total) and leaves every diversity index unchanged to three decimal places. That is itself informative: it confirms reads are consistently oriented within the library, and that the two amplicon ends identified in stage 1 are genuinely distinct sequence, not one region read from both strands. The orientation-sensitive result is used as primary.

## C. Diversity

All indices computed from the **full-dataset** frequency spectrum — no subsampling, no read excluded.

| Index | Value | Reading |
|---|---|---|
| Observed richness | 246,001 | distinct variants seen |
| Shannon H' | 5.6664 | entropy of the abundance distribution |
| Pielou evenness J | 0.4565 | 0 = one variant dominates, 1 = perfectly even |
| Simpson D | 0.04840 | chance two reads share a variant |
| Inverse Simpson | 20.66 | effective number of dominant variants |
| Hill q0 (richness) | 246,001 | all variants counted equally |
| Hill q1 (exp Shannon) | 289.0 | equally-common variants giving the same entropy |
| Hill q2 (inv. Simpson) | 20.7 | weighted toward the abundant |
| Chao1 estimate | 1,113,657 | 4.5x observed |
| Good's coverage | 93.41% | fraction of the community sampled |

**The Hill numbers are the story.** 246,001 variants were observed, but the community behaves like only ~289 equally-common variants (q=1) or ~21 (q=2). Richness and dominance are telling completely different stories, and reporting richness alone would be misleading.

**On Chao1.** The 1,113,657 estimate is driven by 198,919 singletons. In amplicon data singletons are predominantly sequencing error, not rare organisms, so this figure is an upper bound on *molecular* diversity and should not be read as a species estimate. Notably those singletons are 80.9% of variants but only 6.6% of reads.

## D. Community concentration

| Cumulative share of all reads | Variants required |
|---|---|
| 50% | **7** |
| 90% | 9,658 |
| 99% | 215,816 |

| Top N variants | Share of reads |
|---|---|
| 1 | 13.17% |
| 10 | 56.22% |
| 100 | 70.60% |
| 1,000 | 82.19% |

**This community is extraordinarily uneven.** 7 variants account for half of a 3-million-read library, and a single variant accounts for 13.17% of it. Combined with the amplicon structure found in stage 1, this is the profile of a marker-gene library dominated by a small number of source organisms, with a very long rare tail.

## E. Sampling completeness

Analytic (Hurlbert) rarefaction — deterministic, no random subsampling.

| Metric | Value |
|---|---|
| Total reads | 3,018,522 |
| Observed richness at full depth | 246,001 |
| New variants per additional read at full depth | 0.0661 |
| Good's coverage | 93.41% |

The curve has **not** saturated: more sequencing would keep revealing new variants. But because most new variants at this depth are error-derived singletons, this indicates the *variant* space is unsaturated, not necessarily that organisms are being missed. Good's coverage of 93.4% says the abundant community is well captured.

## F. Encoding variants

Both representations were computed on the **same variant set**, with the same seed — the fairness protocol from stage 1 carried forward.

| | Foundation model | Baseline |
|---|---|---|
| Variants encoded | 47,082 | 47,082 |
| Failures | 0 | 0 |
| Embedding dimension | 1,536 | 137 |
| Processing time | 103.4 s | 3.7 s |
| Throughput | 456 /s | 12,600 /s |
| All values finite | True | True |
| Dead dimensions | 0 | 0 |

These 47,082 variants represent 2,819,603 reads — 93.4% of the QC-passed dataset — so the clustering below speaks for the large majority of the library despite operating on a small fraction of the variants.

## G. Community clustering

| | Foundation model | Baseline |
|---|---|---|
| Selected k (max silhouette) | 10 | 10 |
| Silhouette at selected k | 0.2067 | 0.2662 |
| Largest cluster, share of reads | 22.79% | 22.84% |
| Clusters covering 50% of reads | 3 | 3 |
| Clusters covering 90% of reads | 6 | 6 |
| Mean amplicon-end purity | 0.9313 | 0.9201 |

### G.1 Is the structure real, or arbitrary?

With no taxonomy, the only honest check is whether clusters track **measured sequence properties**. GC content and amplicon end are measured, not labelled.

| Property | Foundation model | Baseline |
|---|---|---|
| GC variance explained by clusters | 0.7309 | 0.7440 |
| log-abundance variance explained | 0.0272 | 0.0295 |
| Agreement with amplicon end (ARI) | 0.2410 | 0.2545 |

High GC variance explained means the partition is structured rather than arbitrary. It says nothing about taxonomic correctness.

### G.2 Do the two representations agree?

| Chance-corrected metric | Value |
|---|---|
| Adjusted Rand index | **0.7857** |
| Adjusted mutual information | 0.8176 |
| Fowlkes-Mallows | 0.8109 |

Chance-corrected, so 0 = agreement no better than random and 1 = identical partitions. An adjusted Rand of **0.7857** is strong agreement: the two representations largely recover the **same coarse community structure** from the same molecules.

**This is worth contrasting with stage 1.** At read level, the mean Jaccard overlap of 20-nearest-neighbour sets between the two representations was only 0.2558 — they disagreed about which individual reads are similar. At variant level and coarse granularity they agree strongly (ARI 0.786). The two findings are consistent: the representations differ in fine-grained local neighbourhood structure while converging on the same broad composition-driven partition.

**No foundation-model advantage is detectable on this task.** The baseline produces the higher silhouette at the selected k (0.2662 vs 0.2067) and explains comparable GC variance (0.744 baseline vs 0.731 foundation), using 137 dimensions against 1,536. For unsupervised community structure at this granularity, the 11x wider learned representation buys nothing measurable.

With no labels, neither partition can be declared correct where they do differ — which is precisely why a labelled comparison remains the necessary next step.

### G.3 How much of a read is actually informative?

A question neither stage had answered: this is marker-gene data, so how much of each 151 bp read is conserved anchor and how much is organism-discriminating sequence? Per-position base composition was computed across the abundant variants, weighted by read abundance.

| Amplicon end | Conserved block | Mean conservation there | Variable region | Mean conservation there | Read that varies |
|---|---|---|---|---|---|
| 0 | 90 bp | 97.2% | 61 bp | 57.7% | 34.4% |
| 1 | 16 bp | 100.0% | 135 bp | 63.6% | 44.4% |

**Only 39.7% of each read varies across the community** (abundance-weighted across both ends). End 0 in particular opens with a 90 bp block that is 97.2% invariant.

**This reframes the earlier results.** Positions that do not vary cannot discriminate organisms, whatever encodes them. Both representations were therefore handed reads whose majority content is near-identical across the whole community, which bounds what *any* representation could achieve on mate retrieval or fine-grained clustering. The weak absolute retrieval scores in stage 1 (2–3% top-1) and the absence of a foundation-model advantage here should both be read in that light: the ceiling is set by the data, not only by the encoders.

It also suggests a caution about the clustering above. Clusters came out ~93% pure by amplicon end, and the conserved anchor is exactly the signal that separates the ends. Some of the apparent cluster structure is therefore likely to reflect *which end of the amplicon a read came from* rather than which organism it came from. Re-running the comparison on the variable region alone would isolate the biological signal, and is the obvious follow-up.

**On the identity of the conserved block.** Its architecture — a long, near-invariant anchor followed by a sharply variable region — is the signature of a conserved-region-primed marker gene, and the end-0 consensus shows clear homology to the universal small-subunit rRNA conserved block (motifs `GGGCACCAC`, `GTGGAGCATGTGG`, `TTAATTTGACTCAAC`, `GGATTGACAG`). That is as far as the data alone can go: **distinguishing 16S from 18S from an organellar SSU, and assigning any taxonomy, requires a reference database.** No gene name is asserted here. A web search of the exact primer sequences returned no catalogued match, so these appear to be custom or non-standard primers.

### G.4 Removing the conserved anchor — the follow-up, run

The obvious test of §G.3: trim the conserved anchor from every read and re-run the retrieval comparison on the same pairs, same seed, same amplicon-end control, same strand correction. Only the input sequence changes, retaining 70% of it (mean 105 bp per read).

| Input | Representation | Top-1 | AUROC |
|---|---|---|---|
| Full read (151 bp) | GenomeOcean-500M | 2.159% | 0.6200 |
| Full read (151 bp) | k-mer / TNF | 2.934% | 0.5167 |
| Variable region only | GenomeOcean-500M | 2.323% | 0.5815 |
| Variable region only | k-mer / TNF | 2.385% | 0.4972 |

Top-1 outcomes are **paired** (both representations scored on the same queries), so differences are tested with McNemar's exact test rather than eyeballed:

| Comparison | Difference | 95% CI | p | Verdict |
|---|---|---|---|---|
| Full read: foundation vs baseline | -0.78 pp | [-1.01, -0.54] | 2e-10 | baseline is better (p=2.01e-10) |
| Variable region: foundation vs baseline | -0.06 pp | [-0.30, +0.18] | 0.63 | no significant difference (p=0.635) |
| Foundation: variable vs full read | +0.16 pp | [-0.02, +0.35] | 0.093 | no significant difference (p=0.093) |
| Baseline: variable vs full read | -0.55 pp | [-0.74, -0.36] | 8.6e-09 | full_read is better (p=8.61e-09) |

**This substantially qualifies §G.2's conclusion.** Three findings:

1. **The baseline's full-read advantage is real but anchor-dependent.** It beats the foundation model by 0.78 pp on full reads (p = 2e-10), but loses 0.55 pp (p = 9e-09) once the conserved anchor is removed. Much of what looked like superior retrieval was the near-invariant anchor, not organism signal.

2. **The foundation model loses nothing.** Trimming changes its top-1 by +0.16 pp (p = 0.09, not significant) — it was not relying on the anchor.

3. **On the informative region the two are statistically tied on top-1** (p = 0.63), but their mate/non-mate separation diverges sharply: the foundation model holds AUROC 0.582 while the baseline falls to 0.497 — indistinguishable from chance. On the biologically informative sequence, the k-mer representation retains essentially no mate/non-mate separation; the foundation model does.

**Revised bottom line.** The earlier statement that the foundation model shows *no measurable advantage* holds for full 151 bp reads and for coarse community clustering. It does **not** hold once the uninformative conserved anchor is excluded: there the k-mer baseline's advantage disappears entirely and its separation collapses to chance, while the foundation model's is preserved. Both remain weak in absolute terms, and none of this establishes taxonomic accuracy.

*Caveat: trimming discards real sequence and shortens reads unequally between amplicon ends. It is a diagnostic for locating the signal, not a proposed preprocessing default; the full-read result remains primary.*

## H. What remains unavailable

**Not available with the current dataset/experimental setup.** for all of the following, unchanged from stage 1:

- Any taxonomic identity for any variant or cluster
- Species richness (as opposed to variant richness)
- Classification accuracy, macro/weighted F1, per-class metrics, confusion matrix
- Confidence and calibration analysis
- Whether the foundation model's partition is *better* than the baseline's

No cluster is named. No variant is assigned a taxon. Cluster count is a property of the embedding and the selected k, not a species estimate.

## I. Figures

| File | Content | Scope |
|---|---|---|
| `dereplication.png` | Figure 13 — variants, frequency spectrum, Chao1 | Full dataset |
| `rank_abundance.png` | Figure 14 — rank abundance and concentration | Full dataset |
| `rarefaction.png` | Figure 15 — sampling completeness | Full dataset |
| `community_clusters.png` | Figure 16 — clustering, foundation vs baseline | Encoded variants |
| `community_map.png` | Figure 17 — abundance-weighted community map | Encoded variants |
| `amplicon_architecture.png` | Figure 18 — conserved vs variable read architecture | Abundant variants |

## J. Limitations

1. **Exact-sequence variants, not denoised ASVs.** No error model was applied, so richness is inflated and Chao1 substantially so. A DADA2/UNOISE-style denoiser would materially change every richness figure (though little of the abundance profile).
2. **Variants are not organisms.** One organism can contribute several variants (sequencing error, intragenomic marker copies) and two organisms can share one. Richness is an upper bound on organismal diversity.
3. **Clusters are not taxa.** They are unnamed groups of similar sequences.
4. **Only the abundant variants were encoded.** Variants seen once were excluded from encoding; they are 80.9% of variants but only 6.6% of reads, so the abundance-weighted picture is largely unaffected while the rare tail is not represented in the clustering.
5. **Single sample, single site, single sequencing run.** No spatial, temporal or cross-sample comparison is possible, so beta diversity is not computable.
6. **Neither partition can be validated.** Without labels there is no way to say which representation is right where they disagree.

## K. Conclusions

1. **The library is far less diverse than its read count suggests.** 3,018,522 reads collapse to 246,001 variants, and the effective diversity is smaller still — around 289 equally-common variants.

2. **It is dominated by very few sequences.** 7 variants carry half the library; the single most abundant carries 13.17%.

3. **The rare tail is mostly error, not biology.** 198,919 singletons are 80.9% of variants but only 6.6% of reads, and they are what drives Chao1 to 4.5x the observed richness.

4. **Both representations recover the same coarse community structure** (adjusted Rand = 0.7857). Both select k = 10, both need 3 clusters to cover half the reads, and both produce clusters that are ~93% pure by amplicon end and explain ~73-74% of GC variance. The partitions are structured, not arbitrary.

5. **On full reads and coarse clustering, the foundation model shows no measurable advantage.** The 137-dimensional k-mer baseline achieves a higher silhouette than the 1,536-dimensional GenomeOcean embedding, at ~28x the throughput, and wins full-read top-1 retrieval by a statistically solid margin. Reported as measured.

5b. **But that advantage is anchor-dependent, and reverses on the informative sequence.** Trimming the near-invariant conserved anchor costs the baseline 0.55 pp of top-1 (p = 9e-09) and drops its mate/non-mate AUROC to 0.497 — chance. The foundation model loses nothing (p = 0.09) and holds AUROC 0.582. On the part of the read that actually varies, the learned representation retains signal the k-mer features do not.

6. **Only 40% of each read varies across the community.** The rest is conserved anchor. This bounds what any representation can achieve and is the most likely explanation for both the weak absolute retrieval scores and the absence of a foundation-model advantage — the ceiling is set by the data. It also means some apparent cluster structure probably reflects amplicon end rather than organism.

7. **The unsupervised route has now been taken as far as it honestly goes.** Composition, evenness, dominance and sampling completeness are all characterised from the full dataset. What cannot be resolved without labels is which representation organises the community *correctly* — and that is the question the next stage would need reference taxonomy to answer.

---

*Generated by `python main.py community-report`. Every value is read from a metrics file produced by an executed run.*