# Methodology

Stage 1 of a deep-sea eDNA genomic representation pipeline.
**A TaxDistill-inspired adaptation for short deep-sea eDNA reads.**

Every component below carries one of three labels, and they are never blurred:

| Label | Meaning |
|---|---|
| **PAPER-DERIVED** | Taken from TaxDistill (Ye et al., arXiv:2605.28868) as published. |
| **EXPERIMENTAL ADAPTATION** | The paper's idea, changed because our data is not the paper's data. The change and its reason are stated. |
| **IMPLEMENTATION DECISION** | Not addressed by the paper; chosen by us and justified here. |

This work does **not** reproduce TaxDistill. TaxDistill is a knowledge-distillation
framework whose teacher is GenomeOcean and whose student is a Taxometer-style MLP
trained on noisy retrieval labels. We implement only the **upper/representation
stage** — raw reads to deep genomic embeddings — plus the analysis needed to judge
whether those embeddings are usable. The distillation, the student network, and the
taxonomic annotation task itself are out of scope.

---

## 1. Dataset

| Property | Measured value |
|---|---|
| Source file | `SRR26872904.fasta` |
| Size | 648.87 MB |
| Sequence records | 3,026,920 |
| Unique sequence IDs | 1,513,460 |
| Records per ID | exactly 2, with no exceptions |
| Read length | 151 bp for every record (min = max = median = 151, sd = 0) |
| Total sequenced bases | 457,064,920 |
| Reads with ambiguous bases | 5,804 (0.19%) |
| Reads with invalid characters | 0 |
| Mean GC content | 51.31% |
| Taxonomic labels | **none** |

Full detail: `outputs/metrics/dataset_metrics.json`.

### 1.1 Why every ID appears twice — determined, not assumed

The brief explicitly warns against assuming duplicate IDs mean paired-end reads.
We therefore measured three independent properties of every repeated-ID group:

| Property | Measured |
|---|---|
| The two records are adjacent in file order | **100.00%** |
| The two records sit on the same flowcell tile | **100.00%** |
| The two records carry identical GC content | **1.85%** |

Same cluster coordinate + adjacent + different sequence content is the signature of
the two mates (R1/R2) of one paired-end cluster, written interleaved. The file
contains 1,513,460 clusters × 2 mates. The repeated IDs are **not** duplicated
data: both records are real, distinct sequence. See `outputs/figures/read_id_analysis.png`.

This finding is also what makes the label-free evaluation in §7 possible.

---

## 2. The paper's architecture vs ours

### 2.1 What TaxDistill does

```
CAMI2 assembled contigs (>= 2,000 bp)
    + abundance matrix across K samples
    + noisy initial taxonomy from MMseqs2 / Kraken2 / Metabuli
                  |
      +-----------+-----------+
      |                       |
  TEACHER                  STUDENT
  GenomeOcean (frozen)     Taxometer-style MLP
  + learnable head         input: 103-d TNF + K abundances + total
      |                       |
      +--- KD loss (soft labels, temperature T, weight alpha) ---+
                  |
        corrected species-level taxonomy
```

### 2.2 What we do

```
Raw 151 bp deep-sea eDNA reads (no assembly, no abundance, no labels)
                  |
        streaming QC (full dataset)
                  |
      +-----------+-----------+
      |                       |
  k-mer / TNF baseline    GenomeOcean-500M (frozen)
  137-d                   1536-d
      |                       |
      +---- stored embeddings ----+
                  |
    PCA / UMAP / cluster tendency / mate-pair retrieval
                  |
        [classifier + calibration: NOT EXECUTED, no labels]
```

### 2.3 The differences that matter

| # | TaxDistill | This work | Why |
|---|---|---|---|
| 1 | Assembled contigs ≥ 2,000 bp | Raw 151 bp reads | **This is our data.** See §3. |
| 2 | Contig length filter ≥ 2,000 bp | **No such filter** | Applying it would discard 100% of our reads. |
| 3 | Teacher = GenomeOcean | Teacher = GenomeOcean, **same checkpoint** | Genuinely the paper's model; verified, not claimed. §4. |
| 4 | Student = MLP on TNF + abundance | Baseline = canonical 4-mer/TNF, **no abundance** | Abundance needs multiple samples mapped to a shared assembly. We have one sample of unassembled reads. Not computable. |
| 5 | Ground truth from CAMI2 + retrieval tools | **No ground truth** | Verified absent. §6. |
| 6 | Knowledge distillation, soft labels, hierarchical loss | **Not implemented** | Requires labels. Out of scope for stage 1. |
| 7 | Species-level F1 reported | **Not computable** | No labels. Reported as unavailable, not estimated. |

---

## 3. Why 151 bp changes the problem — EXPERIMENTAL ADAPTATION

The paper filters to contigs ≥ 2,000 bp and states the reason plainly: it
"ensures that deep representation learning can capture sufficient contextual
semantic information."

Our reads are 151 bp — an order of magnitude shorter. Three consequences:

1. **The paper's length filter cannot be applied.** At a ≥ 2,000 bp threshold,
   0 of 3,026,920 reads survive. The pipeline would have no input.

2. **Padding to 2,000 bp would be scientifically indefensible.** Padding adds
   1,849 positions of non-biological content per read — 92.5% of the resulting
   "sequence." Whatever a model computed from that would be a function mostly of
   the padding scheme. We do not pad, anywhere, at any stage.

3. **The k-mer baseline is genuinely handicapped by short reads, and that is
   the point.** A 151 bp read yields 148 4-mers spread over 136 canonical bins —
   roughly one count per bin. A 2,000 bp contig yields 1,997. Tetranucleotide
   frequency is a *statistical* signal that needs length to stabilise. Whether a
   pretrained genomic language model degrades more gracefully than TNF at 151 bp
   is precisely the question this experiment can answer.

**What we do instead:** feed each read to the encoder at its true length. The
model's own limit is 1,024 tokens; our reads occupy ~33. There is no length
problem to solve — only a *context* caveat to state honestly (§4.3).

---

## 4. Encoder — PAPER-DERIVED

### 4.1 Model identity

`pGenomeOcean/GenomeOcean-500M` — the 500M-parameter variant the paper uses for
its main results. Not a substitute, not a look-alike.

| Verified property | Value |
|---|---|
| Architecture | `MistralForCausalLM`, 14 layers |
| Hidden size (= embedding dim) | 1536 |
| Vocabulary | 4096 BPE tokens over ACGT |
| Documented max input | 1024 tokens (~5 kbp) |
| Parameters (counted at load) | 541,109,760 |
| Trainable | **0** |
| Frozen | **541,109,760** |
| Precision | bfloat16 |
| Attention | `sdpa` |
| Device | NVIDIA RTX 4050 Laptop, 6,140 MB |

### 4.2 Compatibility was measured, not assumed

`FoundationModelEncoder.verify_compatibility()` runs on real reads before any
bulk inference and raises if they do not fit:

| Measured on 1,000 real reads | Value |
|---|---|
| Tokens per read | 30 – 37 (mean 33.6) |
| bp per token | 4.49 |
| Model token limit | 1024 |
| Fraction of limit used | **3.3%** |
| Fits without truncation | **yes** |
| `[UNK]` tokens produced | **0** |
| Reads truncated | **0** |
| Padding to a target length | **none** |

Saved to `outputs/metrics/model_compatibility.json`.

### 4.3 The caveat we do not hide

151 bp is well **within spec** for input length but **out of distribution** with
respect to the context GenomeOcean was pretrained to exploit (assembled genomic
sequence). Length compatibility is not the same as representational suitability.
Testing that gap is the experiment; assuming it away would be the error.

### 4.4 Pooling — IMPLEMENTATION DECISION

The paper says the teacher "extracts deep semantic features" and projects them
through a learnable head, but does not specify the pooling operation. We use
**attention-masked mean pooling over the final hidden layer**: standard for
causal-LM sequence embeddings and unbiased with respect to read position.
`last` and `max` pooling are implemented and config-selectable so the choice is
auditable rather than baked in.

### 4.5 Replaceability

`SequenceEncoder` (`models/encoder.py`) is the seam. `FoundationModelEncoder`
and `KmerBaselineEncoder` both implement it. Swapping in a different genomic
foundation model means writing one subclass — preprocessing, storage, analysis
and reporting are untouched.

---

## 5. Preprocessing

### 5.1 Streaming — IMPLEMENTATION DECISION

One generator pass, at most one record resident. Measured: 3,026,920 records in
78.9 s (38,353 rec/s, 8.2 MB/s).

**On the memory figure.** Absolute RSS is dominated by whichever libraries the
entry point imported — the CLI pulls in torch and transformers just to record
the environment, which alone accounts for ~797 MB. The number that actually
supports the constant-memory claim is therefore the *delta*: streaming the
entire 648.9 MB file grew resident memory by only **58.85 MB**. Both figures,
and the reason they differ, are recorded in `preprocessing_metrics.json`.

Per-read metadata goes to Parquet in row groups; exact order statistics are
computed from that table afterwards rather than approximated on the fly. That
second pass is what takes whole-stage peak memory to 1,490 MB — a deliberate
trade of memory for exactness, not a failure of the streaming design.

### 5.2 Quality control — IMPLEMENTATION DECISION

Every threshold lives in `configs/default.json`; every rejected read is recorded
with a reason code; nothing is discarded silently.

| Filter | Setting | Removed |
|---|---|---|
| `min_length` | 100 bp | 0 |
| invalid characters | reject | 0 |
| `max_ambiguous_fraction` | 0.10 | 0 |
| `max_single_base_fraction` | 0.90 | 8,267 |
| `min_effective_length` | 100 bp | 131 |
| **Total** | | **8,398 (0.28%)** |

3,018,522 reads (99.72%) pass.

### 5.3 Ambiguity policy — IMPLEMENTATION DECISION

GenomeOcean's BPE vocabulary covers ACGT only; an `N` would become `[UNK]`.
Four policies are implemented (`keep`, `reject`, `trim_ends`,
`longest_unambiguous_run`). Default: **`longest_unambiguous_run`** — keep the
longest contiguous ACGT stretch of the read.

It never substitutes a base, never imputes, never pads. It is lossy, and the
loss is measured: mean effective length 150.99 bp of 151, i.e. 0.006% of bases.
Ambiguity is concentrated at sequencing cycle 2 (0.131% of reads), the signature
of a base-calling artefact rather than biology — see
`outputs/figures/ambiguity.png`. The measured `[UNK]` count at encode time is 0,
confirming the policy worked rather than assuming it.

---

## 6. Taxonomic labels — verified absent

`analysis/labels.py` checks three sources and records the evidence:

1. **Configured label file** — none set.
2. **FASTA headers** — 20,000 inspected, 0 match any taxonomy pattern. Every
   header has exactly 3 tokens: run accession, Illumina cluster coordinate,
   `length=151`.
3. **Sibling annotation files** — none present.

**Consequence.** Supervised taxonomic classification cannot be scientifically
evaluated here. Per the brief, no labels are invented. The following are
reported as *"Not available with the current dataset/experimental setup."*:
accuracy, macro/weighted precision-recall-F1, per-class metrics, confusion
matrix, training/validation curves, confidence histograms, reliability diagrams,
Brier score, ECE, and supervised silhouette / intra-class / inter-class distances.

The classifier (`models/classifier.py`) and its metrics are implemented and will
run if `classification.labels_path` is supplied — but they were **not executed**,
because there is nothing valid to run them on.

### What would be needed next

- A per-read taxonomic assignment table from an independent reference-based
  classifier (Kraken2 / MMseqs2 / Metabuli) — the same tools TaxDistill corrects.
- **or** a mock-community / CAMI2-style set where each read's source genome is
  known by construction.
- **or** curated reference amplicon sequences (SILVA / PR2) with a stated
  confidence threshold.

In every case the label source, version and its own error rate must be recorded,
because TaxDistill's entire premise is that such labels are noisy.

---

## 7. Label-free evaluation — IMPLEMENTATION DECISION

With no taxonomy, we use the one piece of genuine structure the data supplies
that we did not invent: **the paired-end layout established in §1.1**.

The two mates of a cluster are non-overlapping reads from the *same source DNA
fragment*, hence the same source organism. That yields a retrieval benchmark:
given R1's embedding, is R2 ranked above 99 random distractor reads? Chance
top-1 is 1%. It applies identically to both representations, so the comparison
is fair.

**Stated confound.** Mates of one fragment may share locus-specific context. A
high score demonstrates same-fragment / same-locus signal — necessary but *not
sufficient* for species-level taxonomy. It is never reported as taxonomic accuracy.

### 7.1 Two confounds found during the run, and how they were controlled

The naive form of this benchmark scored **below chance**. Rather than discard
the result, we diagnosed it. Two distinct confounds were found; both are
controlled for, and both uncontrolled results remain in the report.

**Confound 1 — amplicon end.** Prefix analysis of the paired subset shows this
library is **amplicon (metabarcoding) data, not shotgun**: 90.0% of reads begin
with one of just two conserved primer sequences (`AAGGGCACCACAAGAACG…`,
`CGGTCACGTTCGTTGCCT…`), and the two mates of a cluster are *always* the two
opposite ends of the amplicon. A uniformly-drawn distractor is therefore ~50%
likely to share the **query's** end — and its conserved primer sequence — making
it spuriously more similar than the true mate.

*Control:* `evaluation.structure.amplicon_end_groups` assigns each read to an
end group by Hamming distance to the two dominant 18 bp prefixes; distractors
are then drawn only from reads sharing the **true mate's** end. The only thing
distinguishing the mate from a distractor becomes which DNA fragment it came
from — which is what the benchmark is meant to measure. The end group is a
technical covariate derived from primer sequence, exactly like the flowcell
tile. **It is not a taxonomic label and is not used as one.**

**Confound 2 — strand orientation.** The two mates are sequenced from opposite
strands. The canonical k-mer baseline is reverse-complement invariant *by
construction*; a causal genomic language model is not — GenomeOcean reads
sequence in one direction and has no built-in notion that a strand and its
complement are the same molecule.

*Control:* reverse-complement mate 2, placing both mates on the same strand.
This is an exact, information-preserving operation on DNA — no information is
added or invented. Measured effect: the baseline is unchanged **bit-for-bit**
(max absolute difference 0.0 across 25,000 transformed rows — verified
numerically, not asserted), while the foundation model's top-1 accuracy nearly
doubles (1.113% → 2.159%) and its AUROC rises from 0.5642 to 0.6200.

Because the transformation is provably a no-op for the baseline, applying it to
both representations keeps the comparison fair. The strand-corrected numbers are
therefore the primary result.

**Finding:** a substantial part of GenomeOcean's apparent weakness on paired
short reads is strand orientation, not absence of fragment-level signal. Any
future use of a causal genomic LM on paired-end reads should normalise mate
orientation first.

### 7.2 Supporting statistics

PCA intrinsic dimensionality; k-means silhouette **paired with a
dimension-shuffled null** (shuffling each dimension independently preserves all
marginals while destroying joint structure, so only the gap is interpretable);
and k-NN neighbourhood agreement between the two representations.

---

## 7.3 Stage 2 — unsupervised community structure — IMPLEMENTATION DECISION

Stage 1 concluded that supervised evaluation is impossible without labels.
Stage 2 takes the honest alternative: characterise the community from sequence
structure alone. Nothing in it is paper-derived — TaxDistill does not do
community ecology — so the whole stage is labelled an implementation decision.

### The unit of analysis changes

Stage 1 encoded randomly sampled **reads**. For community structure that is the
wrong unit: 3 million reads are a re-sampling of a much smaller set of distinct
molecules at very uneven depth. Stage 2 dereplicates the full QC-passed dataset
into **unique sequence variants**, carrying read count as an explicit abundance
weight on every statistic.

| | Stage 1 | Stage 2 |
|---|---|---|
| Unit | read | unique sequence variant |
| Selection | systematic sample (200,000) | full-dataset dereplication |
| Abundance | uncontrolled sampling bias | explicit weight |

### What a variant is, and is not

Exact-sequence variants: **no denoising error model** (no DADA2/UNOISE), **no
similarity clustering**, **no taxonomy**. Sequencing error inflates the count,
predominantly as singletons — measured (80.9% of variants, 6.6% of reads), never
corrected for silently. Richness is an upper bound on organismal diversity, not
an estimate of it, and Chao1 is explicitly flagged as singleton-driven.

### Memory

Counting 3M sequence strings would cost several hundred MB. Pass 1 counts 8-byte
BLAKE2b digests in a dict; pass 2 recovers sequences only for variants used
downstream. The 64-bit collision probability (~2.4e-7 at this scale) is recorded
in the output rather than assumed away. Measured peak: 626 MB.

### Reverse-complement sensitivity check

Dereplication was re-run with each sequence collapsed against its reverse
complement. Collapsing merges only 721 of 246,001 variants (0.29%) and leaves
every diversity index unchanged to three decimals. This confirms reads are
consistently oriented and that the two amplicon ends are genuinely distinct
sequence rather than one region read from both strands. The orientation-sensitive
result is used as primary.

### Clustering, and what it cannot claim

Clusters are groups of similar sequences. They are **not taxa, not species, not
OTUs** (no similarity threshold, no denoising). No cluster is named, because
nothing in this dataset could name one. Cluster count is a property of the
embedding and the selected k, not a species estimate.

The only available check that clusters are not arbitrary is whether they track
**measured** sequence properties. They do: both partitions explain ~73–74% of GC
variance and are ~92–93% pure by amplicon end. GC content and amplicon end are
measurements, not labels.

### Result

Both representations recover the same coarse structure (adjusted Rand 0.786;
both select k = 10). The 137-d k-mer baseline achieves a **higher** silhouette
(0.266) than the 1,536-d GenomeOcean embedding (0.207) at ~28× the throughput.
No foundation-model advantage is detectable on this task. Reported as measured.

---

## 8. Fairness protocol

Both representations are computed from the same subset Parquet file, after the
same QC, with the same seed, and are scored with the same metrics on the same
samples. Throughput is reported as *realised* cost on this machine (foundation
model on GPU, baseline on CPU) and labelled as such — it is not a claim about
equivalent hardware.

---

## 9. Sample sizes — three distinct numbers, never conflated

| Number | Value | Used for |
|---|---|---|
| **Full dataset** | 3,026,920 records | All dataset metrics and Figures 1–4 |
| **Encoding subset** | 200,000 reads (+ 25,000 pairs = 50,000 reads) | Embedding extraction, PCA, comparison |
| **Visualization sample** | 20,000 reads | UMAP / PCA scatter plots only |

Subsets are drawn by systematic sampling with a fixed stride over QC-passed reads
in file order. Reads are ordered by flowcell coordinate, not by organism, so a
fixed stride spreads the sample across the whole run without holding 3M reads in
memory. Full-dataset encoding was **not** executed; its cost is reported as a
projection from measured throughput and labelled as a projection.

---

## 10. Reproducibility

- Fixed seed (42) for Python, NumPy and Torch; cuDNN deterministic.
- All configuration in `configs/default.json`; overridable via `--set key=value`.
- No absolute or machine-specific paths — everything resolves from the project root.
- Per-stage CLI entry points; expensive stages reuse existing outputs unless `--force`.
- Embedding extraction is shard-checkpointed and resumable.
- Environment (Python, torch, CUDA, GPU, library versions) recorded in every
  metrics file.

---

## 11. Limitations

1. **No ground-truth taxonomy** — the dominant limitation. No supervised metric
   is computable, so "useful for downstream taxonomic analysis" is assessed by
   proxy, not directly.
2. **Single sample** — no abundance features, so the paper's student input
   cannot be reproduced even in principle.
3. **Unassembled reads** — the paper's premise is that assembly supplies context;
   we deliberately test the regime where it does not.
4. **Foundation-model inference was run on a subset**, not all 3M reads. The
   full-dataset figure is a projection, clearly labelled.
5. **Out-of-distribution context length** — GenomeOcean was pretrained on
   assembled sequence. 151 bp is in-spec for length, not for training context.
6. **Mate-pair retrieval is a proxy**, with the locus-sharing confound stated.
   Absolute scores are low for both representations (top-1 2.16% and 2.93%
   against 1% chance): the recoverable same-fragment signal in 151 bp amplicon
   reads is weak, whichever representation is used.
7. **Amplicon, not shotgun.** This was discovered during the run (§7.1), not
   known in advance. Conserved primer regions occupy a substantial fraction of
   every read, which limits how much organism-discriminating sequence either
   representation has to work with.
8. **Single foundation model tested** — no claim is made about genomic
   foundation models in general.
9. **One sequencing run, one site** — no claim about deep-sea eDNA in general.
