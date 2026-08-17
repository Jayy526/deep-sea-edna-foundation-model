# Deep-sea eDNA — genomic foundation model representation pipeline

A four-stage pipeline that encodes raw deep-sea eDNA short reads with a
genomic foundation model and tests whether the resulting embeddings support
downstream taxonomic analysis.

**A TaxDistill-inspired adaptation for short reads, not a reproduction.**
The paper ([arXiv:2605.28868](https://arxiv.org/abs/2605.28868)) uses
assembled contigs ≥ 2,000 bp; this dataset is 151 bp raw reads. Differences
are documented in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

---

## Research question

> Can raw deep-sea eDNA short reads be processed through a genomic
> foundation-model pipeline to obtain embeddings that support downstream
> taxonomic analysis?

**Answer: yes, technically — but the binding constraint is reference-database
coverage, not the encoder.**

## Headline findings

| | |
|---|---|
| Dataset | 3,026,920 reads · 151 bp · 648.9 MB · 1,513,460 paired-end clusters |
| **Identified as** | **deep-sea benthic Foraminifera 18S rRNA metabarcoding (s14F1/s15, 37F region)** |
| Encoder | `pGenomeOcean/GenomeOcean-500M` — 541 M params, 0 trainable |
| Compatibility | 151 bp → ~34 BPE tokens (3.6% of the model's limit); 0 `[UNK]`, 0 truncation, **0 padding** |
| Throughput | 476 reads/s on a 6 GB laptop GPU, peak 1.2 GB VRAM |
| Unique variants | 246,001 from 3M reads — **7 variants are 50% of the library** |
| Taxonomic labels | reference-derived, **only 5.2% assignable at genus level** |
| Classification | 97–98% accuracy at class/order on both representations |
| Distillation | **no significant benefit** — task saturated at 99% |

Four reports in [`outputs/reports/`](outputs/reports/) carry every number.

---

## Target architecture

This repo's own finding — **reference-database coverage, not the encoder,
caps taxonomic resolution** (only 5.2% of variants reach genus) — motivates a
target design for a full open-world eDNA pipeline: screen cheaply against
references first, spend foundation-model compute only on what references
can't resolve, and treat "no match" as a discovery signal to cluster and
validate rather than a dead end.

**This is a design document, not a claim about this repo's code.** See
*Status* below.

```
                         RAW eDNA
                       18S / COI reads
                              │
                              ▼
                  QC + preprocessing
                  filtering · trimming · ASVs
                              │
                              ▼
                  Unique representative sequences
                        (dereplication)
                              │
                              ▼
                            MMseq2
                  fast reference screening
                              │
                 ┌────────────┴─────────────┐
                 │                           │
          high-confidence              low / no match
                 │                           │
                 ▼                           ▼
        Known taxonomic                GenomeOcean encoder
          assignment                     (DNA → embedding)
     (hierarchical taxonomic                  │
             head)               ┌────────────┴────────────┐
                 │                │                         │
                 │          Known-taxon                  HDBSCAN
                 │          classification            (unsupervised
                 │          (taxonomic head)            clustering)
                 │                │                         │
                 │                │              Candidate novel clusters
                 │                │                         │
                 │                │                         ▼
                 │                │              Novelty validation
                 │                │        (cluster stability · similarity
                 │                │         search · phylogenetic evidence)
                 │                │                         │
                 └────────────────┴────────────┬────────────┘
                                                 │
                                                 ▼
                                       XAI — AttnLRP
                              important sequence regions / nucleotides
                                                 │
                                                 ▼
                                   Interpretable evidence
                        (prediction + confidence + attributed positions)
                                                 │
                 ┌───────────────────────────────┼───────────────────────────────┐
                 ▼                               ▼                               ▼
            Abundance                      Biodiversity                  Environmental context
            estimation                     assessment                     WOA23 + INCOIS
         (bias-corrected)             (richness, α/β diversity)      (decoupled from taxonomy —
                 │                               │                    applied post-classification)
                 └───────────────────────────────┴───────────────────────────────┘
                                                 │
                                                 ▼
                                   Ecological interpretation
```

**Why each stage exists**

| Stage | Problem it solves |
|---|---|
| QC + dereplication | Don't send millions of redundant reads into a foundation model |
| MMseq2 | Cheap conventional method first — skip foundation-model compute on strong reference matches |
| GenomeOcean encoder | Poor reference coverage — learn a representation instead of calling divergent sequences "unclassified" |
| HDBSCAN | Genuinely unsupervised discovery — candidate novel groups, no predefined labels |
| Novelty validation | A cluster isn't "novel" just because HDBSCAN drew a boundary — needs stability, similarity search, phylogenetic support |
| AttnLRP | Mandatory XAI, not optional — every prediction ships with the positions that drove it |
| WOA23 + INCOIS | Kept **out of** the taxonomy/novelty branch, applied only after classification — "where/how deep does this lineage occur" is defensible; "DNA + ocean conditions → taxon" is not |

**Tech stack:** Illumina/FASTQ · DADA2/VSEARCH · MMseq2 · PyTorch/ONNX ·
HDBSCAN/Leiden · AttnLRP · PostgreSQL/pgvector · FastAPI/React/TS ·
Snakemake/Docker

**Problem → component:** poor reference DBs → GenomeOcean representation ·
unassigned sequences → foundation-model branch · novel taxa → HDBSCAN ·
classification → MMseq2 + taxonomic head · abundance → read/ASV aggregation
· biodiversity → richness + α/β diversity · compute cost → dedup + MMseq2
early filtering · explainability → AttnLRP · verification → cluster
stability + similarity + phylogenetics · ecological context → WOA23 +
INCOIS

The intended contribution isn't a new neural architecture — it's an
efficient open-world pipeline that uses conventional methods for the easy
cases and learned representations + unsupervised discovery for the
poorly-referenced ones, with interpretable evidence on every prediction.

**Status.** Implemented and measured today (the [Results](#results) below):
QC/dereplication, the GenomeOcean encoder, a k-mer baseline, reference-derived
taxonomic assignment, plus a TaxDistill-style KD experiment outside the
target design. **Not yet implemented:** MMseq2 triage, HDBSCAN + novelty
validation, AttnLRP, abundance/biodiversity/environmental-context modules,
and the database/API/pipeline layers.

---

## Quick start

```bash
pip install -r requirements.txt
python main.py all
```

Expensive stages reuse existing outputs; add `--force` to recompute. Any
config value can be overridden without editing files:

```bash
python main.py distill --set distillation.alpha=0.3 --set distillation.temperature=3
```

Run a single stage (24 available, `python main.py --help` lists them):

```bash
python main.py preprocess    # full-dataset streaming QC + metrics
python main.py encode        # foundation-model embeddings
python main.py dereplicate   # variants + diversity
python main.py assign        # reference + noisy taxonomic labels
python main.py distill       # TaxDistill teacher/student KD
```

**Requirements:** Python 3.13, ~8 GB RAM, ~5 GB disk. CUDA GPU recommended
(CPU fallback works, foundation-model encoding is far slower). ~250 MB
downloaded on first run: GenomeOcean (HuggingFace) + PR2/SILVA references.

---

## Results

### Stage 1 — representation

GenomeOcean-500M accepts 151 bp reads directly: 200,000 reads encoded, 0
failures, 0 dead dimensions. With no taxonomy available, representations
were compared on **paired-end mate retrieval** — given one mate's embedding,
find its partner among 99 distractors (1% chance).

| Metric | k-mer / TNF baseline | GenomeOcean-500M |
|---|---|---|
| Top-1 retrieval | **2.93%** | 2.16% |
| AUROC (mate vs random) | 0.5167 | **0.6200** |
| Throughput | **15,365 reads/s** (CPU) | 434 reads/s (GPU) |

![Baseline vs foundation model: retrieval, AUROC, throughput, representation width](https://raw.githubusercontent.com/Jayy526/deep-sea-edna-foundation-model/main/outputs/figures/model_comparison.png)

Two confounds were controlled for: the data is **amplicon, not shotgun**
(uncontrolled, both scores fall *below* chance), and GenomeOcean is **not
reverse-complement invariant** (correcting mate orientation nearly doubled
its top-1). The baseline is provably RC-invariant, verified bit-for-bit.

### Stage 2 — unsupervised community structure

3,018,522 reads dereplicate to **246,001 unique variants**. Extraordinarily
uneven: **7 variants carry half the library**, one carries 13.17%. 80.9% of
variants are singletons but only 6.6% of reads — mostly sequencing error,
so Chao1 (4.5× observed) is an upper bound, not a species count.

![Dereplication frequency spectrum and observed vs Chao1-estimated richness](https://raw.githubusercontent.com/Jayy526/deep-sea-edna-foundation-model/main/outputs/figures/dereplication.png)

**Only 39.7% of each read varies across the community** — amplicon end 0
opens with a 90 bp block that is 97.2% invariant, bounding what *any*
representation can achieve. Trimming that anchor reverses the stage-1
comparison (McNemar's exact test, paired):

| Comparison | Difference | p | Verdict |
|---|---|---|---|
| Full read: foundation vs baseline | −0.78 pp | 2e−10 | baseline genuinely better |
| Variable region: foundation vs baseline | −0.06 pp | 0.63 | **statistically tied** |
| Baseline: variable vs full read | −0.55 pp | 9e−09 | **loses the anchor's help** |
| Foundation: variable vs full read | +0.16 pp | 0.09 | unaffected |

On the informative region GenomeOcean holds AUROC 0.582 while the baseline
falls to 0.497 — chance.

### Stage 3 — dataset identified, supervised classification

Three lines of evidence identify the library: the conserved anchor matches
**66 of 510,495** SILVA SSU sequences (all Eukaryota, overwhelmingly
Foraminifera); it's `AAGGGCACCACAAGAACGC` — published primer **s14F1**, 37F
region of 18S rRNA; all **1,547** PR2 sequences carrying that primer are
Foraminifera. SILVA was tried and **rejected on evidence** (69 Foraminifera
in half a million); PR2 has 1,547.

Labels come from an RDP-style naive Bayes k-mer classifier, cross-validated
before use: 94.4% at class, 84.1% at order. **Assignment falls off steeply
with depth** — 58.5% of variants labelled at class, 37.4% at order, **5.2%
at genus**.

| Rank | | Accuracy | Macro F1 | ECE |
|---|---|---|---|---|
| class (3, n=9,109) | GenomeOcean | 0.9751 | **0.9679** | 0.021 |
| | k-mer / TNF | 0.9737 | 0.9652 | 0.036 |
| order (5, n=5,773) | GenomeOcean | **0.9792** | **0.8970** | 0.024 |
| | k-mer / TNF | 0.9664 | 0.8311 | 0.041 |

Neither accuracy difference is significant (McNemar p=0.885, p=0.061). The
foundation model leads on macro F1 — the baseline misassigns
*Globothalamea_X* to *Robertinida*, which GenomeOcean resolves.

![Row-normalised confusion matrices at class rank, held-out test split](https://raw.githubusercontent.com/Jayy526/deep-sea-edna-foundation-model/main/outputs/figures/confusion_matrix_class.png)

### Stage 4 — TaxDistill's distillation loop

Three branches, same split/seed/student init:

| Rank | Teacher | Student + KD | Student alone | KD effect |
|---|---|---|---|---|
| class (3) | 0.9876 | 0.9912 | 0.9905 | +0.07 pp, p=1.000 |
| order (5) | 0.9907 | 0.9907 | 0.9896 | +0.11 pp, p=1.000 |

**No significant KD benefit.** The α/T ablation rules out a hyperparameter
artefact — KD peaks near α≈0.2–0.3 then degrades above 0.4; at α=0 the KD
student reproduces student-alone exactly. **Why:** the task is saturated —
all branches sit at 98.8–99.1%, a 138-d k-mer MLP already solves 3–5 coarse
classes almost perfectly.

**This does not refute TaxDistill** — the paper evaluates species-level
annotation, where the student has real headroom. Reference coverage caps
this dataset at class/order, so it can't test the claim where it matters.

---

## Project structure

```
main.py                  24-stage CLI; every stage reusable and resumable
configs/default.json     every threshold, size, seed and hyperparameter

preprocessing/           streaming FASTA parser, QC, statistics
models/                  encoder interface, GenomeOcean, k-mer baseline, classifier
embeddings/              sharded storage with checkpoint/resume, batch inference
analysis/                PCA, UMAP, dereplication, diversity, community, amplicon
evaluation/              retrieval benchmarks, comparison, performance
taxonomy/                reference building, naive Bayes classifier, assignment
training/                TaxDistill knowledge distillation
visualization/           all figures + architecture diagram
reporting/               the four reports
tests/                   61 tests
docs/METHODOLOGY.md      paper-vs-adaptation, labelled component by component
```

**Swapping the encoder** is a one-class change: implement `SequenceEncoder`
(`models/encoder.py`) and register it. Preprocessing, storage, analysis and
reporting are untouched.

---

## Data availability

Large artefacts are **not** in this repository:

| Not committed | Size | How to obtain |
|---|---|---|
| `SRR26872904.fasta/` | 649 MB | NCBI SRA accession `SRR26872904` |
| `data/reference/` | 245 MB | auto-downloaded by `python main.py assign` (PR2 5.0.0, SILVA 138.2) |
| `outputs/embeddings/` | 2.6 GB | regenerated by `python main.py encode baseline` |

Committed: all source, configs, tests, docs, **33 figures**, **41 metrics
files**, **4 reports** — every result is inspectable without re-running
anything.

---

## Tests

```bash
python -m pytest tests/ -q     # 61 tests, ~9 s
```

Hand-implemented quantities are each validated against an **independent**
source:

| Implementation | Validated against |
|---|---|
| Hurlbert rarefaction | brute-force Monte Carlo subsampling |
| Rank-based AUROC | `sklearn.metrics.roc_auc_score` |
| Shannon entropy | `scipy.stats.entropy` |
| Chao1, Good's coverage, Hill numbers | closed-form values computed by hand |
| Classification metrics | `sklearn` accuracy / macro-F1 / weighted-F1 |
| Expected calibration error | synthetic calibrated & overconfident predictors |
| Streaming digest dereplication | a naive `Counter` over full sequences |
| Canonical k-mer vectors | RC-invariance, plus a non-canonical control |

---

## Scientific integrity

Deliberately **not** done: padding reads toward the paper's 2,000 bp
threshold; a `≥2,000 bp` filter (would discard 100% of the reads); inventing
labels (stages 1–2 report supervised metrics as unavailable; stage 3 labels
are reference-derived and marked **noisy, not ground truth**); fabricating
compatibility (151 bp support is measured, not assumed); declaring a winner
on noise (paired McNemar throughout); claiming to reproduce TaxDistill.

Anything not measurable is reported verbatim as *"Not available with the
current dataset/experimental setup."*

Two errors found in setup and corrected, documented in the reports: feature
standardisation (without it the baseline collapsed to majority-class, a
spurious 3.5× macro-F1 gap) and teacher head capacity (a linear teacher vs
MLP student made "KD doesn't help" an artefact).

## Limitations

1. Labels are reference-derived, not ground truth (5.6% / 15.9% measured CV error).
2. Only 5.2% of variants assignable at genus — conclusions apply to a biased subset.
3. Coarse ranks only (3 and 5 classes) vs. the paper's species level.
4. Partly circular evaluation: labels come from a k-mer model, one representation is k-mer based.
5. Single sample, single site, single sequencing run — no cross-environment abundance.
6. Foundation-model inference on subsets, not all 3M reads (full cost is a measured projection).
7. Deep hierarchical loss not implemented.

## Reproducibility

Fixed seed (42) across Python/NumPy/Torch, cuDNN deterministic, all settings
in `configs/default.json`, no machine-specific paths, per-stage entry
points, resumable embedding extraction, full environment recorded in every
metrics file. Re-running the full pipeline reproduces every reported number
exactly (verified).
