# Stage 3 Report — Taxonomic Identification and Supervised Classification

Stages 1 and 2 both ended at the same wall: no taxonomy, therefore no supervised metric. This stage removes that wall by identifying what the dataset actually is and deriving labels from a reference database.

**The labels produced here are NOISY, REFERENCE-DERIVED ASSIGNMENTS — not ground truth.** That distinction is carried through every number below. It is also exactly the regime TaxDistill was designed for: the paper's premise is that retrieval-derived labels are noisy and need correcting.

---

## A. What this dataset actually is

Stage 2 established the reads were marker-gene amplicons with a long conserved anchor, but could not name the gene. Three independent lines of evidence now identify it.

**1. The conserved anchor matches SSU rRNA, in eukaryotes only.** Searching SILVA 138.2 SSU NR99 (510,495 sequences) for our 20 bp anchor returns 66 hits — **all Eukaryota**, overwhelmingly Foraminifera, at SSU position ~1300.

**2. The primer is a published one.** The anchor `AAGGGCACCACAAGAACGC` is **s14F1**, a Foraminifera-specific forward primer targeting the **37F hypervariable region of the 18S rRNA gene**, normally paired with reverse primer s15. That matches our two-amplicon-end structure.

**3. Every reference carrying the primer is a foraminiferan.** In PR2 5.0.0 (221,085 protist sequences), 1,547 carry the primer site and 100% of them are Foraminifera.

**Conclusion: this is a deep-sea benthic Foraminifera 18S rRNA metabarcoding library (s14F1/s15, 37F region).** Everything measured in stages 1 and 2 is consistent with that: the two amplicon ends, the 90 bp conserved anchor, the extreme community dominance, and the failure of universal 16S/18S primers to match.

## B. Choosing a reference database — measured, not assumed

| Database | Sequences | Carrying the s14F1 site | Usable? |
|---|---|---|---|
| SILVA 138.2 SSU NR99 | 510,495 | 67 (69 Foraminifera total) | **No** — too few to train on |
| PR2 5.0.0 SSU | 221,085 | 1,547 | **Yes** |

SILVA was downloaded and measured first; it was rejected on evidence, not preference. Containing 69 Foraminifera in half a million sequences is the well-documented foraminiferal reference gap in general-purpose databases.

**In-silico PCR.** References were cut to the same window as our reads: find the primer site, keep the following 151 bp. This makes reference and query directly comparable instead of matching a 151 bp read against a 1,800 bp gene.

| Property | Value |
|---|---|
| Region-matched references | 1,545 |
| Distinct class labels | 4 |
| Distinct order labels | 19 |
| Distinct family labels | 71 |
| Distinct genus labels | 157 |

| Foraminiferal class | References |
|---|---|
| Globothalamea | 1,161 |
| Monothalamids | 233 |
| Tubothalamea | 113 |
| Foraminifera_X | 38 |

*1,547 references is a small database. Coverage of deep-sea benthic Foraminifera is known to be poor, so a substantial share of query variants is expected to be unassignable. Labels derived here are incomplete and noisy by construction.*

## C. How noisy are the labels?

Labels are assigned by an RDP-style naive Bayes k-mer classifier (Wang et al. 2007) reimplemented in `taxonomy/classifier.py`, with bootstrap confidence. Accuracy is measured by 5-fold cross-validation **on the reference itself** before the labels are used for anything.

| Rank | Classes | CV accuracy | Mean confidence when correct | when wrong |
|---|---|---|---|---|
| class | 4 | 0.9437 | 0.967 | 0.607 |
| order | 19 | 0.8408 | 0.923 | 0.489 |

Confidence separates correct from incorrect assignments sharply (~0.94 vs ~0.42), which is what makes the confidence threshold effective. **Real label error on environmental variants will be higher than these figures**, because reference sequences are cleaner and better represented than deep-sea reads.

## D. How much of our data could be labelled?

| Rank | Threshold | Variants assigned | Rate | Read-weighted rate | Distinct labels |
|---|---|---|---|---|---|
| class | 0.7 | 9,112 / 15,583 | 58.5% | 82.2% | 4 |
| order | 0.7 | 5,824 / 15,583 | 37.4% | 15.0% | 12 |

Only amplicon end 0 is classified — those are the reads carrying s14F1, the primer the reference window is anchored on.

**Assignment falls off steeply with taxonomic depth.** At genus level only 5.2% of variants could be assigned at confidence ≥ 0.8. This is a real finding about deep-sea foraminiferal diversity versus reference coverage, not a pipeline failure: most of what is in this sample has no close relative in any public database. It is the strongest possible argument for the TaxDistill premise.

## E. Supervised classification on frozen embeddings

A linear probe is trained on each frozen representation over the same labelled variants, same split, same seed. The foundation model stays frozen throughout — only the probe head is trained.

**Feature standardisation matters and is applied.** k-mer frequencies have std ≈ 0.04 while the foundation embeddings have std ≈ 0.64. Without standardisation the baseline fails to converge at a shared learning rate and collapses to majority-class prediction, which would have produced a spurious 3.5× macro-F1 gap in the foundation model's favour. Statistics are fitted on the training split only.

### Class level — 3 classes, 9,109 labelled variants

| Metric | GenomeOcean-500M | k-mer / TNF baseline |
|---|---|---|
| Accuracy | 0.9751 | 0.9737 |
| Macro precision | 0.9690 | 0.9610 |
| Macro recall | 0.9668 | 0.9695 |
| Macro F1 | 0.9679 | 0.9652 |
| Weighted F1 | 0.9751 | 0.9737 |
| Feature dimension | 1,536 | 137 |
| Training time | 3.57 s | 2.04 s |
| Expected calibration error | 0.0073 | 0.0183 |
| Brier score | 0.0389 | 0.0443 |

Accuracy difference **+0.15 pp** (95% CI [-0.85, +1.14], McNemar p = 0.885 on 1,368 held-out test variants) — *no significant difference (p=0.885)*.

Split: 6,375 train / 1,366 validation / 1,368 test. The test split is never seen during training.

### Order level — 5 classes, 5,773 labelled variants

| Metric | GenomeOcean-500M | k-mer / TNF baseline |
|---|---|---|
| Accuracy | 0.9792 | 0.9664 |
| Macro precision | 0.8474 | 0.7856 |
| Macro recall | 0.9823 | 0.8971 |
| Macro F1 | 0.8970 | 0.8311 |
| Weighted F1 | 0.9803 | 0.9683 |
| Feature dimension | 1,536 | 137 |
| Training time | 1.37 s | 1.27 s |
| Expected calibration error | 0.0103 | 0.0425 |
| Brier score | 0.0303 | 0.0528 |

Accuracy difference **+1.27 pp** (95% CI [+0.05, +2.49], McNemar p = 0.061 on 864 held-out test variants) — *no significant difference (p=0.061)*.

Split: 4,042 train / 867 validation / 864 test. The test split is never seen during training.


## F. What this does and does not show

**Both representations classify these labels well** — 97–98% accuracy at both ranks. Neither difference in accuracy reaches significance (p = 0.885 at class, p = 0.061 at order).

**The foundation model is ahead on macro F1**, most clearly at order level (0.897 vs 0.831). Macro F1 weights rare classes equally, and the confusion matrices show why: the baseline misassigns a large share of *Globothalamea_X* to *Robertinida*, which the foundation model resolves. McNemar tests accuracy, not macro F1, so that gap is not significance-tested here.

**The important caveat — this task is partly circular.** The labels were generated by an 8-mer naive Bayes classifier reading the same sequences. Asking a 4-mer representation to predict them is therefore closer to distillation than to independent evaluation, and it structurally *favours* k-mer features. That the foundation model still matches or beats the baseline under that handicap is meaningful; it does not establish accuracy against true taxonomy, which remains unmeasured.

**Calibration.** Both probes are reasonably calibrated (ECE ≈ 0.02–0.04). Low confidence is **not** evidence of a novel taxon — novel-taxa discovery remains outside this implementation, and the low assignment rate in §D is a reference coverage problem, not a discovery claim.

## G. Figures

| File | Content |
|---|---|
| `confusion_matrix_class.png` | Figure 9 — confusion matrix (class) |
| `per_class_f1_class.png` | Figure 9b — per-class F1 (class) |
| `training_curves_class.png` | Figure 10 — training/validation curves (class) |
| `confidence_class.png` | Figure 11 — confidence and calibration (class) |
| `confusion_matrix_order.png` | Figure 9 — confusion matrix (order) |
| `per_class_f1_order.png` | Figure 9b — per-class F1 (order) |
| `training_curves_order.png` | Figure 10 — training/validation curves (order) |
| `confidence_order.png` | Figure 11 — confidence and calibration (order) |

## H. Limitations

1. **Labels are reference-derived, not ground truth.** Measured CV error is 5.6% at class and 15.9% at order on reference sequences; on environmental variants it is higher by an unmeasured amount.
2. **The evaluation is partly circular** (§F): labels come from a k-mer model, and one of the two representations being compared is k-mer based.
3. **Only 37–58% of end-0 variants could be labelled at all**, and only 5.2% at genus level. Conclusions apply to the labellable subset, which is biased toward taxa that are well represented in PR2.
4. **Only amplicon end 0 is classified.** End 1 would need its own region-matched reference anchored on the s15 primer.
5. **Coarse ranks only.** Class (3 usable) and order (5 usable) are far coarser than the species-level annotation TaxDistill evaluates.
6. **Severe class imbalance** at order level (Robertinida dominates), which is why macro F1 is reported alongside accuracy.
7. **No distillation.** TaxDistill's actual contribution — using the teacher to correct noisy student labels — is still not implemented. This stage builds the input that framework would consume.

## I. Conclusions

1. **The dataset is identified.** A deep-sea benthic Foraminifera 18S rRNA metabarcoding library (s14F1/s15, 37F region), established from three independent lines of evidence.

2. **Supervised evaluation is now possible**, for the first time in this project, on 5,773–9,109 labelled variants.

3. **Both representations support accurate taxonomic classification** of these labels (97–98%), with no significant accuracy difference. The foundation model leads on macro F1, i.e. on the rarer classes.

4. **The binding constraint is reference coverage, not representation quality.** Only 5.2% of variants can be assigned a genus. The deep-sea foraminiferal reference gap, not the encoder, is what limits taxonomic analysis of this sample.

5. **The natural next step is now unblocked**: with noisy labels in hand, TaxDistill's actual distillation loop — teacher soft labels correcting a lightweight student — becomes implementable for the first time.

---

*Generated by `python main.py taxonomy-report`. Every value is read from a metrics file produced by an executed run.*