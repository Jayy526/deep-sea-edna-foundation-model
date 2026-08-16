# Stage 4 Report — Knowledge Distillation (TaxDistill's core mechanism)

Stage 3 produced noisy reference-derived labels. That unblocked the one part of TaxDistill that had not been implementable: the **distillation loop the paper is named for**.

---

## A. What was implemented

| Branch | Input | Trained by | Label |
|---|---|---|---|
| **Teacher** | frozen GenomeOcean embedding (1,536-d) + head | its own cross-entropy only | PAPER-DERIVED |
| **Student + KD** | TNF + total abundance (138-d) MLP | cross-entropy **+ KD against the teacher** | PAPER-DERIVED |
| **Student alone** | identical MLP, identical init | cross-entropy only | our control |

KD loss is `T² · KL(softmax(student/T) ‖ softmax(teacher/T))`, weighted by α. The KD gradient reaches the student only; the teacher never learns from the student, exactly as the paper specifies.

**The third branch is the point.** The paper compares distilled against undistilled baselines; without a student trained on identical data with an identical initialisation and no KD term, any improvement could be capacity or seed. That control is what makes the claim testable.

### Two fairness decisions worth stating

1. **The teacher head matches the student's architecture.** A first run gave the teacher a linear head and the student an MLP; the teacher then lost to its own student, and "KD does not help" would have been an artefact of head capacity rather than a statement about representations. Both now use the same MLP, isolating foundation embedding vs hand-crafted features.
2. **Both students share an initialisation and a seed**, so the only difference between them is the KD term.

### Adaptations from the paper

- **Abundance:** the paper's student uses per-environment abundances across K samples plus a total. We have one sample, so only `log10(read count)` is computable. Not omitted for convenience — it does not exist in this data.
- **Deep hierarchical loss NOT implemented.** The paper uses Valmadre (2022) hierarchical loss over the taxonomy tree; we use plain cross-entropy.
- **Coarse ranks.** The paper evaluates at species level; reference coverage limits us to class (3 classes) and order (5).

## B. Results

### Class level — 3 classes, 9,109 labelled variants

| Branch | Accuracy | Macro F1 | Weighted F1 | ECE |
|---|---|---|---|---|
| Teacher (GenomeOcean) | 0.9876 | 0.9821 | 0.9875 | 0.0108 |
| Student + KD | 0.9912 | 0.9886 | 0.9912 | 0.0041 |
| Student alone | 0.9905 | 0.9877 | 0.9905 | 0.0059 |

**Effect of the KD term:** +0.07 pp accuracy (95% CI [-0.36, +0.50], McNemar p = 1.000), macro F1 +0.0009. 5 test items corrected by KD, 4 broken. **no significant KD effect (p=1.000)**.

### Order level — 5 classes, 5,773 labelled variants

| Branch | Accuracy | Macro F1 | Weighted F1 | ECE |
|---|---|---|---|---|
| Teacher (GenomeOcean) | 0.9907 | 0.9429 | 0.9912 | 0.0081 |
| Student + KD | 0.9907 | 0.9112 | 0.9908 | 0.0063 |
| Student alone | 0.9896 | 0.9201 | 0.9894 | 0.0080 |

**Effect of the KD term:** +0.12 pp accuracy (95% CI [-0.39, +0.62], McNemar p = 1.000), macro F1 -0.0088. 3 test items corrected by KD, 2 broken. **no significant KD effect (p=1.000)**.

Parameter counts: teacher head 394,243, student 36,355, foundation backbone trainable **0**.

## C. Ablations

Rank order, 5,773 variants, 5 classes. Each row is a full retrain from an identical initialisation.

| α | Student+KD accuracy | Student+KD macro F1 | Student alone macro F1 |
|---|---|---|---|
| 0.00 | 0.9896 | 0.9201 | 0.9201 |
| 0.20 | 0.9931 | 0.9258 | 0.9201 |
| 0.30 | 0.9931 | 0.9258 | 0.9201 |
| 0.40 | 0.9907 | 0.9112 | 0.9201 |
| 0.50 | 0.9919 | 0.9123 | 0.9201 |
| 0.60 | 0.9919 | 0.9123 | 0.9201 |
| 0.70 | 0.9907 | 0.9100 | 0.9201 |
| 0.80 | 0.9896 | 0.9076 | 0.9201 |

| T | Student+KD accuracy | Student+KD macro F1 |
|---|---|---|
| 1.0 | 0.9896 | 0.9201 |
| 2.0 | 0.9907 | 0.9212 |
| 3.0 | 0.9919 | 0.9235 |
| 4.0 | 0.9907 | 0.9112 |
| 5.0 | 0.9896 | 0.9089 |
| 6.0 | 0.9896 | 0.9089 |

**Implementation sanity check:** at α = 0 the KD student reproduces the student-alone scores *exactly* (0.9896 / 0.9201), confirming the KD term is the only thing separating the two branches.

KD peaks around α ≈ 0.2 (macro F1 0.9258 vs 0.9201 undistilled) and **degrades above α ≈ 0.4**, where the soft targets start to outweigh the labels. Temperature shows the same shape, peaking near T ≈ 3.

## D. Interpretation

**Distillation gives at best a marginal, non-significant improvement here, and hurts at high α.** That is a null result, and it is reported as one.

The mechanism is visible in the numbers: **the task is saturated.** All three branches sit at 98.8–99.1% accuracy. A 138-dimensional k-mer MLP already solves 3–5 coarse foraminiferal classes almost perfectly, so there is nearly no headroom for teacher soft labels to add anything. Distillation helps when the student is capacity- or information-limited relative to the teacher; here it is neither.

**This does not refute TaxDistill.** The paper evaluates species-level annotation across seven CAMI2 datasets, where discrimination is far harder and the student has real headroom. Our test is at class and order level on a single sample, because that is the finest resolution the reference database supports (§ Stage 3: only 5.2% of variants can be assigned a genus). The honest conclusion is that **this dataset cannot test the paper's claim in the regime where the claim is interesting**.

**What would make it testable:** finer taxonomic resolution — which needs better reference coverage of deep-sea benthic Foraminifera, not a better encoder or a better distillation scheme.

## E. Limitations

1. **Labels are noisy and reference-derived**, with measured CV error of 5.6% (class) and 15.9% (order) on reference sequences, higher on real variants.
2. **The task is saturated at 99% accuracy**, which is the direct cause of the null result and limits what any conclusion can mean.
3. **Coarse ranks only** (3 and 5 classes) versus the paper's species level.
4. **Partly circular evaluation**: labels come from an 8-mer classifier, and the student is k-mer based, so the student's input is closer to the label-generating process than the teacher's is.
5. **Single sample**, so the paper's K-environment abundance features do not exist.
6. **No deep hierarchical loss**, so this is not a complete reimplementation of the paper's objective.
7. **One seed per configuration.** Differences of a few tenths of a percentage point are within run-to-run variation and should not be over-read; this is why the McNemar test rather than the point estimate carries the conclusion.

## F. Conclusions

1. **TaxDistill's distillation loop is now implemented and runs end to end** — frozen teacher, lightweight student, temperature-scaled KD, with the undistilled control the claim requires.
2. **The implementation is verified** by the α = 0 identity check.
3. **On this dataset, KD does not significantly help** (p = 1.000 at both ranks), peaking at roughly +0.35 pp accuracy near α ≈ 0.2–0.3 and degrading beyond α ≈ 0.4.
4. **The reason is task saturation, not a failure of the method.** At 99% accuracy there is nothing left to distil.
5. **The binding constraint remains reference coverage** — the same conclusion stage 3 reached, now reinforced from a second direction.

---

*Generated by `python main.py distill-report`.*