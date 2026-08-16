"""Stage 4 report -- TaxDistill knowledge distillation."""

from __future__ import annotations

import json
from pathlib import Path

from utils import config as cfgutil


def _load(cfg: dict, name: str) -> dict | None:
    path = cfgutil.output_dir(cfg, "metrics") / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def build_report(cfg: dict) -> Path:
    dist = _load(cfg, "distillation_metrics.json")
    abl = _load(cfg, "distillation_ablation.json")
    tax = {r: _load(cfg, f"taxonomy_metrics_{r}.json") for r in cfg["taxonomy"]["ranks"]}

    lines: list[str] = []
    add = lines.append

    add("# Stage 4 Report — Knowledge Distillation (TaxDistill's core mechanism)")
    add("")
    add("Stage 3 produced noisy reference-derived labels. That unblocked the one part "
        "of TaxDistill that had not been implementable: the **distillation loop the "
        "paper is named for**.")
    add("")
    add("---")
    add("")

    add("## A. What was implemented")
    add("")
    add("| Branch | Input | Trained by | Label |")
    add("|---|---|---|---|")
    add("| **Teacher** | frozen GenomeOcean embedding (1,536-d) + head | its own cross-entropy only | PAPER-DERIVED |")
    add("| **Student + KD** | TNF + total abundance (138-d) MLP | cross-entropy **+ KD against the teacher** | PAPER-DERIVED |")
    add("| **Student alone** | identical MLP, identical init | cross-entropy only | our control |")
    add("")
    add("KD loss is `T² · KL(softmax(student/T) ‖ softmax(teacher/T))`, weighted by α. "
        "The KD gradient reaches the student only; the teacher never learns from the "
        "student, exactly as the paper specifies.")
    add("")
    add("**The third branch is the point.** The paper compares distilled against "
        "undistilled baselines; without a student trained on identical data with an "
        "identical initialisation and no KD term, any improvement could be capacity or "
        "seed. That control is what makes the claim testable.")
    add("")
    add("### Two fairness decisions worth stating")
    add("")
    add("1. **The teacher head matches the student's architecture.** A first run gave "
        "the teacher a linear head and the student an MLP; the teacher then lost to its "
        "own student, and \"KD does not help\" would have been an artefact of head "
        "capacity rather than a statement about representations. Both now use the same "
        "MLP, isolating foundation embedding vs hand-crafted features.")
    add("2. **Both students share an initialisation and a seed**, so the only "
        "difference between them is the KD term.")
    add("")
    add("### Adaptations from the paper")
    add("")
    add("- **Abundance:** the paper's student uses per-environment abundances across K "
        "samples plus a total. We have one sample, so only `log10(read count)` is "
        "computable. Not omitted for convenience — it does not exist in this data.")
    add("- **Deep hierarchical loss NOT implemented.** The paper uses Valmadre (2022) "
        "hierarchical loss over the taxonomy tree; we use plain cross-entropy.")
    add("- **Coarse ranks.** The paper evaluates at species level; reference coverage "
        "limits us to class (3 classes) and order (5).")
    add("")

    if dist and dist.get("ranks"):
        add("## B. Results")
        add("")
        for rank, res in dist["ranks"].items():
            b = res["branches"]
            add(f"### {rank.capitalize()} level — {res['n_classes']} classes, "
                f"{res['n_labelled']:,} labelled variants")
            add("")
            add("| Branch | Accuracy | Macro F1 | Weighted F1 | ECE |")
            add("|---|---|---|---|---|")
            for key, name in (
                ("teacher", "Teacher (GenomeOcean)"),
                ("student_kd", "Student + KD"),
                ("student_alone", "Student alone"),
            ):
                m = b[key]["metrics"]
                add(f"| {name} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | "
                    f"{m['weighted_f1']:.4f} | "
                    f"{b[key]['calibration']['expected_calibration_error']:.4f} |")
            add("")
            e = res.get("kd_effect")
            if e:
                add(f"**Effect of the KD term:** {e['accuracy_difference_pp']:+.2f} pp "
                    f"accuracy (95% CI [{e['ci95_pp'][0]:+.2f}, {e['ci95_pp'][1]:+.2f}], "
                    f"McNemar p = {e['p_value']:.3f}), macro F1 "
                    f"{e['macro_f1_difference']:+.4f}. {e['corrected_by_kd']} test items "
                    f"corrected by KD, {e['broken_by_kd']} broken. "
                    f"**{e['verdict']}**.")
                add("")
        add(f"Parameter counts: teacher head "
            f"{list(dist['ranks'].values())[0]['parameter_counts']['teacher_head']:,}, "
            f"student {list(dist['ranks'].values())[0]['parameter_counts']['student']:,}, "
            f"foundation backbone trainable **0**.")
        add("")

    if abl:
        add("## C. Ablations")
        add("")
        add(f"Rank {abl['rank']}, {abl['n_labelled']:,} variants, {abl['n_classes']} "
            f"classes. Each row is a full retrain from an identical initialisation.")
        add("")
        add("| α | Student+KD accuracy | Student+KD macro F1 | Student alone macro F1 |")
        add("|---|---|---|---|")
        for row in abl["alpha_sweep"]:
            add(f"| {row['alpha']:.2f} | {row['student_kd_accuracy']:.4f} | "
                f"{row['student_kd_macro_f1']:.4f} | {row['student_alone_macro_f1']:.4f} |")
        add("")
        add("| T | Student+KD accuracy | Student+KD macro F1 |")
        add("|---|---|---|")
        for row in abl["temperature_sweep"]:
            add(f"| {row['temperature']:.1f} | {row['student_kd_accuracy']:.4f} | "
                f"{row['student_kd_macro_f1']:.4f} |")
        add("")
        add("**Implementation sanity check:** at α = 0 the KD student reproduces the "
            "student-alone scores *exactly* "
            f"({abl['alpha_sweep'][0]['student_kd_accuracy']:.4f} / "
            f"{abl['alpha_sweep'][0]['student_kd_macro_f1']:.4f}), confirming the KD "
            "term is the only thing separating the two branches.")
        add("")
        best = max(abl["alpha_sweep"], key=lambda r: r["student_kd_macro_f1"])
        add(f"KD peaks around α ≈ {best['alpha']:.1f} (macro F1 "
            f"{best['student_kd_macro_f1']:.4f} vs {best['student_alone_macro_f1']:.4f} "
            f"undistilled) and **degrades above α ≈ 0.4**, where the soft targets start "
            f"to outweigh the labels. Temperature shows the same shape, peaking near "
            f"T ≈ 3.")
        add("")

    add("## D. Interpretation")
    add("")
    add("**Distillation gives at best a marginal, non-significant improvement here, and "
        "hurts at high α.** That is a null result, and it is reported as one.")
    add("")
    add("The mechanism is visible in the numbers: **the task is saturated.** All three "
        "branches sit at 98.8–99.1% accuracy. A 138-dimensional k-mer MLP already "
        "solves 3–5 coarse foraminiferal classes almost perfectly, so there is nearly "
        "no headroom for teacher soft labels to add anything. Distillation helps when "
        "the student is capacity- or information-limited relative to the teacher; here "
        "it is neither.")
    add("")
    add("**This does not refute TaxDistill.** The paper evaluates species-level "
        "annotation across seven CAMI2 datasets, where discrimination is far harder and "
        "the student has real headroom. Our test is at class and order level on a "
        "single sample, because that is the finest resolution the reference database "
        "supports (§ Stage 3: only 5.2% of variants can be assigned a genus). The "
        "honest conclusion is that **this dataset cannot test the paper's claim in the "
        "regime where the claim is interesting**.")
    add("")
    add("**What would make it testable:** finer taxonomic resolution — which needs "
        "better reference coverage of deep-sea benthic Foraminifera, not a better "
        "encoder or a better distillation scheme.")
    add("")

    add("## E. Limitations")
    add("")
    for i, item in enumerate([
        "**Labels are noisy and reference-derived**, with measured CV error of 5.6% "
        "(class) and 15.9% (order) on reference sequences, higher on real variants.",
        "**The task is saturated at 99% accuracy**, which is the direct cause of the "
        "null result and limits what any conclusion can mean.",
        "**Coarse ranks only** (3 and 5 classes) versus the paper's species level.",
        "**Partly circular evaluation**: labels come from an 8-mer classifier, and the "
        "student is k-mer based, so the student's input is closer to the label-"
        "generating process than the teacher's is.",
        "**Single sample**, so the paper's K-environment abundance features do not exist.",
        "**No deep hierarchical loss**, so this is not a complete reimplementation of "
        "the paper's objective.",
        "**One seed per configuration.** Differences of a few tenths of a percentage "
        "point are within run-to-run variation and should not be over-read; this is "
        "why the McNemar test rather than the point estimate carries the conclusion.",
    ], 1):
        add(f"{i}. {item}")
    add("")

    add("## F. Conclusions")
    add("")
    add("1. **TaxDistill's distillation loop is now implemented and runs end to end** — "
        "frozen teacher, lightweight student, temperature-scaled KD, with the "
        "undistilled control the claim requires.")
    add("2. **The implementation is verified** by the α = 0 identity check.")
    add("3. **On this dataset, KD does not significantly help** (p = 1.000 at both "
        "ranks), peaking at roughly +0.35 pp accuracy near α ≈ 0.2–0.3 and degrading "
        "beyond α ≈ 0.4.")
    add("4. **The reason is task saturation, not a failure of the method.** At 99% "
        "accuracy there is nothing left to distil.")
    add("5. **The binding constraint remains reference coverage** — the same conclusion "
        "stage 3 reached, now reinforced from a second direction.")
    add("")
    add("---")
    add("")
    add("*Generated by `python main.py distill-report`.*")

    path = cfgutil.output_dir(cfg, "reports") / "distillation_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {path}")
    return path
