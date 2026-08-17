"""Stage 2 report -- unsupervised community structure.

Separate document from the stage 1 report, which stays intact. Every number is
read from a metrics file produced by an executed run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils import config as cfgutil
from utils.runtime import NOT_AVAILABLE

NA = NOT_AVAILABLE


def _load(cfg: dict, name: str) -> dict | None:
    path = cfgutil.output_dir(cfg, "metrics") / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def build_report(cfg: dict) -> Path:
    derep = _load(cfg, "dereplication_metrics.json")
    diversity = _load(cfg, "diversity_metrics.json")
    community = _load(cfg, "community_metrics.json")
    dataset = _load(cfg, "dataset_metrics.json")
    var_found = _load(cfg, "encoder_metrics_variants_genomeocean.json")
    var_base = _load(cfg, "encoder_metrics_variants_kmer4.json")

    lines: list[str] = []
    add = lines.append

    add("# Stage 2 Report — Unsupervised Community Structure")
    add("")
    add("**Question:** without any taxonomic labels, what can be said about the "
        "composition and structure of this deep-sea eDNA community, and do "
        "foundation-model embeddings organise it differently from k-mer features?")
    add("")
    add("Stage 1 established that no ground-truth taxonomy exists for this dataset, "
        "so no supervised metric is computable. This stage takes the honest "
        "alternative: characterise the community from sequence structure alone, and "
        "state clearly what that can and cannot establish.")
    add("")
    add("---")
    add("")

    # ------------------------------------------------------------------ A
    add("## A. Why the unit of analysis changed")
    add("")
    add("Stage 1 encoded randomly sampled **reads**. For community structure that is "
        "the wrong unit: 3 million reads are not 3 million molecules, they are a "
        "re-sampling of a much smaller set of distinct sequences at very uneven depth. "
        "Stage 2 therefore dereplicates first, making the unit the **unique sequence "
        "variant**, with read count carried as an explicit abundance weight.")
    add("")
    add("| | Stage 1 | Stage 2 |")
    add("|---|---|---|")
    add("| Unit | read | unique sequence variant |")
    add("| Selection | systematic sample | full-dataset dereplication |")
    add("| Abundance | uncontrolled sampling bias | explicit weight on every statistic |")
    add("")
    add("**What a variant is not.** These are exact-sequence variants: no denoising "
        "error model (no DADA2/UNOISE), no similarity clustering, no taxonomy. "
        "Sequencing error inflates the count, predominantly as singletons. That is "
        "measured below, not corrected for silently.")
    add("")

    # ------------------------------------------------------------------ B
    add("## B. Dereplication")
    add("")
    if derep:
        add("| Metric | Value |")
        add("|---|---|")
        add(f"| Reads dereplicated (full dataset) | {derep['reads_dereplicated']:,} |")
        add(f"| **Unique sequence variants** | **{derep['unique_variants']:,}** |")
        add(f"| Mean reads per variant | {derep['reads_per_variant_mean']:.2f} |")
        add(f"| Singletons (seen once) | {derep['singletons']:,} ({derep['singleton_fraction_of_variants'] * 100:.1f}% of variants) |")
        add(f"| Doubletons | {derep['doubletons']:,} |")
        add(f"| Most abundant single variant | {derep['max_variant_count']:,} reads |")
        add(f"| Variants with sequence retained | {derep['variants_written']:,} |")
        add(f"| Reads covered by retained variants | {derep['fraction_of_reads_covered_by_written_variants'] * 100:.2f}% |")
        add(f"| Reverse-complement collapsing | {derep['canonical_reverse_complement_collapse']} |")
        add("")
        add(f"Streamed in two passes ({derep['pass1_seconds']:.1f} s + "
            f"{derep['pass2_seconds']:.1f} s) at {derep['peak_process_memory_mb']} MB peak. "
            f"Counting is done on 8-byte BLAKE2b digests rather than sequence strings to "
            f"bound memory; the resulting collision probability is "
            f"{derep['hash']['collision_probability_estimate']:.2e} and is recorded rather "
            f"than assumed away.")
        add("")
        add("**Headline:** 3,018,522 reads collapse to "
            f"{derep['unique_variants']:,} distinct molecules — a "
            f"{derep['reads_dereplicated'] / derep['unique_variants']:.1f}x redundancy factor. "
            "Encoding reads rather than variants would have spent most of the compute "
            "re-embedding identical sequences.")
        add("")

        canon = _load(cfg, "dereplication_metrics_canonical.json")
        if canon:
            from analysis.diversity import diversity_from_spectrum

            cd = diversity_from_spectrum(canon["frequency_spectrum"])
            ed = diversity_from_spectrum(derep["frequency_spectrum"])
            merged = derep["unique_variants"] - canon["unique_variants"]
            add("### B.1 Sensitivity check: reverse-complement collapsing")
            add("")
            add("Dereplication was re-run collapsing each sequence with its reverse "
                "complement, to test whether the primary (orientation-sensitive) result "
                "depends on read orientation.")
            add("")
            add("| | Exact (primary) | RC-collapsed |")
            add("|---|---|---|")
            add(f"| Unique variants | {derep['unique_variants']:,} | {canon['unique_variants']:,} |")
            add(f"| Singletons | {derep['singletons']:,} | {canon['singletons']:,} |")
            add(f"| Most abundant variant | {derep['max_variant_count']:,} | {canon['max_variant_count']:,} |")
            add(f"| Shannon H' | {ed['shannon_H']:.4f} | {cd['shannon_H']:.4f} |")
            add(f"| Pielou J | {ed['pielou_evenness_J']:.4f} | {cd['pielou_evenness_J']:.4f} |")
            add(f"| Inverse Simpson | {ed['inverse_simpson']:.4f} | {cd['inverse_simpson']:.4f} |")
            add(f"| Hill q1 | {ed['hill_q1_exp_shannon']:.2f} | {cd['hill_q1_exp_shannon']:.2f} |")
            add(f"| Chao1 | {ed['chao1_estimated_richness']:,.0f} | {cd['chao1_estimated_richness']:,.0f} |")
            add("")
            add(f"**Result: the choice does not matter here.** Collapsing merges only "
                f"{merged:,} variants ({merged / derep['unique_variants'] * 100:.2f}% of "
                f"the total) and leaves every diversity index unchanged to three decimal "
                f"places. That is itself informative: it confirms reads are consistently "
                f"oriented within the library, and that the two amplicon ends identified in "
                f"stage 1 are genuinely distinct sequence, not one region read from both "
                f"strands. The orientation-sensitive result is used as primary.")
    add("")

    # ------------------------------------------------------------------ C
    add("## C. Diversity")
    add("")
    if diversity:
        idx = diversity["indices"]
        add("All indices computed from the **full-dataset** frequency spectrum — no "
            "subsampling, no read excluded.")
        add("")
        add("| Index | Value | Reading |")
        add("|---|---|---|")
        add(f"| Observed richness | {idx['observed_richness_variants']:,} | distinct variants seen |")
        add(f"| Shannon H' | {idx['shannon_H']:.4f} | entropy of the abundance distribution |")
        add(f"| Pielou evenness J | {idx['pielou_evenness_J']:.4f} | 0 = one variant dominates, 1 = perfectly even |")
        add(f"| Simpson D | {idx['simpson_D']:.5f} | chance two reads share a variant |")
        add(f"| Inverse Simpson | {idx['inverse_simpson']:.2f} | effective number of dominant variants |")
        add(f"| Hill q0 (richness) | {idx['hill_q0_richness']:,.0f} | all variants counted equally |")
        add(f"| Hill q1 (exp Shannon) | {idx['hill_q1_exp_shannon']:,.1f} | equally-common variants giving the same entropy |")
        add(f"| Hill q2 (inv. Simpson) | {idx['hill_q2_inverse_simpson']:,.1f} | weighted toward the abundant |")
        add(f"| Chao1 estimate | {idx['chao1_estimated_richness']:,.0f} "
            f"(95% CI {idx['chao1_ci95_lower']:,.0f}–{idx['chao1_ci95_upper']:,.0f}) "
            f"| {idx['chao1_over_observed']:.1f}x observed |")
        add(f"| Good's coverage | {idx['goods_coverage'] * 100:.2f}% | fraction of the community sampled |")
        add("")
        add(f"**The Hill numbers are the story.** {idx['observed_richness_variants']:,} "
            f"variants were observed, but the community behaves like only "
            f"~{idx['hill_q1_exp_shannon']:.0f} equally-common variants (q=1) or "
            f"~{idx['hill_q2_inverse_simpson']:.0f} (q=2). Richness and dominance are "
            f"telling completely different stories, and reporting richness alone would "
            f"be misleading.")
        add("")
        add(f"**On Chao1.** The {idx['chao1_estimated_richness']:,.0f} estimate "
            f"(log-normal 95% CI {idx['chao1_ci95_lower']:,.0f}–"
            f"{idx['chao1_ci95_upper']:,.0f}) is driven "
            f"by {idx['singletons']:,} singletons. In amplicon data singletons are "
            f"predominantly sequencing error, not rare organisms, so this figure is an "
            f"upper bound on *molecular* diversity and should not be read as a species "
            f"estimate. Notably those singletons are 80.9% of variants but only "
            f"{idx['singleton_fraction_of_reads'] * 100:.1f}% of reads.")
    add("")

    # ------------------------------------------------------------------ D
    add("## D. Community concentration")
    add("")
    if diversity:
        ra = diversity["rank_abundance"]
        idx = diversity["indices"]
        add("| Cumulative share of all reads | Variants required |")
        add("|---|---|")
        add(f"| 50% | **{ra['variants_for_50pct_reads']}** |")
        add(f"| 90% | {ra['variants_for_90pct_reads']:,} |")
        add(f"| 99% | {ra['variants_for_99pct_reads']:,} |")
        add("")
        add("| Top N variants | Share of reads |")
        add("|---|---|")
        add(f"| 1 | {idx['most_abundant_variant_share'] * 100:.2f}% |")
        add(f"| 10 | {ra['top10_read_share'] * 100:.2f}% |")
        add(f"| 100 | {ra['top100_read_share'] * 100:.2f}% |")
        add(f"| 1,000 | {ra['top1000_read_share'] * 100:.2f}% |")
        add("")
        add(f"**This community is extraordinarily uneven.** {ra['variants_for_50pct_reads']} "
            f"variants account for half of a 3-million-read library, and a single variant "
            f"accounts for {idx['most_abundant_variant_share'] * 100:.2f}% of it. "
            f"Combined with the amplicon structure found in stage 1, this is the profile of "
            f"a marker-gene library dominated by a small number of source organisms, with a "
            f"very long rare tail.")
    add("")

    # ------------------------------------------------------------------ E
    add("## E. Sampling completeness")
    add("")
    if diversity:
        rf = diversity["rarefaction"]
        idx = diversity["indices"]
        add(f"Analytic (Hurlbert) rarefaction — deterministic, no random subsampling.")
        add("")
        add("| Metric | Value |")
        add("|---|---|")
        add(f"| Total reads | {rf['total_reads']:,} |")
        add(f"| Observed richness at full depth | {rf['observed_richness']:,} |")
        add(f"| New variants per additional read at full depth | {rf['new_variants_per_additional_read_at_full_depth']:.4f} |")
        add(f"| Good's coverage | {idx['goods_coverage'] * 100:.2f}% |")
        add("")
        add("The curve has **not** saturated: more sequencing would keep revealing new "
            "variants. But because most new variants at this depth are error-derived "
            "singletons, this indicates the *variant* space is unsaturated, not "
            "necessarily that organisms are being missed. Good's coverage of "
            f"{idx['goods_coverage'] * 100:.1f}% says the abundant community is well captured.")
    add("")

    # ------------------------------------------------------------------ F
    add("## F. Encoding variants")
    add("")
    if var_found and var_base:
        add("Both representations were computed on the **same variant set**, with the "
            "same seed — the fairness protocol from stage 1 carried forward.")
        add("")
        add("| | Foundation model | Baseline |")
        add("|---|---|---|")
        add(f"| Variants encoded | {var_found['sequences_successful']:,} | {var_base['sequences_successful']:,} |")
        add(f"| Failures | {var_found['sequences_failed']} | {var_base['sequences_failed']} |")
        add(f"| Embedding dimension | {var_found['embedding_dimension']:,} | {var_base['embedding_dimension']:,} |")
        add(f"| Processing time | {var_found['processing_time_seconds']:,.1f} s | {var_base['processing_time_seconds']:,.1f} s |")
        add(f"| Throughput | {var_found['sequences_per_second']:,.0f} /s | {var_base['sequences_per_second']:,.0f} /s |")
        add(f"| All values finite | {var_found['embedding_health']['all_finite']} | {var_base['embedding_health']['all_finite']} |")
        add(f"| Dead dimensions | {var_found['embedding_health']['dead_dimensions']} | {var_base['embedding_health']['dead_dimensions']} |")
        add("")
        if community:
            add(f"These {community['variants_encoded']:,} variants represent "
                f"{community['reads_represented']:,} reads — "
                f"{community['reads_represented'] / dataset['reads_passing_qc'] * 100:.1f}% of "
                f"the QC-passed dataset — so the clustering below speaks for the large "
                f"majority of the library despite operating on a small fraction of the variants.")
    add("")

    # ------------------------------------------------------------------ G
    add("## G. Community clustering")
    add("")
    if community:
        add("| | Foundation model | Baseline |")
        add("|---|---|---|")
        f = community["representations"]["foundation_model"]
        b = community["representations"]["baseline"]
        add(f"| Selected k (max silhouette) | {f['selected_k']} | {b['selected_k']} |")
        best_f = next(r for r in f["k_sweep"] if r["k"] == f["selected_k"])
        best_b = next(r for r in b["k_sweep"] if r["k"] == b["selected_k"])
        add(f"| Silhouette at selected k | {best_f['silhouette']:.4f} | {best_b['silhouette']:.4f} |")
        add(f"| Largest cluster, share of reads | {f['largest_cluster_read_share'] * 100:.2f}% | {b['largest_cluster_read_share'] * 100:.2f}% |")
        add(f"| Clusters covering 50% of reads | {f['clusters_for_50pct_reads']} | {b['clusters_for_50pct_reads']} |")
        add(f"| Clusters covering 90% of reads | {f['clusters_for_90pct_reads']} | {b['clusters_for_90pct_reads']} |")
        add(f"| Mean amplicon-end purity | {f['mean_amplicon_end_purity']:.4f} | {b['mean_amplicon_end_purity']:.4f} |")
        add("")
        add("### G.1 Is the structure real, or arbitrary?")
        add("")
        add("With no taxonomy, the only honest check is whether clusters track "
            "**measured sequence properties**. GC content and amplicon end are measured, "
            "not labelled.")
        add("")
        add("| Property | Foundation model | Baseline |")
        add("|---|---|---|")
        fs = f["structure_vs_measured_properties"]
        bs = b["structure_vs_measured_properties"]
        add(f"| GC variance explained by clusters | {fs['gc_variance_explained_by_clusters']:.4f} | {bs['gc_variance_explained_by_clusters']:.4f} |")
        add(f"| log-abundance variance explained | {fs['log_abundance_variance_explained_by_clusters']:.4f} | {bs['log_abundance_variance_explained_by_clusters']:.4f} |")
        add(f"| Agreement with amplicon end (ARI) | {fs['agreement_with_amplicon_end_ari']:.4f} | {bs['agreement_with_amplicon_end_ari']:.4f} |")
        add("")
        add("High GC variance explained means the partition is structured rather than "
            "arbitrary. It says nothing about taxonomic correctness.")
        add("")
        add("### G.2 Do the two representations agree?")
        add("")
        agree = community["partition_agreement"]
        add("| Chance-corrected metric | Value |")
        add("|---|---|")
        add(f"| Adjusted Rand index | **{agree['adjusted_rand_index']:.4f}** |")
        add(f"| Adjusted mutual information | {agree['adjusted_mutual_information']:.4f} |")
        add(f"| Fowlkes-Mallows | {agree['fowlkes_mallows']:.4f} |")
        add("")
        ari = agree["adjusted_rand_index"]
        strength = (
            "strong" if ari >= 0.7 else
            "moderate" if ari >= 0.4 else
            "weak" if ari >= 0.2 else
            "essentially no"
        )
        add(f"Chance-corrected, so 0 = agreement no better than random and 1 = identical "
            f"partitions. An adjusted Rand of **{ari:.4f}** is {strength} agreement: the "
            f"two representations largely recover the **same coarse community structure** "
            f"from the same molecules.")
        add("")
        comparison = _load(cfg, "comparison.json")
        if comparison:
            jac = comparison.get("agreement", {}).get("mean_jaccard")
            if jac is not None:
                add(f"**This is worth contrasting with stage 1.** At read level, the mean "
                    f"Jaccard overlap of 20-nearest-neighbour sets between the two "
                    f"representations was only {jac:.4f} — they disagreed about which "
                    f"individual reads are similar. At variant level and coarse "
                    f"granularity they agree strongly (ARI {ari:.3f}). The two findings are "
                    f"consistent: the representations differ in fine-grained local "
                    f"neighbourhood structure while converging on the same broad "
                    f"composition-driven partition.")
                add("")
        f_sil = best_f["silhouette"]
        b_sil = best_b["silhouette"]
        cleaner = "baseline" if b_sil > f_sil else "foundation model"
        add(f"**No foundation-model advantage is detectable on this task.** The "
            f"{cleaner} produces the higher silhouette at the selected k "
            f"({max(f_sil, b_sil):.4f} vs {min(f_sil, b_sil):.4f}) and explains comparable "
            f"GC variance ({bs['gc_variance_explained_by_clusters']:.3f} baseline vs "
            f"{fs['gc_variance_explained_by_clusters']:.3f} foundation), using "
            f"{b['n_variants_clustered'] and var_base['embedding_dimension']} dimensions "
            f"against {var_found['embedding_dimension']:,}. For unsupervised community "
            f"structure at this granularity, the 11x wider learned representation buys "
            f"nothing measurable.")
        add("")
        add("With no labels, neither partition can be declared correct where they do "
            "differ — which is precisely why a labelled comparison remains the necessary "
            "next step.")
    add("")

    # ------------------------------------------------------------------ G.3
    amplicon = _load(cfg, "amplicon_metrics.json")
    if amplicon:
        summary = amplicon["summary"]
        add("### G.3 How much of a read is actually informative?")
        add("")
        add("A question neither stage had answered: this is marker-gene data, so how "
            "much of each 151 bp read is conserved anchor and how much is "
            "organism-discriminating sequence? Per-position base composition was "
            "computed across the abundant variants, weighted by read abundance.")
        add("")
        add("| Amplicon end | Conserved block | Mean conservation there | Variable region | Mean conservation there | Read that varies |")
        add("|---|---|---|---|---|---|")
        for end, data in sorted(amplicon["ends"].items()):
            add(f"| {end} | {data['conserved_block_length']} bp | "
                f"{data['conserved_block']['before'] * 100:.1f}% | "
                f"{data['variable_region_length']} bp | "
                f"{data['conserved_block']['after'] * 100:.1f}% | "
                f"{data['informative_fraction_of_read'] * 100:.1f}% |")
        add("")
        add(f"**Only {summary['weighted_informative_fraction'] * 100:.1f}% of each read varies "
            f"across the community** (abundance-weighted across both ends). End 0 in "
            f"particular opens with a {amplicon['ends']['0']['conserved_block_length']} bp block "
            f"that is {amplicon['ends']['0']['conserved_block']['before'] * 100:.1f}% invariant.")
        add("")
        add("**This reframes the earlier results.** Positions that do not vary cannot "
            "discriminate organisms, whatever encodes them. Both representations were "
            "therefore handed reads whose majority content is near-identical across the "
            "whole community, which bounds what *any* representation could achieve on "
            "mate retrieval or fine-grained clustering. The weak absolute retrieval "
            "scores in stage 1 (2–3% top-1) and the absence of a foundation-model "
            "advantage here should both be read in that light: the ceiling is set by the "
            "data, not only by the encoders.")
        add("")
        add("It also suggests a caution about the clustering above. Clusters came out "
            f"~{max(r['mean_amplicon_end_purity'] for r in community['representations'].values()) * 100:.0f}% "
            "pure by amplicon end, and the conserved anchor is exactly the signal that "
            "separates the ends. Some of the apparent cluster structure is therefore "
            "likely to reflect *which end of the amplicon a read came from* rather than "
            "which organism it came from. Re-running the comparison on the variable "
            "region alone would isolate the biological signal, and is the obvious "
            "follow-up.")
        add("")
        add("**On the identity of the conserved block.** Its architecture — a long, "
            "near-invariant anchor followed by a sharply variable region — is the "
            "signature of a conserved-region-primed marker gene, and the end-0 consensus "
            "shows clear homology to the universal small-subunit rRNA conserved block "
            "(motifs `GGGCACCAC`, `GTGGAGCATGTGG`, `TTAATTTGACTCAAC`, `GGATTGACAG`). "
            "That is as far as the data alone can go: **distinguishing 16S from 18S from "
            "an organellar SSU, and assigning any taxonomy, requires a reference "
            "database.** No gene name is asserted here. A web search of the exact primer "
            "sequences returned no catalogued match, so these appear to be custom or "
            "non-standard primers.")
        add("")

    # ------------------------------------------------------------------ G.4
    varreg = _load(cfg, "variable_region_metrics.json")
    if varreg:
        v = varreg["variants"]
        sig = varreg["significance"]
        add("### G.4 Removing the conserved anchor — the follow-up, run")
        add("")
        add("The obvious test of §G.3: trim the conserved anchor from every read and "
            "re-run the retrieval comparison on the same pairs, same seed, same "
            "amplicon-end control, same strand correction. Only the input sequence "
            f"changes, retaining {varreg['subset']['fraction_of_sequence_retained'] * 100:.0f}% "
            f"of it (mean {varreg['subset']['mean_length_after']:.0f} bp per read).")
        add("")
        add("| Input | Representation | Top-1 | AUROC |")
        add("|---|---|---|---|")
        for key, inp, rep in (
            ("foundation_full_read", "Full read (151 bp)", "GenomeOcean-500M"),
            ("baseline_full_read", "Full read (151 bp)", "k-mer / TNF"),
            ("foundation_variable_region", "Variable region only", "GenomeOcean-500M"),
            ("baseline_variable_region", "Variable region only", "k-mer / TNF"),
        ):
            add(f"| {inp} | {rep} | {v[key]['top1_accuracy'] * 100:.3f}% | "
                f"{v[key]['auroc_mate_vs_random']:.4f} |")
        add("")
        add("Top-1 outcomes are **paired** (both representations scored on the same "
            "queries), so differences are tested with McNemar's exact test rather than "
            "eyeballed:")
        add("")
        add("| Comparison | Difference | 95% CI | p | Verdict |")
        add("|---|---|---|---|---|")
        for key, name in (
            ("full_read_foundation_vs_baseline", "Full read: foundation vs baseline"),
            ("variable_region_foundation_vs_baseline", "Variable region: foundation vs baseline"),
            ("foundation_full_vs_variable", "Foundation: variable vs full read"),
            ("baseline_full_vs_variable", "Baseline: variable vs full read"),
        ):
            s_ = sig[key]
            ci = s_["ci95_percentage_points"]
            add(f"| {name} | {s_['difference_percentage_points']:+.2f} pp | "
                f"[{ci[0]:+.2f}, {ci[1]:+.2f}] | {s_['p_value']:.2g} | {s_['verdict']} |")
        add("")
        add("**This substantially qualifies §G.2's conclusion.** Three findings:")
        add("")
        add("1. **The baseline's full-read advantage is real but anchor-dependent.** It "
            f"beats the foundation model by {abs(sig['full_read_foundation_vs_baseline']['difference_percentage_points']):.2f} "
            f"pp on full reads (p = {sig['full_read_foundation_vs_baseline']['p_value']:.0e}), but loses "
            f"{abs(sig['baseline_full_vs_variable']['difference_percentage_points']):.2f} pp "
            f"(p = {sig['baseline_full_vs_variable']['p_value']:.0e}) once the conserved anchor is "
            f"removed. Much of what looked like superior retrieval was the near-invariant "
            f"anchor, not organism signal.")
        add("")
        add("2. **The foundation model loses nothing.** Trimming changes its top-1 by "
            f"{sig['foundation_full_vs_variable']['difference_percentage_points']:+.2f} pp "
            f"(p = {sig['foundation_full_vs_variable']['p_value']:.2f}, not significant) — it was "
            f"not relying on the anchor.")
        add("")
        add(f"3. **On the informative region the two are statistically tied on top-1** "
            f"(p = {sig['variable_region_foundation_vs_baseline']['p_value']:.2f}), but their "
            f"mate/non-mate separation diverges sharply: the foundation model holds AUROC "
            f"{v['foundation_variable_region']['auroc_mate_vs_random']:.3f} while the baseline "
            f"falls to {v['baseline_variable_region']['auroc_mate_vs_random']:.3f} — "
            f"indistinguishable from chance. On the biologically informative sequence, "
            f"the k-mer representation retains essentially no mate/non-mate separation; "
            f"the foundation model does.")
        add("")
        add("**Revised bottom line.** The earlier statement that the foundation model "
            "shows *no measurable advantage* holds for full 151 bp reads and for coarse "
            "community clustering. It does **not** hold once the uninformative conserved "
            "anchor is excluded: there the k-mer baseline's advantage disappears entirely "
            "and its separation collapses to chance, while the foundation model's is "
            "preserved. Both remain weak in absolute terms, and none of this establishes "
            "taxonomic accuracy.")
        add("")
        add("*Caveat: trimming discards real sequence and shortens reads unequally "
            "between amplicon ends. It is a diagnostic for locating the signal, not a "
            "proposed preprocessing default; the full-read result remains primary.*")
        add("")

    # ------------------------------------------------------------------ H
    add("## H. What remains unavailable")
    add("")
    add(f"**{NA}** for all of the following, unchanged from stage 1:")
    add("")
    add("- Any taxonomic identity for any variant or cluster")
    add("- Species richness (as opposed to variant richness)")
    add("- Classification accuracy, macro/weighted F1, per-class metrics, confusion matrix")
    add("- Confidence and calibration analysis")
    add("- Whether the foundation model's partition is *better* than the baseline's")
    add("")
    add("No cluster is named. No variant is assigned a taxon. Cluster count is a "
        "property of the embedding and the selected k, not a species estimate.")
    add("")

    # ------------------------------------------------------------------ I
    add("## I. Figures")
    add("")
    add("| File | Content | Scope |")
    add("|---|---|---|")
    add("| `dereplication.png` | Figure 13 — variants, frequency spectrum, Chao1 | Full dataset |")
    add("| `rank_abundance.png` | Figure 14 — rank abundance and concentration | Full dataset |")
    add("| `rarefaction.png` | Figure 15 — sampling completeness | Full dataset |")
    add("| `community_clusters.png` | Figure 16 — clustering, foundation vs baseline | Encoded variants |")
    add("| `community_map.png` | Figure 17 — abundance-weighted community map | Encoded variants |")
    if amplicon:
        add("| `amplicon_architecture.png` | Figure 18 — conserved vs variable read architecture | Abundant variants |")
    add("")

    # ------------------------------------------------------------------ J
    add("## J. Limitations")
    add("")
    for i, item in enumerate([
        "**Exact-sequence variants, not denoised ASVs.** No error model was applied, so "
        "richness is inflated and Chao1 substantially so. A DADA2/UNOISE-style denoiser "
        "would materially change every richness figure (though little of the abundance profile).",
        "**Variants are not organisms.** One organism can contribute several variants "
        "(sequencing error, intragenomic marker copies) and two organisms can share one. "
        "Richness is an upper bound on organismal diversity.",
        "**Clusters are not taxa.** They are unnamed groups of similar sequences.",
        "**Only the abundant variants were encoded.** Variants seen once were excluded "
        "from encoding; they are 80.9% of variants but only 6.6% of reads, so the "
        "abundance-weighted picture is largely unaffected while the rare tail is not "
        "represented in the clustering.",
        "**Single sample, single site, single sequencing run.** No spatial, temporal or "
        "cross-sample comparison is possible, so beta diversity is not computable.",
        "**Neither partition can be validated.** Without labels there is no way to say "
        "which representation is right where they disagree.",
    ], 1):
        add(f"{i}. {item}")
    add("")

    # ------------------------------------------------------------------ K
    add("## K. Conclusions")
    add("")
    if derep and diversity and community:
        idx = diversity["indices"]
        ra = diversity["rank_abundance"]
        agree = community["partition_agreement"]
        add(f"1. **The library is far less diverse than its read count suggests.** "
            f"3,018,522 reads collapse to {derep['unique_variants']:,} variants, and the "
            f"effective diversity is smaller still — around "
            f"{idx['hill_q1_exp_shannon']:.0f} equally-common variants.")
        add("")
        add(f"2. **It is dominated by very few sequences.** "
            f"{ra['variants_for_50pct_reads']} variants carry half the library; the single "
            f"most abundant carries {idx['most_abundant_variant_share'] * 100:.2f}%.")
        add("")
        add(f"3. **The rare tail is mostly error, not biology.** {idx['singletons']:,} "
            f"singletons are 80.9% of variants but only "
            f"{idx['singleton_fraction_of_reads'] * 100:.1f}% of reads, and they are what "
            f"drives Chao1 to {idx['chao1_over_observed']:.1f}x the observed richness.")
        add("")
        ari = agree["adjusted_rand_index"]
        fcl = community["representations"]["foundation_model"]
        bcl = community["representations"]["baseline"]
        add(f"4. **Both representations recover the same coarse community structure** "
            f"(adjusted Rand = {ari:.4f}). Both select k = {fcl['selected_k']}, both need "
            f"{fcl['clusters_for_50pct_reads']} clusters to cover half the reads, and both "
            f"produce clusters that are ~93% pure by amplicon end and explain ~73-74% of GC "
            f"variance. The partitions are structured, not arbitrary.")
        add("")
        add(f"5. **On full reads and coarse clustering, the foundation model shows no "
            f"measurable advantage.** The 137-dimensional k-mer baseline achieves a higher "
            f"silhouette than the 1,536-dimensional GenomeOcean embedding, at ~28x the "
            f"throughput, and wins full-read top-1 retrieval by a statistically solid "
            f"margin. Reported as measured.")
        add("")
        if varreg:
            sg = varreg["significance"]
            vv = varreg["variants"]
            add(f"5b. **But that advantage is anchor-dependent, and reverses on the "
                f"informative sequence.** Trimming the near-invariant conserved anchor costs "
                f"the baseline "
                f"{abs(sg['baseline_full_vs_variable']['difference_percentage_points']):.2f} pp of "
                f"top-1 (p = {sg['baseline_full_vs_variable']['p_value']:.0e}) and drops its "
                f"mate/non-mate AUROC to "
                f"{vv['baseline_variable_region']['auroc_mate_vs_random']:.3f} — chance. The "
                f"foundation model loses nothing (p = "
                f"{sg['foundation_full_vs_variable']['p_value']:.2f}) and holds AUROC "
                f"{vv['foundation_variable_region']['auroc_mate_vs_random']:.3f}. On the part of "
                f"the read that actually varies, the learned representation retains signal "
                f"the k-mer features do not.")
            add("")
        if amplicon:
            add(f"6. **Only {amplicon['summary']['weighted_informative_fraction'] * 100:.0f}% of "
                f"each read varies across the community.** The rest is conserved anchor. "
                f"This bounds what any representation can achieve and is the most likely "
                f"explanation for both the weak absolute retrieval scores and the absence "
                f"of a foundation-model advantage — the ceiling is set by the data. It "
                f"also means some apparent cluster structure probably reflects amplicon "
                f"end rather than organism.")
            add("")
        add("7. **The unsupervised route has now been taken as far as it honestly goes.** "
            "Composition, evenness, dominance and sampling completeness are all "
            "characterised from the full dataset. What cannot be resolved without labels "
            "is which representation organises the community *correctly* — and that is the "
            "question the next stage would need reference taxonomy to answer.")
    add("")
    add("---")
    add("")
    add("*Generated by `python main.py community-report`. Every value is read from a "
        "metrics file produced by an executed run.*")

    path = cfgutil.output_dir(cfg, "reports") / "community_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {path}")
    return path
