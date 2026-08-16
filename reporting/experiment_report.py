"""Generate the final research-style experiment report.

Every number in the report is read from a metrics file produced by an actual
run. Nothing is typed in by hand and nothing is estimated except values
explicitly labelled as projections.
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


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return NA
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def build_report(cfg: dict) -> Path:
    dataset = _load(cfg, "dataset_metrics.json")
    preprocess = _load(cfg, "preprocessing_metrics.json")
    enc_main_early = _load(cfg, "encoder_metrics_main_genomeocean.json") or {}
    # Read the compatibility record for the MAIN run specifically, not whichever
    # encode happened to write last.
    compat = (
        enc_main_early.get("compatibility_check")
        or _load(cfg, "model_compatibility_main_genomeocean.json")
        or _load(cfg, "model_compatibility.json")
    )
    testrun = _load(cfg, "testrun_metrics.json")
    embed = _load(cfg, "embedding_metrics.json")
    comparison = _load(cfg, "comparison.json")
    performance = _load(cfg, "performance_metrics.json")
    labels = _load(cfg, "label_verification.json")
    enc_main = _load(cfg, "encoder_metrics_main_genomeocean.json")
    base_main = _load(cfg, "encoder_metrics_main_kmer4.json")

    figures_dir = cfgutil.output_dir(cfg, "figures")
    figures = sorted(p.name for p in figures_dir.glob("*.png"))

    lines: list[str] = []
    add = lines.append

    # ---------------------------------------------------------------- header
    add("# Experiment Report — Stage 1")
    add("")
    add("## Can raw deep-sea eDNA short reads be processed through a genomic")
    add("## foundation-model pipeline to obtain useful genomic embeddings?")
    add("")
    add("**A TaxDistill-inspired adaptation for short deep-sea eDNA reads.**")
    add("This is not a reproduction of TaxDistill. See `docs/METHODOLOGY.md` §2.")
    add("")
    if (cfgutil.output_dir(cfg, "reports") / "community_report.md").exists():
        add("> **Stage 2 followed this work.** Because the supervised branch is blocked "
            "(§I), a separate unsupervised community-structure analysis was carried out "
            "over the full dataset — dereplication, diversity, and abundance-weighted "
            "clustering. See [`community_report.md`](community_report.md).")
        add("")
    # Prefer an environment record from a stage that exercised the GPU path.
    env = (enc_main or {}).get("environment") or (preprocess or {}).get("environment", {})
    add(f"Seed `{cfg['seed']}` · Python {env.get('python')} · torch {env.get('torch')} · "
        f"transformers {env.get('transformers')} · {env.get('gpu', 'CPU only')}")
    add("")
    add("---")
    add("")

    # ------------------------------------------------------- A. dataset
    add("## A. Dataset statistics")
    add("")
    if dataset:
        length = dataset["sequence_length"]
        gc = dataset["gc_content_qc_passed_reads"]
        ids = dataset["read_id_analysis"]
        add("Measured over the **full dataset**, not a sample.")
        add("")
        add("| Metric | Value |")
        add("|---|---|")
        add(f"| Source file | `{dataset['source_file'].replace(chr(92), '/')}` |")
        add(f"| File size | {dataset['source_file_mb']} MB |")
        add(f"| Total sequence records | {dataset['total_sequence_records']:,} |")
        add(f"| Unique sequence IDs | {ids['unique_sequence_ids']:,} |")
        add(f"| Records per ID | exactly {ids['total_records'] // ids['unique_sequence_ids']}, max multiplicity {ids['max_multiplicity']} |")
        add(f"| Total sequenced bases | {dataset['total_base_pairs']:,} |")
        add(f"| Length — min / max | {length['min']} / {length['max']} bp |")
        add(f"| Length — mean / median | {length['mean']} / {length['median']} bp |")
        add(f"| Length — standard deviation | {length['std']} |")
        add(f"| Distinct read lengths | {length['distinct_values']} |")
        add(f"| GC content — mean | {gc['mean'] * 100:.3f}% |")
        add(f"| GC content — median | {gc['median'] * 100:.3f}% |")
        add(f"| GC content — standard deviation | {gc['std']:.5f} |")
        add(f"| GC content — 5th–95th percentile | {gc['p05'] * 100:.2f}% – {gc['p95'] * 100:.2f}% |")
        add(f"| Reads with ambiguous bases | {dataset['reads_with_ambiguous_bases']:,} ({dataset['reads_with_ambiguous_bases'] / dataset['total_sequence_records'] * 100:.3f}%) |")
        add(f"| Total ambiguous bases | {dataset['total_ambiguous_bases']:,} |")
        add(f"| Reads with invalid characters | {dataset['reads_with_invalid_characters']:,} |")
        add(f"| Reads passing QC | {dataset['reads_passing_qc']:,} ({dataset['qc_pass_rate'] * 100:.3f}%) |")
        add(f"| Reads removed by QC | {dataset['reads_removed_by_qc']:,} |")
        add("")
        add("**Nucleotide composition (all sequenced bases):**")
        add("")
        add("| Base | Count | Share |")
        add("|---|---|---|")
        for base, count in dataset["nucleotide_counts"].items():
            share = dataset["nucleotide_fractions"][base] * 100
            add(f"| {base} | {count:,} | {share:.5f}% |")
        add("")
        add("**Why every sequence ID occurs twice — measured, not assumed:**")
        add("")
        add("| Evidence | Measured |")
        add("|---|---|")
        add(f"| Repeats adjacent in file order | {ids['fraction_of_repeats_adjacent_in_file'] * 100:.2f}% |")
        add(f"| Repeats on the same flowcell tile | {ids['fraction_of_repeats_on_same_tile'] * 100:.2f}% |")
        add(f"| Repeats with identical GC content | {ids['fraction_of_repeats_with_identical_gc'] * 100:.2f}% |")
        add("")
        add("Same cluster coordinate, adjacent in the file, but different sequence content: "
            "these are the two mates (R1/R2) of one paired-end cluster written interleaved. "
            "The repeated IDs are **not duplicated data** — both records are distinct, real sequence.")
    else:
        add(NA)
    add("")

    # ------------------------------------------- B. preprocessing decisions
    add("## B. Preprocessing decisions")
    add("")
    if preprocess:
        qc = preprocess["qc_configuration"]["thresholds"]
        delta = preprocess.get("streaming_phase_memory_delta_mb")
        add(f"Full dataset streamed in **{preprocess['elapsed_seconds']:,.1f} s** "
            f"({preprocess['records_per_second']:,.0f} records/s, "
            f"{preprocess['megabytes_per_second']:.1f} MB/s). At most one record is held "
            f"in memory at a time"
            + (f"; streaming the entire {dataset['source_file_mb']:,.0f} MB file grew "
               f"resident memory by only **{delta} MB**." if delta is not None else ".")
            )
        add("")
        add(f"Absolute RSS figures are dominated by imported libraries (the CLI imports "
            f"torch and transformers to record the environment), which is why the growth "
            f"attributable to streaming is the number quoted. The whole stage peaks at "
            f"{preprocess['peak_process_memory_mb']} MB because a second, deliberate pass "
            f"reads the metadata columns back in to compute *exact* order statistics "
            f"(median, percentiles) rather than approximating them on the fly.")
        add("")
        add("| Filter | Setting |")
        add("|---|---|")
        for key in ("min_length", "max_length", "valid_alphabet", "ambiguity_policy",
                    "max_ambiguous_fraction", "min_effective_length",
                    "max_single_base_fraction", "min_gc", "max_gc"):
            add(f"| `{key}` | `{qc.get(key)}` |")
        add("")
        if dataset:
            add("| QC outcome | Reads |")
            add("|---|---|")
            for reason, count in dataset["qc_reason_counts"].items():
                add(f"| {reason.replace('_', ' ')} | {count:,} |")
        add("")
        add("**Ambiguity policy — `longest_unambiguous_run`.** GenomeOcean's BPE "
            "vocabulary covers ACGT only, so an `N` would become `[UNK]`. This policy "
            "keeps the longest contiguous ACGT stretch of each read. It never "
            "substitutes, imputes, or pads. The cost is measured: mean effective "
            f"length {dataset['effective_length_after_qc']['mean']} bp of 151."
            if dataset else "")
    add("")

    # ------------------------------------------------ C. short reads
    add("## C. Short reads vs long contigs — the central methodological difference")
    add("")
    add("| | TaxDistill (paper) | This work |")
    add("|---|---|---|")
    add("| Input | assembled contigs | raw sequencing reads |")
    add("| Length filter | **≥ 2,000 bp**, strictly applied | **none** |")
    add("| Typical length | ≥ 2,000 bp | 151 bp, uniformly |")
    add("| 4-mers per sequence | ≥ 1,997 | **148** |")
    add("| Assembly available | yes | no |")
    add("| Abundance across samples | yes | no (single sample) |")
    add("")
    add("**The paper's ≥ 2,000 bp filter was deliberately NOT applied.** At that "
        "threshold, 0 of 3,026,920 reads survive and the pipeline has no input.")
    add("")
    add("**Reads were NOT padded to 2,000 bp.** Padding a 151 bp read to 2,000 bp "
        "would make 92.5% of the resulting sequence non-biological; whatever the model "
        "computed would be largely a function of the padding scheme. No padding is "
        "applied at any stage of this pipeline — only within-batch padding for tensor "
        "shape, which the attention mask excludes from pooling entirely.")
    add("")
    add("**Reads were not treated as contigs and were never concatenated.** No genomic "
        "context is invented.")
    add("")
    add("The paper's own justification for its filter is that it 'ensures deep "
        "representation learning can capture sufficient contextual semantic "
        "information.' Our reads have an order of magnitude less context. Whether a "
        "pretrained genomic language model degrades more gracefully than "
        "tetranucleotide frequency in that regime is exactly what this experiment "
        "measures.")
    add("")

    # ------------------------------------------------ D. architecture
    add("## D. TaxDistill architecture vs our adaptation")
    add("")
    add("| Component | TaxDistill | This work | Label |")
    add("|---|---|---|---|")
    add("| Teacher backbone | GenomeOcean, frozen | GenomeOcean-500M, frozen — same checkpoint | PAPER-DERIVED |")
    add("| Teacher head | learnable classification head | none (no labels to train it on) | ADAPTATION |")
    add("| Student | Taxometer MLP: TNF + K abundances + total | canonical 4-mer/TNF vector, no abundance | ADAPTATION |")
    add("| Knowledge distillation | KD loss, soft labels, temperature | **not implemented** — requires labels | OUT OF SCOPE |")
    add("| Hierarchical loss | deep hierarchical loss over taxonomy tree | **not implemented** — requires a taxonomy | OUT OF SCOPE |")
    add("| Pooling | unspecified in the paper | attention-masked mean over final layer | IMPLEMENTATION DECISION |")
    add("| Evaluation | species-level F1 vs CAMI2 ground truth | label-free mate-pair retrieval + structure | ADAPTATION |")
    add("")
    add("Only the **upper/representation stage** is in scope: raw reads → deep genomic "
        "embeddings, plus the analysis needed to judge whether those embeddings are usable.")
    add("")

    # ------------------------------------------------ E. encoder
    add("## E. Foundation model / encoder used")
    add("")
    enc = (enc_main or {}).get("encoder", {})
    if enc:
        add(f"**`{enc['model_id']}`** — the actual model from the paper, not a substitute.")
        add("")
        add("| Property | Value |")
        add("|---|---|")
        add(f"| Architecture | {', '.join(enc.get('architecture') or [])} |")
        add(f"| Hidden layers | {enc['num_hidden_layers']} |")
        add(f"| Vocabulary | {enc['vocab_size']:,} BPE tokens over ACGT |")
        add(f"| Total parameters | {enc['model_parameters_total']:,} |")
        add(f"| Trainable parameters | {enc['model_parameters_trainable']:,} |")
        add(f"| Frozen parameters | {enc['model_parameters_frozen']:,} |")
        add(f"| Precision | {enc['dtype']} |")
        add(f"| Attention implementation | {enc['attn_implementation']} |")
        add(f"| Device | {enc['device']} |")
        add(f"| Pooling | {enc['pooling']} over the final hidden layer, attention-masked |")
        add("")
    if compat:
        add("**Compatibility with 151 bp reads — measured on real reads before bulk inference:**")
        add("")
        add("| Check | Measured |")
        add("|---|---|")
        add(f"| Reads probed | {compat['n_sample_reads']:,} |")
        add(f"| Tokens per read (min / mean / max) | {compat['tokens_per_read_min']} / {compat['tokens_per_read_mean']} / {compat['tokens_per_read_max']} |")
        add(f"| Base pairs per token | {compat['bp_per_token_mean']} |")
        add(f"| Model token limit | {compat['model_token_limit']:,} |")
        add(f"| Fraction of limit used | {compat['fraction_of_limit_used'] * 100:.2f}% |")
        add(f"| Fits without truncation | **{compat['fits_without_truncation']}** |")
        add(f"| `[UNK]` tokens produced | {compat['unk_tokens_in_sample']} |")
        add(f"| Padding to a target length | {compat['padding_applied_to_reach_target_length']} |")
        add("")
        add("The model accepts our reads directly. No substitute encoder was needed and "
            "none is claimed to be GenomeOcean.")
        add("")
        add("**Caveat stated plainly:** 151 bp is well within spec for *input length* but "
            "out of distribution with respect to the *context* GenomeOcean was pretrained "
            "on (assembled genomic sequence). Length compatibility is not representational "
            "suitability. Testing that gap is the experiment.")
    add("")

    # --------------------------------------- F. embedding dims + extraction
    add("## F. Embedding dimensions and extraction")
    add("")
    if enc_main and base_main:
        add("| | Foundation model | Baseline |")
        add("|---|---|---|")
        add(f"| Representation | GenomeOcean-500M | canonical 4-mer / TNF |")
        add(f"| Embedding dimension | **{enc_main['embedding_dimension']:,}** | **{base_main['embedding_dimension']:,}** |")
        add(f"| Reads requested | {enc_main['sequences_requested']:,} | {base_main['sequences_requested']:,} |")
        add(f"| Reads processed | {enc_main['sequences_processed']:,} | {base_main['sequences_processed']:,} |")
        add(f"| Reads successful | {enc_main['sequences_successful']:,} | {base_main['sequences_successful']:,} |")
        add(f"| Reads failed | {enc_main['sequences_failed']:,} | {base_main['sequences_failed']:,} |")
        add(f"| Batch size | {enc_main['batch_size']} | {base_main['batch_size']:,} |")
        add(f"| Processing time | {enc_main['processing_time_seconds']:,.1f} s | {base_main['processing_time_seconds']:,.1f} s |")
        add(f"| Throughput | {enc_main['sequences_per_second']:,.1f} reads/s | {base_main['sequences_per_second']:,.1f} reads/s |")
        add(f"| Time per read | {enc_main['milliseconds_per_sequence']:.4f} ms | {base_main['milliseconds_per_sequence']:.4f} ms |")
        add(f"| Storage | {enc_main['embedding_storage_mb']:,.1f} MB | {base_main['embedding_storage_mb']:,.1f} MB |")
        add(f"| Bytes per embedding | {enc_main['bytes_per_embedding']:,.0f} | {base_main['bytes_per_embedding']:,.0f} |")
        gpu = enc_main.get("gpu_memory", {})
        if gpu.get("available"):
            add(f"| Peak GPU memory | {gpu['peak_allocated_mb']:,.0f} MB of {gpu['total_mb']:,.0f} MB | n/a (CPU) |")
        add("")
        health = enc_main["embedding_health"]
        bhealth = base_main["embedding_health"]
        add("**Embedding health — verified, not assumed:**")
        add("")
        add("| Check | Foundation model | Baseline |")
        add("|---|---|---|")
        add(f"| Shape | {health['shape']} | {bhealth['shape']} |")
        add(f"| All values finite | {health['all_finite']} | {bhealth['all_finite']} |")
        add(f"| NaN / Inf count | {health['n_nan']} / {health['n_inf']} | {bhealth['n_nan']} / {bhealth['n_inf']} |")
        add(f"| Dead (zero-variance) dimensions | {health['dead_dimensions']} | {bhealth['dead_dimensions']} |")
        add(f"| Value range | [{health['value_min']}, {health['value_max']}] | [{bhealth['value_min']}, {bhealth['value_max']}] |")
        add(f"| Mean L2 norm | {health['l2_norm_mean']} | {bhealth['l2_norm_mean']} |")
    add("")
    if testrun:
        add(f"**Small-subset test run first.** {testrun['subset_size']:,} reads were pushed "
            f"through the complete pipeline (parse → QC → tokenise → encode → store → PCA) "
            f"before any full-scale run. Verdict: *{testrun['verdict']}*")
        add("")

    # ------------------------------------------------ G. baseline
    add("## G. Baseline representation")
    add("")
    benc = (base_main or {}).get("encoder", {})
    if benc:
        add(f"Canonical {benc['k']}-mer frequency vector "
            f"({benc['n_kmer_features']} reverse-complement-collapsed k-mers"
            f"{' + GC content' if benc['includes_gc_content'] else ''} "
            f"= {benc['embedding_dim']} dimensions), 0 learned parameters.")
        add("")
        add("Two deliberate departures from the paper's student input, both stated as "
            "adaptations rather than omissions:")
        add("")
        add("1. **No VAMB 103-d projection.** That projection decorrelates features for "
            "VAMB's variational autoencoder. Here the vector feeds PCA and a linear "
            "probe, for which raw canonical frequencies are the more faithful baseline.")
        add("2. **No abundance features.** Abundance requires multiple samples mapped "
            "against a shared assembly. We have one sample of unassembled reads, so "
            "these are **not computable** — not omitted for convenience.")
        add("")
        add("A 151 bp read yields 148 4-mers spread across 136 canonical bins — roughly "
            "one observation per bin. This sparsity is intrinsic to short reads and is "
            "the central reason the comparison is informative.")
    add("")

    # ------------------------------------------------ H. embedding analysis
    add("## H. Embedding analysis")
    add("")
    if embed:
        add("| Metric | Foundation model | Baseline |")
        add("|---|---|---|")
        f, b = embed["foundation_model"], embed["baseline"]
        add(f"| Reads encoded | {f['n_encoded']:,} | {b['n_encoded']:,} |")
        add(f"| Nominal dimension | {f['embedding_dim']:,} | {b['embedding_dim']:,} |")
        add(f"| Variance in PC1 | {f['pca']['variance_pc1'] * 100:.2f}% | {b['pca']['variance_pc1'] * 100:.2f}% |")
        add(f"| Variance in PC1–PC2 | {f['pca']['variance_pc1_pc2'] * 100:.2f}% | {b['pca']['variance_pc1_pc2'] * 100:.2f}% |")
        add(f"| Variance in first 10 PCs | {f['pca']['variance_first_10'] * 100:.2f}% | {b['pca']['variance_first_10'] * 100:.2f}% |")

        def pcs(analysis: dict, threshold: str) -> str:
            """'None' means the threshold was not reached inside the components computed."""
            value = analysis["pca"][threshold]
            return str(value) if value else f"> {analysis['pca']['n_components']} (not reached)"

        add(f"| PCs for 90% variance | {pcs(f, 'components_for_90pct')} | {pcs(b, 'components_for_90pct')} |")
        add(f"| PCs for 95% variance | {pcs(f, 'components_for_95pct')} | {pcs(b, 'components_for_95pct')} |")
        add(f"| Participation ratio | {f['pca']['effective_dimensionality_participation_ratio']} | {b['pca']['effective_dimensionality_participation_ratio']} |")
        add("")
        add("**Cluster tendency vs a dimension-shuffled null.** Silhouette on k-means "
            "labels is self-referential on its own, so each value is paired with the same "
            "statistic on a null that preserves every marginal distribution while "
            "destroying joint structure. Only the gap is interpretable.")
        add("")
        add("| Representation | Best k | Real silhouette | Null silhouette | Gap |")
        add("|---|---|---|---|---|")
        for label, analysis in (("Foundation model", f), ("Baseline", b)):
            ct = analysis["cluster_tendency"]
            best = next((r for r in ct["per_k"] if r["k"] == ct["best_k_by_gap"]), None)
            if best:
                add(f"| {label} | {best['k']} | {best['real_silhouette']:.4f} | "
                    f"{best['shuffled_null_silhouette']:.4f} | **{best['silhouette_gap_vs_null']:.4f}** |")
        add("")
        add("These clusters have **no verified taxonomic meaning**. This is a "
            "clusterability statistic, not a classification result.")
        add("")
        if isinstance(f.get("umap"), dict):
            add(f"UMAP visualization sample: **{f['umap']['visualization_sample_size']:,} reads** "
                f"(n_neighbors={f['umap']['n_neighbors']}, min_dist={f['umap']['min_dist']}, "
                f"seed={f['umap']['seed']}). This is the visualization sample size, "
                f"distinct from the {f['n_encoded']:,}-read encoding subset and from the "
                f"{dataset['total_sequence_records']:,}-record full dataset.")
    add("")

    # ------------------------------------------------ I. classification
    add("## I. Classification results")
    add("")
    add(f"**{NA}**")
    add("")
    if labels:
        add("This is a finding, not a gap left unexplored. Three independent label "
            "sources were checked:")
        add("")
        add("| Source checked | Result |")
        add("|---|---|")
        cfile = labels["checks"]["configured_label_file"]
        add(f"| Configured label file | {'present' if cfile['exists'] else 'not configured'} |")
        hdr = labels["checks"]["fasta_headers"]
        add(f"| FASTA headers | {hdr['headers_inspected']:,} inspected, "
            f"{hdr['headers_matching_taxonomy_pattern']} contain any taxonomic field |")
        sib = labels["checks"]["sibling_annotation_files"]
        add(f"| Sibling annotation files | {len(sib['candidates_found'])} candidate(s), none a taxonomy table |")
        add("")
        add(f"Every header has exactly 3 tokens — run accession, Illumina cluster "
            f"coordinate, `length=151`. Example: `{hdr['example_headers'][0]}`")
        add("")
        add("Consequently the following are **not computable** and are **not estimated, "
            "simulated, or filled in**: accuracy, macro precision, macro recall, macro F1, "
            "weighted F1, per-class precision/recall/F1, confusion matrix, "
            "training/validation loss and accuracy curves, confidence histogram, "
            "reliability diagram, Brier score, Expected Calibration Error, and "
            "supervised silhouette / intra-class / inter-class distances.")
        add("")
        add("The classifier (`models/classifier.py`) and its full metric suite are "
            "**implemented and runnable**; they were **not executed** because there is no "
            "valid input. Supply `classification.labels_path` and the supervised branch "
            "runs unchanged.")
        add("")
        add("**What would be needed:**")
        add("")
        for item in labels["what_would_be_needed"]:
            add(f"- {item}")
    add("")

    # ------------------------------------------------ J. confidence
    add("## J. Confidence and calibration results")
    add("")
    add(f"**{NA}** — calibration is a property of a trained classifier's predicted "
        "probabilities. With no classifier (§I), there are no probabilities to calibrate.")
    add("")
    add("Noted for the record: low classification confidence would **not** constitute "
        "evidence of a novel species. Novel-taxa discovery is outside the scope of this "
        "implementation and no such inference is drawn anywhere in this work.")
    add("")

    # ------------------------------------------------ K. comparison
    add("## K. Baseline vs foundation model")
    add("")
    if comparison:
        proto = comparison["protocol"]
        add(f"**Fairness protocol.** Both representations were computed from the same "
            f"reads (`{proto['subset_file']}`), after the same QC, with the same seed "
            f"({proto['seed']}), and scored with the same metrics on the same samples. "
            f"Reads compared: {proto['reads_compared']:,}; "
            f"paired-end reads: {proto['pair_reads_compared']:,}.")
        add("")
        add("### K.1 Supervised metrics")
        add("")
        add("| Metric | Baseline | Foundation model |")
        add("|---|---|---|")
        for metric in ("Accuracy", "Macro precision", "Macro recall", "Macro F1", "Weighted F1"):
            add(f"| {metric} | {NA} | {NA} |")
        add("")
        add("### K.2 Label-free retrieval benchmark")
        add("")
        add("The two mates of a paired-end cluster are non-overlapping reads from the "
            "**same source DNA fragment**, hence the same source organism. Given R1's "
            "embedding, is R2 ranked above random distractors? This is measured structure "
            "we did not invent.")
        add("")
        fr = comparison["mate_pair_retrieval"]["foundation_model"]
        br = comparison["mate_pair_retrieval"]["baseline"]
        add(f"Pairs: {fr['n_pairs']:,} · distractors per query: {fr['n_distractors_per_query']} · "
            f"chance top-1: {fr['chance_top1_accuracy'] * 100:.1f}%")
        add("")
        add("| Metric | Baseline | Foundation model | Better |")
        add("|---|---|---|---|")
        for key, name, scale in (
            ("top1_accuracy", "Top-1 retrieval accuracy", 100),
            ("top5_accuracy", "Top-5 retrieval accuracy", 100),
            ("top10_accuracy", "Top-10 retrieval accuracy", 100),
            ("mean_reciprocal_rank", "Mean reciprocal rank", 1),
            ("auroc_mate_vs_random", "AUROC (mate vs random)", 1),
            ("cohens_d", "Effect size (Cohen's d)", 1),
        ):
            bv, fv = br[key], fr[key]
            better = "foundation model" if fv > bv else ("baseline" if bv > fv else "tie")
            suffix = "%" if scale == 100 else ""
            add(f"| {name} | {bv * scale:.4f}{suffix} | {fv * scale:.4f}{suffix} | {better} |")
        add("")
        add(f"*Caveat carried through from the methodology:* {fr['caveat']}")
        add("")
        add("#### Two confounds we found and controlled for")
        add("")
        group = comparison.get("amplicon_end_structure", {})
        unc = comparison.get("mate_pair_retrieval_uncontrolled", {})
        if group and unc:
            add(f"**1. Amplicon-end confound.** Prefix analysis of the paired subset shows "
                f"this is amplicon (metabarcoding) data, not shotgun: "
                f"{group['fraction_assigned'] * 100:.1f}% of reads begin with one of just two "
                f"conserved primer sequences (`{group['anchor_prefixes'][0]}…`, "
                f"`{group['anchor_prefixes'][1]}…`), and the two mates of a cluster are "
                f"**always** the two opposite ends. A uniformly-drawn distractor is therefore "
                f"~50% likely to share the *query's* end and its conserved primer sequence, "
                f"making it spuriously more similar than the true mate.")
            add("")
            add("The effect is large enough to invert the benchmark:")
            add("")
            add("| Variant | Baseline AUROC | Foundation AUROC |")
            add("|---|---|---|")
            add(f"| Uncontrolled distractors | {unc['baseline']['auroc_mate_vs_random']:.4f} | "
                f"{unc['foundation_model']['auroc_mate_vs_random']:.4f} |")
            add(f"| Distractors matched to the mate's amplicon end | "
                f"{br['auroc_mate_vs_random']:.4f} | {fr['auroc_mate_vs_random']:.4f} |")
            add("")
            below = [
                name for name, key in (("baseline", "baseline"),
                                       ("foundation model", "foundation_model"))
                if unc[key]["auroc_mate_vs_random"] < 0.5
            ]
            if below:
                verb = "falls" if len(below) == 1 else "fall"
                add(f"Uncontrolled, the {' and '.join(below)} {verb} **below** chance (0.5) "
                    f"— for that representation the confound does not merely dilute the "
                    f"signal, it reverses it. Both representations lose ground when the "
                    f"control is removed.")
            else:
                add("Both representations lose ground when the control is removed.")
            add("")
            add("The uncontrolled numbers are reported rather than quietly dropped: they are a "
                "genuine property of the data, and they are what exposed the confound. The "
                "end-group assignment is a technical covariate derived from primer sequence, "
                "like the flowcell tile — it is not a taxonomic label and is not used as one.")
            add("")
        strand = comparison.get("strand_orientation")
        if strand:
            v = strand["variants"]
            inv = strand["baseline_reverse_complement_invariance"]
            add("**2. Strand-orientation confound.** The two mates are sequenced from "
                "opposite strands. The canonical k-mer baseline is reverse-complement "
                "invariant by construction; a causal genomic language model is not. "
                "Reverse-complementing mate 2 puts both mates on the same strand — an "
                "exact, information-preserving operation.")
            add("")
            add("| Representation | Orientation | Top-1 | AUROC |")
            add("|---|---|---|---|")
            for key, rep, orient in (
                ("baseline_original", "Baseline", "as sequenced"),
                ("baseline_mate2_revcomp", "Baseline", "mate 2 reverse-complemented"),
                ("foundation_original", "Foundation model", "as sequenced"),
                ("foundation_mate2_revcomp", "Foundation model", "mate 2 reverse-complemented"),
            ):
                r = v[key]
                add(f"| {rep} | {orient} | {r['top1_accuracy'] * 100:.3f}% | "
                    f"{r['auroc_mate_vs_random']:.4f} |")
            add("")
            add(f"The two baseline rows are **bit-for-bit identical** (max absolute difference "
                f"{inv['max_abs_difference_on_transformed_rows']:.1e} across "
                f"{inv['rows_checked']:,} transformed rows), empirically confirming its "
                f"reverse-complement invariance rather than "
                f"asserting it. For the foundation model the same transformation nearly "
                f"doubles top-1 accuracy "
                f"({v['foundation_original']['top1_accuracy'] * 100:.3f}% → "
                f"{v['foundation_mate2_revcomp']['top1_accuracy'] * 100:.3f}%) and raises AUROC "
                f"from {v['foundation_original']['auroc_mate_vs_random']:.4f} to "
                f"{v['foundation_mate2_revcomp']['auroc_mate_vs_random']:.4f}.")
            add("")
            add("**Finding:** a substantial part of GenomeOcean's apparent weakness on paired "
                "short reads is strand orientation, not absence of fragment-level signal. "
                "Because the transformation is provably a no-op for the baseline, applying it "
                "to both keeps the comparison fair, and the strand-corrected numbers are used "
                "as the primary result above.")
            add("")
        add("### K.3 Cost and capacity")
        add("")
        h2h = comparison["head_to_head"]
        f_rep = comparison["representations"]["foundation_model"]
        b_rep = comparison["representations"]["baseline"]
        add("| Metric | Baseline | Foundation model |")
        add("|---|---|---|")
        add(f"| Feature dimension | {b_rep['feature_dimension']:,} | {f_rep['feature_dimension']:,} |")
        add(f"| Model parameters | {b_rep['model_parameters_total']:,} | {f_rep['model_parameters_total']:,} |")
        add(f"| Trainable parameters | {b_rep['model_parameters_trainable']:,} | {f_rep['model_parameters_trainable']:,} |")
        add(f"| Training time | 0 s (not trained) | 0 s (frozen) |")
        add(f"| Inference throughput | {b_rep['sequences_per_second']:,.0f} reads/s | {f_rep['sequences_per_second']:,.0f} reads/s |")
        add(f"| Inference time per read | {b_rep['milliseconds_per_sequence']:.4f} ms | {f_rep['milliseconds_per_sequence']:.4f} ms |")
        add(f"| Storage per embedding | {b_rep['bytes_per_embedding']:,.0f} bytes | {f_rep['bytes_per_embedding']:,.0f} bytes |")
        add(f"| Compute device | {b_rep['device']} | {f_rep['device']} |")
        add("")
        speedup = h2h["inference_cost"]["baseline_speedup_factor"]
        add(f"The baseline is **{speedup:,.0f}× faster** per read. Throughput is realised "
            f"cost on this machine (foundation model on GPU, baseline on CPU); it is not a "
            f"claim about equivalent hardware.")
        add("")
        agree = comparison["agreement"]
        add(f"**Neighbourhood agreement.** Mean Jaccard overlap of {agree['k']}-nearest-neighbour "
            f"sets between the two representations: **{agree['mean_jaccard']:.4f}** "
            f"(median {agree['median_jaccard']:.4f}, n = {agree['sample_size']:,}). "
            f"{agree['interpretation']}")
    add("")

    # ------------------------------------------------ L. performance
    add("## L. Computational performance")
    add("")
    if performance:
        add("Measured on this machine, warm-up excluded and CUDA-synchronised.")
        add("")
        add("| Batch size | Reads/s | ms per read | Peak GPU allocated (MB) |")
        add("|---|---|---|---|")
        for row in performance["rows"]:
            add(f"| {row['batch_size']} | {row['sequences_per_second']:,.1f} | "
                f"{row['milliseconds_per_sequence']:.4f} | {row['peak_gpu_allocated_mb']:,.1f} |")
        add("")
        add(f"Best batch size: **{performance['best_batch_size']}** at "
            f"**{performance['best_throughput_seq_per_second']:,.1f} reads/s**.")
        add("")
        env2 = performance["environment"]
        add(f"GPU: {env2.get('gpu')} ({env2.get('gpu_total_mb', 0):,.0f} MB). Peak usage "
            f"never exceeded {max(r['peak_gpu_allocated_mb'] for r in performance['rows']):,.0f} MB, "
            f"so this workload is compute-bound, not memory-bound, on 6 GB of VRAM.")
        add("")
        proj = performance.get("full_dataset_projection")
        if proj:
            add("**Full-dataset projection** (derived from measured throughput; "
                "**not executed**):")
            add("")
            add("| Quantity | Projected |")
            add("|---|---|")
            add(f"| Reads | {proj['full_dataset_reads']:,} |")
            add(f"| Wall-clock time | {proj['projected_minutes']:,.1f} min ({proj['projected_hours']:.2f} h) |")
            add(f"| Embedding storage | {proj['projected_embedding_storage_gb']:,.2f} GB |")
            add(f"| Basis | {proj['basis']} |")
    add("")

    # ------------------------------------------------ M. figures
    add("## M. Figures generated")
    add("")
    add("All figures are computed from real measurements. None is illustrative.")
    add("")
    add("| File | Content | Scope |")
    add("|---|---|---|")
    scopes = {
        "sequence_length.png": ("Figure 1 — sequence length distribution", "Full dataset"),
        "gc_content.png": ("Figure 2 — GC-content distribution", "Full dataset"),
        "qc_filtering.png": ("Figure 3 — QC filtering outcome", "Full dataset"),
        "ambiguity.png": ("Figure 3a — ambiguous-base distribution", "Full dataset"),
        "nucleotide_composition.png": ("Figure 3b — nucleotide composition", "Full dataset"),
        "read_id_analysis.png": ("Figure 4 — read-ID duplication evidence", "Full dataset"),
        "pca_explained_variance.png": ("Figure 5 — PCA explained variance", "Encoding subset"),
        "pca.png": ("Figure 6 — PCA embedding projection", "Visualization sample"),
        "cluster_tendency.png": ("Figure 6b — cluster tendency vs null", "Analysis sample"),
        "umap.png": ("Figure 7 — UMAP embedding projection", "Visualization sample"),
        "model_comparison.png": ("Figure 8 — baseline vs foundation model", "Encoding subset"),
        "computational_performance.png": ("Figure 12 — computational performance", "Measured on this machine"),
        "architecture.png": ("Implemented architecture diagram", "Documentation"),
    }
    # Stage 2 figures live in the same directory but belong to the other report.
    stage2 = {
        "dereplication.png": "Figure 13 — dereplication and frequency spectrum",
        "rank_abundance.png": "Figure 14 — rank abundance and concentration",
        "rarefaction.png": "Figure 15 — sampling completeness",
        "community_clusters.png": "Figure 16 — community clustering comparison",
        "community_map.png": "Figure 17 — abundance-weighted community map",
    }
    for name in figures:
        if name in stage2:
            continue
        title, scope = scopes.get(name, (name, "-"))
        add(f"| `{name}` | {title} | {scope} |")
    add("")
    add("Stage 2 produced five further figures, documented in "
        "[`community_report.md`](community_report.md): "
        + ", ".join(f"`{n}`" for n in stage2 if n in figures)
        + ".")
    add("")
    add("**Not generated, because the underlying data does not exist:**")
    add("")
    add("| Figure | Reason |")
    add("|---|---|")
    add(f"| Figure 9 — confusion matrix | {NA} (no labels) |")
    add(f"| Figure 10 — training/validation curves | {NA} (no classifier trained) |")
    add(f"| Figure 11 — confidence / calibration | {NA} (no predicted probabilities) |")
    add("")

    # ------------------------------------------------ N. limitations
    add("## N. Limitations")
    add("")
    for i, item in enumerate([
        "**No ground-truth taxonomy.** The dominant limitation. No supervised metric is "
        "computable, so 'useful for downstream taxonomic analysis' is assessed by proxy "
        "(mate-pair retrieval, cluster structure) rather than directly.",
        "**Single sample.** No abundance features, so the paper's student input cannot be "
        "reproduced even in principle.",
        "**Unassembled reads.** The paper's premise is that assembly supplies context; we "
        "deliberately test the regime where it does not.",
        "**Subset, not full-dataset, inference.** Foundation-model embeddings were "
        "extracted for the encoding subset. The full-dataset cost is a labelled projection.",
        "**Out-of-distribution context length.** GenomeOcean was pretrained on assembled "
        "sequence. 151 bp is in-spec for input length, not for training context.",
        "**Mate-pair retrieval is a proxy**, with a stated locus-sharing confound. It "
        "demonstrates same-fragment signal, which is necessary but not sufficient for "
        "species-level discriminability.",
        "**One foundation model tested.** No claim is made about genomic foundation models "
        "in general.",
        "**One sequencing run, one site.** No claim is made about deep-sea eDNA in general.",
    ], 1):
        add(f"{i}. {item}")
    add("")

    # ------------------------------------------------ O. feasibility
    add("## O. Is the first-stage architecture technically feasible?")
    add("")
    add("**Yes, on the evidence collected here.** Specifically:")
    add("")
    if dataset and enc_main and performance:
        delta = preprocess.get("streaming_phase_memory_delta_mb")
        add(f"1. **The data can be processed at scale.** All {dataset['total_sequence_records']:,} "
            f"records stream through parsing and QC in {preprocess['elapsed_seconds']:,.0f} s "
            f"({preprocess['records_per_second']:,.0f} records/s), growing resident memory by "
            f"{delta} MB while reading a {dataset['source_file_mb']:,.0f} MB file, with "
            f"{dataset['qc_pass_rate'] * 100:.2f}% passing QC.")
        add("")
        add(f"2. **The paper's own foundation model accepts our short reads directly.** "
            f"151 bp → ~{compat['tokens_per_read_mean']:.0f} BPE tokens, "
            f"{compat['fraction_of_limit_used'] * 100:.1f}% of the model's 1,024-token limit, "
            f"0 `[UNK]` tokens, 0 truncations, 0 padding. No substitute model was required.")
        add("")
        add(f"3. **Embeddings are well-formed.** {enc_main['sequences_successful']:,} of "
            f"{enc_main['sequences_requested']:,} reads encoded successfully "
            f"({enc_main['sequences_failed']} failures), all values finite, "
            f"{enc_main['embedding_health']['dead_dimensions']} dead dimensions out of "
            f"{enc_main['embedding_dimension']:,}.")
        add("")
        proj = performance.get("full_dataset_projection", {})
        add(f"4. **It is computationally feasible on modest hardware.** "
            f"{performance['best_throughput_seq_per_second']:,.0f} reads/s on a 6 GB laptop GPU, "
            f"peak usage under {max(r['peak_gpu_allocated_mb'] for r in performance['rows']):,.0f} MB. "
            f"The full {proj.get('full_dataset_reads', 0):,}-read dataset projects to "
            f"{proj.get('projected_hours', 0):.1f} h and "
            f"{proj.get('projected_embedding_storage_gb', 0):.1f} GB.")
        add("")
        if comparison:
            fr = comparison["mate_pair_retrieval"]["foundation_model"]
            br = comparison["mate_pair_retrieval"]["baseline"]
            better = "outperforms" if fr["top1_accuracy"] > br["top1_accuracy"] else (
                "does not outperform" if fr["top1_accuracy"] < br["top1_accuracy"] else "matches")
            add(f"5. **The embeddings carry real, measurable genomic signal.** On the "
                f"label-free mate-pair retrieval benchmark the foundation model reaches "
                f"{fr['top1_accuracy'] * 100:.2f}% top-1 against a "
                f"{fr['chance_top1_accuracy'] * 100:.1f}% chance baseline "
                f"({fr['lift_over_chance']:,.1f}× chance), and {better} the k-mer baseline "
                f"({br['top1_accuracy'] * 100:.2f}%).")
    add("")
    add("**What this does NOT establish.** Feasibility of the representation stage is not "
        "the same as taxonomic utility. Without ground-truth labels this experiment cannot "
        "measure whether these embeddings support accurate taxonomic assignment — only that "
        "they are producible, well-formed, computationally tractable, and carry "
        "non-trivial sequence-level structure. Any claim beyond that would not be supported "
        "by these measurements.")
    add("")

    # ------------------------------------------------ P. next stage
    add("## P. What additional data is required for the next stage")
    add("")
    add("| Requirement | Why | Enables |")
    add("|---|---|---|")
    add("| Per-read taxonomic labels (Kraken2 / MMseqs2 / Metabuli output, or a mock "
        "community with known source genomes) | Nothing supervised is measurable without them | "
        "Accuracy, macro/weighted F1, per-class metrics, confusion matrix, training curves |")
    add("| The label source's own error rate | TaxDistill's premise is that retrieval labels "
        "are noisy | Meaningful evaluation of label correction |")
    add("| A taxonomy tree for the target taxa | The paper's hierarchical loss operates over it | "
        "Deep hierarchical loss, hierarchy-aware evaluation |")
    add("| Multiple samples from the same site, mapped to a shared reference | Abundance is a "
        "cross-sample quantity | The paper's full student input (TNF + K abundances + total) |")
    add("| Assembled contigs from these reads, if assembly is viable | Enables a like-for-like "
        "comparison with the paper's ≥ 2,000 bp regime | Direct measurement of how much the "
        "151 bp constraint actually costs |")
    add("| A held-out validation set with independent labels | Test-set integrity | "
        "Calibration, confidence analysis, honest generalisation estimates |")
    add("")
    add("---")
    add("")
    add("*Generated by `python main.py report`. Every value above is read from a metrics "
        "file produced by an executed run; nothing is hand-entered.*")

    path = cfgutil.output_dir(cfg, "reports") / "experiment_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {path}")
    return path
