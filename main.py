"""Deep-sea eDNA genomic representation pipeline -- stage 1.

A TaxDistill-inspired adaptation for short deep-sea eDNA reads.

Usage
-----
    python main.py preprocess              # full-dataset streaming QC + metrics
    python main.py figures                 # dataset figures from stored metrics
    python main.py subsets                 # build encoding subsets
    python main.py testrun                 # small-subset validation of the encoder
    python main.py encode                  # foundation-model embeddings
    python main.py baseline                # k-mer/TNF embeddings
    python main.py strandtest              # strand-orientation control experiment
    python main.py labels                  # verify whether taxonomy exists
    python main.py assign                  # STAGE 3: reference + noisy taxonomic labels
    python main.py train                   # STAGE 3: supervised probe on those labels
    python main.py distill                 # STAGE 4: TaxDistill teacher/student KD
    python main.py distill-ablation        # STAGE 4: alpha / temperature sweeps
    python main.py dereplicate             # STAGE 2: full-dataset variants + diversity
    python main.py community               # STAGE 2: encode variants, cluster community
    python main.py amplicon                # STAGE 2: conserved/variable read architecture
    python main.py varregion               # STAGE 2: retrieval on the variable region only
    python main.py analyze                 # PCA / UMAP / cluster tendency
    python main.py compare                 # baseline vs foundation model
    python main.py performance             # batch-size sweep + projection
    python main.py architecture            # architecture diagram
    python main.py report                  # stage 1 experiment report
    python main.py community-report        # STAGE 2 community structure report
    python main.py taxonomy-report         # STAGE 3 taxonomy + classification report
    python main.py distill-report          # STAGE 4 knowledge-distillation report
    python main.py all                     # every stage, in order

Options
-------
    --config PATH        alternative JSON config
    --set key.path=val   override any config value (JSON-parsed)
    --force              recompute a stage even if its output exists

Expensive stages reuse existing outputs unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import config as cfgutil
from utils.runtime import environment_report, set_seed


def _metrics_dir(cfg):
    return cfgutil.output_dir(cfg, "metrics")


def _read(cfg, name):
    path = _metrics_dir(cfg) / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------


def stage_preprocess(cfg, force=False):
    from preprocessing.run import run_preprocess

    return run_preprocess(cfg, force=force)


def stage_figures(cfg, force=False):
    from visualization.dataset_figures import generate_all

    return generate_all(cfg)


def stage_subsets(cfg, force=False):
    from preprocessing.run import build_pair_subset, build_subset

    return {
        "test": build_subset(cfg, "test", cfg["subset"]["test_run_size"], force),
        "main": build_subset(cfg, "main", cfg["subset"]["encode_size"], force),
        "pairs": build_pair_subset(cfg, "pairs", cfg["subset"]["pair_subset_pairs"], force),
    }


def _encode(cfg, which, subset_name, store_name, run_label, force=False):
    """Shared encode driver for both encoder families."""
    from embeddings.run import extract
    from models.foundation_encoder import build_baseline, build_encoder
    from preprocessing.run import load_subset, subset_path

    path = subset_path(cfg, subset_name)
    if which == "foundation":
        encoder = build_encoder(cfg)
        _, sequences = load_subset(path)
        compatibility = encoder.verify_compatibility(sequences[: min(1000, len(sequences))])
        print(f"[compat] {compatibility['tokens_per_read_mean']:.1f} tokens/read, "
              f"limit {compatibility['model_token_limit']}, "
              f"fits={compatibility['fits_without_truncation']}, "
              f"UNK={compatibility['unk_tokens_in_sample']}")
        # Per-store, so successive encodes do not clobber each other's record.
        # The canonical copy is also embedded in encoder_metrics_<store>.json.
        cfgutil.save_json(
            compatibility,
            _metrics_dir(cfg) / f"model_compatibility_{store_name}.json",
        )
        batch = cfg["encoder"]["batch_size"]
        shard = cfg["encoder"]["shard_size"]
    else:
        encoder = build_baseline(cfg)
        compatibility = None
        batch = cfg["baseline"]["batch_size"]
        shard = cfg["baseline"]["shard_size"]

    try:
        return extract(cfg, encoder, path, store_name, batch, shard,
                       run_label, compatibility, force)
    finally:
        encoder.close()


def stage_testrun(cfg, force=False):
    """Mandatory small-subset validation before any expensive run."""
    print("=" * 70)
    print("TEST RUN -- small subset validation. This is NOT the full dataset.")
    print("=" * 70)
    from preprocessing.run import build_subset

    build_subset(cfg, "test", cfg["subset"]["test_run_size"], force)
    foundation = _encode(cfg, "foundation", "test", "test_genomeocean", "TEST RUN", force)
    baseline = _encode(cfg, "baseline", "test", "test_kmer4", "TEST RUN", force)

    from embeddings.store import load_embeddings
    from analysis.pca import run_pca

    root = cfgutil.output_dir(cfg, "embeddings")
    checks = {}
    for label, store in (("foundation", "test_genomeocean"), ("baseline", "test_kmer4")):
        _, matrix, _ = load_embeddings(root, store)
        pca_result, _ = run_pca(matrix, 10, cfg["seed"])
        checks[label] = {
            "shape": list(matrix.shape),
            "all_finite": bool(__import__("numpy").isfinite(matrix).all()),
            "pca_ran": True,
            "variance_first_10": pca_result["variance_first_10"],
        }
    report = {
        "stage": "testrun",
        "run_label": "TEST RUN",
        "subset_size": cfg["subset"]["test_run_size"],
        "foundation": foundation,
        "baseline": baseline,
        "validation_checks": checks,
        "verdict": "All test-run checks passed; safe to scale up."
        if all(c["all_finite"] for c in checks.values())
        else "TEST RUN FAILED -- do not scale up.",
    }
    cfgutil.save_json(report, _metrics_dir(cfg) / "testrun_metrics.json")
    print(f"[testrun] {report['verdict']}")
    return report


def stage_encode(cfg, force=False):
    print("=" * 70)
    print("FULL SUBSET RUN -- foundation-model embedding extraction")
    print("=" * 70)
    main = _encode(cfg, "foundation", "main", "main_genomeocean", "FULL SUBSET RUN", force)
    pairs = _encode(cfg, "foundation", "pairs", "pairs_genomeocean", "FULL SUBSET RUN", force)
    return {"main": main, "pairs": pairs}


def stage_baseline(cfg, force=False):
    main = _encode(cfg, "baseline", "main", "main_kmer4", "FULL SUBSET RUN", force)
    pairs = _encode(cfg, "baseline", "pairs", "pairs_kmer4", "FULL SUBSET RUN", force)
    return {"main": main, "pairs": pairs}


def stage_strandtest(cfg, force=False):
    """Does reverse-complementing mate 2 change retrieval? Isolates strand effects."""
    import numpy as np

    from embeddings.store import load_embeddings
    from evaluation import structure
    from preprocessing.run import build_rc_pair_subset, load_subset, subset_path

    build_rc_pair_subset(cfg)
    _encode(cfg, "foundation", "pairs_rc", "pairs_rc_genomeocean", "FULL SUBSET RUN", force)
    _encode(cfg, "baseline", "pairs_rc", "pairs_rc_kmer4", "FULL SUBSET RUN", force)

    # Group assignment comes from the ORIGINAL orientation so the control is
    # identical across both variants.
    _, original = load_subset(subset_path(cfg, "pairs"))
    groups, group_meta = structure.amplicon_end_groups(original)

    root = cfgutil.output_dir(cfg, "embeddings")
    n_distractors = cfg["analysis"].get("retrieval_distractors", 99)
    results = {"amplicon_end_structure": group_meta, "variants": {}}
    for label, store in (
        ("foundation_original", "pairs_genomeocean"),
        ("foundation_mate2_revcomp", "pairs_rc_genomeocean"),
        ("baseline_original", "pairs_kmer4"),
        ("baseline_mate2_revcomp", "pairs_rc_kmer4"),
    ):
        _, matrix, _ = load_embeddings(root, store)
        results["variants"][label] = structure.strip_internal(
            structure.mate_pair_retrieval(matrix, n_distractors, cfg["seed"], groups=groups)
        )

    # Empirical check of the claim that canonical k-mers are RC-invariant.
    _, base_a, _ = load_embeddings(root, "pairs_kmer4")
    _, base_b, _ = load_embeddings(root, "pairs_rc_kmer4")
    odd = np.arange(1, min(base_a.shape[0], base_b.shape[0]), 2)
    results["baseline_reverse_complement_invariance"] = {
        "claim": "canonical k-mer frequencies are invariant under reverse complement",
        "max_abs_difference_on_transformed_rows": round(
            float(np.abs(base_a[odd] - base_b[odd]).max()), 8
        ),
        "rows_checked": int(odd.size),
        "verified": bool(np.allclose(base_a[odd], base_b[odd], atol=1e-6)),
    }
    cfgutil.save_json(results, _metrics_dir(cfg) / "strand_orientation_metrics.json")
    for label, res in results["variants"].items():
        print(f"[strand] {label:28} top1={res['top1_accuracy']:.5f} "
              f"auroc={res['auroc_mate_vs_random']:.5f}")
    print(f"[strand] baseline RC-invariance verified: "
          f"{results['baseline_reverse_complement_invariance']['verified']}")
    return results


def stage_dereplicate(cfg, force=False):
    """Stage 2 -- full-dataset dereplication + diversity statistics."""
    from analysis.dereplicate import dereplicate
    from analysis.diversity import diversity_from_spectrum, rank_abundance, rarefaction

    derep = dereplicate(cfg, force=force)
    spectrum = derep["frequency_spectrum"]

    diversity = {
        "stage": "diversity",
        "scope": "FULL DATASET",
        "unit": "exact sequence variant (not taxon, not denoised ASV, not OTU)",
        "indices": diversity_from_spectrum(spectrum),
        "rank_abundance": rank_abundance(spectrum),
        "rarefaction": rarefaction(spectrum, cfg["community"]["rarefaction_points"]),
        "seed": cfg["seed"],
    }
    cfgutil.save_json(diversity, _metrics_dir(cfg) / "diversity_metrics.json")

    from visualization.community_figures import (
        figure_dereplication,
        figure_rank_abundance,
        figure_rarefaction,
    )
    from visualization import style as S

    S.apply_style()
    figure_dereplication(cfg, derep, diversity)
    figure_rank_abundance(cfg, diversity)
    figure_rarefaction(cfg, diversity)

    idx = diversity["indices"]
    print(f"[derep] {idx['observed_richness_variants']:,} variants | "
          f"Shannon {idx['shannon_H']:.3f} | "
          f"{diversity['rank_abundance']['variants_for_50pct_reads']} variants = 50% of reads")
    return {"dereplication": derep, "diversity": diversity}


def stage_community(cfg, force=False):
    """Stage 2 -- encode variants and profile unsupervised community structure."""
    import numpy as np
    import pyarrow.parquet as pq

    from analysis import community as com
    from embeddings.store import load_embeddings
    from evaluation import structure
    from preprocessing.run import load_subset

    ccfg = cfg["community"]
    coords_dir = cfgutil.output_dir(cfg, "embeddings", "coords")
    cached = _read(cfg, "community_metrics.json")
    label_files = {
        "foundation_model": coords_dir / "variant_labels_foundation.npy",
        "baseline": coords_dir / "variant_labels_baseline.npy",
    }
    if cached and not force and all(p.exists() for p in label_files.values()):
        # k-means over 47k x 1536 across the k grid costs minutes; never redo it
        # just to redraw a figure.
        print("[community] Reusing community_metrics.json (use --force to recluster).")
        labels = {k: np.load(p) for k, p in label_files.items()}
        counts, gc = com.variant_weights(cfg, cached["variants_encoded"])
        from visualization.community_figures import (
            figure_community_clusters,
            figure_community_map,
        )
        from visualization import style as S

        S.apply_style()
        figure_community_clusters(cfg, cached)
        figure_community_map(cfg, cached, labels, counts, gc)
        return cached

    subset = com.build_variant_subset(cfg, "variants", ccfg["encode_variants"])
    n = pq.read_metadata(subset).num_rows

    _encode(cfg, "foundation", "variants", "variants_genomeocean", "FULL SUBSET RUN", force)
    _encode(cfg, "baseline", "variants", "variants_kmer4", "FULL SUBSET RUN", force)

    counts, gc = com.variant_weights(cfg, n)
    _, sequences = load_subset(subset)
    end_groups, end_meta = structure.amplicon_end_groups(sequences)

    root = cfgutil.output_dir(cfg, "embeddings")
    results = {
        "stage": "community",
        "unit": "exact sequence variant",
        "variants_encoded": int(n),
        "reads_represented": int(counts.sum()),
        "amplicon_end_structure": end_meta,
        "representations": {},
    }

    labels = {}
    for label, store in (
        ("foundation_model", "variants_genomeocean"),
        ("baseline", "variants_kmer4"),
    ):
        _, matrix, _ = load_embeddings(root, store)
        matrix = matrix[:n]
        res = com.cluster_community(
            matrix, counts, gc, end_groups,
            ccfg["cluster_k_grid"], cfg["seed"], cfg["analysis"]["silhouette_sample"],
        )
        labels[label] = np.array(res.pop("labels"))
        res["structure_vs_measured_properties"] = com.structure_vs_properties(
            labels[label], gc, counts, end_groups
        )
        results["representations"][label] = res
        print(f"[community] {label}: k={res['selected_k']}, "
              f"largest cluster = {res['largest_cluster_read_share'] * 100:.1f}% of reads")

    results["partition_agreement"] = com.compare_partitions(
        labels["foundation_model"], labels["baseline"]
    )
    cfgutil.save_json(results, _metrics_dir(cfg) / "community_metrics.json")
    np.save(cfgutil.output_dir(cfg, "embeddings", "coords") / "variant_labels_foundation.npy",
            labels["foundation_model"])
    np.save(cfgutil.output_dir(cfg, "embeddings", "coords") / "variant_labels_baseline.npy",
            labels["baseline"])

    from visualization.community_figures import (
        figure_community_clusters,
        figure_community_map,
    )
    from visualization import style as S

    S.apply_style()
    figure_community_clusters(cfg, results)
    figure_community_map(cfg, results, labels, counts, gc)
    print(f"[community] partition agreement ARI = "
          f"{results['partition_agreement']['adjusted_rand_index']:.4f}")
    return results


def stage_amplicon(cfg, force=False):
    """Stage 2 -- how much of each read actually varies across the community?"""
    from analysis.amplicon import run

    result = run(cfg)

    from visualization.community_figures import figure_amplicon_architecture
    from visualization import style as S

    S.apply_style()
    figure_amplicon_architecture(cfg, result)
    return result


def stage_varregion(cfg, force=False):
    """Re-run the retrieval comparison on the variable region only.

    Isolates whether the foundation model's lack of advantage is a property of
    the model or an artefact of the near-invariant conserved anchor.
    """
    from analysis.amplicon import characterize
    from analysis.variable_region import build_variable_region_pairs
    from embeddings.store import load_embeddings
    from evaluation import structure
    from preprocessing.run import load_subset, subset_path

    amplicon = _read(cfg, "amplicon_metrics.json") or characterize(cfg)
    subset_meta = build_variable_region_pairs(cfg, amplicon)

    _encode(cfg, "foundation", "pairs_var", "pairs_var_genomeocean",
            "FULL SUBSET RUN", force)
    _encode(cfg, "baseline", "pairs_var", "pairs_var_kmer4", "FULL SUBSET RUN", force)

    # Groups from the ORIGINAL untrimmed reads, so the control is identical to
    # the full-read experiment and the two are directly comparable.
    _, original = load_subset(subset_path(cfg, "pairs"))
    groups, _ = structure.amplicon_end_groups(original)

    root = cfgutil.output_dir(cfg, "embeddings")
    n_distractors = cfg["analysis"].get("retrieval_distractors", 99)
    results = {
        "stage": "variable_region",
        "question": "Does removing the conserved anchor change which representation wins?",
        "subset": subset_meta,
        "variants": {},
    }
    for label, store in (
        ("foundation_variable_region", "pairs_var_genomeocean"),
        ("baseline_variable_region", "pairs_var_kmer4"),
        ("foundation_full_read", "pairs_rc_genomeocean"),
        ("baseline_full_read", "pairs_rc_kmer4"),
    ):
        _, matrix, _ = load_embeddings(root, store)
        results["variants"][label] = structure.mate_pair_retrieval(
            matrix, n_distractors, cfg["seed"], groups=groups
        )

    v = results["variants"]
    # Paired significance tests: the differences are small, so state explicitly
    # whether each is distinguishable from noise.
    results["significance"] = {
        "full_read_foundation_vs_baseline": structure.compare_retrieval(
            v["foundation_full_read"], v["baseline_full_read"], "foundation", "baseline"),
        "variable_region_foundation_vs_baseline": structure.compare_retrieval(
            v["foundation_variable_region"], v["baseline_variable_region"],
            "foundation", "baseline"),
        "foundation_full_vs_variable": structure.compare_retrieval(
            v["foundation_variable_region"], v["foundation_full_read"],
            "variable_region", "full_read"),
        "baseline_full_vs_variable": structure.compare_retrieval(
            v["baseline_variable_region"], v["baseline_full_read"],
            "variable_region", "full_read"),
    }
    results["effect_of_removing_the_anchor"] = {
        "foundation_top1_delta": round(
            v["foundation_variable_region"]["top1_accuracy"]
            - v["foundation_full_read"]["top1_accuracy"], 6),
        "baseline_top1_delta": round(
            v["baseline_variable_region"]["top1_accuracy"]
            - v["baseline_full_read"]["top1_accuracy"], 6),
        "foundation_auroc_delta": round(
            v["foundation_variable_region"]["auroc_mate_vs_random"]
            - v["foundation_full_read"]["auroc_mate_vs_random"], 6),
        "baseline_auroc_delta": round(
            v["baseline_variable_region"]["auroc_mate_vs_random"]
            - v["baseline_full_read"]["auroc_mate_vs_random"], 6),
        "winner_full_read": "foundation"
        if v["foundation_full_read"]["top1_accuracy"]
        > v["baseline_full_read"]["top1_accuracy"] else "baseline",
        "winner_variable_region": "foundation"
        if v["foundation_variable_region"]["top1_accuracy"]
        > v["baseline_variable_region"]["top1_accuracy"] else "baseline",
    }
    for res in v.values():
        structure.strip_internal(res)
    cfgutil.save_json(results, _metrics_dir(cfg) / "variable_region_metrics.json")

    for label, res in v.items():
        print("[varregion] %-32s top1=%.5f auroc=%.5f"
              % (label, res["top1_accuracy"], res["auroc_mate_vs_random"]))

    from visualization.community_figures import figure_variable_region
    from visualization import style as S

    S.apply_style()
    figure_variable_region(cfg, results)
    return results


def stage_train(cfg, force=False):
    """Supervised branch: linear probe on frozen embeddings.

    Uses the NOISY reference-derived labels produced by `python main.py assign`.
    If no labels exist it reports unavailability and exits cleanly -- it never
    invents labels to have something to train on.
    """
    import numpy as np
    from collections import Counter

    from embeddings.store import load_embeddings
    from models.classifier import (
        LinearProbe, classification_metrics, confidence_and_calibration,
        stratified_split,
    )
    from taxonomy.assign import load_labels
    from utils.runtime import NOT_AVAILABLE, Timer

    ccfg = cfg["classification"]
    stores = ccfg["embedding_stores"]
    root = cfgutil.output_dir(cfg, "embeddings")
    seed = cfg["seed"]
    min_class = ccfg.get("min_class_size", 30)

    all_results = {"stage": "train", "ranks": {}}
    for rank in cfg["taxonomy"]["ranks"]:
        try:
            labels_map = load_labels(cfg, rank)
        except FileNotFoundError:
            print("[train] No labels for rank " + rank + ". Run: python main.py assign")
            continue
        if not labels_map:
            continue

        rank_results = {
            "rank": rank,
            "label_provenance": (
                "NOISY reference-derived labels from a naive Bayes classifier "
                "against PR2. NOT ground truth. See "
                "outputs/metrics/taxonomy_metrics_" + rank + ".json for the "
                "measured classifier error rate."
            ),
            "representations": {},
        }
        for label, store in stores.items():
            ids, matrix, _ = load_embeddings(root, store)
            index_of = {sid: i for i, sid in enumerate(ids)}
            pairs = [(index_of[k], v) for k, v in labels_map.items() if k in index_of]
            if not pairs:
                raise ValueError(
                    "No label key matched a sequence id in store " + store
                )
            counts = Counter(v for _, v in pairs)
            usable = {c for c, n in counts.items() if n >= min_class}
            pairs = [(i, v) for i, v in pairs if v in usable]

            classes = sorted({v for _, v in pairs})
            cls_index = {c: i for i, c in enumerate(classes)}
            rows = np.array([i for i, _ in pairs])
            x = matrix[rows]
            y = np.array([cls_index[v] for _, v in pairs])

            train, val, test = stratified_split(y, tuple(ccfg["split"]), seed)

            # FAIRNESS: standardise features before the probe. k-mer frequencies
            # have std ~0.04 while the foundation embeddings have std ~0.64, so at
            # a shared learning rate the baseline silently fails to converge and
            # collapses to majority-class prediction. Statistics are fitted on the
            # TRAIN split only, so no information leaks from val/test.
            mean = x[train].mean(axis=0)
            std = x[train].std(axis=0)
            std[std == 0] = 1.0
            x = (x - mean) / std

            probe = LinearProbe(x.shape[1], len(classes), ccfg["model"],
                                ccfg["hidden_dim"], seed)
            timer = Timer()
            history = probe.fit(
                x[train], y[train], x[val], y[val],
                ccfg["epochs"], ccfg["lr"], ccfg["weight_decay"], ccfg["batch_size"],
            )
            train_seconds = timer.stop()
            probabilities = probe.predict_proba(x[test])
            rank_results.setdefault("_test_pred", {})[label] = probabilities.argmax(1)
            rank_results["_test_truth"] = y[test]

            rank_results["representations"][label] = {
                "store": store,
                "n_labelled": len(pairs),
                "n_classes": len(classes),
                "classes": classes,
                "class_counts": {c: int(counts[c]) for c in classes},
                "classes_dropped_below_min": sorted(set(counts) - usable),
                "min_class_size": min_class,
                "embedding_dim": int(x.shape[1]),
                "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
                "training_seconds": round(train_seconds, 3),
                "features_standardised": "fitted on the train split only",
                "parameter_counts": probe.parameter_counts(),
                "history": history,
                "metrics": classification_metrics(y[test], probabilities, classes),
                "calibration": confidence_and_calibration(y[test], probabilities),
            }
            m = rank_results["representations"][label]["metrics"]
            print("[train] %-6s %-16s n=%5d k=%2d acc=%.4f macroF1=%.4f weightedF1=%.4f"
                  % (rank, label, len(pairs), len(classes), m["accuracy"],
                     m["macro_f1"], m["weighted_f1"]))

        preds = rank_results.pop("_test_pred", {})
        truth = rank_results.pop("_test_truth", None)
        if len(preds) == 2 and truth is not None:
            from scipy.stats import binomtest
            import math as _math
            a = preds["foundation_model"] == truth
            b = preds["baseline"] == truth
            only_a = int((a & ~b).sum()); only_b = int((b & ~a).sum())
            disc = only_a + only_b
            if disc:
                t = binomtest(only_a, disc, 0.5)
                diff = float(a.mean() - b.mean())
                se = _math.sqrt(disc) / len(truth)
                rank_results["significance"] = {
                    "test": "McNemar exact on the shared held-out test split",
                    "n_test": int(len(truth)),
                    "accuracy_difference_pp": round(diff * 100, 4),
                    "ci95_pp": [round((diff - 1.96 * se) * 100, 4),
                                round((diff + 1.96 * se) * 100, 4)],
                    "correct_only_by_foundation": only_a,
                    "correct_only_by_baseline": only_b,
                    "p_value": float(t.pvalue),
                    "significant_at_0.05": bool(t.pvalue < 0.05),
                    "verdict": ("foundation model better" if diff > 0 else "baseline better")
                    + (" (p=%.2e)" % t.pvalue) if t.pvalue < 0.05
                    else "no significant difference (p=%.3f)" % t.pvalue,
                }
                print("[train] %-6s significance: %s" % (rank, rank_results["significance"]["verdict"]))

        reps = rank_results["representations"]
        if len(reps) == 2:
            f = reps["foundation_model"]["metrics"]
            b = reps["baseline"]["metrics"]
            rank_results["head_to_head"] = {
                "accuracy": {"foundation_model": f["accuracy"], "baseline": b["accuracy"]},
                "macro_f1": {"foundation_model": f["macro_f1"], "baseline": b["macro_f1"]},
                "weighted_f1": {"foundation_model": f["weighted_f1"],
                                "baseline": b["weighted_f1"]},
                "winner_accuracy": "foundation_model"
                if f["accuracy"] > b["accuracy"] else "baseline",
                "winner_macro_f1": "foundation_model"
                if f["macro_f1"] > b["macro_f1"] else "baseline",
            }
        all_results["ranks"][rank] = rank_results

    if not all_results["ranks"]:
        all_results.update({"executed": False, "accuracy": NOT_AVAILABLE,
                            "macro_f1": NOT_AVAILABLE})
    else:
        all_results["executed"] = True
    cfgutil.save_json(all_results, _metrics_dir(cfg) / "classification_metrics.json")

    if all_results.get("executed"):
        from visualization.classification_figures import generate_all
        from visualization import style as S
        S.apply_style()
        for rank, res in all_results["ranks"].items():
            generate_all(
                cfg,
                {"executed": True, "representations": res["representations"]},
                suffix="_" + rank,
            )
    return all_results


def stage_assign(cfg, force=False):
    """Stage 3 -- build the reference and assign noisy taxonomic labels."""
    from taxonomy.assign import assign
    from taxonomy.reference import build_reference

    build_reference(cfg, force=force)
    return {
        rank: assign(cfg, force=force, rank=rank, threshold=threshold)
        for rank, threshold in cfg["taxonomy"]["ranks"].items()
    }


def stage_distill(cfg, force=False):
    """Stage 4 -- TaxDistill's knowledge-distillation loop.

    Three-way comparison per rank: teacher (frozen foundation model + head),
    student alone (TNF + abundance MLP, hard labels), and student + KD.
    """
    import numpy as np
    import pyarrow.parquet as pq
    from collections import Counter

    from analysis.dereplicate import variants_path
    from embeddings.store import load_embeddings
    from models.classifier import (
        classification_metrics, confidence_and_calibration, stratified_split,
    )
    from taxonomy.assign import load_labels
    from training.distill import DistillationExperiment, build_student_features
    from utils.runtime import Timer

    dcfg = cfg["distillation"]
    ccfg = cfg["classification"]
    root = cfgutil.output_dir(cfg, "embeddings")
    seed = cfg["seed"]
    min_class = ccfg.get("min_class_size", 30)

    counts_table = pq.read_table(variants_path(cfg), columns=["variant_id", "count"])
    count_of = dict(zip(counts_table.column("variant_id").to_pylist(),
                        counts_table.column("count").to_pylist()))

    results = {"stage": "distill", "ranks": {}}
    for rank in cfg["taxonomy"]["ranks"]:
        try:
            labels_map = load_labels(cfg, rank)
        except FileNotFoundError:
            continue
        if not labels_map:
            continue

        t_ids, teacher_x, _ = load_embeddings(root, ccfg["embedding_stores"]["foundation_model"])
        s_ids, kmer_x, _ = load_embeddings(root, ccfg["embedding_stores"]["baseline"])
        assert t_ids == s_ids, "teacher and student stores must be row-aligned"
        index_of = {sid: i for i, sid in enumerate(t_ids)}

        pairs = [(index_of[k], v) for k, v in labels_map.items() if k in index_of]
        counter = Counter(v for _, v in pairs)
        usable = {c for c, n in counter.items() if n >= min_class}
        pairs = [(i, v) for i, v in pairs if v in usable]
        classes = sorted({v for _, v in pairs})
        cls_index = {c: i for i, c in enumerate(classes)}

        rows = np.array([i for i, _ in pairs])
        y = np.array([cls_index[v] for _, v in pairs])
        reads = np.array([count_of.get(int(t_ids[i].split("_")[1]), 1) for i in rows])

        xt = teacher_x[rows]
        xs = build_student_features(kmer_x[rows], reads, dcfg["include_abundance"])
        train, val, test = stratified_split(y, tuple(ccfg["split"]), seed)

        experiment = DistillationExperiment(
            xt.shape[1], xs.shape[1], len(classes), ccfg["hidden_dim"], seed
        )
        timer = Timer()
        run = experiment.run(
            xt, xs, y, train, val, test,
            alpha=dcfg["alpha"], temperature=dcfg["temperature"],
            epochs=dcfg["epochs"], lr=ccfg["lr"],
            weight_decay=ccfg["weight_decay"], batch_size=ccfg["batch_size"],
            teacher_head=dcfg["teacher_head"],
        )
        elapsed = timer.stop()

        rank_result = {
            "rank": rank,
            "n_labelled": len(pairs),
            "n_classes": len(classes),
            "classes": classes,
            "student_feature_dim": int(xs.shape[1]),
            "teacher_feature_dim": int(xt.shape[1]),
            "includes_abundance": dcfg["include_abundance"],
            "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
            "training_seconds": round(elapsed, 2),
            "hyperparameters": run["hyperparameters"],
            "parameter_counts": run["parameter_counts"],
            "history": run["history"],
            "branches": {},
        }
        for branch, probabilities in run["test_probabilities"].items():
            rank_result["branches"][branch] = {
                "metrics": classification_metrics(run["y_test"], probabilities, classes),
                "calibration": confidence_and_calibration(run["y_test"], probabilities),
            }

        # The claim: does KD improve the student? Paired McNemar on the shared test set.
        from scipy.stats import binomtest
        import math as _math
        truth = run["y_test"]
        kd_ok = run["test_probabilities"]["student_kd"].argmax(1) == truth
        alone_ok = run["test_probabilities"]["student_alone"].argmax(1) == truth
        only_kd = int((kd_ok & ~alone_ok).sum()); only_alone = int((alone_ok & ~kd_ok).sum())
        disc = only_kd + only_alone
        if disc:
            t = binomtest(only_kd, disc, 0.5)
            diff = float(kd_ok.mean() - alone_ok.mean())
            se = _math.sqrt(disc) / len(truth)
            rank_result["kd_effect"] = {
                "test": "McNemar exact, student+KD vs student alone, shared test split",
                "accuracy_difference_pp": round(diff * 100, 4),
                "ci95_pp": [round((diff - 1.96 * se) * 100, 4),
                            round((diff + 1.96 * se) * 100, 4)],
                "macro_f1_difference": round(
                    rank_result["branches"]["student_kd"]["metrics"]["macro_f1"]
                    - rank_result["branches"]["student_alone"]["metrics"]["macro_f1"], 6),
                "corrected_by_kd": only_kd,
                "broken_by_kd": only_alone,
                "p_value": float(t.pvalue),
                "significant_at_0.05": bool(t.pvalue < 0.05),
            }
            verdict = ("KD helps" if diff > 0 else "KD hurts") if t.pvalue < 0.05 \
                else "no significant KD effect"
            rank_result["kd_effect"]["verdict"] = "%s (p=%.3f)" % (verdict, t.pvalue)

        for branch, data in rank_result["branches"].items():
            m = data["metrics"]
            print("[distill] %-6s %-14s acc=%.4f macroF1=%.4f"
                  % (rank, branch, m["accuracy"], m["macro_f1"]))
        if "kd_effect" in rank_result:
            print("[distill] %-6s KD effect: %s" % (rank, rank_result["kd_effect"]["verdict"]))

        results["ranks"][rank] = rank_result

    cfgutil.save_json(results, _metrics_dir(cfg) / "distillation_metrics.json")

    from visualization.distill_figures import generate_all
    from visualization import style as S
    S.apply_style()
    generate_all(cfg, results)
    return results


def stage_distill_ablation(cfg, force=False):
    """Stage 4 -- alpha and temperature ablations, as the paper reports."""
    import numpy as np
    import pyarrow.parquet as pq
    from collections import Counter

    from analysis.dereplicate import variants_path
    from embeddings.store import load_embeddings
    from models.classifier import classification_metrics, stratified_split
    from taxonomy.assign import load_labels
    from training.distill import DistillationExperiment, build_student_features

    dcfg = cfg["distillation"]
    ccfg = cfg["classification"]
    root = cfgutil.output_dir(cfg, "embeddings")
    seed = cfg["seed"]
    rank = dcfg["ablation_rank"]

    labels_map = load_labels(cfg, rank)
    counts_table = pq.read_table(variants_path(cfg), columns=["variant_id", "count"])
    count_of = dict(zip(counts_table.column("variant_id").to_pylist(),
                        counts_table.column("count").to_pylist()))

    t_ids, teacher_x, _ = load_embeddings(root, ccfg["embedding_stores"]["foundation_model"])
    _, kmer_x, _ = load_embeddings(root, ccfg["embedding_stores"]["baseline"])
    index_of = {sid: i for i, sid in enumerate(t_ids)}
    pairs = [(index_of[k], v) for k, v in labels_map.items() if k in index_of]
    counter = Counter(v for _, v in pairs)
    usable = {c for c, n in counter.items() if n >= ccfg.get("min_class_size", 30)}
    pairs = [(i, v) for i, v in pairs if v in usable]
    classes = sorted({v for _, v in pairs})
    cls_index = {c: i for i, c in enumerate(classes)}
    rows = np.array([i for i, _ in pairs])
    y = np.array([cls_index[v] for _, v in pairs])
    reads = np.array([count_of.get(int(t_ids[i].split("_")[1]), 1) for i in rows])
    xt = teacher_x[rows]
    xs = build_student_features(kmer_x[rows], reads, dcfg["include_abundance"])
    train, val, test = stratified_split(y, tuple(ccfg["split"]), seed)

    out = {"stage": "distill_ablation", "rank": rank, "n_labelled": len(pairs),
           "n_classes": len(classes), "alpha_sweep": [], "temperature_sweep": []}

    def evaluate(alpha, temperature):
        experiment = DistillationExperiment(
            xt.shape[1], xs.shape[1], len(classes), ccfg["hidden_dim"], seed)
        run = experiment.run(xt, xs, y, train, val, test, alpha=alpha,
                             temperature=temperature, epochs=dcfg["epochs"],
                             lr=ccfg["lr"], weight_decay=ccfg["weight_decay"],
                             batch_size=ccfg["batch_size"],
                             teacher_head=dcfg["teacher_head"])
        m = classification_metrics(
            run["y_test"], run["test_probabilities"]["student_kd"], classes)
        base = classification_metrics(
            run["y_test"], run["test_probabilities"]["student_alone"], classes)
        return m, base

    for alpha in dcfg["alpha_grid"]:
        m, base = evaluate(alpha, dcfg["temperature"])
        out["alpha_sweep"].append({
            "alpha": alpha, "temperature": dcfg["temperature"],
            "student_kd_accuracy": m["accuracy"], "student_kd_macro_f1": m["macro_f1"],
            "student_alone_accuracy": base["accuracy"],
            "student_alone_macro_f1": base["macro_f1"],
        })
        print("[ablation] alpha=%.2f  acc=%.4f macroF1=%.4f (alone %.4f/%.4f)"
              % (alpha, m["accuracy"], m["macro_f1"], base["accuracy"], base["macro_f1"]))

    for temperature in dcfg["temperature_grid"]:
        m, base = evaluate(dcfg["alpha"], temperature)
        out["temperature_sweep"].append({
            "alpha": dcfg["alpha"], "temperature": temperature,
            "student_kd_accuracy": m["accuracy"], "student_kd_macro_f1": m["macro_f1"],
            "student_alone_accuracy": base["accuracy"],
            "student_alone_macro_f1": base["macro_f1"],
        })
        print("[ablation] T=%.1f     acc=%.4f macroF1=%.4f"
              % (temperature, m["accuracy"], m["macro_f1"]))

    cfgutil.save_json(out, _metrics_dir(cfg) / "distillation_ablation.json")

    from visualization.distill_figures import figure_ablation
    from visualization import style as S
    S.apply_style()
    figure_ablation(cfg, out)
    return out


def stage_labels(cfg, force=False):
    from analysis.labels import verify

    result = verify(cfg)
    cfgutil.save_json(result, _metrics_dir(cfg) / "label_verification.json")
    print(f"[labels] labels_available = {result['labels_available']}")
    if not result["labels_available"]:
        print("[labels] Supervised classification will NOT run. "
              "No labels will be fabricated.")
    return result


def stage_analyze(cfg, force=False):
    from analysis.embedding_analysis import analyse_representation

    cfg["analysis"]["visualization_sample_size"] = cfg["subset"]["visualization_size"]
    analyses = {
        "foundation_model": analyse_representation(cfg, "main_genomeocean", "GenomeOcean-500M"),
        "baseline": analyse_representation(cfg, "main_kmer4", "canonical 4-mer / TNF"),
    }
    cfgutil.save_json(analyses, _metrics_dir(cfg) / "embedding_metrics.json")

    from visualization.embedding_figures import (
        figure_cluster_tendency,
        figure_pca_explained_variance,
        figure_projection,
    )
    from visualization import style as S

    S.apply_style()
    figure_pca_explained_variance(cfg, analyses)
    figure_projection(cfg, "pca", analyses, "main", "6")
    figure_projection(cfg, "umap", analyses, "main", "7")
    figure_cluster_tendency(cfg, analyses)
    return analyses


def stage_compare(cfg, force=False):
    import csv

    from evaluation.comparison import compare, comparison_table

    # Retrieval uses the strand-corrected pair subset (mate 2 reverse-complemented).
    # That is the methodologically correct handling of paired-end reads, and it is
    # applied to BOTH representations -- it is provably a no-op for the canonical
    # k-mer baseline, so the comparison stays fair. See stage_strandtest.
    result = compare(cfg, "main_genomeocean", "main_kmer4",
                     "pairs_rc_genomeocean", "pairs_rc_kmer4")
    strand = _read(cfg, "strand_orientation_metrics.json")
    if strand:
        result["strand_orientation"] = strand
    cfgutil.save_json(result, _metrics_dir(cfg) / "comparison.json")

    rows = comparison_table(result)
    csv_path = _metrics_dir(cfg) / "comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[compare] Wrote comparison.json and comparison.csv")

    from visualization.embedding_figures import figure_model_comparison
    from visualization import style as S

    S.apply_style()
    figure_model_comparison(cfg, result)
    return result


def stage_performance(cfg, force=False):
    from evaluation.performance import batch_size_sweep, full_dataset_projection
    from models.foundation_encoder import build_encoder
    from preprocessing.run import load_subset, subset_path

    _, sequences = load_subset(subset_path(cfg, "main"))
    sequences = sequences[: cfg["performance"]["sweep_reads"]]
    encoder = build_encoder(cfg)
    try:
        perf = batch_size_sweep(
            encoder, sequences, cfg["performance"]["batch_size_sweep"]
        )
    finally:
        encoder.close()

    dataset = _read(cfg, "dataset_metrics.json")
    projection = full_dataset_projection(
        perf["best_throughput_seq_per_second"],
        perf["encoder"]["embedding_dim"],
        dataset["reads_passing_qc"],
        "GenomeOcean-500M over all QC-passed reads",
    )
    perf["full_dataset_projection"] = projection
    cfgutil.save_json(perf, _metrics_dir(cfg) / "performance_metrics.json")

    from visualization.embedding_figures import figure_computational_performance
    from visualization import style as S

    S.apply_style()
    figure_computational_performance(cfg, perf, projection)
    return perf


def stage_architecture(cfg, force=False):
    from visualization.architecture import draw_architecture

    return draw_architecture(cfg)


def stage_report(cfg, force=False):
    from reporting.experiment_report import build_report

    return build_report(cfg)


def stage_taxonomy_report(cfg, force=False):
    """Stage 3 report."""
    from reporting.taxonomy_report import build_report

    return build_report(cfg)


def stage_distill_report(cfg, force=False):
    """Stage 4 report."""
    from reporting.distill_report import build_report

    return build_report(cfg)


def stage_community_report(cfg, force=False):
    """Stage 2 report. Separate document; the stage 1 report stays intact."""
    from reporting.community_report import build_report

    return build_report(cfg)


STAGES = {
    "preprocess": stage_preprocess,
    "figures": stage_figures,
    "subsets": stage_subsets,
    "testrun": stage_testrun,
    "encode": stage_encode,
    "baseline": stage_baseline,
    "strandtest": stage_strandtest,
    "dereplicate": stage_dereplicate,
    "community": stage_community,
    "amplicon": stage_amplicon,
    "varregion": stage_varregion,
    "labels": stage_labels,
    "assign": stage_assign,
    "train": stage_train,
    "distill": stage_distill,
    "distill-ablation": stage_distill_ablation,
    "analyze": stage_analyze,
    "compare": stage_compare,
    "performance": stage_performance,
    "architecture": stage_architecture,
    "report": stage_report,
    "community-report": stage_community_report,
    "taxonomy-report": stage_taxonomy_report,
    "distill-report": stage_distill_report,
}

ALL_ORDER = [
    "preprocess", "figures", "subsets", "testrun", "encode", "baseline",
    "strandtest", "labels", "analyze", "compare", "performance",
    "dereplicate", "community", "amplicon", "varregion", "assign", "train", "distill", "distill-ablation", "architecture", "report", "community-report", "taxonomy-report", "distill-report",
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stage", choices=list(STAGES) + ["all"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    overrides = dict(cfgutil.parse_override(text) for text in args.overrides)
    cfg = cfgutil.load_config(args.config, overrides)
    set_seed(cfg["seed"])

    print(f"[config] {cfg['_config_path']}  seed={cfg['seed']}")
    env = environment_report()
    print(f"[env] python {env['python']} | torch {env.get('torch')} | "
          f"cuda={env.get('cuda_available')} | {env.get('gpu', 'CPU only')}")

    stages = ALL_ORDER if args.stage == "all" else [args.stage]
    for name in stages:
        print(f"\n{'=' * 70}\n[stage] {name}\n{'=' * 70}")
        STAGES[name](cfg, force=args.force)
    print("\n[done]")


if __name__ == "__main__":
    main()
