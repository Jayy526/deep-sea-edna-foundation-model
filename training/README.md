# training/

**This directory is intentionally empty of training scripts.**

No training was performed in this experiment, and that is a result rather than
an omission:

- **The foundation model is frozen.** Following TaxDistill, GenomeOcean's
  backbone is used as a fixed feature extractor. All 541,109,760 parameters have
  `requires_grad=False`; the measured trainable-parameter count is **0**.
- **The baseline has no parameters.** Canonical k-mer frequencies are a
  deterministic hand-crafted transform.
- **The classifier was never trained**, because this dataset contains no
  ground-truth taxonomy (verified — see `outputs/metrics/label_verification.json`).
  Training a classifier would have required inventing labels.

The training code exists, is wired to a CLI stage, and is covered by tests:

| What | Where |
|---|---|
| CLI stage (`python main.py train`) | `main.py::stage_train` |
| Linear probe / MLP head, training loop, stratified split | `models/classifier.py` |
| Classification metrics (accuracy, macro/weighted F1, per-class, confusion matrix) | `models/classifier.py::classification_metrics` |
| Confidence, Brier score, ECE, reliability bins | `models/classifier.py::confidence_and_calibration` |
| Figures 9–11 (confusion matrix, training curves, calibration) | `visualization/classification_figures.py` |
| Tests, on synthetic data only | `tests/test_classification.py` |

Run today, `python main.py train` detects that no labels exist, reports the
unavailability, writes `classification_metrics.json` with every supervised
metric marked *"Not available with the current dataset/experimental setup."*,
and exits cleanly. It never invents labels in order to have something to train on.

**The branch is verified working.** Because it cannot be run on this dataset,
`tests/test_classification.py` exercises the entire path — split, training,
metrics against sklearn, calibration against closed-form values, and figure
rendering — on clearly-synthetic data generated inside the test. Nothing there
touches the project's outputs. That way the branch is known-good on the day real
reference labels arrive.

To activate the supervised branch, supply a verified per-read label table:

```bash
python main.py train --set classification.labels_path='"data/labels.tsv"'
```

where `labels.tsv` is two columns, `read_id<TAB>taxon`. The label source, its
version, and its own error rate should be recorded alongside it — TaxDistill's
premise is precisely that such labels are noisy.
