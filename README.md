# Dermoscopy Melanoma Research

A TensorFlow research baseline for **melanoma versus non-melanoma** in
dermoscopic images. Non-melanoma includes other skin cancers: a negative
prediction does **not** mean that a lesion is benign or that a person is healthy.

**Research prototype, not a diagnostic product.** Not validated for phone photos,
screening decisions, treatment decisions, or use without clinical assessment.
No claim of 95% overall accuracy or clinical readiness is made.

## Retained Model

`melanoma-effnetv2b1-v0.1.0-research` is the retained EfficientNetV2B1 checkpoint,
not the newest experimental candidate. It uses 300 x 300 RGB images, an image-only
binary head, and an auxiliary eight-category diagnosis head. The archived
research inference recipe uses unknown metadata, legacy eight-view augmentation,
a validation-selected output blend, temperature scaling, and threshold 0.1.

The 144,230,612-byte original checkpoint is distributed as a **release asset**,
not in Git history. `models/model_manifest.json` records its exact SHA-256 and
inference configuration. No dataset, patient-level table, or PanDerm checkpoint
is included.

## Verified Historical Results

These are archived observations, not a fresh independent confirmation:

| Evaluation protocol | Lesions | Accuracy | Precision | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical internal held-out split, eight-view calibrated recipe | 4,178 | 75.18% | 21.00% | 92.78% | 73.86% |
| Reused development reference, one-view fixed-blend group-cross-fitted thresholds | 4,177 | 73.95% | 20.40% | 94.83% | 72.40% |

Do not compare the two rows as an improvement or regression: their inference
and threshold protocols differ. The inference command below implements the
first recipe, not the cross-fitted development experiment. At the first recipe's
operating point, 270 melanomas were detected, 21 missed, and 1,016 non-melanomas
flagged. Average precision was 0.5799 and ROC AUC 0.9160.

See [MODEL_CARD.md](MODEL_CARD.md) and [reports/research_status.json](reports/research_status.json)
for limitations, provenance, and the newer unsuccessful experiments.

## Run Research Inference

Use Python 3.12. CPU inference works; NVIDIA GPU use with current TensorFlow on
Windows uses WSL2. The dependency file records the tested CPU environment.
Keras 3.15.1 is pinned to the checkpoint's serialization version; TensorFlow's
minimum Keras dependency alone is not sufficient to load this artifact.

```bash
python -m venv .venv
# Activate the environment using your operating system's usual command.
python -m pip install -r requirements.txt
```

Download `melanoma-effnetv2b1-v0.1.0-research.zip` from the repository's
`v0.1.0-research` release, and extract it into `models/`. Then run:

```bash
python predict_research.py --images /path/to/dermoscopic-lesion.jpg --acknowledge-research-use
```

Multiple `--images` arguments must depict **the same lesion**; they are aggregated
into one research score. Only JPEG input is supported by this reproducibility
entry point. It performs no validated image-quality or out-of-distribution check.
It does not upload images or contact a server. Output is a research flag, not a
diagnosis or a calibrated estimate of an individual's cancer risk.

The command refuses a checkpoint whose SHA-256 differs from the retained model.
It deliberately preserves the archived `legacy_v1` transform sequence, including
its duplicate orientation, instead of silently changing evaluated preprocessing.

## Training and Tests

The original training, data-preparation, evaluation, and experiment source is
included for inspection. Obtain data separately from its original providers
under their terms; see [DATA_SOURCES.md](DATA_SOURCES.md). Configure your own
manifest and grouped split files. Example:

```bash
python train_multitask.py --manifest data/combined_manifest.csv --binary-target melanoma --backbone efficientnetv2b1 --image-size 300 --output-root artifacts/research_run
python -m unittest discover -p 'test_*.py' -q
```

Publication verification: all 151 unit tests passed with the pinned TensorFlow
and Keras versions. The actual retained checkpoint also passed a local CPU
inference smoke check. These checks establish packaging functionality, not
new medical-performance evidence; see `reports/software_verification.json`.

This example starts a new experiment; it is **not** a claim of reproducing the
retained checkpoint. That run was warm-started from an earlier local checkpoint.
Full historical reconstruction needs those artifacts and the original split
assignments, neither of which is included. Recovery scripts require their
explicitly named local run artifacts and are not turnkey downloads.
The shell launchers preserve the original local experiment layout; inspect and
adapt their paths before running them. Optional GPU dependencies are separate
in `requirements-gpu.txt`; release packaging is verified on CPU.

The diagnosis-aware experimental continuation stopped at an optimizer iteration
limit before final validation. Do not treat it as a validated replacement.

## Access and Licensing

This is a public research snapshot. An original-source-code license has not yet
been selected by the owner; see [LICENSE_STATUS.md](LICENSE_STATUS.md).
Public visibility alone is not an open-source license grant. Code, model weights,
datasets, and third-party dependencies have distinct provenance and terms.

No images or patient-level predictions should be submitted in issues. Report
software problems with synthetic examples and aggregate results instead.
