# v0.1.0-research

Initial research snapshot of the retained melanoma-versus-non-melanoma model.
This is a versioned publication, not a newly trained or clinically approved model.

## Included

- Original EfficientNetV2B1 checkpoint as a separate release asset, with SHA-256
  verification and the archived eight-view inference configuration.
- Training, preprocessing, experiment, and evaluation source; unit tests; a local
  JPEG inference command requiring explicit acknowledgement of research use.
- Model card, dataset provenance, aggregate reports, and limitations.

## Performance and Limitations

Archived internal held-out results: 75.18% accuracy, 21.00% precision, 92.78%
sensitivity, and 73.86% specificity. These numbers describe one documented
operating point, not phone-camera screening or an independent clinical study.
Non-melanoma includes other skin cancers. A negative flag is not reassurance.

Later experiments have not replaced this checkpoint. The full-frame comparison
retained the baseline under its joint acceptance checks; the diagnosis-aware
follow-up stopped at its solver iteration limit before final validation.

No datasets, patient-level records, personal paths, credentials, or PanDerm
checkpoint are included. Checkpoint weights are not stored in Git history.
Source-code and weight licensing remain distinct and unresolved. The owner has
approved public visibility; public access alone is not an open-source license.

## Asset

`melanoma-effnetv2b1-v0.1.0-research.zip`: extract into `models/` after installing
`requirements.txt`. Check the archive against the separately attached
`SHA256SUMS.txt`; the inference command also checks the original checkpoint hash.
See the README for the exact command and the model card for the evaluated recipe.
