# Melanoma EfficientNetV2B1 Research Baseline

## Identity and Intended Use

- Model ID: `melanoma-effnetv2b1-v0.1.0-research`.
- Original run: `melanoma_v1/run_20260822_111527`.
- Checkpoint SHA-256: `62c7603e41ee4d30f43f6b4cb2c129cedb3fe16efd8cd1a9fae9fef967538171`.
- TensorFlow 2.21.0; Keras 3.15.1 serialization; EfficientNetV2B1; 300 x 300 RGB input.
- Research on dermoscopic lesion images, not clinical decision-making.
- Target: melanoma versus all other labeled lesions, including BCC and SCC.

This card describes the archived v0.1 release asset. The current development
reference is a separate, newer run; this code update does not replace the release
asset or its inference configuration.

Ordinary phone photographs, body overview images, multiple lesions per image,
and unvalidated devices/workflows are unsupported. A low score must not be used
as reassurance or as a reason to delay medical assessment.

## Data and Training

The archived dataset contains 32,216 images representing 20,882 recorded lesion
IDs from HAM10000, BCN20000, and DERM12345. Training used 19,328 images and 12,527
recorded lesion IDs, including 874 melanoma IDs. DERM12345 image IDs serve as
lesion proxies, so these counts do not establish distinct biological lesions.
Validation contains 4,177 recorded lesion IDs;
the historical held-out split contains 4,178 lesions.

Splits group lesions, and patient identifiers where available, and check exact
SHA-256 overlap. Patient separation is only as complete as available identities.
Multiple photographs are not independent positive cases. Dataset providers'
terms and labels are not replaced by this release.

**Confirmed ancestry exposure:** the October review found that an ancestor
checkpoint trained on 3,053 images assigned to this run's validation split and
3,138 assigned to its test split. The descendant's grouped splits therefore do
not establish independent holdouts. The historical metrics below must not be
used as unbiased estimates of generalization. See
[the provenance review](PIPELINE_REVIEW_20261005.md).

The retained run used focal loss (gamma 1.5), 45% positive sampling, label smoothing
0.02, and auxiliary diagnosis supervision. Its image-only primary head was
warm-started through a prior locally trained encoder lineage. No claim of
bit-for-bit retraining from the released source alone is made.

## Inference and Evaluation

The checkpoint expects the archived preprocessing implemented by
`train_multitask.resize_and_normalize_image`: TensorFlow antialiased resize to
300 pixels and division by 255. Preserve the model's own internal rescaling.
Do not substitute a pretrained backbone's generic preprocessing function.

Unknown metadata is supplied consistently to the auxiliary branch. Each image
uses eight archived `legacy_v1` views, then predictions are averaged. Same-lesion
direct scores are combined by averaging logits; diagnosis probabilities are
averaged. The lesion score is 0.35 direct-head plus 0.65 MEL diagnosis probability,
followed by temperature 0.76 and the validation-selected threshold 0.1.

Historical internal held-out results: 75.18% accuracy, 21.00% precision,
92.78% sensitivity, 73.86% specificity, 0.5799 average precision, and 0.9160 ROC
AUC. Counts: TP 270, FN 21, FP 1,016, TN 2,871. These observations inherit the
ancestry exposure described above. They are not the repeatedly
reported one-view development cross-fit results, which use a different protocol.
Aggregate metric files are included; individual records are not.
The checkpoint retains its mixed-float16 policy. CPU and GPU arithmetic may
produce small score differences; software smoke checks do not re-estimate the
historical metrics or establish bit-for-bit platform equivalence.

PH2 and DERM12345 external evaluations have already informed development and
must not be presented as untouched final confirmation. Validation has been
reused extensively. No new held-out evaluation was run merely to publish this
snapshot.

## Limitations and Current Experiments

False alarms remain common. Performance changes differ by data source, and skin
tone/device fairness is not established. There is no validated image-quality
gate, abstention system, or out-of-distribution detector. Temperature scaling
does not establish calibration in a new clinical population.

Historical feature-fusion studies used the exposed lineage and cannot establish
independent performance gains. The diagnosis-aware follow-up stopped at its
2,000-iteration solver limit before final fitting/evaluation. Its convergence
check has since been repaired, but the contaminated study was not resumed.

New training runs avoid that known local ancestry exposure. The latest revised
sampling run is the development reference: under matched group-separated
threshold calibration/scoring it detected 276 of 290 melanomas and flagged 1,056
non-melanomas, versus 274 and 1,284 for the clean reference. This is 95.17%
sensitivity, 72.83% specificity, and 20.72% precision on 4,177 reused development
lesions. DERM12345 sensitivity was only 92.06%; the aggregate target is not a
source-specific guarantee. Checkpoint selection and repeated experimentation
already used this validation cohort. These results do not isolate sampling's
causal effect or constitute an independent confirmation. The feature-head
follow-ups did not surpass this reference while meeting the overall sensitivity
target. See [the aggregate development report](reports/development_progress_20261007.json).

The v0.1 release continues to distribute its original checkpoint for historical
reproducibility, not the newer development reference.

Research distribution is not regulatory authorization, diagnostic validation,
or approval for patient-facing use. A genuinely independent, patient-separated
confirmation cohort remains necessary for further performance claims.
