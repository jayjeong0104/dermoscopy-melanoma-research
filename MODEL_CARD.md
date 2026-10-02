# Melanoma EfficientNetV2B1 Research Baseline

## Identity and Intended Use

- Model ID: `melanoma-effnetv2b1-v0.1.0-research`.
- Original run: `melanoma_v1/run_20260822_111527`.
- Checkpoint SHA-256: `62c7603e41ee4d30f43f6b4cb2c129cedb3fe16efd8cd1a9fae9fef967538171`.
- TensorFlow 2.21.0; Keras 3.15.1 serialization; EfficientNetV2B1; 300 x 300 RGB input.
- Research on dermoscopic lesion images, not clinical decision-making.
- Target: melanoma versus all other labeled lesions, including BCC and SCC.

Ordinary phone photographs, body overview images, multiple lesions per image,
and unvalidated devices/workflows are unsupported. A low score must not be used
as reassurance or as a reason to delay medical assessment.

## Data and Training

The retained dataset contains 32,216 images representing 20,882 lesions from
HAM10000, BCN20000, and DERM12345. Training used 19,328 images and 12,527 lesions,
including only 874 distinct melanoma lesions. Validation contains 4,177 lesions;
the historical held-out split contains 4,178 lesions.

Splits group lesions, and patient identifiers where available, and check exact
SHA-256 overlap. Patient separation is only as complete as available identities.
Multiple photographs are not independent positive cases. Dataset providers'
terms and labels are not replaced by this release.

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
AUC. Counts: TP 270, FN 21, FP 1,016, TN 2,871. These are not the repeatedly
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

Converged feature fusion improved development ranking but failed its original
high-sensitivity gate. The full-frame follow-up also retained the baseline.
The diagnosis-aware follow-up stopped at its 2,000-iteration solver limit
before final fitting/evaluation; no performance is claimed for that unfinished
candidate. This release intentionally retains the original checkpoint.

Research distribution is not regulatory authorization, diagnostic validation,
or approval for patient-facing use. A genuinely independent, patient-separated
confirmation cohort remains necessary for further performance claims.
