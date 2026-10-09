# Why accuracy and precision remain limited

Publication note: aggregate results are available in [reports/development_progress_20261007.json](reports/development_progress_20261007.json). Artifact paths below identify local-only audit evidence; datasets, individual predictions, training outputs and checkpoints are excluded from this Git snapshot.

This review follows the completed clean ImageNet run. The central problems are limited independent melanoma diversity, a training distribution that overexposes rare diagnoses and underexposes ordinary moles, strong source differences, and an evaluation process that has repeatedly reused its development cohorts. The implementation also contained several reproducible bugs. Fixing those bugs makes the experiments more trustworthy; it does not by itself establish a large accuracy gain.

The working objective is melanoma sensitivity of at least 95% while reducing false positives, with source-specific results reported. That is a research objective, not a medical-use acceptance standard. Current data support dermoscopy; ordinary smartphone photographs have not been established as a supported input. The target remains melanoma versus non-melanoma, which does not rule out other cancers.

## The completed clean reference

The ImageNet-only run finished on October 6 at 05:44 UTC (01:44 Eastern), after four classifier-head epochs and eleven fine-tuning epochs. Fine-tuning epoch six supplied the selected checkpoint. Its validation callback reported 95.17% sensitivity, 66.20% specificity, 17.36% precision, 68.21% accuracy and average precision 0.4944: 276 detected melanomas, 14 missed melanomas, 1,314 false positives and 2,573 true negatives.

These are results on 4,177 repeatedly inspected development lesions, with checkpoint, blend and threshold selected on that cohort. They are not an independent test result. Group-disjoint calibration and the feature-head comparison below use a different, explicitly recorded evaluation protocol; their numbers must not be substituted into the old report as if the protocol were unchanged.

Evidence: completed run (`artifacts/clean_melanoma_20261005/complete.json`; local-only), training history (`artifacts/clean_melanoma_20261005/training/run_20261005_210209/training_finetune.csv`; local-only).

## Findings that explain the performance ceiling

### 1. Sampling spends too little effort on the main source of false alarms

Nevus lesions comprise 70.08% of training lesions, but receive only 30.31% of training draws. Within each source/diagnosis stratum, lesions are sampled uniformly; across strata, the allocation still uses the square root of the number of photographs. Thus repeated photographs and rare-diagnosis balancing continue to distort exposure even when `sampling_unit=lesion`.

At the current full-epoch budget, expected visits range from 0.46 per DERM nevus to 11.84 per BCN vascular lesion. This is a credible mechanism for inefficient learning, not proof that changing it will improve held-out performance.

| Exposure | Clean reference | Natural lesion recipe |
| --- | ---: | ---: |
| Melanoma draws | 25.00% | 25.00% |
| Nevus draws | 30.31% | 56.50% |
| Dermatofibroma + vascular draws | 9.13% | 2.35% |
| Effective sampled lesions, probability-based calculation | 5,394 | 8,349 |

**Action:** added a backward-compatible `--sampling-mass-unit` option. The controlled new recipe uses lesion mass with exponent one and preserves 25% melanoma exposure. Negative lesions then receive equal sampling probability. The original default remains reproducible. Sampling audit (`artifacts/precision_rebuild_20261006/sampling_audit.json`; local-only).

### 2. Image count substantially overstates independent positive-case diversity

The training split contains 19,328 photographs and 12,527 lesions, including only 874 distinct melanoma lesions. DERM's 191 melanoma training images represent 43 patients; its largest contributing patient supplies 22 images. HAM and BCN patient identifiers are unavailable, so lesion-disjoint splits cannot establish complete patient independence.

Adding the local ISIC 2019 archive wholesale would mainly add overlap. Of 25,331 image entries, 10,013 have HAM filename matches. Among 12,412 apparently new IDs whose file sizes match BCN images, a deterministic 100-entry SHA-256 sample matched existing BCN photographs 99 times. This sample is conditional on the size-match screen; it is not a 99% estimate for the entire archive.

**Action:** retained the existing cleaned images and labels; did not count renamed copies as new cases. Future data acquisition should prioritize independent, patient-linked melanomas and diagnostically difficult benign lesions from the intended devices and population. Data review (`artifacts/precision_rebuild_20261006/data_review.json`; local-only).

### 3. Source-specific appearance and case mix remain major problems

The new error analysis reports false alarms by source and diagnosis rather than hiding them inside overall accuracy. BCN photographs are 1,024-pixel squares, HAM photographs are 600×450, and DERM spans wider dimensions/aspect ratios. A fixed 90-image training sample found average near-black pixel fractions of 11.10% in BCN, 0.71% in DERM and 0.17% in HAM. Some BCN images devote over half their pixels to dark background.

With the saved 5%/95% blend and grouped calibration on the newly extracted validation outputs, 843 of 1,285 false alarms are nevi and another 206 are benign keratoses. Sensitivity/specificity are 95.19%/31.91% for BCN, 96.83%/79.21% for DERM, and 92.68%/65.50% for HAM. The aggregate 94.48% sensitivity also illustrates that a 95% calibration target does not guarantee 95% sensitivity on the separately scored groups. Checkpoint diagnostics (`artifacts/precision_rebuild_20261006/clean_checkpoint_diagnostics.json`; local-only).

These observations support investigating acquisition differences and wasted input area. They do not prove that the model uses a particular shortcut or that automatic cropping would help. Aggressive crops can remove lesion context. The earlier crop studies also inherited the contaminated encoder and cannot settle this question. Research has demonstrated shortcut effects in skin-lesion classifiers, but that is background evidence, not a causal diagnosis of this checkpoint. [Primary study](https://arxiv.org/abs/1904.08818).

**Action:** no automatic cropping, image deletion or relabeling was applied. Existing perceptual duplicate candidates were screened more carefully: none was demonstrated to be an exact duplicate, and the closest inspected pair contained different lesions with similar dark borders. Expert adjudication remains necessary before changing uncertain labels.

### 4. Historical achievements need a different interpretation

The older retained model's ancestors trained on 3,053 of its later validation photographs and 3,138 of its later test photographs. Resetting the binary head did not remove information from its backbone. Frozen-feature, fusion and continuation studies using that encoder inherited the exposure. Their measured changes remain historical development observations, not unbiased evidence of improvement.

The new clean reference excludes those local ancestor weights. Reusing validation for checkpoint selection still prevents independent confirmation. Cross-fitting calibration separates threshold fitting from scored groups, but cannot undo earlier architecture/checkpoint selection. The distinction is consistent with [scikit-learn's model-selection guidance](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

**Action:** preserved historical artifacts while withdrawing independent-performance interpretations. No test/HIBA inference was used in this investigation. [Previous ancestry evidence](PIPELINE_REVIEW_20261005.md).

### 5. The objective and selection policy are not fully aligned

The clean reference trains binary and auxiliary diagnosis heads, but the selected score gives only 5% weight to the direct binary head and 95% to diagnosis-derived melanoma probability. This shows that multiclass supervision contributes useful signal. It does not establish that deleting either branch or changing the auxiliary loss weight will help.

Average precision summarizes ranking across thresholds. Maximizing it does not necessarily minimize false positives at 95% sensitivity, especially within each source. The historical fresh-process finalizer also reselects checkpoints by blended AP even when training used an operating-point objective, then retunes blend, temperature and threshold on the same validation data.

**Action:** added a separate development evaluator with explicit, group-disjoint blend/threshold fitting and source/diagnosis breakdowns. The new training driver disables automatic test finalization. The bounded feature experiment tests natural lesion-weighted binary heads before another architectural redesign. Its comparison reports every declared arm rather than silently selecting a validation winner.

## Reproduced code defects and repairs

| Defect | Consequence | Repair |
| --- | --- | --- |
| Threshold candidates restricted to 0.001–0.999 | Could miss a feasible recall target or select unnecessary false positives, particularly for small calibrated scores | Exact empirical thresholds; historical grid explicitly available |
| Resume guard omitted row-level labels/supervision/metadata | A generic resume could silently change training semantics | Per-image identity, label and metadata comparison |
| Backup occurred before learning-rate reduction | Interrupted run could restore the old learning rate | Backup moved last; scheduler/early-stop/checkpoint state restored |
| CSV history could be ahead of the actual weight backup | Resume could skip an epoch's unsaved training | Backup epoch takes precedence; discarded history is preserved for audit |
| Masked `UNKNOWN` diagnoses were still scored as dummy AK labels | Auxiliary diagnosis reports counted invented labels | Carry supervision availability through aggregation and exclude masked rows from diagnosis metrics |
| Corrected high-risk DERM manifest could fail melanoma conversion | Valid non-melanoma malignancies were rejected before their explicit melanoma label was used | Correct raw-taxonomy/explicit-label precedence and roundtrip tests |
| External evaluator used a generic older training manifest | DERM overlap audit could omit the selected model's real exposure | Audit selected run plus recursive ancestor train/validation manifests |
| External checkpoint override reused another model's calibration | Reported scores could use an incompatible inference recipe | Bind checkpoint and available inference-artifact hashes; reject unsupported overrides |
| Readiness report passed independence/licensing gates without adequate evidence | A model card could overstate what had actually been established | Missing/unknown evidence stays unverified; unsupported patient/independence claims removed |
| Smoke test used fewer microbatches than its accumulation interval | A reported training test could finish without an optimizer update | Execute a complete accumulation interval; assert optimizer/weight update and save/reload agreement |

The threshold-grid defect was not the cause of the clean reference's logged lesion metrics: its callback already used an exact operating threshold. The unavailable auxiliary labels affect only a small fraction of draws. These repairs should not be marketed as explanations for the entire precision deficit.

Gradient tests verified that fine-tuning updates convolutional weights while frozen BatchNorm statistics remain fixed, and that accumulation over four physical batches produces an update on the fourth microbatch. Image normalization is consistent with the EfficientNetV2 input contract. Keeping pretrained BatchNorm in inference mode is supported by [TensorFlow's transfer-learning guide](https://www.tensorflow.org/guide/keras/transfer_learning). The saturation augmentation is centered around Keras's neutral value 0.5; `(0.4, 0.6)` is not a constant 40–60% saturation multiplier. [Keras documentation](https://keras.io/api/layers/preprocessing_layers/image_augmentation/random_saturation/).

## Experiments and restructuring

The feature experiment extracts the clean checkpoint's pooled image features and both heads on training and reused validation only. It verifies input image SHA-256 values, uses the saved mixed-precision policy, checks prediction equivalence with the original model, and publishes resumable, integrity-checked chunks. A preliminary extraction with a different global precision policy was stopped and retained separately; it is excluded from the final study.

Original-model and extractor predictions agree exactly on the four-image equivalence check. Re-extracted full-cohort, same-data blend metrics differ slightly from the saved epoch log: AP 0.49388 versus 0.49438 and 1,316 versus 1,314 false positives at the same 276/290 detected melanomas. The exact numerical cause has not been isolated; this is not a bit-identical reproduction. All new candidate comparisons use the same re-extracted reference and calibration protocol, and such small differences are not counted as improvement. Extraction and resumes use batch size four.

Two predeclared classifier arms use clean B1 features alone or clean B1 plus frozen PanDerm full-frame features. Regularization is chosen within training using separate fitting, calibration and scoring groups. Both final optimization starts must converge and agree. The B1 encoder has already seen all training groups, so these inner head scores are conditional and optimistic; the reused validation remains development evidence. Private PanDerm pretraining overlap is not fully auditable.

**Completed result: neither replacement head improved precision.** All 25,771 training/validation image hashes were verified during extraction. Both heads selected L2=1 using only the declared training criterion; neither training selection met the 95% sensitivity floor in every source. Final optimizer starts agreed within 1.21e-7 probability.

| Same reused-development protocol | Precision | Sensitivity | Specificity | Accuracy | AP | False positives / missed melanomas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Re-extracted clean reference, saved blend | 17.58% | 94.48% | 66.94% | 68.85% | 0.4939 | 1,285 / 16 |
| Natural lesion-weighted B1 linear head | 15.73% | 95.52% | 61.82% | 64.16% | 0.3926 | 1,484 / 13 |
| Natural lesion-weighted B1 + PanDerm head | 17.50% | 95.17% | 66.53% | 68.52% | 0.4273 | 1,301 / 14 |

Both alternatives detected slightly more melanomas but increased false alarms and reduced ranking quality. The combined head's source sensitivity was 100% BCN, 90.48% DERM and 93.50% HAM. Its conditional paired 95% interval for specificity change spans -2.61 to +1.61 percentage points; this provides no convincing evidence of a gain and is not an independent significance test. Both replacement heads are rejected. No checkpoint was promoted. The negative result supports retaining the nonlinear/multitask reference while testing a change to image training, rather than assuming that a simpler head or feature fusion solves the problem.

Evidence: fixed plan (`artifacts/precision_rebuild_20261006/head_study/experiment_plan.json`; local-only), training-only selection (`artifacts/precision_rebuild_20261006/head_study/selection.json`; local-only), completed comparison (`artifacts/precision_rebuild_20261006/head_study/development_report.json`; local-only), integrity marker (`artifacts/precision_rebuild_20261006/head_study/complete.json`; local-only).

The follow-on image-training experiment changes sampling distribution only, preserving the clean reference's images, folds, ImageNet initialization, architecture, losses, seed, learning rates, 4+12 maximum epochs and 25% positive sampling. It is run through the existing driver with `bash launch_clean_training.sh natural_lesion`, with an isolated output directory and frozen inputs/code/runtime. This is a single-seed controlled experiment, not proof of a general improvement.

**Training started October 6 at 21:40 UTC (17:40 Eastern), and entered head epoch 1/4.** It can then train up to 12 fine-tuning epochs, subject to early stopping. The previous full run took about 4 hours 47 minutes; roughly five hours is a planning estimate, not a guaranteed finish time. This run has no completed performance result yet. Its result must be compared under a matched validation protocol and source breakdown before any replacement decision. Status (`artifacts/natural_lesion_20261006/status.json`; local-only), training log (`artifacts/natural_lesion_20261006/training.log`; local-only), frozen plan (`artifacts/natural_lesion_20261006/experiment_plan.json`; local-only).

The useful restructuring is separation of responsibilities: training and recovery state, immutable provenance, inference output extraction, group-disjoint calibration/scoring, and evidence-bound reporting. New modules implement those boundaries; the existing training driver supports named recipes. A framework rewrite or indiscriminate increase in architecture size is not supported by the current evidence.

## Evidence required before a medical-use claim

Some achievements must be surrendered: contaminated validation/test headlines, treating repeated photos as independent cases, and the assumption that a high overall accuracy or a tuned threshold establishes clinical usefulness. The current work does not justify sacrificing melanoma sensitivity simply to inflate accuracy or precision.

A credible next confirmation requires a locked model and operating policy evaluated on newly obtained, representative, patient-separated data with adequate source/subgroup coverage. The FDA's guiding material emphasizes independent datasets, representative intended populations, and clinically relevant testing conditions; it does not supply a universal accuracy percentage that makes this model ready. [FDA guiding principles](https://www.fda.gov/media/160125/download), [FDA on repeated test-data reuse](https://www.fda.gov/medical-devices/medical-device-regulatory-science-research-programs-conducted-osel/performance-evaluation-methods-evolving-artificial-intelligence-ai-enabled-medical-devices).

## Verification and review coverage

The final broad regression run passed 243 tests. Six focused head-study checks and a duplicate-lock test passed after the final execution-lock addition. Sandbox temporary-directory rename restrictions were resolved by placing test temporary files inside the workspace; this was not a model or cache-read defect. Final regression log (`artifacts/precision_rebuild_20261006/final_regression_tests.log`; local-only).

All 61 root project Python implementation files and 23 root launcher/dependency files were read in full. Consolidated coverage (`artifacts/precision_rebuild_20261006/review_coverage.json`; local-only) records hashes, line counts and review ownership. Detailed findings are in data review (`artifacts/precision_rebuild_20261006/data_review.json`; local-only), methodology review (`artifacts/precision_rebuild_20261006/methodology_review.json`; local-only), TensorFlow/Python review (`artifacts/precision_rebuild_20261006/tensorflow_line_review.json`; local-only), and ancillary reporting review (`artifacts/precision_rebuild_20261006/ancillary_line_review.json`; local-only). Coverage excludes TensorFlow/Keras internals, immutable third-party pretrained code and historical snapshots. Tests were executed; no claim is made that every test line was separately audited. Source inspection cannot establish that every medical label is correct; that requires expert review.

Legacy launchers still encode historical workflows, including rebuilding shared inputs and automatically evaluating previously inspected external cohorts. They are not the active route for these experiments. The named clean driver and explicit development evaluator are the supported investigation path. Dependency manifests omit some transitive versions; the new experiment plans therefore record the actual runtime instead of claiming the requirements file alone reproduces it.
