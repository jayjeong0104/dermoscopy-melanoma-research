# TensorFlow melanoma pipeline review

Publication note: aggregate results are available in [reports/development_progress_20261007.json](reports/development_progress_20261007.json). Artifact paths below identify local-only audit evidence; datasets, individual predictions, training outputs and checkpoints are excluded from this Git snapshot.

**October 6 update:** the clean ImageNet run has finished. See the [follow-up review and corrective experiments](PIPELINE_REVIEW_20261006.md) for the completed result, deeper sampling/data audit, and subsequent repairs. Statements below describing training as pending reflect the October 5 review and are superseded by that update.

The largest confirmed flaw is **training leakage through checkpoint ancestry**. The retained melanoma model inherited an encoder trained on thousands of its later validation and test images. Its current split files are disjoint, but its learned weights are not independent of those holdouts. Resetting its binary classifier did not remove that knowledge. Historical internal scores and subsequent experiments using this encoder cannot establish unbiased generalization or a significant accuracy improvement.

This review covers the TensorFlow trainer and finalizer, preparation and split code, inherited checkpoints, sampling, calibration and selection, frozen feature studies, and saved external results. Review date: October 5, 2026, America/New_York. The prediction target is melanoma versus nonmelanoma; a negative prediction does not rule out another skin cancer.

## Critical findings

### The retained checkpoint inherited its holdouts

The immediate parent, `artifacts/multitask_v2/run_20260817_182119`, and two earlier ancestors used different splits from `artifacts/melanoma_v1/run_20260822_111527`. The child's `warm_start.json` confirms that the backbone, image projection, shared dense layer, and diagnosis head were copied.

| Exposure in the immediate parent | Current validation | Current test |
| --- | ---: | ---: |
| Total photographs | 6,443 | 6,445 |
| Photographs used in ancestor training | 3,053 (47.4%) | 3,138 (48.7%) |
| Lesions used in ancestor training | 1,540 of 4,177 | 1,537 of 4,178 |
| Melanoma lesions used in ancestor training | 156 of 290 | 150 of 291 |
| Additional photographs used in ancestor validation | 707 | 674 |

Ancestor validation also influenced checkpoint selection. Three ancestors have the same historical splits; their counts must not be added together. The evidence establishes contamination, not the exact amount by which any metric is inflated. Fusion, convergence, SIIM continuation, and recovery experiments using the retained encoder inherit this problem.

**Correction:** new warm starts recursively audit ancestor configurations and train/validation/test identities, image hashes, lesions, and groups. Missing ancestry fails closed; ancestor training or validation overlap rejects the new warm start. Existing runs can resume only with unchanged identities and configuration, retaining their historical interpretation. The new reference initializes from ImageNet, with no project checkpoint.

Evidence: machine-readable audit (`artifacts/pipeline_review_20261005/data_audit.json`; local-only), [provenance checks](training_provenance.py), and [trainer](train_multitask.py).

### Accuracy alone rewards missing melanomas

The historical test results report 91.02% accuracy at threshold 0.5 with only 68.04% melanoma sensitivity, meaning 93 missed melanomas. Predicting every lesion as nonmelanoma would score 93.03% accuracy. These historical results are contaminated, but the arithmetic still demonstrates why a raw accuracy target is inappropriate for this imbalanced screening task.

The working objective remains at least 95% melanoma sensitivity while reducing false positives, alongside average precision, ROC AUC, and source-specific performance. A threshold change can increase raw accuracy while degrading detection. No threshold or promotion gate was relaxed to manufacture a gain.

Evidence: historical test metrics (`artifacts/melanoma_v1/run_20260822_111527/test_risk_metrics.json`; local-only).

### Generalization and effective sample size are limiting

There are 19,328 training photographs but only **874 distinct training melanoma lesions**, represented by 2,567 positive photographs. Repeated photographs do not supply the diversity of new patients or lesions. HAM10000 and BCN20000 supply 22,377 cohort images without patient IDs; patient independence cannot be established for them.

Current split checks find no cross-split image, lesion, group, or exact SHA-256 overlap. Identical dHash values span splits in 78 groups containing 266 images, including 31 groups with conflicting melanoma labels. These are candidates for review, not proof of duplicates: low-resolution perceptual hashes collide. No images were removed or automatically relabeled from these matches.

The completed HIBA evaluation provides a more realistic warning about transfer: at the unchanged threshold, the retained model detected 138 of 157 melanomas, for **87.90% sensitivity**, 63.64% specificity, and 66.83% accuracy across 1,191 lesions from 584 patients. It was not retuned or rescored during this review. Its available-ID overlap audit cannot establish globally complete patient independence. Its 13.18% prevalence also makes average precision and precision unsuitable for direct comparison with the 6.94% internal validation prevalence.

Evidence: data audit (`artifacts/pipeline_review_20261005/data_audit.json`; local-only), duplicate candidates (`artifacts/pipeline_review_20261005/near_duplicate_candidates.csv`; local-only), and saved HIBA results (`artifacts/hiba_external_20261003/results.json`; local-only).

### Metadata training and inference did not match

Training supplied available age, sex, and site to the auxiliary diagnosis branch; inference supplied unknown values. Dropping the encoded metadata branch is not equivalent to passing unknown raw metadata through its learned layers. This matters even with an image-only binary head: the retained inference blend gives 65% weight to diagnosis-derived melanoma probability. DERM12345 has unknown metadata throughout, creating an additional possible source shortcut.

**Correction:** new runs use unknown metadata during training as well as inference. Legacy resumes keep their original policy. The clean recipe sets metadata dropout to zero because the input policy is already fixed. This removes a distribution mismatch; its standalone accuracy effect has not been established.

### Unsupported diagnoses were assigned misleading auxiliary labels

The DERM mapper assigned Kaposi sarcoma to benign VASC and mapped cutaneous horn to SCC based on broad malignancy categories. The retained manifest contains 12 KS and 9 cutaneous-horn rows affected by these mappings. Their melanoma-versus-rest targets remain nonmelanoma; the defect concerns diagnosis supervision and broader cancer semantics.

**Correction:** the supported eight-category vocabulary no longer supplies invented category labels for these entries. They remain in binary training as confirmed nonmelanoma, use `UNKNOWN` with zero auxiliary diagnosis weight, and retain original malignancy and separate high-risk labels. The same rule handles other unsupported source categories. The new derived manifest preserves all 32,216 image identities, paths, hashes, lesions, groups, and fold membership. Original manifests and images are unchanged.

Evidence: taxonomy changes (`artifacts/pipeline_review_20261005/taxonomy_corrections.json`; local-only), changed rows (`artifacts/pipeline_review_20261005/taxonomy_corrections.csv`; local-only), and [preparation code](prepare_derm12345_training.py).

## TensorFlow implementation and system design

The loader's 0–1 image scaling followed by multiplication by 255 and EfficientNetV2 preprocessing is consistent. The pretrained backbone's `training=False` preserves BatchNorm statistics while selected convolutional layers can still receive gradients; this is consistent with the [TensorFlow transfer-learning guidance](https://www.tensorflow.org/guide/keras/transfer_learning). It should not be changed merely because the word “training” appears false.

Physical batches of four still make new classifier BatchNorm estimates noisy; gradient accumulation does not increase the batch used for those statistics. An optional LayerNorm head now supports clean initialization without depending on historical learned BatchNorm state. Existing model formats and legacy defaults remain supported.

Previous code already addressed exact duplicate removal, lesion grouping, corrected dihedral test-time augmentation, small-memory GPU operation, sampling controls, and interruption recovery. Repeating these as newly discovered solutions would overstate the changes. Source balancing, focal loss, positive oversampling, label smoothing, and metadata blending create several interacting choices; previous short warm-start studies do not isolate their effects and inherit the ancestry defect.

The clean run uses binary cross-entropy without label smoothing, lesion-uniform sampling within source/diagnosis strata, 25% positive exposure, LayerNorm heads, and matching metadata policies. These are predeclared reference choices, not separately proven improvements or optimal hyperparameters. The run retains the 300-pixel EfficientNetV2B1 input and a full epoch budget, with four head epochs and up to twelve fine-tuning epochs subject to early stopping.

A separate numerical bug stopped the diagnosis study at 2,000 iterations although its measured maximum gradient, 1.88e-8, satisfied the existing 1e-6 acceptance criterion. The corrected classifier independently verifies finite parameters, loss, and gradient; it accepts an iteration-limit termination only when that criterion is met. Other unsuccessful termination modes remain errors. [SciPy documents iteration limits separately from gradient stopping](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html). The contaminated fusion study was not resumed merely because this software defect was fixed.

The workspace is not a Git repository. Existing experiment snapshots help reproduce individual runs, but shared scripts, many historical launchers, and mutable defaults make lineage difficult to inspect. The new training driver records input, source, pretrained-weight and runtime fingerprints, preserves a source snapshot, uses the existing process lock, and refuses silently changed resume inputs. It runs training in a separate process and disables test finalization and automatic replacement. Windows CPU tests use different NumPy and scikit-learn versions from WSL training; WSL matches the existing pinned requirements. The new plan records actual training package versions. Direct SciPy, h5py, and threadpoolctl use should also be declared explicitly in a future dependency cleanup.

## Completed experiment and active training

A new bounded experiment used **only the frozen full-frame PanDerm embedding**, excluding every incumbent feature. Regularization was selected using five grouped training rotations with disjoint fitting, calibration, and scoring groups. Both optimization starts converged and produced identical outer decisions. TensorFlow export/reload matched 512 real training-feature rows within 1.22e-15.

| Metric | PanDerm-only reused validation |
| --- | ---: |
| Lesions / melanoma lesions | 4,177 / 290 |
| Accuracy | 67.92% |
| Melanoma sensitivity | 95.17% |
| Specificity | 65.89% |
| Precision | 17.23% |
| Average precision | 0.4535 |
| ROC AUC | 0.9046 |
| False positives / missed melanomas | 1,326 / 14 |

No training configuration met the 95% sensitivity floor in every source. Validation sensitivity was 98.08% for BCN, 92.06% for DERM, and 94.31% for HAM. This is a completed negative result, not a significant accuracy improvement. It removes the identified local ancestor contamination by construction, but PanDerm's private pretraining overlap remains unverified and outer validation has been reused in development. Its exported head accepts embeddings, not raw JPEGs.

Evidence: experiment plan (`artifacts/clean_panderm_review_20261005/experiment_plan.json`; local-only), selection (`artifacts/clean_panderm_review_20261005/selection.json`; local-only), and results (`artifacts/clean_panderm_review_20261005/validation_report.json`; local-only).

The new ImageNet-initialized TensorFlow run is recorded under clean training (`artifacts/clean_melanoma_20261005`; local-only). Read status (`artifacts/clean_melanoma_20261005/status.json`; local-only), training log (`artifacts/clean_melanoma_20261005/training.log`; local-only), and the fixed plan (`artifacts/clean_melanoma_20261005/experiment_plan.json`; local-only). It is a new reference training job; no completed accuracy gain is claimed while it runs. Historical checkpoints remain available for audit.

To resume an interrupted run after checking that no driver or training child remains, run `bash launch_clean_training.sh` from this directory in WSL. The launcher does not schedule a service or guarantee restart after a reboot. Completed runs and changed fingerprints are checked before reuse.

## Achievements that must be reconsidered

1. Withdraw the interpretation of inherited-model internal validation/test scores as independent performance estimates. Keep the artifacts as historical evidence.
2. Give up the speed advantage of reusing the contaminated project encoder for this reference. Resetting only its risk head is insufficient.
3. Accept that a more credible evaluation may initially produce lower scores. A threshold-selected 90–95% headline accuracy is not a useful achievement when it misses many melanomas.
4. Stop calling melanoma-negative predictions benign or cancer-free. Unsupported diagnoses must not be forced into the auxiliary taxonomy.
5. Stop treating repeatedly consulted validation, PH2, DERM, or HIBA results as new independent confirmation. Future credible claims need a separately obtained, patient-linked confirmation cohort and a locked evaluation protocol.

The strongest next evidence after clean training is whether false alarms fall at preserved sensitivity across sources. The existing review found that NV and BKL account for 902 of 1,073 historical false alarms; additional independently labeled melanoma and lookalike lesions are more relevant than indiscriminately adding easy negative photographs. Expert adjudication is needed before treating prediction errors as label errors. No clinically justified sacrifice of melanoma sensitivity has been established by this review.

## Verification

The full regression suite passed 185 tests after the main repairs, including provenance rejection, normalization serialization, metadata masking, split/resume checks, optimizer failure handling, and real TensorFlow head export/reload. A subsequent 18-test run passed the final DERM/SIIM compatibility and both clean-experiment suites; these overlap the full-suite count and must not be added as unique tests. Logs are full regression (`artifacts/pipeline_review_20261005/regression_tests.log`; local-only) and final targeted checks (`artifacts/pipeline_review_20261005/final_targeted_tests.log`; local-only). Synthetic test scores are software checks, not model-performance measurements. The actual new PanDerm experiment completed and verified its saved artifact hashes; the ImageNet training result remains pending.
