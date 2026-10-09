# Progress review: 95% melanoma sensitivity with fewer false alarms

Publication note: aggregate results are available in [reports/development_progress_20261007.json](reports/development_progress_20261007.json). Artifact paths below identify local-only audit evidence; datasets, individual predictions, training outputs and checkpoints are excluded from this Git snapshot.

Review date: October 6 Eastern / October 7 UTC, 2026. The user confirmed clinical dermoscopy as the intended image modality and 95% sensitivity with fewer false positives as the development objective. This is melanoma versus non-melanoma, not a general cancer-clearance decision.

## Assessment

The evidence does not establish a technological ceiling. We have demonstrated correctable experimental and sampling problems, and correcting them has already produced measurable development gains. There is no evidence that replacing TensorFlow or rewriting the entire application would solve the remaining diagnostic errors. Nor do the present results justify promising rapid clinical readiness.

High sensitivity is technologically achievable, but its specificity depends on the population, acquisition conditions and case difficulty. For example, the prospective PROVE-AI study reported 96.8% sensitivity and 37.4% specificity on 603 biopsied lesions at a predetermined threshold. That supports feasibility of high sensitivity while illustrating the false-positive tradeoff; it is not a directly comparable benchmark or an estimate of this project's ceiling. [Primary study](https://www.nature.com/articles/s41746-023-00872-1).

## Why progress has been slow

1. **The recent head-selection rule rewarded the wrong behavior when its constraint failed.** If no candidate achieved 95% sensitivity in every source, it selected the highest worst-source sensitivity before considering specificity. A classifier calling every case positive would win that fallback, despite zero specificity. This selected excessive regularization in the recent clean feature experiments. The corrected criterion measures within-source specificity at 95% recall on training-only inner scoring folds. Those descriptive thresholds select ranking quality only; disjoint calibration still determines the reported operating decisions. Failed sensitivity constraints remain failures. The criterion was corrected after earlier development inspection, so this is a retrospective methodological repair, not independent confirmation.

2. **The old sampling allocation underexposed the common benign lesions responsible for most false alarms.** Nevi are 70.08% of training lesion IDs but received 30.31% of draws. The completed controlled recipe increases their exposure to 56.50%, while retaining 25% melanoma draws. Thus “natural lesion” means equal lesion allocation within the negative class, not restoring natural disease prevalence. The training callback reported 262 fewer false positives at the same 276/290 detected melanomas. The comparison below checks this with separate calibration groups.

   This is a single-seed recipe comparison, not isolated causal proof of sampling. Trainer recovery/reporting repairs also changed between runs; the reference stopped after 4+11 epochs and the new run completed 4+12.

3. **Photograph count exaggerates independent case diversity.** There are 2,567 melanoma training photographs but only 874 recorded melanoma lesion IDs. Even that is an upper bound on proven unique lesions: DERM uses image IDs as lesion proxies, and its 191 positive IDs come from only 43 patients. BCN's 1,709 melanoma photos represent 315 lesions. HAM and BCN lack patient IDs in this project, preventing a complete patient-disjointness claim. Repeated views do not provide the same information as new patients.

4. **Case mix and acquisition vary substantially by source.** A source-only prior fitted on training metadata obtains validation ROC AUC 0.6663. The canonical clean-reference scores obtain 0.9138, with within-source AUCs of 0.8485 (BCN), 0.9570 (DERM) and 0.8784 (HAM), verified in the corrected comparison below. The older cohort audit used an arithmetic-mean-direct proxy; its reported AUCs round to the same values but it remains a distinct computation. The model learns useful within-source information, but source association and uneven difficulty remain material. A bounded image sample confirms source-specific black framing and non-square photographs being resized to squares. Neither observation proves that cropping or changing aspect-ratio handling will improve performance; aggressive cropping could remove useful lesion context.

5. **Some earlier achievements were overstated by the evaluation design.** The old encoder ancestry included thousands of current validation/test photographs. Those historical checkpoints and inherited-feature experiments remain preserved but cannot serve as independent performance evidence. Current clean checkpoints avoid those project ancestors, yet checkpoint selection and repeated experimentation still reuse validation. Grouped threshold cross-fitting cannot undo that exposure. Reliable confirmation requires a locked candidate and new representative, patient-separated evaluation data.

6. **The experiment cycle contains avoidable computation, but hardware is not the only learning bottleneck.** The completed natural run used 77,312 training microbatches across 16 epochs. Keras validation and the lesion callback each processed the 6,443 validation photos every epoch, totaling 206,176 validation forwards. The GTX 1650 makes whole-image experiments expensive. A warm native-Windows decode benchmark measured about 153 images/s with two workers; it does not support blaming file I/O for the much slower complete training cycle. This benchmark is not a WSL profiler or a measured training speedup.

## Actions completed in this review

- Replaced the flawed head-selection fallback in a separate, bounded experiment. Added a regression case demonstrating the all-positive failure. Historical experiment drivers and outputs remain reproducible.
- Refit the training-selected clean B1 and B1+PanDerm heads with converged optimization and two numerical starts. Every fixed candidate is reported on identical grouped calibration/scoring folds, without adjusting its hyperparameters from those results.
- Generalized the clean checkpoint extractor to bind the selected run, completed-artifact hashes, batch size, mixed-precision runtime and source snapshot. It verifies image bytes and permits only training/validation extraction.
- Found and repaired a multi-image scoring mismatch: the feature export averaged direct probabilities, while the training callback applies sigmoid to the mean of clipped image logits. Canonical checkpoint scores are rebuilt from verified per-image chunks without rerunning the GPU. Feature embeddings and their fitted linear heads are unaffected. Historical proxy reports remain preserved and are identified as such.
- Prepared a matched evaluation of the completed natural-sampling run, including one predeclared 50:50 ensemble with the clean reference. No ensemble weight search is performed on these outcomes.
- Added an opt-in path that omits the redundant Keras validation pass while retaining the lesion validation callback. It is restricted to lesion-based selection with automatic finalization disabled, and bound into resume compatibility. The default remains unchanged. A small CPU equivalence test verifies unchanged learned weights and lesion selection metrics; whole-run speedup has not been measured.
- Audited available SIIM training data for a potential diversity experiment, without adding images or changing labels/folds. It contains 581 retained melanoma IDs from 428 known positive patients, zero known exact hash overlap with the current clean cohort, and no rows assigned to current validation/test folds. All retained binary labels match the original CSV. Source/resolution effects and absent verification-method metadata still require careful handling.

## Matched development results

The completed comparisons are recorded in `artifacts/progress_review_20261007`. Results use the same 4,177 validation lesion IDs (290 positive), five grouped calibration/scoring folds, a 95% calibration sensitivity target and seed 20261006. These are reused development results. A 95% calibration target does not guarantee 95% sensitivity on the separately scored groups.

The initial corrected-head comparison found:

| Candidate | Detected / missed melanoma | False alarms | Sensitivity | Specificity | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean reference blend, earlier arithmetic-mean-direct proxy | 274 / 16 | 1,285 | 94.48% | 66.94% | 0.4939 |
| Original B1 linear head, L2=1 | 277 / 13 | 1,484 | 95.52% | 61.82% | 0.3926 |
| Corrected B1 linear head, L2=0.1 | 277 / 13 | 1,158 | 95.52% | 70.21% | 0.4826 |
| Corrected B1+PanDerm, L2=0.01 | 275 / 15 | 1,060 | 94.83% | 72.73% | 0.5575 |

The corrected B1 head removes 326 false alarms relative to its old selection, with the same melanoma count. The fusion improves ranking but misses the aggregate 95% sensitivity target and must not be presented as passing. Source sensitivity remains uneven: corrected B1 has 98.08% BCN, 93.65% DERM and 94.31% HAM; corrected fusion has 95.19%, 85.71% and 99.19%, respectively. A strong overall ranking score does not establish reliable source-specific detection.

The canonical checkpoint comparison now reproduces the training aggregation rule from per-image chunks and ties each cache to its actual checkpoint and split. The earlier arithmetic-mean proxy reports remain preserved. Small remaining differences from training-time callback numbers are not counted as improvements: reconstructed reference AP is 0.494268 versus logged 0.494381; reconstructed natural AP is 0.493035 versus logged 0.493164. Four-image model/extractor equivalence is exact, but this does not establish bitwise equivalence to the entire original inference pass.

| Canonical fixed candidate | Detected / missed melanoma | False alarms | Sensitivity | Specificity | Precision | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean reference saved blend | 274 / 16 | 1,284 | 94.48% | 66.97% | 17.59% | 0.4943 |
| Natural-sampling saved checkpoint | 276 / 14 | 1,056 | 95.17% | 72.83% | 20.72% | 0.4930 |
| Fixed 50:50 checkpoint ensemble | 275 / 15 | 1,038 | 94.83% | 73.30% | 20.94% | 0.4986 |
| Natural diagnosis head | 275 / 15 | 1,040 | 94.83% | 73.24% | 20.91% | 0.4878 |

The new checkpoint removes 228 of 1,284 false positives (17.76%) while detecting two additional melanomas on these scoring folds. This is a useful development gain at the intended aggregate operating point, despite essentially unchanged average precision. The fixed ensemble and diagnosis-only score do not meet the aggregate 95% sensitivity target and were not selected as replacements.

| Source | Reference sensitivity / specificity | Natural sensitivity / specificity | Reference → natural false alarms |
| --- | ---: | ---: | ---: |
| BCN | 95.19% / 32.08% | 97.12% / 29.30% | 415 → 432 |
| DERM | 96.83% / 79.21% | 92.06% / 90.13% | 396 → 188 |
| HAM | 92.68% / 65.50% | 95.12% / 68.20% | 473 → 436 |

Most false-positive reduction comes from DERM, where three additional melanomas are missed. BCN gains two detections but has 17 additional false alarms; HAM gains three detections and removes 37 false alarms. Thus aggregate progress does not establish a uniform gain or reliable 95% sensitivity in every source. A conditional paired group bootstrap gives specificity improvement bounds of +4.42 to +7.20 percentage points and sensitivity change bounds of -1.65 to +3.30 points. These intervals condition on fixed selected checkpoints/decisions and omit threshold refitting, training variation and repeated development inspection; they are not independent significance evidence.

By diagnosis, nevus false positives fall from 842 to 571 and benign keratosis false positives from 206 to 203; other non-melanoma categories produce more false positives. This supports the exposure rationale without proving it caused the gain. “False positive” here means non-melanoma: it includes some other cancers, and is not synonymous with a clinically harmless lesion. DERM average precision falls from 0.5634 to 0.4601, whereas HAM rises from 0.4274 to 0.4821. Aggregate averages conceal these tradeoffs.

The bounded natural-feature head experiment also completed. It used the frozen training-only ranking rule and L2 grid `[0.01, 0.1, 1]` before any validation feature access. It selected 0.1 for natural B1 and 0.01 for natural B1+PanDerm. A separate training-only reanalysis fixed pure PanDerm at 0.01. All 36 fits converged, numerical starts agreed within 1.59e-6 probability, and all 101 completed-output hashes verified.

| Additional fixed candidate | Detected / missed melanoma | False alarms | Sensitivity | Specificity | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Natural B1 feature head | 275 / 15 | 1,109 | 94.83% | 71.47% | 0.5017 |
| Natural B1+PanDerm feature head | 274 / 16 | 1,072 | 94.48% | 72.42% | 0.5575 |
| Corrected pure PanDerm feature head | 274 / 16 | 1,081 | 94.48% | 72.19% | 0.4990 |

None improves on the natural checkpoint while meeting the aggregate target. The large fusion AP increase still fails to deliver the desired operating behavior, with DERM sensitivity only 85.71%. The completed natural checkpoint remains the development reference; none of these additional heads was promoted. This is a useful stopping result for this bounded family of cheap head changes, not proof that representation learning has reached its technological limit.

## What should change next

The implementation should keep the clean provenance and cached-feature experiment path, with the operating objective explicitly separated from calibration reliability. Small head experiments now take minutes and can reject poor hypotheses before another multi-hour image-training run. The next image experiment should change one justified factor, retain equal negative-lesion exposure, and use the optional redundant-validation removal after its tests pass.

A monotonic probability-calibration change cannot improve the underlying sensitivity/specificity tradeoff: it changes the scale of scores and the corresponding threshold. Achieving fewer false alarms at the same sensitivity requires improved ranking or better information, followed by reliable calibration. The current scores should not be presented as calibrated clinical melanoma probabilities.

Among locally audited sources, the strongest available new-data opportunity is a controlled SIIM diversity experiment using the cleaned base manifest, explicit binary labels, supervision masks, existing training-only patient groups and within-source negative controls. Its 428 named positive-patient groups are not proven globally independent of patients in other sources: no linked cross-source patient identifiers are available. Adding only SIIM melanomas would create a source shortcut; adding all 32,693 images without controlling exposure would swamp the experiment with negatives. Its image-resolution groups also have very different melanoma prevalence. The prior contaminated SIIM continuation cannot settle whether a clean, controlled use of these patients helps.

Current PanDerm results concern frozen Base embeddings and small linear heads. The published model uses a ViT-Large encoder and reports both linear-probe and task-specific fine-tuning experiments. Our negative probes do not establish that all dermatology foundation models are ineffective, but the paper also does not guarantee that a bigger checkpoint will meet our objective. [PanDerm methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC12353815/).

What must be relinquished is confidence in contaminated historical scores, aggregate-only success claims and the idea that more epochs on the same photographs must produce rapid gains. The data, checkpoints, useful features and verified TensorFlow implementation can be retained. Expert review of difficult/conflicting labels and representative new patient data remain work that code inspection cannot replace.

## Evidence and scope

This follow-up builds on the full root implementation review of 61 Python files and 23 launcher/dependency files, documented in [the October 6 review](PIPELINE_REVIEW_20261006.md). It adds focused objective, source-prior, patient-diversity, runtime and new-checkpoint analyses. It does not claim a new line-by-line audit of TensorFlow internals, third-party weights, or clinical label correctness.

- Training-selection audit (`artifacts/training_selection_objective_audit/training_rank_quality.json`; local-only)
- Corrected clean-head results (`artifacts/progress_review_20261007/corrected_clean_heads/development_report.json`; local-only)
- Cohort and source-prior audit (`artifacts/progress_review_20261007/cohort_signal_review.json`; local-only)
- SIIM training diversity audit (`artifacts/progress_review_20261007/siim_training_diversity_review.json`; local-only)
- Canonical natural checkpoint comparison (`artifacts/progress_review_20261007/natural_comparison_v2/development_report.json`; local-only)
- Completed natural-feature head comparison (`artifacts/progress_review_20261007/natural_heads/development_report.json`; local-only)
- Runtime, aggregation and regression evidence (`artifacts/progress_review_20261007/runtime_and_score_review.json`; local-only)

After the active studies completed, the future extractor was corrected to write canonical mean-logit direct scores itself. Plan version 4 explicitly records embedding, direct-score and diagnosis aggregation rules. Existing version-2/3 caches remain immutable; the canonical reconstruction helper handles them correctly. The original feature-head comparison and earlier report are not silently rewritten.

Final focused verification passed **70 tests**, covering aggregation, cache integrity, group/label alignment, head-selection failure cases, training-before-validation ordering, validation-pass equivalence, trainer helpers and provenance. All **147 output hash bindings** across the new completed studies and extraction markers verified. Regression log (`artifacts/progress_review_20261007/final_regression_tests.log`; local-only), completion verification (`artifacts/progress_review_20261007/completion_verification.json`; local-only). All bounded experiments described above have finished; no new long training run was started as a substitute for resolving these findings.

No test/HIBA images were used for these new experiments, and no clinical deployment or automatic model replacement is authorized by their results.
