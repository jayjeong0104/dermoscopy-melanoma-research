"""Keep the primary result separate from explicitly exploratory threshold tradeoffs."""
import json

import pandas as pd

from compare_model_runs import file_hash
from foundation_common import ROOT, REFERENCE, save_json
from operating_policy import crossfit_policy
from run_convergence_study import OUTPUT
from study_precision import plot_precision


def main():
    report = json.loads((OUTPUT / "validation_report.json").read_text())
    complete = json.loads((OUTPUT / "complete.json").read_text())
    if complete["report_sha256"] != file_hash(OUTPUT / "validation_report.json"):
        raise ValueError("Completed convergence report changed.")
    fusion_path = OUTPUT / "fusion_start0_validation_predictions.csv"
    if file_hash(fusion_path) != report["results"]["fusion_start0"]["predictions_sha256"]:
        raise ValueError("Fusion predictions changed.")
    baseline, fusion = pd.read_csv(REFERENCE), pd.read_csv(fusion_path)
    rows = []
    for name, frame in (("retained", baseline), ("converged_fusion", fusion)):
        for target in (.95, .955, .96, .97, .98, .99):
            metrics, _ = crossfit_policy(frame, frame.probability.to_numpy(), minimum_recall=target,
                                          folds=frame.policy_fold.to_numpy())
            rows.append({"model": name, "calibration_target": target,
                **{key: metrics[key] for key in ("recall", "precision", "specificity", "false_positives", "false_negatives")}})
    save_json(OUTPUT / "exploratory_threshold_tradeoffs.json", {
        "exploratory_only": True, "primary_decision_unchanged": report["decision"],
        "nominated_arm": report["train_only_nominated_arm"], "baseline_predictions_sha256": file_hash(REFERENCE),
        "fusion_predictions_sha256": file_hash(fusion_path), "rows": rows,
        "limitation": "Thresholds inspected after primary results; fusion was not the train-only nominated arm. These rows cannot convert the failed primary study into a confirmed improvement. No test data or clinical use."})
    plot_precision(OUTPUT, baseline, {"Converged feature fusion": fusion})
    save_json(OUTPUT / "crop_audit/visual_observations.json", {
        "review": "Engineering visual review of the fixed 18-image training sample, not dermatologist adjudication",
        "examples": [
            {"image_id": "DERM_931536", "observation": "Visible pigmented area extends beyond the center crop's top/left boundaries."},
            {"image_id": "DERM_208508", "observation": "The off-center pigmented region reaches the cropped view's right boundary."},
            {"image_id": "ISIC_0026517", "observation": "Center crop removes part of the visible pigmented region near the upper-left boundary."}],
        "inference": "Preserving the complete image extent is a justified preprocessing ablation, not a proven accuracy fix.",
        "dataset_wide_truncation_rate_estimated": False, "labels_changed": False})
    print("Primary report preserved; exploratory tradeoffs, crop observations and precision curve saved.")


if __name__ == "__main__":
    main()
