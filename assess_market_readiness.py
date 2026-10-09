from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DEFAULT_MANIFEST = (
    Path("data") / "combined_ham_bcn_derm12345" / "combined_manifest.csv"
)
PERFORMANCE_TARGETS = {
    "accuracy": 0.95,
    "balanced_accuracy": 0.95,
    "recall": 0.95,
    "specificity": 0.95,
}
PROJECT_ROOT = Path(__file__).resolve().parent
LICENSE_INVENTORY = {
    "HAM10000": {
        "commercial_status": "blocked",
        "reason": "The distributed HAM10000 terms restrict use to non-commercial work.",
        "source": "https://doi.org/10.7910/DVN/DBW86T",
        "required_action": (
            "Remove HAM10000 and retrain from commercially cleared data, or obtain a "
            "separate written commercial license from the rights holder."
        ),
    },
    "BCN20000": {
        "commercial_status": "conditional",
        "reason": "The published dataset is CC BY 4.0 and requires attribution.",
        "source": "https://doi.org/10.1038/s41597-024-03387-w",
        "required_action": "Preserve image-level provenance and satisfy attribution terms.",
    },
    "DERM12345": {
        "commercial_status": "conditional",
        "reason": "The supplied metadata identifies the images as CC-BY.",
        "source": "https://arxiv.org/abs/2406.07426",
        "required_action": "Have counsel verify the exact license version and attribution chain.",
    },
    "PH2": {
        "commercial_status": "blocked",
        "reason": "The official terms prohibit redistribution and commercial use.",
        "source": "https://www.fc.up.pt/addi/ph2%20database.html",
        "required_action": (
            "Do not use PH2 in a commercial development or release claim without separate "
            "written permission."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed technical and evidence audit for a trained medical AI run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gate(
    gate_id: str,
    category: str,
    status: str,
    evidence: str,
    remediation: str,
) -> dict[str, str]:
    if status not in {"passed", "failed", "missing", "manual_review_required"}:
        raise ValueError(f"Unsupported gate status: {status}")
    return {
        "id": gate_id,
        "category": category,
        "status": status,
        "evidence": evidence,
        "remediation": remediation,
    }


def metric_gates(
    prefix: str,
    label: str,
    metrics: dict[str, object] | None,
) -> list[dict[str, str]]:
    gates: list[dict[str, str]] = []
    for metric_name, target in PERFORMANCE_TARGETS.items():
        value = None if metrics is None else metrics.get(metric_name)
        if value is None:
            status = "missing"
            evidence = f"{label} {metric_name} is unavailable."
        else:
            numeric = float(value)
            status = "passed" if math.isfinite(numeric) and target <= numeric <= 1 else "failed"
            evidence = f"{label} {metric_name}={numeric:.4f}; target>={target:.2f}."
        gates.append(
            gate(
                f"{prefix}_{metric_name}",
                "performance",
                status,
                evidence,
                (
                    "Improve and revalidate on a locked, representative cohort; do not tune "
                    "against this test result."
                ),
            )
        )
    return gates


def current_sources(manifest: Path) -> dict[str, int]:
    if not manifest.exists():
        return {}
    frame = pd.read_csv(manifest, usecols=["source"])
    return {str(key): int(value) for key, value in frame["source"].value_counts().items()}


def build_report(run_dir: Path, manifest: Path) -> dict[str, object]:
    finalization = read_json(run_dir / "finalization_status.json")
    model_selection = read_json(run_dir / "model_selection.json")
    internal = read_json(run_dir / "test_risk_metrics.json")
    derm = read_json(run_dir / "derm12345_external_metrics.json")
    ph2 = read_json(run_dir / "ph2_external_metrics.json")
    subgroup = read_json(run_dir / "test_subgroup_metrics.json")
    decode_audit = read_json(run_dir / "test_jpeg_decode_findings.summary.json")
    model_path = run_dir / "best_model.keras"
    saved_training_manifest = run_dir / "train_split.csv"
    source_manifest = saved_training_manifest if saved_training_manifest.exists() else manifest
    sources = current_sources(source_manifest)

    gates: list[dict[str, str]] = []
    finalized = bool(finalization and finalization.get("status") == "complete")
    gates.append(
        gate(
            "reproducible_finalization",
            "engineering",
            "passed" if finalized else "missing",
            "Finalization completed with restartable cached inference."
            if finalized
            else "No complete finalization_status.json is present.",
            "Run finalize_multitask_run.py to completion in the verified environment.",
        )
    )

    hash_matches = False
    if model_path.exists() and model_selection:
        expected_hash = model_selection.get("selected_model_sha256")
        hash_matches = bool(expected_hash and sha256_file(model_path) == expected_hash)
    gates.append(
        gate(
            "checkpoint_integrity",
            "engineering",
            "passed" if hash_matches else "failed",
            "The selected checkpoint SHA-256 matches model_selection.json."
            if hash_matches
            else "The selected checkpoint is missing or its SHA-256 is not verified.",
            "Re-finalize the run and preserve the model plus signed provenance manifest.",
        )
    )

    split_integrity = bool(
        finalization
        and "split_integrity" in finalization.get("completed_stages", [])
    )
    gates.append(
        gate(
            "patient_lesion_hash_separation",
            "data",
            "manual_review_required" if split_integrity else "missing",
            "Recorded split checks cover available groups, lesions, and exact hashes; "
            "complete patient linkage and checkpoint-ancestor separation are not verified by this report."
            if split_integrity
            else "No completed split-integrity stage is recorded.",
            "Rebuild patient-safe folds and rerun exact-hash leakage checks.",
        )
    )
    decode_clean = bool(
        decode_audit
        and int(decode_audit.get("decode_errors", 1)) == 0
        and int(decode_audit.get("dimension_mismatches", 1)) == 0
    )
    gates.append(
        gate(
            "held_out_jpeg_decode_integrity",
            "data",
            "passed" if decode_clean else "missing",
            (
                f"All {decode_audit.get('images_checked')} held-out images decoded; "
                f"recoverable warnings={decode_audit.get('decoder_warnings', 0)}, hard "
                "errors=0, dimension mismatches=0."
            )
            if decode_clean
            else "No complete held-out TensorFlow JPEG decode audit is present.",
            "Run audit_jpeg_decode.py and investigate every warning or hard failure.",
        )
    )

    internal_operating = None
    if internal:
        internal_operating = (
            internal.get("threshold_metrics", {}).get("operating")
        )
    gates.extend(metric_gates("internal", "Internal held-out lesion-level", internal_operating))

    derm_operating = None
    if derm:
        derm_operating = (
            derm.get("patient_level_max_probability", {})
            .get("operating_threshold_metrics")
        )
    gates.extend(metric_gates("external_derm", "DERM12345 patient-level", derm_operating))
    gates.append(
        gate(
            "independent_external_test",
            "clinical_evidence",
            "manual_review_required" if derm is not None else "missing",
            "DERM12345 results exist, but cohort independence is not verified. "
            f"Recorded development status: {derm.get('development_status', 'unrecorded')}. "
            "Local overlap checks and checkpoint bindings do not establish an untouched confirmatory cohort."
            if derm is not None
            else "Locked DERM12345 results are absent.",
            "Run the locked external evaluator once, then freeze the result from tuning.",
        )
    )
    gates.append(
        gate(
            "ph2_research_stress_test",
            "clinical_evidence",
            "passed" if ph2 is not None else "missing",
            "PH2 research-only stress-test results exist."
            if ph2
            else "PH2 stress-test results are absent.",
            "Evaluate for research diagnostics only; PH2 cannot support commercial use.",
        )
    )

    missingness = {} if not subgroup else subgroup.get("missingness", {})
    demographic_ready = bool(
        subgroup
        and int(missingness.get("skin_tone", 1)) == 0
        and int(missingness.get("capture_device", 1)) == 0
    )
    gates.append(
        gate(
            "representative_subgroup_validation",
            "clinical_evidence",
            "manual_review_required" if demographic_ready else "failed",
            "Skin-tone and capture-device labels are absent for the evaluated cohort."
            if not demographic_ready
            else "Required demographic and device labels are available; representative subgroup "
            "performance and statistical power remain unverified.",
            (
                "Acquire a prospectively powered, multi-site cohort representative of the "
                "intended population and report subgroup confidence intervals."
            ),
        )
    )

    commercial_blockers = [
        source
        for source in sources
        if LICENSE_INVENTORY.get(source, {}).get("commercial_status") == "blocked"
    ]
    if ph2 is not None:
        commercial_blockers.append("PH2")
    commercial_blockers = sorted(set(commercial_blockers))
    unknown_sources = sorted(source for source in sources if source not in LICENSE_INVENTORY)
    if commercial_blockers:
        rights_status = "failed"
        rights_evidence = "Commercial-use blockers: " + ", ".join(commercial_blockers)
    elif not sources or unknown_sources:
        rights_status = "missing"
        rights_evidence = ("No usable training-source inventory is available." if not sources else
                           "Training-source rights are unknown: " + ", ".join(unknown_sources))
    else:
        rights_status = "manual_review_required"
        rights_evidence = "Listed source terms are conditional; this report does not verify " \
                          "attribution, commercial permissions, or the complete checkpoint lineage."
    if not saved_training_manifest.exists():
        rights_evidence += " The supplied manifest is not bound to this run's saved training split."
    gates.append(
        gate(
            "commercial_data_rights",
            "legal",
            rights_status,
            rights_evidence,
            (
                "Obtain written commercial rights or retrain and revalidate using only data "
                "with a documented commercial provenance chain."
            ),
        )
    )

    manual_gates = (
        (
            "prospective_multicenter_clinical_validation",
            "clinical_evidence",
            "No prospective, protocol-registered, multi-center clinical study is present.",
            "Run a statistically powered study in the intended workflow and population.",
        ),
        (
            "intended_use_and_regulatory_pathway",
            "regulatory",
            "The intended user, patient population, workflow, claims, and jurisdiction are not frozen.",
            "Freeze intended use and obtain regulatory counsel/agency feedback on the pathway.",
        ),
        (
            "quality_management_system",
            "quality",
            "No audited medical-device quality management system evidence is present.",
            "Develop under an applicable QMS with design controls and controlled records.",
        ),
        (
            "risk_management_and_human_factors",
            "safety",
            "No clinical hazard analysis or human-AI usability validation is present.",
            "Complete risk management, usability engineering, and clinician override testing.",
        ),
        (
            "input_quality_ood_and_abstention",
            "safety",
            "The classifier has no validated image-quality, out-of-distribution, or abstention layer.",
            "Build and clinically validate capture QC, OOD detection, and safe failure behavior.",
        ),
        (
            "software_lifecycle_and_cybersecurity",
            "engineering",
            "No controlled software lifecycle, threat model, SBOM, or security evidence is present.",
            "Implement medical-device software lifecycle and current cybersecurity controls.",
        ),
        (
            "postmarket_monitoring_and_change_control",
            "operations",
            "No drift monitoring, incident response, rollback, or approved model-change plan is present.",
            "Define production monitoring, complaint handling, rollback, and change control before release.",
        ),
        (
            "regulatory_authorization",
            "regulatory",
            "No regulator authorization for a melanoma detection claim is present.",
            "Do not market the product for diagnosis or screening until the applicable authorization is obtained.",
        ),
    )
    for gate_id, category, evidence, remediation in manual_gates:
        gates.append(
            gate(
                gate_id,
                category,
                "manual_review_required",
                evidence,
                remediation,
            )
        )

    blocker_statuses = {"failed", "missing", "manual_review_required"}
    blockers = [item for item in gates if item["status"] in blocker_statuses]
    return {
        "assessment_time_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "overall_status": "NOT_MARKET_READY" if blockers else "TECHNICAL_GATES_PASSED",
        "release_blocked": bool(blockers),
        "warning": (
            "This automated audit is not regulatory approval, legal advice, clinical "
            "certification, or a substitute for an accredited quality system."
        ),
        "performance_targets": PERFORMANCE_TARGETS,
        "training_sources": sources,
        "training_source_manifest": str(source_manifest),
        "training_sources_bound_to_saved_split": saved_training_manifest.exists(),
        "license_inventory": LICENSE_INVENTORY,
        "summary": {
            "passed": sum(item["status"] == "passed" for item in gates),
            "failed": sum(item["status"] == "failed" for item in gates),
            "missing": sum(item["status"] == "missing" for item in gates),
            "manual_review_required": sum(
                item["status"] == "manual_review_required" for item in gates
            ),
        },
        "gates": gates,
    }


def write_markdown(path: Path, report: dict[str, object]) -> None:
    blockers = [
        item
        for item in report["gates"]
        if item["status"] != "passed"
    ]
    passed = [item for item in report["gates"] if item["status"] == "passed"]
    lines = [
        "# Market Readiness Assessment",
        "",
        f"**Status: {report['overall_status']}**",
        "",
        str(report["warning"]),
        "",
        "## Release blockers",
        "",
    ]
    for item in blockers:
        lines.append(
            f"- **{item['id']}** ({item['status']}): {item['evidence']} "
            f"Required: {item['remediation']}"
        )
    lines.extend(["", "## Controls passed", ""])
    for item in passed:
        lines.append(f"- **{item['id']}**: {item['evidence']}")
    lines.extend(
        [
            "",
            "## Regulatory basis",
            "",
            "- FDA/IMDRF Good Machine Learning Practice requires representative data, "
            "independent test sets, clinically relevant testing, clear user information, "
            "and deployed-model monitoring: "
            "https://www.fda.gov/medical-devices/software-medical-device-samd/"
            "good-machine-learning-practice-medical-device-development-guiding-principles",
            "- FDA's digital-health guidance index lists applicable AI lifecycle, clinical "
            "decision support, cybersecurity, and change-control guidance: "
            "https://www.fda.gov/medical-devices/digital-health-center-excellence/"
            "guidances-digital-health-content",
            "",
            "The current model remains a research prototype. It must not be represented as "
            "a diagnostic device or used to replace a dermatologist.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def format_metric(value: object) -> str:
    return "not measured" if value is None else f"{float(value):.3f}"


def write_model_card(path: Path, run_dir: Path) -> None:
    config = read_json(run_dir / "run_config.json") or {}
    finalization = read_json(run_dir / "finalization_status.json") or {}
    split_summary = read_json(run_dir / "split_summary.json") or {}
    internal = read_json(run_dir / "test_risk_metrics.json") or {}
    derm = read_json(run_dir / "derm12345_external_metrics.json") or {}
    ph2 = read_json(run_dir / "ph2_external_metrics.json") or {}
    decode_audit = read_json(run_dir / "test_jpeg_decode_findings.summary.json") or {}
    internal_operating = internal.get("threshold_metrics", {}).get("operating", {})
    derm_operating = (
        derm.get("patient_level_max_probability", {})
        .get("operating_threshold_metrics", {})
    )
    ph2_operating = ph2.get("operating_threshold_metrics", {})
    full_dataset = split_summary.get("full_dataset", {})
    lines = [
        "# Model Card",
        "",
        "## Status",
        "",
        "Research prototype only. Not clinically validated, regulator-authorized, or cleared "
        "for commercial deployment. It must not replace examination, biopsy, pathology, or "
        "a dermatologist's judgment.",
        "",
        "## Intended task",
        "",
        f"- Endpoint: {config.get('binary_target', 'unknown')} classification.",
        "- Input: one centered dermoscopic image of one lesion.",
        "- Output: model score and recorded threshold, when available; calibration quality "
        "and threshold-selection provenance require separate verification.",
        "- Unsupported inputs: ordinary phone photographs, overview/body images, multiple "
        "lesions, mucosal/ocular lesions, poor-quality captures, or images from an unvalidated "
        "device/workflow.",
        "",
        "## Model",
        "",
        f"- Backbone: {config.get('backbone', 'unknown')}.",
        f"- Input resolution: {config.get('image_size', 'unknown')} x "
        f"{config.get('image_size', 'unknown')} pixels.",
        f"- Primary evaluation unit: {config.get('primary_evaluation_unit', 'lesion')}.",
        f"- Test-time views: {finalization.get('tta_transforms', config.get('tta_transforms', 'unrecorded'))}.",
        "",
        "## Development data",
        "",
        f"- Images: {full_dataset.get('images', 'unknown')}.",
        f"- Lesions: {full_dataset.get('lesions', 'unknown')}.",
        f"- Split groups/patients: {full_dataset.get('split_groups', 'unknown')}.",
        f"- Sources: {json.dumps(full_dataset.get('sources', {}), sort_keys=True)}.",
        "- Patient independence, exact-image separation, and checkpoint-ancestor exposure "
        "require their recorded audits; this model card does not establish them. Available "
        "split groups are not necessarily patient identifiers.",
        f"- Held-out decode audit: {decode_audit.get('images_checked', 'not run')} images, "
        f"{decode_audit.get('decoder_warnings', 'unknown')} recoverable warnings, "
        f"{decode_audit.get('decode_errors', 'unknown')} hard failures.",
        "",
        "## Performance",
        "",
        "| Cohort/unit | Accuracy | Balanced accuracy | Sensitivity | Specificity |",
        "| --- | ---: | ---: | ---: | ---: |",
        "| Internal recorded lesion split (independence unverified) | "
        f"{format_metric(internal_operating.get('accuracy'))} | "
        f"{format_metric(internal_operating.get('balanced_accuracy'))} | "
        f"{format_metric(internal_operating.get('recall'))} | "
        f"{format_metric(internal_operating.get('specificity'))} |",
        "| DERM12345 development patients | "
        f"{format_metric(derm_operating.get('accuracy'))} | "
        f"{format_metric(derm_operating.get('balanced_accuracy'))} | "
        f"{format_metric(derm_operating.get('recall'))} | "
        f"{format_metric(derm_operating.get('specificity'))} |",
        "| PH2 images (research-only) | "
        f"{format_metric(ph2_operating.get('accuracy'))} | "
        f"{format_metric(ph2_operating.get('balanced_accuracy'))} | "
        f"{format_metric(ph2_operating.get('recall'))} | "
        f"{format_metric(ph2_operating.get('specificity'))} |",
        "",
        "Metrics use recorded thresholds; this card does not independently verify threshold "
        "selection or cohort independence. PH2 and DERM12345 have informed development and "
        "are not fresh confirmatory cohorts.",
        "",
        "## Known limitations",
        "",
        "- No prospectively powered multi-center clinical validation in the intended workflow.",
        "- No reliable skin-tone or capture-device labels for fairness and device-stratified testing.",
        "- No validated image-quality gate, out-of-distribution detector, or abstention mechanism.",
        "- Training labels and acquisition conditions differ by source and may encode shortcuts.",
        "- Melanoma prevalence is low; raw accuracy can be high while sensitivity is unacceptable.",
        "- HAM10000 and PH2 impose non-commercial restrictions relevant to product development.",
        "- Performance outside centered dermoscopy images is unknown.",
        "",
        "## Release rule",
        "",
        "Release is blocked whenever MARKET_READINESS.md reports NOT_MARKET_READY. A human "
        "regulatory, clinical, legal, security, and quality review remains mandatory even if "
        "all automated technical gates pass.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_evidence_manifest(path: Path, run_dir: Path) -> None:
    run_artifacts = (
        "best_model.keras",
        "run_config.json",
        "metadata_schema.json",
        "train_split.csv",
        "val_split.csv",
        "test_split.csv",
        "finalization_status.json",
        "model_selection.json",
        "calibration.json",
        "selected_threshold.json",
        "test_risk_metrics.json",
        "test_subgroup_metrics.json",
        "derm12345_external_metrics.json",
        "derm12345_external_exact_duplicates.csv",
        "ph2_external_metrics.json",
        "test_jpeg_decode_findings.csv",
        "test_jpeg_decode_findings.summary.json",
        "market_readiness_report.json",
        "MARKET_READINESS.md",
        "MODEL_CARD.md",
    )
    project_artifacts = (
        "train_multitask.py",
        "finalize_multitask_run.py",
        "evaluate_derm12345.py",
        "evaluate_multitask_ph2.py",
        "audit_jpeg_decode.py",
        "assess_market_readiness.py",
        "test_train_multitask_helpers.py",
        "requirements.txt",
        "requirements-gpu.txt",
    )
    files: dict[str, dict[str, object]] = {}
    for name in run_artifacts:
        artifact = run_dir / name
        if artifact.exists():
            files[f"run/{name}"] = {
                "bytes": int(artifact.stat().st_size),
                "sha256": sha256_file(artifact),
            }
    for name in project_artifacts:
        artifact = PROJECT_ROOT / name
        if artifact.exists():
            files[f"code/{name}"] = {
                "bytes": int(artifact.stat().st_size),
                "sha256": sha256_file(artifact),
            }
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Research evidence integrity manifest; not a regulatory signature.",
        "files": files,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    report = build_report(run_dir, args.manifest.resolve())
    output_json = run_dir / "market_readiness_report.json"
    output_md = run_dir / "MARKET_READINESS.md"
    model_card = run_dir / "MODEL_CARD.md"
    evidence_manifest = run_dir / "evidence_manifest.json"
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(output_md, report)
    write_model_card(model_card, run_dir)
    write_evidence_manifest(evidence_manifest, run_dir)
    print(json.dumps(report["summary"], indent=2))
    print("Overall status:", report["overall_status"])
    print("Report:", output_md)
    print("Model card:", model_card)
    print("Evidence manifest:", evidence_manifest)


if __name__ == "__main__":
    main()
