"""Export allowlisted LiTS metrics without private paths, scans or annotations."""

import argparse
import csv
from pathlib import Path
import re

from evaluation_common import read_json, sha256, write_json_new
from evaluate_predictions import CLASS_LABELS, describe, export_csv


METRIC_KEYS = (
    "dice", "iou", "precision", "recall", "prediction_voxels", "reference_voxels",
    "true_positive_voxels", "false_positive_voxels", "false_negative_voxels",
    "prediction_volume_ml", "reference_volume_ml", "empty_status", "hd95_mm",
    "assd_mm", "surface_status",
)
COMPONENT_KEYS = (
    "reference_lesions", "predicted_lesions", "matched_lesions", "missed_lesions",
    "false_positive_lesions", "lesion_recall", "lesion_precision", "case_tumor_detected",
)
SETTINGS_KEYS = ("profile", "device", "use_mirroring", "tile_step_size", "threads")
SAFE_CASE = re.compile(r"LITS_\d{3}\Z")


def public_fast_report(current, historical, paired):
    """Fail closed on unverified current results; do not copy raw provenance."""
    if (current.get("checkpoint_attribution") != "verified_prediction_sidecars"
            or current.get("current_provenance_verified_all_cases") is not True
            or current.get("completed_cohort_run_verified") is not True):
        raise ValueError("Public current report requires verified sidecars and completed cohort")
    if historical.get("checkpoint_attribution") != "historical_unpinned" or historical.get("checkpoint_sha256") is not None:
        raise ValueError("Historical comparator must explicitly retain unknown checkpoint identity")
    if current["manifest_sha256"] != historical["manifest_sha256"] or current["protocol"] != historical["protocol"]:
        raise ValueError("Public comparison requires the same cohort and metric protocol")
    ids = [case["case_id"] for case in current["per_case"]]
    if (len(ids) != len(set(ids)) or any(not SAFE_CASE.fullmatch(case_id) for case_id in ids)
            or sorted(ids) != paired["case_ids"]
            or sorted(ids) != sorted(case["case_id"] for case in historical["per_case"])):
        raise ValueError("Expected unique anonymized LiTS case IDs and exact paired membership")
    if (paired.get("left_checkpoint_sha256") != current["checkpoint_sha256"]
            or paired.get("left_checkpoint_attribution") != "verified_prediction_sidecars"
            or paired.get("right_checkpoint_sha256") is not None
            or paired.get("right_checkpoint_attribution") != "historical_unpinned"):
        raise ValueError("Paired attribution differs from the evaluated artifacts")
    cases = []
    for case in current["per_case"]:
        metadata = case["prediction_provenance"]["contents"]
        cases.append({
            "case_id": case["case_id"], "prediction_sha256": case["prediction_sha256"],
            "prediction_sidecar_sha256": case["prediction_provenance"]["sidecar_sha256"],
            "end_to_end_seconds": metadata["end_to_end_seconds"],
            "classes": {name: {key: case["classes"][name][key] for key in METRIC_KEYS} for name in CLASS_LABELS},
            "lesions": {key: case["lesions"][key] for key in COMPONENT_KEYS},
            "failure_flags": case["failure_flags"],
        })
    settings = {key: current["per_case"][0]["prediction_provenance"]["contents"]["settings"][key] for key in SETTINGS_KEYS}
    return {
        "schema_version": 1, "analysis_kind": "current_pinned_fast_complete_volume_retrospective_audit",
        "cohort_id": current["cohort_id"], "manifest_sha256": current["manifest_sha256"],
        "held_out_certified": False,
        "limitations": [
            "Previously exposed internal thesis test cases; not a new untouched or external test.",
            "Historical comparator checkpoint identity is unknown; this is not a controlled same-weights fast-versus-reference equivalence experiment.",
            "26-connected annotation islands are engineering components, not adjudicated distinct clinical lesions.",
            "No retraining, annotation-derived inference ROI, equivalence, superiority or clinical-readiness claim.",
        ],
        "current": {
            "model": current["model"], "checkpoint_sha256": current["checkpoint_sha256"],
            "checkpoint_attribution": current["checkpoint_attribution"],
            "current_provenance_verified_all_cases": True, "completed_cohort_run_verified": True,
            "inference_settings": settings, "cohort_run_sha256": current["inference_settings"]["sha256"],
            "software": current["software"], "summary": current["summary"], "per_case": cases,
            "end_to_end_seconds": describe([case["end_to_end_seconds"] for case in cases]),
        },
        "historical_comparator": {
            "model": historical["model"], "checkpoint_sha256": None,
            "checkpoint_attribution": "historical_unpinned", "summary": historical["summary"],
        },
        "metric_protocol": current["protocol"],
        "paired": {key: paired[key] for key in (
            "left_model", "right_model", "difference_direction", "left_checkpoint_sha256",
            "right_checkpoint_sha256", "left_checkpoint_attribution", "right_checkpoint_attribution",
            "cohort_id", "case_ids", "held_out_certified", "bootstrap", "paired_differences",
            "left_evaluation_sha256", "right_evaluation_sha256",
        )},
    }


def export_components(path, current):
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        fields = ("case_id", "reference_component_id", "reference_voxels", "reference_volume_mm3", "detected", "matched_prediction_component_id", "matched_iou")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in current["per_case"]:
            for component in case["lesions"]["reference_lesion_details"]:
                writer.writerow(dict(zip(fields, (
                    case["case_id"], component["reference_lesion_id"], component["reference_voxels"],
                    component["reference_volume_mm3"], component["detected"],
                    component["matched_prediction_id"], component["matched_iou"],
                ))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", required=True)
    parser.add_argument("--historical", required=True)
    parser.add_argument("--paired", required=True)
    parser.add_argument("--output-dir", required=True, help="Directory for four new sanitized result files; existing files are not overwritten")
    args = parser.parse_args()
    current, historical, paired = read_json(args.current), read_json(args.historical), read_json(args.paired)
    if paired["left_evaluation_sha256"] != sha256(args.current) or paired["right_evaluation_sha256"] != sha256(args.historical):
        raise ValueError("Paired comparison hashes do not match the source evaluation reports")
    public = public_fast_report(current, historical, paired)
    output = Path(args.output_dir)
    targets = ("current_fast_statistics.json", "current_fast_per_case_metrics.csv", "current_fast_failures.csv", "current_fast_component_details.csv")
    if any((output / name).exists() for name in targets):
        raise FileExistsError("Public result artifacts already exist; refusing to overwrite them")
    output.mkdir(parents=True, exist_ok=True)
    write_json_new(output / "current_fast_statistics.json", public)
    export_csv(output, public["current"]["per_case"], prefix="current_fast_")
    export_components(output / "current_fast_component_details.csv", current)
    print("Saved allowlisted statistics, per-case metrics, failure and component tables; no raw provenance paths or images.")


if __name__ == "__main__":
    main()
