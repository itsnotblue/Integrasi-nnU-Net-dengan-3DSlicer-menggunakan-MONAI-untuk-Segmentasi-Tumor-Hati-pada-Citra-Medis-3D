"""Recover per-case statistics from saved nnU-Net summary; no new inference claim."""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from evaluation_common import assert_geometry, geometry, read_json, resolve_path, sha256, validate_labels, write_json_new
from evaluate_predictions import describe


def extract_cases(summary, manifest, manifest_path):
    locked = {case["case_id"]: case for case in manifest["cases"]}
    cases = []
    seen = set()
    for raw in summary["metric_per_case"]:
        prediction_id = Path(raw["prediction_file"].replace("\\", "/")).name.removesuffix(".nii.gz")
        reference_id = Path(raw["reference_file"].replace("\\", "/")).name.removesuffix(".nii.gz")
        if prediction_id != reference_id or prediction_id not in locked or prediction_id in seen:
            raise ValueError("Historical summary IDs differ, repeat, or are outside locked cohort")
        case = locked[prediction_id]
        reference_path = resolve_path(manifest_path, case["reference"])
        if sha256(reference_path) != case["reference_sha256"]:
            raise ValueError(f"{prediction_id}: reference changed after lock")
        reference_itk = sitk.ReadImage(str(reference_path))
        assert_geometry(geometry(reference_itk), case["geometry"], "reference")
        array = sitk.GetArrayFromImage(reference_itk)
        validate_labels(array, "reference")
        result = {"case_id": prediction_id, "classes": {}, "failure_flags": []}
        for label, name in (("1", "liver_label_1"), ("2", "tumor_label_2")):
            metrics = raw["metrics"][label]
            tp, fp, fn, tn = (int(metrics[key]) for key in ("TP", "FP", "FN", "TN"))
            if min(tp, fp, fn, tn) < 0 or tp + fp + fn + tn != array.size:
                raise ValueError(f"{prediction_id}: summary voxel totals do not match full reference")
            pred_count, ref_count = tp + fp, tp + fn
            if int(metrics["n_ref"]) != ref_count or int(metrics["n_pred"]) != pred_count:
                raise ValueError(f"{prediction_id}: inconsistent confusion counts")
            if np.count_nonzero(array == int(label)) != ref_count:
                raise ValueError(f"{prediction_id}: historical reference count differs from locked reference")
            dice = 2 * tp / (pred_count + ref_count) if pred_count + ref_count else 1.0
            if not np.isclose(dice, metrics["Dice"], atol=1e-10, rtol=0):
                raise ValueError(f"{prediction_id}: historical Dice differs from recomputed counts")
            result["classes"][name] = {
                "dice": dice, "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
                "precision": tp / pred_count if pred_count else None,
                "recall": tp / ref_count if ref_count else None,
                "prediction_voxels": pred_count, "reference_voxels": ref_count,
                "true_positive_voxels": tp, "false_positive_voxels": fp, "false_negative_voxels": fn,
            }
        tumor = result["classes"]["tumor_label_2"]
        if tumor["reference_voxels"] and not tumor["true_positive_voxels"]:
            result["failure_flags"].append("complete_tumor_miss")
        if tumor["reference_voxels"] and tumor["dice"] < 0.5:
            result["failure_flags"].append("tumor_dice_below_0.5")
        if not tumor["reference_voxels"] and tumor["prediction_voxels"]:
            result["failure_flags"].append("false_positive_only")
        if result["classes"]["liver_label_1"]["dice"] < 0.9:
            result["failure_flags"].append("liver_label_dice_below_0.9")
        seen.add(prediction_id)
        cases.append(result)
    if set(locked) != seen:
        raise ValueError("Historical summary does not include every locked case")
    return sorted(cases, key=lambda case: case["case_id"])


def summarize(cases, bootstrap_samples, seed):
    result = {}
    for name in ("liver_label_1", "tumor_label_2"):
        result[name] = {metric: describe([case["classes"][name][metric] for case in cases], bootstrap_samples, seed)
                        for metric in ("dice", "iou", "precision", "recall")}
    positive = [case for case in cases if case["classes"]["tumor_label_2"]["reference_voxels"]]
    result["tumor_label_2_reference_positive_cases"] = {
        metric: describe([case["classes"]["tumor_label_2"][metric] for case in positive], bootstrap_samples, seed)
        for metric in ("dice", "iou", "precision", "recall")
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--comparison-summary")
    parser.add_argument("--comparison-model", default="model_c")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("Use at least 100 bootstrap replicates")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output directory")
    manifest = read_json(args.manifest)
    cases = extract_cases(read_json(args.summary), manifest, args.manifest)
    report = {
        "schema_version": 1, "analysis_kind": "historical_summary_reanalysis",
        "analyzed_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "source_summary_sha256": sha256(args.summary),
        "manifest_sha256": sha256(args.manifest), "cohort_id": manifest["cohort_id"],
        "case_count": len(cases), "split_evidence": manifest["split_evidence"],
        "held_out_certified": False,
        "validation": "Confusion counts, total voxels, per-label reference counts and Dice checked against locked complete local reference volumes",
        "limitations": [
            "Historical prediction files were not rerun or independently rehashed by this analysis",
            "Counts confirm compatible references but do not prove spatial prediction/reference alignment",
            "Checkpoint used in the old run requires authoritative provenance; no inferred checkpoint hash attached",
            "Surface metrics, combined liver-region Dice and lesion detection cannot be reconstructed from classwise confusion counts",
            "These are previously evaluated cases; confidence intervals describe this cohort, not external generalization",
        ],
        "bootstrap": {"samples": args.bootstrap_samples, "seed": args.seed, "unit": "case", "method": "percentile 95% CI of macro mean"},
        "summary": summarize(cases, args.bootstrap_samples, args.seed), "per_case": cases,
    }
    if args.comparison_summary:
        comparison = extract_cases(read_json(args.comparison_summary), manifest, args.manifest)
        report["comparison"] = {"model": args.comparison_model, "source_summary_sha256": sha256(args.comparison_summary),
                                "summary": summarize(comparison, args.bootstrap_samples, args.seed), "per_case": comparison}
        report["paired_difference_primary_minus_comparison"] = {
            name: {metric: describe([first["classes"][name][metric] - second["classes"][name][metric]
                                    if first["classes"][name][metric] is not None and second["classes"][name][metric] is not None else None
                                    for first, second in zip(cases, comparison)], args.bootstrap_samples, args.seed)
                   for metric in ("dice", "iou", "precision", "recall")}
            for name in ("liver_label_1", "tumor_label_2")
        }
    output.mkdir(parents=True, exist_ok=True)
    write_json_new(output / "historical_evaluation.json", report)
    models = [(args.model, cases)]
    if args.comparison_summary:
        models.append((args.comparison_model, comparison))
    rows = []
    for model, model_cases in models:
        for case in model_cases:
            row = {"model": model, "case_id": case["case_id"], "failure_flags": ";".join(case["failure_flags"])}
            for name, metrics in case["classes"].items():
                row.update({f"{name}_{key}": value for key, value in metrics.items()})
            rows.append(row)
    with (output / "historical_per_case_metrics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "historical_failures.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("model", "case_id", "failure_flags", "liver_dice", "tumor_dice", "tumor_precision", "tumor_recall"))
        writer.writeheader()
        for row in sorted(rows, key=lambda value: (value["model"], value["tumor_label_2_dice"])):
            writer.writerow({"model": row["model"], "case_id": row["case_id"], "failure_flags": row["failure_flags"],
                             "liver_dice": row["liver_label_1_dice"], "tumor_dice": row["tumor_label_2_dice"],
                             "tumor_precision": row["tumor_label_2_precision"], "tumor_recall": row["tumor_label_2_recall"]})
    print(f"Validated and summarized {len(cases)} historical cases: {output}")
    print(f"Tumor Dice mean/CI: {report['summary']['tumor_label_2']['dice']}")


if __name__ == "__main__":
    main()
