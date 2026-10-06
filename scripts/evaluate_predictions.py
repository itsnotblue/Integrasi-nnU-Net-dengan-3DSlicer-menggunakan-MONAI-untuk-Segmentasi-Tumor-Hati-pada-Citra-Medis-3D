"""Evaluate locked full-volume segmentations; never crop/resample predictions."""

import argparse
import csv
from datetime import datetime, timezone
import math
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree
import SimpleITK as sitk

from evaluation_common import (
    assert_geometry, geometry, read_json, resolve_path, sha256,
    validate_labels, write_json_new,
)


CLASS_LABELS = {"liver_label_1": (1,), "tumor_label_2": (2,), "liver_region_1_or_2": (1, 2)}


def binary_metrics(prediction, reference, spacing_zyx):
    tp = int(np.count_nonzero(prediction & reference))
    pred_count, ref_count = int(prediction.sum()), int(reference.sum())
    fp, fn = pred_count - tp, ref_count - tp
    total = pred_count + ref_count
    metrics = {
        "dice": 2 * tp / total if total else 1.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "precision": tp / pred_count if pred_count else None,
        "recall": tp / ref_count if ref_count else None,
        "prediction_voxels": pred_count, "reference_voxels": ref_count,
        "true_positive_voxels": tp, "false_positive_voxels": fp,
        "false_negative_voxels": fn,
        "prediction_volume_ml": pred_count * float(np.prod(spacing_zyx)) / 1000,
        "reference_volume_ml": ref_count * float(np.prod(spacing_zyx)) / 1000,
        "empty_status": "both_empty" if not total else "prediction_empty" if not pred_count else "reference_empty" if not ref_count else "neither_empty",
    }
    if not total:
        metrics.update(hd95_mm=0.0, assd_mm=0.0, surface_status="both_empty")
    elif not pred_count or not ref_count:
        metrics.update(hd95_mm=None, assd_mm=None, surface_status="undefined_one_empty")
    else:
        # Limit temporary metric arrays to the foreground union bounding box.
        # This does not change the input/prediction or evaluation voxel counts.
        regions = ndimage.find_objects((prediction | reference).astype(np.uint8))
        bounds = tuple(slice(max(0, part.start - 1), min(prediction.shape[i], part.stop + 1))
                       for i, part in enumerate(regions[0]))
        pred, ref = prediction[bounds], reference[bounds]
        structure = ndimage.generate_binary_structure(3, 1)
        pred_surface = pred ^ ndimage.binary_erosion(pred, structure=structure, border_value=0)
        ref_surface = ref ^ ndimage.binary_erosion(ref, structure=structure, border_value=0)
        pred_points = np.argwhere(pred_surface) * np.asarray(spacing_zyx)
        ref_points = np.argwhere(ref_surface) * np.asarray(spacing_zyx)
        # Exact nearest boundary-voxel distances; avoid volume-sized float64
        # distance transforms on hundreds-of-slice CTs. Both sets use the same
        # translated coordinate frame, so bounding-box translation cancels.
        pred_distances = cKDTree(ref_points).query(pred_points, k=1, eps=0, workers=1)[0]
        ref_distances = cKDTree(pred_points).query(ref_points, k=1, eps=0, workers=1)[0]
        distances = np.concatenate((pred_distances, ref_distances))
        metrics.update(hd95_mm=float(np.percentile(distances, 95)),
                       assd_mm=float(np.mean(distances)), surface_status="measured")
    return metrics


def lesion_metrics(prediction, reference, spacing_zyx, iou_threshold=0.1):
    """26-connected lesions, one-to-one maximum-cardinality IoU matching."""
    structure = ndimage.generate_binary_structure(3, 3)
    pred_components, pred_n = ndimage.label(prediction, structure)
    ref_components, ref_n = ndimage.label(reference, structure)
    pred_sizes = np.bincount(pred_components.ravel(), minlength=pred_n + 1)
    ref_sizes = np.bincount(ref_components.ravel(), minlength=ref_n + 1)
    overlap = (pred_components > 0) & (ref_components > 0)
    pair_codes, counts = np.unique(
        ref_components[overlap].astype(np.int64) * (pred_n + 1) + pred_components[overlap],
        return_counts=True,
    )
    edges, pair_ious = [], {}
    for code, count in zip(pair_codes, counts):
        ref_id, pred_id = divmod(int(code), pred_n + 1)
        iou = int(count) / int(ref_sizes[ref_id] + pred_sizes[pred_id] - count)
        pair_ious[(ref_id, pred_id)] = float(iou)
        if iou >= iou_threshold:
            edges.append((ref_id - 1, pred_id - 1))
    matches = np.full(ref_n, -1, dtype=int)
    if ref_n and pred_n and edges:
        rows, columns = zip(*edges)
        graph = csr_matrix((np.ones(len(edges), dtype=np.uint8), (rows, columns)), shape=(ref_n, pred_n))
        matches = maximum_bipartite_matching(graph, perm_type="column")
    matched_pred = {int(value) + 1 for value in matches if value >= 0}
    voxel_volume = float(np.prod(spacing_zyx))
    details = []
    for ref_id in range(1, ref_n + 1):
        pred_id = int(matches[ref_id - 1]) + 1
        details.append({"reference_lesion_id": ref_id, "reference_voxels": int(ref_sizes[ref_id]),
                        "reference_volume_mm3": int(ref_sizes[ref_id]) * voxel_volume,
                        "detected": pred_id > 0, "matched_prediction_id": pred_id or None,
                        "matched_iou": pair_ious.get((ref_id, pred_id)) if pred_id else None})
    tp = len(matched_pred)
    return {
        "reference_lesions": ref_n, "predicted_lesions": pred_n, "matched_lesions": tp,
        "missed_lesions": ref_n - tp, "false_positive_lesions": pred_n - tp,
        "lesion_recall": tp / ref_n if ref_n else None,
        "lesion_precision": tp / pred_n if pred_n else None,
        "case_tumor_detected": bool(np.any(prediction & reference)) if ref_n else None,
        "reference_lesion_details": details,
        "false_positive_prediction_ids": sorted(set(range(1, pred_n + 1)) - matched_pred),
    }


def describe(values, bootstrap_samples=2000, seed=20261006):
    finite = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    result = {"n_total": len(values), "n_defined": len(finite), "n_undefined": len(values) - len(finite),
              "mean": None, "sample_sd": None, "median": None, "q25": None, "q75": None,
              "mean_ci95": None}
    if len(finite):
        result.update(mean=float(finite.mean()), median=float(np.median(finite)),
                      q25=float(np.percentile(finite, 25)), q75=float(np.percentile(finite, 75)))
    if len(finite) >= 2:
        means = np.random.default_rng(seed).choice(finite, (bootstrap_samples, len(finite)), replace=True).mean(axis=1)
        result.update(sample_sd=float(finite.std(ddof=1)),
                      mean_ci95=[float(value) for value in np.percentile(means, [2.5, 97.5])])
    return result


def summarize(cases, bootstrap_samples=2000, seed=20261006):
    metric_names = ("dice", "iou", "precision", "recall", "hd95_mm", "assd_mm")
    classes = {name: {metric: describe([case["classes"][name][metric] for case in cases], bootstrap_samples, seed)
                      for metric in metric_names} for name in CLASS_LABELS}
    positive = [case for case in cases if case["classes"]["tumor_label_2"]["reference_voxels"] > 0]
    classes["tumor_label_2_reference_positive_cases"] = {
        metric: describe([case["classes"]["tumor_label_2"][metric] for case in positive], bootstrap_samples, seed)
        for metric in metric_names
    }
    lesion = {key: sum(case["lesions"][key] for case in cases)
              for key in ("reference_lesions", "predicted_lesions", "matched_lesions", "missed_lesions", "false_positive_lesions")}
    lesion["pooled_lesion_recall"] = lesion["matched_lesions"] / lesion["reference_lesions"] if lesion["reference_lesions"] else None
    lesion["pooled_lesion_precision"] = lesion["matched_lesions"] / lesion["predicted_lesions"] if lesion["predicted_lesions"] else None
    # Resample patients/cases, preserving all their lesions within a bootstrap unit.
    for numerator, denominator, metric in (
            ("matched_lesions", "reference_lesions", "pooled_lesion_recall_ci95"),
            ("matched_lesions", "predicted_lesions", "pooled_lesion_precision_ci95")):
        lesion[metric] = None
        if len(cases) >= 2:
            sampled = np.random.default_rng(seed).integers(0, len(cases), (bootstrap_samples, len(cases)))
            numerators = np.array([case["lesions"][numerator] for case in cases])[sampled].sum(axis=1)
            denominators = np.array([case["lesions"][denominator] for case in cases])[sampled].sum(axis=1)
            defined = denominators > 0
            if np.any(defined):
                lesion[metric] = [float(value) for value in np.percentile(numerators[defined] / denominators[defined], [2.5, 97.5])]
    return {"cases": len(cases), "tumor_reference_positive_cases": len(positive),
            "classes": classes, "lesions": lesion}


def evaluate_case(case, manifest_path, prediction_path, iou_threshold):
    image_path = resolve_path(manifest_path, case["image"])
    reference_path = resolve_path(manifest_path, case["reference"])
    for key, path in (("image", image_path), ("reference", reference_path)):
        if sha256(path) != case[f"{key}_sha256"]:
            raise ValueError(f"{case['case_id']}: {key} changed after cohort lock")
    image_itk = sitk.ReadImage(str(image_path))
    reference_itk, prediction_itk = sitk.ReadImage(str(reference_path)), sitk.ReadImage(str(prediction_path))
    for name, itk in (("image", image_itk), ("reference", reference_itk), ("prediction", prediction_itk)):
        assert_geometry(geometry(itk), case["geometry"], f"{case['case_id']} {name}")
    prediction, reference = sitk.GetArrayFromImage(prediction_itk), sitk.GetArrayFromImage(reference_itk)
    prediction_labels, reference_labels = validate_labels(prediction, "prediction"), validate_labels(reference, "reference")
    spacing_zyx = tuple(reversed(reference_itk.GetSpacing()))
    classes = {name: binary_metrics(np.isin(prediction, labels), np.isin(reference, labels), spacing_zyx)
               for name, labels in CLASS_LABELS.items()}
    lesions = lesion_metrics(prediction == 2, reference == 2, spacing_zyx, iou_threshold)
    tumor = classes["tumor_label_2"]
    flags = []
    if tumor["reference_voxels"] and not tumor["true_positive_voxels"]:
        flags.append("complete_tumor_miss")
    if not tumor["reference_voxels"] and tumor["prediction_voxels"]:
        flags.append("false_positive_only")
    if tumor["reference_voxels"] and tumor["dice"] < 0.5:
        flags.append("tumor_dice_below_0.5")
    if lesions["missed_lesions"]:
        flags.append("unmatched_reference_lesions")
    if lesions["false_positive_lesions"]:
        flags.append("unmatched_prediction_lesions")
    if classes["liver_region_1_or_2"]["dice"] < 0.9:
        flags.append("liver_region_dice_below_0.9")
    return {"case_id": case["case_id"], "prediction_sha256": sha256(prediction_path),
            "prediction_labels": prediction_labels, "reference_labels": reference_labels,
            "geometry": case["geometry"], "classes": classes, "lesions": lesions,
            "failure_flags": flags}


def export_csv(output_dir, cases, prefix=""):
    rows = []
    for case in cases:
        row = {"case_id": case["case_id"], "failure_flags": ";".join(case["failure_flags"])}
        for name, metrics in case["classes"].items():
            row.update({f"{name}_{key}": value for key, value in metrics.items()})
        row.update({f"lesion_{key}": value for key, value in case["lesions"].items()
                    if key not in ("reference_lesion_details", "false_positive_prediction_ids")})
        rows.append(row)
    with (output_dir / f"{prefix}per_case_metrics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / f"{prefix}failures.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("case_id", "failure_flags", "tumor_dice", "missed_lesions", "false_positive_lesions"))
        writer.writeheader()
        for case in sorted(cases, key=lambda value: value["classes"]["tumor_label_2"]["dice"]):
            writer.writerow({"case_id": case["case_id"], "failure_flags": ";".join(case["failure_flags"]),
                             "tumor_dice": case["classes"]["tumor_label_2"]["dice"],
                             "missed_lesions": case["lesions"]["missed_lesions"],
                             "false_positive_lesions": case["lesions"]["false_positive_lesions"]})


def verify_historical_counts(result, raw):
    for label, name in (("1", "liver_label_1"), ("2", "tumor_label_2")):
        current, previous = result["classes"][name], raw["metrics"][label]
        mapping = {"TP": "true_positive_voxels", "FP": "false_positive_voxels", "FN": "false_negative_voxels",
                   "n_pred": "prediction_voxels", "n_ref": "reference_voxels"}
        if any(int(previous[source]) != current[target] for source, target in mapping.items()):
            raise ValueError(f"{result['case_id']}: historical label {label} confusion counts differ")
        total = int(np.prod(result["geometry"]["size_xyz"]))
        tn = total - current["true_positive_voxels"] - current["false_positive_voxels"] - current["false_negative_voxels"]
        if tn != int(previous["TN"]):
            raise ValueError(f"{result['case_id']}: historical label {label} TN differs")
        for source, target in (("Dice", "dice"), ("IoU", "iou")):
            if current["prediction_voxels"] + current["reference_voxels"] and not np.isclose(previous[source], current[target], rtol=0, atol=1e-10):
                raise ValueError(f"{result['case_id']}: historical label {label} {source} differs")
    result["historical_confusion_counts_verified"] = True


def validate_cohort_run(record, manifest, manifest_hash, checkpoint_hash):
    """Reject a partial/misattributed batch before any expensive image read."""
    if record.get("completed") is not True or not record.get("completed_at_utc"):
        raise ValueError("Cohort inference is incomplete; do not evaluate a partial batch")
    if record.get("manifest_sha256") != manifest_hash or record.get("cohort_id") != manifest["cohort_id"]:
        raise ValueError("Cohort run does not match the locked manifest")
    if record.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Cohort run checkpoint differs from supplied checkpoint")
    if record.get("reference_annotation_used") is not False:
        raise ValueError("Cohort run lacks explicit no-reference-annotation provenance")
    rows = record.get("cases", [])
    run_cases = {row["case_id"]: row for row in rows}
    if len(run_cases) != len(rows) or set(run_cases) != {case["case_id"] for case in manifest["cases"]}:
        raise ValueError("Completed run case membership differs from locked cohort")
    if not isinstance(record.get("settings"), dict) or record.get("resolved_device") not in ("cpu", "cuda"):
        raise ValueError("Completed run lacks settings/resolved device provenance")
    return run_cases


def validate_prediction_metadata(metadata, case, checkpoint_hash, output_hash, run_record=None, run_case=None):
    if metadata.get("input_sha256") != case["image_sha256"]:
        raise ValueError(f"{case['case_id']}: prediction sidecar input hash differs from locked image")
    if metadata.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError(f"{case['case_id']}: prediction sidecar checkpoint differs from supplied checkpoint")
    if metadata.get("output_sha256") != output_hash:
        raise ValueError(f"{case['case_id']}: prediction sidecar output hash mismatch")
    if metadata.get("case_id", case["case_id"]) != case["case_id"]:
        raise ValueError("Prediction sidecar case ID differs")
    if metadata.get("manual_roi") is not False or metadata.get("reference_annotation_used") is not False:
        raise ValueError("Prediction lacks explicit full-volume/no-reference provenance")
    if metadata.get("input_size_xyz") != case["geometry"]["size_xyz"]:
        raise ValueError("Prediction sidecar input dimensions differ from full locked image")
    try:
        declared = {"size_xyz": metadata["output_size_xyz"], "spacing_xyz_mm": metadata["spacing_xyz"],
                    "origin_xyz_mm": metadata["origin_xyz"], "direction": metadata["direction"]}
    except KeyError as exc:
        raise ValueError("Prediction sidecar lacks physical geometry") from exc
    assert_geometry(declared, case["geometry"], "Prediction sidecar")
    seconds = metadata.get("end_to_end_seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (float, int)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Prediction sidecar timing is invalid")
    settings = metadata.get("settings", {})
    if settings.get("profile") not in ("reference", "fast") or settings.get("device") not in ("cpu", "cuda"):
        raise ValueError("Prediction sidecar lacks a supported profile/resolved device")
    if settings.get("use_mirroring") is not (settings["profile"] == "reference"):
        raise ValueError("Prediction sidecar mirroring disagrees with profile")
    step, threads = settings.get("tile_step_size"), settings.get("threads")
    if isinstance(step, bool) or not isinstance(step, (float, int)) or not 0 < step <= 1 or type(threads) is not int or threads < 1:
        raise ValueError("Prediction sidecar tile/thread settings are invalid")
    if not isinstance(metadata.get("labels"), list) or not metadata["labels"] or any(type(value) is not int or value not in (0, 1, 2) for value in metadata["labels"]):
        raise ValueError("Prediction sidecar lacks valid label declarations")
    if run_record is not None:
        expected = run_record["settings"]
        for key in ("profile", "tile_step_size", "threads"):
            if key not in expected or settings.get(key) != expected[key]:
                raise ValueError(f"Prediction sidecar {key} differs from cohort run")
        if settings["device"] != run_record["resolved_device"]:
            raise ValueError("Prediction sidecar device differs from cohort run")
        if run_case["output_sha256"] != output_hash or run_case["seconds"] != seconds:
            raise ValueError("Prediction sidecar output/timing differs from completed run")
    return metadata


def preflight_current_predictions(manifest, manifest_path, predictions, checkpoint_hash, run_record=None):
    if manifest.get("input_extent") != "full_volume_as_supplied":
        raise ValueError("Current attributed evaluation requires a locked full-volume cohort")
    run_cases = validate_cohort_run(run_record, manifest, sha256(manifest_path), checkpoint_hash) if run_record is not None else None
    validated = {}
    for case in manifest["cases"]:
        prediction = predictions[case["case_id"]]
        sidecar = Path(str(prediction) + ".json")
        if not sidecar.is_file():
            raise ValueError(f"{case['case_id']}: current prediction requires its output provenance sidecar")
        metadata = read_json(sidecar)
        validate_prediction_metadata(metadata, case, checkpoint_hash, sha256(prediction),
                                     run_record, run_cases[case["case_id"]] if run_cases is not None else None)
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(prediction))
        reader.ReadImageInformation()
        if reader.GetDimension() != 3 or reader.GetNumberOfComponents() != 1:
            raise ValueError("Current prediction is not a scalar three-dimensional labelmap")
        actual_geometry = {"size_xyz": list(reader.GetSize()), "spacing_xyz_mm": list(reader.GetSpacing()),
                           "origin_xyz_mm": list(reader.GetOrigin()), "direction": list(reader.GetDirection())}
        assert_geometry(actual_geometry, case["geometry"], "Current prediction header")
        validated[case["case_id"]] = {"sidecar_sha256": sha256(sidecar), "contents": metadata}
    settings = [record["contents"]["settings"] for record in validated.values()]
    if any(value != settings[0] for value in settings[1:]):
        raise ValueError("Current cohort mixes inference settings across cases")
    return validated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--prediction-pattern", default="{case_id}.nii.gz")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", help="Required for current predictions; do not infer an old checkpoint identity from a filename")
    parser.add_argument("--historical-predictions", action="store_true", help="Old masks with unknown checkpoint hash; forbids --checkpoint")
    parser.add_argument("--historical-summary", help="Verify every current confusion count against this saved nnU-Net summary")
    parser.add_argument("--inference-settings", help="Current mode: optional COMPLETED run_cohort_inference.py report; historical mode: old inference metadata")
    parser.add_argument("--lesion-iou-threshold", type=float, default=0.1)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args()
    if not 0 < args.lesion_iou_threshold <= 1 or args.bootstrap_samples < 100:
        parser.error("IoU must be in (0, 1]; bootstrap samples must be >= 100")
    if args.historical_predictions and args.checkpoint:
        parser.error("Historical unpinned predictions must not be attributed to a supplied checkpoint")
    if not args.historical_predictions and not args.checkpoint:
        parser.error("Current predictions require --checkpoint")
    manifest = read_json(args.manifest)
    if manifest.get("schema_version") != 1 or not manifest.get("cases"):
        raise ValueError("Expected nonempty cohort manifest schema 1")
    ids = [case["case_id"] for case in manifest["cases"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest contains duplicate cases")
    if any(not case_id or Path(case_id).name != case_id or any(character in case_id for character in ("/", "\\", ":")) for case_id in ids):
        raise ValueError("Manifest contains unsafe case IDs")
    predictions = {case_id: Path(args.prediction_dir) / args.prediction_pattern.format(case_id=case_id) for case_id in ids}
    missing = [case_id for case_id, path in predictions.items() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ValueError(f"Incomplete prediction cohort; missing/empty files: {missing}")
    inference_record = read_json(args.inference_settings) if args.inference_settings else None
    checkpoint_hash = sha256(args.checkpoint) if args.checkpoint else None
    current_provenance = None
    if not args.historical_predictions:
        current_provenance = preflight_current_predictions(manifest, args.manifest, predictions, checkpoint_hash, inference_record)
    historical = None
    if args.historical_summary:
        historical = {}
        for raw in read_json(args.historical_summary)["metric_per_case"]:
            prediction_id = Path(raw["prediction_file"].replace("\\", "/")).name.removesuffix(".nii.gz")
            reference_id = Path(raw["reference_file"].replace("\\", "/")).name.removesuffix(".nii.gz")
            if prediction_id != reference_id or prediction_id in historical:
                raise ValueError("Historical summary IDs differ or repeat")
            historical[prediction_id] = raw
        if set(historical) != set(ids):
            raise ValueError("Historical summary membership differs from locked cohort")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new output directory; refusing to overwrite {output}")
    cases = []
    for case in manifest["cases"]:
        prediction = predictions[case["case_id"]]
        result = evaluate_case(case, args.manifest, prediction, args.lesion_iou_threshold)
        if current_provenance is not None:
            provenance = current_provenance[case["case_id"]]
            if result["prediction_sha256"] != provenance["contents"]["output_sha256"] or sorted(result["prediction_labels"]) != sorted(provenance["contents"]["labels"]):
                raise ValueError("Evaluated prediction differs from validated sidecar")
            result["prediction_provenance"] = provenance
        if historical is not None:
            verify_historical_counts(result, historical[case["case_id"]])
        cases.append(result)
        print(f"{case['case_id']}: tumor Dice {result['classes']['tumor_label_2']['dice']:.4f}; flags {result['failure_flags']}", flush=True)
    report = {
        "schema_version": 1, "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "checkpoint_sha256": checkpoint_hash,
        "checkpoint_attribution": "historical_unpinned" if args.historical_predictions else "verified_prediction_sidecars",
        "manifest_sha256": sha256(args.manifest), "cohort_id": manifest["cohort_id"],
        "purpose": manifest["purpose"], "split_evidence": manifest["split_evidence"],
        "held_out_certified": False,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "simpleitk": sitk.Version_VersionString()},
        "protocol": {
            "label_definitions": CLASS_LABELS,
            "empty_policy": "Both empty: Dice/IoU=1, HD95/ASSD=0. Zero precision/recall denominator or exactly one empty surface: null. Report undefined counts; report tumor-positive subset separately.",
            "surface": "6-connected boundary voxels; HD95 of pooled bidirectional distances; ASSD pooled mean; physical millimetres",
            "lesions": "26-connected components; one-to-one maximum-cardinality match; no small-component filtering",
            "lesion_iou_threshold": args.lesion_iou_threshold,
            "bootstrap": "Percentile 95% CI of macro mean, case unit, undefined values excluded; no CI for n<2. Pooled lesion ratios resample whole cases.",
            "bootstrap_samples": args.bootstrap_samples, "bootstrap_seed": args.seed,
        },
        "summary": summarize(cases, args.bootstrap_samples, args.seed), "per_case": cases,
    }
    if args.inference_settings:
        report["inference_settings"] = {"sha256": sha256(args.inference_settings), "contents": inference_record}
    if current_provenance is not None:
        if sha256(args.checkpoint) != checkpoint_hash:
            raise ValueError("Supplied checkpoint changed during evaluation")
        report["current_provenance_verified_all_cases"] = True
        report["completed_cohort_run_verified"] = inference_record is not None
    if args.historical_summary:
        report["historical_summary_sha256"] = sha256(args.historical_summary)
        report["historical_confusion_counts_verified_all_cases"] = True
    output.mkdir(parents=True, exist_ok=True)
    write_json_new(output / "evaluation.json", report)
    export_csv(output, cases)
    print(f"Saved evaluation.json, per_case_metrics.csv, failures.csv in {output}")


if __name__ == "__main__":
    main()
