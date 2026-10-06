"""Validate and summarize a completed locked-cohort run without reading CTs/references."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


TIMINGS = ("preprocessing_seconds", "prediction_seconds", "reconstruction_seconds",
           "runtime_seconds_before_file_save", "end_to_end_seconds")
METRICS = TIMINGS + ("outside_predict_seconds", "other_runtime_seconds", "tiles",
                    "peak_cuda_allocated_bytes", "peak_cuda_allocated_gib")
CACHE_STATES = ("cold_first_new_case", "cached_runtime_case",
                "reused_original_cache_state_unknown")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique_keys,
                       parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def valid_hash(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise ValueError(f"Missing or invalid SHA-256: {label}")
    return value


def number(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"Invalid finite numeric value: {label}")
    return value


def positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Expected a positive integer: {label}")
    return value


def case_ids(cases, label):
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"Expected nonempty {label} cases")
    ids = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError(f"Invalid {label} case")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", case_id) is None:
            raise ValueError(f"Unsafe or missing {label} case ID")
        ids.append(case_id)
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate {label} case IDs")
    return ids


def timestamp(value, label):
    if not isinstance(value, str):
        raise ValueError(f"Missing timestamp: {label}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid timestamp: {label}") from exc
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"Expected a UTC timestamp: {label}")
    return parsed


def validate_geometry(metadata, expected, case_id):
    if not isinstance(expected, dict):
        raise ValueError(f"Missing locked geometry: {case_id}")
    size = expected.get("size_xyz")
    if not isinstance(size, list) or len(size) != 3:
        raise ValueError(f"Invalid locked dimensions: {case_id}")
    for value in size:
        positive_int(value, f"{case_id} dimension")
    if metadata.get("input_size_xyz") != size or metadata.get("output_size_xyz") != size:
        raise ValueError(f"Missing/different input or output dimensions: {case_id}")
    for key in ("input_size_xyz", "output_size_xyz"):
        for value in metadata[key]:
            positive_int(value, f"{case_id} {key}")
    for sidecar_key, manifest_key, length in (("spacing_xyz", "spacing_xyz_mm", 3),
                                              ("origin_xyz", "origin_xyz_mm", 3),
                                              ("direction", "direction", 9)):
        actual, locked = metadata.get(sidecar_key), expected.get(manifest_key)
        if not isinstance(actual, list) or not isinstance(locked, list) or len(actual) != length or len(locked) != length:
            raise ValueError(f"Missing/invalid {sidecar_key}: {case_id}")
        for left, right in zip(actual, locked):
            minimum = 0 if sidecar_key == "spacing_xyz" else -math.inf
            number(left, f"{case_id} {sidecar_key}", minimum)
            number(right, f"{case_id} locked {sidecar_key}", minimum)
            if sidecar_key == "spacing_xyz" and (left == 0 or right == 0):
                raise ValueError(f"Nonpositive spacing: {case_id}")
            if not math.isclose(left, right, rel_tol=0, abs_tol=1e-6):
                raise ValueError(f"Different {sidecar_key}: {case_id}")
    return list(size)


def metric_statistics(rows):
    result = {}
    for key in METRICS:
        values = [row[key] for row in rows if row[key] is not None]
        result[key] = {"count": len(values), "mean": statistics.mean(values) if values else None,
                       "median": statistics.median(values) if values else None,
                       "min": min(values) if values else None, "max": max(values) if values else None}
    return result


def summarize(run_settings, manifest_path):
    run_settings, manifest_path = Path(run_settings).resolve(), Path(manifest_path).resolve()
    report, manifest = read_json(run_settings), read_json(manifest_path)
    if type(report.get("schema_version")) is not int or report["schema_version"] != 1:
        raise ValueError("Expected a schema-1 inference run")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1 or manifest.get("input_extent") != "full_volume_as_supplied":
        raise ValueError("Expected a schema-1 locked full-volume cohort")
    if report.get("completed") is not True:
        raise ValueError("Inference run is incomplete")
    started = timestamp(report.get("started_at_utc"), "started_at_utc")
    completed = timestamp(report.get("completed_at_utc"), "completed_at_utc")
    if completed < started:
        raise ValueError("Completion precedes run start")
    manifest_hash = valid_hash(report.get("manifest_sha256"), "manifest")
    if manifest_hash != sha256(manifest_path) or not manifest.get("cohort_id") or report.get("cohort_id") != manifest["cohort_id"]:
        raise ValueError("Run differs from the locked cohort manifest")
    expected_ids, recorded_ids = case_ids(manifest.get("cases"), "locked"), case_ids(report.get("cases"), "recorded")
    if recorded_ids != expected_ids:
        raise ValueError("Completed run has missing, extra, different or reordered cases")
    if report.get("reference_annotation_used") is not False:
        raise ValueError("Missing no-reference-inference provenance")
    checkpoint_hash = valid_hash(report.get("checkpoint_sha256"), "checkpoint")
    settings = report.get("settings")
    if not isinstance(settings, dict) or settings.get("profile") not in ("reference", "fast"):
        raise ValueError("Missing/invalid inference settings")
    if settings.get("device") not in ("auto", "cpu", "cuda"):
        raise ValueError("Invalid requested device")
    device = report.get("resolved_device")
    if device not in ("cpu", "cuda") or (settings["device"] != "auto" and settings["device"] != device):
        raise ValueError("Missing/inconsistent resolved device")
    tile_step = number(settings.get("tile_step_size"), "tile_step_size")
    if not 0 < tile_step <= 1:
        raise ValueError("Invalid tile_step_size")
    positive_int(settings.get("threads"), "threads")
    candidate = settings["profile"] == "fast"
    if report.get("candidate_profile") is not candidate:
        raise ValueError("Inconsistent candidate profile claim")
    run_dir = run_settings.parent
    expected_sidecars = {f"{case_id}.nii.gz.json" for case_id in expected_ids}
    if {path.name for path in run_dir.glob("*.nii.gz.json")} != expected_sidecars:
        raise ValueError("Missing or unexpected per-case sidecars")
    if {path.name for path in run_dir.glob("*.nii.gz")} != {f"{case_id}.nii.gz" for case_id in expected_ids}:
        raise ValueError("Missing or unexpected per-case prediction files")
    rows, uniform_signature, fresh_count = [], None, 0
    for locked, recorded in zip(manifest["cases"], report["cases"]):
        case_id = locked["case_id"]
        output = run_dir / f"{case_id}.nii.gz"
        sidecar = run_dir / f"{case_id}.nii.gz.json"
        if any(not path.is_file() or not path.resolve().is_relative_to(run_dir) for path in (output, sidecar)):
            raise ValueError(f"Missing or out-of-directory result: {case_id}")
        metadata = read_json(sidecar)
        if type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 1 or metadata.get("case_id") != case_id:
            raise ValueError(f"Sidecar differs from recorded case: {case_id}")
        if valid_hash(metadata.get("input_sha256"), case_id + " input") != valid_hash(locked.get("image_sha256"), case_id + " locked input"):
            raise ValueError(f"Different input hash: {case_id}")
        if valid_hash(metadata.get("checkpoint_sha256"), case_id + " checkpoint") != checkpoint_hash:
            raise ValueError(f"Different checkpoint: {case_id}")
        output_hash = valid_hash(metadata.get("output_sha256"), case_id + " output")
        if output_hash != valid_hash(recorded.get("output_sha256"), case_id + " report output") or output_hash != sha256(output):
            raise ValueError(f"Different output hash/content: {case_id}")
        actual = metadata.get("settings")
        if not isinstance(actual, dict):
            raise ValueError(f"Missing sidecar settings: {case_id}")
        positive_int(actual.get("threads"), case_id + " threads")
        number(actual.get("tile_step_size"), case_id + " tile_step_size")
        for key, expected in (("profile", settings["profile"]), ("device", device),
                              ("tile_step_size", tile_step), ("threads", settings["threads"]),
                              ("use_mirroring", not candidate), ("use_gaussian", True),
                              ("network_precision", "float32" if candidate or device == "cpu" else "cuda_autocast_float16"),
                              ("aggregation", "nnunet_cpu_float16")):
            if actual.get(key) != expected or (isinstance(expected, bool) and actual.get(key) is not expected):
                raise ValueError(f"Different/inconsistent {key}: {case_id}")
        if metadata.get("candidate_profile") is not candidate or metadata.get("manual_roi") is not False or metadata.get("reference_annotation_used") is not False:
            raise ValueError(f"Missing full-volume/no-reference profile provenance: {case_id}")
        size = validate_geometry(metadata, locked.get("geometry"), case_id)
        if metadata.get("folds") != [0] or type(metadata["folds"][0]) is not int or not isinstance(metadata.get("configuration"), dict) or metadata.get("checkpoint_name") != "checkpoint_best.pth":
            raise ValueError(f"Missing/inconsistent nnU-Net configuration: {case_id}")
        software = metadata.get("software")
        if not isinstance(software, dict) or not software or any(not isinstance(k, str) or not isinstance(v, str) or not v for k, v in software.items()):
            raise ValueError(f"Missing/invalid software versions: {case_id}")
        gpu, peak = metadata.get("gpu"), metadata.get("peak_cuda_allocated_bytes")
        if device == "cuda":
            if not isinstance(gpu, str) or not gpu:
                raise ValueError(f"Missing CUDA hardware: {case_id}")
            positive_int(peak, case_id + " peak CUDA allocated bytes")
        elif gpu is not None or peak is not None:
            raise ValueError(f"CUDA data recorded for a CPU case: {case_id}")
        signature = {"software": software, "gpu": gpu, "configuration": metadata["configuration"],
                     "checkpoint_name": metadata.get("checkpoint_name"), "settings": actual}
        if uniform_signature is not None and signature != uniform_signature:
            raise ValueError(f"Different software/hardware/configuration/settings across cases: {case_id}")
        uniform_signature = signature
        row = {"case_id": case_id, "input_sha256": metadata["input_sha256"], "output_sha256": output_hash,
               "sidecar_sha256": sha256(sidecar), "input_size_xyz": size}
        for key in TIMINGS:
            row[key] = number(metadata.get(key), case_id + " " + key)
        if not math.isclose(number(recorded.get("seconds"), case_id + " report seconds"), row["end_to_end_seconds"], rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"Report/sidecar timing mismatch: {case_id}")
        stage_sum = sum(row[key] for key in TIMINGS[:3])
        runtime, end_to_end = row["runtime_seconds_before_file_save"], row["end_to_end_seconds"]
        if stage_sum > runtime + 1e-3 or runtime > end_to_end + 1e-3:
            raise ValueError(f"Inconsistent stage/end-to-end timings: {case_id}")
        if type(recorded.get("reused")) is not bool:
            raise ValueError(f"Missing reuse status: {case_id}")
        if recorded["reused"]:
            cache_state = CACHE_STATES[2]
        else:
            cache_state = CACHE_STATES[0] if fresh_count == 0 else CACHE_STATES[1]
            fresh_count += 1
        row.update({"reused": recorded["reused"], "cache_state": cache_state,
                    "outside_predict_seconds": max(0, end_to_end - runtime),
                    "other_runtime_seconds": max(0, runtime - stage_sum),
                    "tiles": positive_int(metadata.get("tiles"), case_id + " tiles"),
                    "peak_cuda_allocated_bytes": peak,
                    "peak_cuda_allocated_gib": peak / 1024**3 if peak is not None else None})
        rows.append(row)
    return {"schema_version": 1, "report_type": "completed_cohort_inference_latency",
            "cohort_id": manifest["cohort_id"], "case_count": len(rows),
            "manifest_sha256": manifest_hash, "run_settings_sha256": sha256(run_settings),
            "checkpoint_sha256": checkpoint_hash, "settings": settings, "resolved_device": device,
            "candidate_profile": candidate, "gpu": uniform_signature["gpu"],
            "software": uniform_signature["software"],
            "configuration_sha256": hashlib.sha256(json.dumps(uniform_signature["configuration"], sort_keys=True).encode()).hexdigest(),
            "output_hashes_verified": True, "ct_or_reference_files_read": False,
            "run_report_wall_seconds": (completed - started).total_seconds(),
            "sum_recorded_case_end_to_end_seconds": sum(row["end_to_end_seconds"] for row in rows),
            "statistics_all_cases": metric_statistics(rows),
            "statistics_by_cache_state": {state: {"case_count": len(group), "metrics": metric_statistics(group)}
                for state in CACHE_STATES for group in [[row for row in rows if row["cache_state"] == state]]},
            "notes": [
                "First non-reused case includes lazy runtime/model loading; subsequent non-reused cases use that runtime.",
                "Reused case timings belong to prior execution; their original cold/cache status cannot be recovered from this run report.",
                "outside_predict_seconds includes initialization (first new case), output serialization and hashing, not a separately instrumented initialization stage.",
                "Stage timings exclude cohort input preflight and Slicer/server transport; sum of reused timings is not current run wall time.",
                "Only JSON metadata and prediction-file bytes were read; this is an integrity/latency report, not an accuracy or decoded-image geometry evaluation.",
                "Reports retain caller-supplied case IDs; review identifiers and redistribution rights before publication."
            ], "cases": rows}


def export_summary(summary, output_dir):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("Use a new/empty output directory; latency reports are never overwritten")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "inference_latency.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    fields = ("case_id", "cache_state", "reused") + METRICS + ("input_size_xyz", "input_sha256", "output_sha256", "sidecar_sha256")
    with (output_dir / "per_case_latency.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summary["cases"]:
            writer.writerow({**row, "input_size_xyz": "x".join(map(str, row["input_size_xyz"]))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-settings", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New private report directory; reports preserve the supplied case IDs")
    args = parser.parse_args()
    result = summarize(args.run_settings, args.manifest)
    export_summary(result, args.output_dir)
    print(json.dumps({"case_count": result["case_count"], "profile": result["settings"]["profile"],
                      "device": result["resolved_device"],
                      "end_to_end_seconds": result["statistics_all_cases"]["end_to_end_seconds"],
                      "cache_state_counts": {key: value["case_count"] for key, value in result["statistics_by_cache_state"].items()}}, indent=2))


if __name__ == "__main__":
    main()
