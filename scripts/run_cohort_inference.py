"""Run/restart a locked full-volume cohort without reading reference masks."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import SimpleITK as sitk
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "radiology"))
from lib.infers.nnunet_runtime import FullVolumeRuntime, InferenceSettings, sha256
from evaluation_common import assert_geometry, geometry, read_json, write_json_new


def header_geometry(path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    if reader.GetDimension() != 3 or reader.GetNumberOfComponents() != 1:
        raise ValueError(f"Expected a scalar 3D image: {path}")
    return {"size_xyz": list(reader.GetSize()), "spacing_xyz_mm": list(reader.GetSpacing()),
            "origin_xyz_mm": list(reader.GetOrigin()), "direction": list(reader.GetDirection())}


def validate_existing(output, sidecar, image_hash, checkpoint_hash, settings,
                      input_geometry, resolved_device):
    if not output.is_file() or not sidecar.is_file():
        raise ValueError(f"Incomplete existing result, use a separate output directory: {output}")
    metadata = read_json(sidecar)
    expected = asdict(settings)
    actual = metadata.get("settings", {})
    for key in ("profile", "tile_step_size", "threads"):
        if actual.get(key) != expected[key]:
            raise ValueError(f"Existing result has different {key}: {output}")
    if actual.get("device") != resolved_device:
        raise ValueError(f"Existing result has different device: {output}")
    if metadata.get("input_sha256") != image_hash or metadata.get("checkpoint_sha256") != checkpoint_hash or metadata.get("output_sha256") != sha256(output):
        raise ValueError(f"Existing result hash mismatch: {output}")
    if metadata.get("manual_roi") is not False or metadata.get("reference_annotation_used") is not False:
        raise ValueError(f"Existing result lacks a full-volume/no-annotation provenance claim: {output}")
    if actual.get("use_mirroring") is not (settings.profile == "reference"):
        raise ValueError(f"Existing result has inconsistent mirroring settings: {output}")
    seconds = metadata.get("end_to_end_seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"Existing result has invalid timing metadata: {output}")
    if metadata.get("input_size_xyz") != input_geometry["size_xyz"]:
        raise ValueError(f"Existing result has missing or different input dimensions: {output}")
    try:
        sidecar_geometry = {"size_xyz": metadata["output_size_xyz"],
                            "spacing_xyz_mm": metadata["spacing_xyz"],
                            "origin_xyz_mm": metadata["origin_xyz"], "direction": metadata["direction"]}
    except KeyError as exc:
        raise ValueError(f"Existing result has incomplete geometry metadata: {output}") from exc
    assert_geometry(sidecar_geometry, input_geometry, "Existing result sidecar")
    assert_geometry(header_geometry(output), input_geometry, "Existing result image")
    return metadata


def write_report(path, report):
    # Preserve the previous checkpointed report if writing is interrupted.
    descriptor, temporary = tempfile.mkstemp(prefix="run-settings-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", choices=["reference", "fast"], default="reference")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = read_json(manifest_path)
    manifest_hash = sha256(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("input_extent") != "full_volume_as_supplied":
        raise ValueError("Expected a schema-1 locked full-volume cohort")
    cases = manifest.get("cases", [])
    ids = [c["case_id"] for c in cases]
    if not cases or len(set(ids)) != len(ids):
        raise ValueError("Expected nonempty unique cases")
    output_dir = Path(args.output_dir).resolve()
    settings = InferenceSettings(args.profile, args.device, 0.5, args.threads)
    resolved_device = settings.device
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_hash = sha256(Path(args.model_dir).resolve() / "fold_0/checkpoint_best.pth")
    report_path = output_dir / "run_settings.json"
    if report_path.exists():
        if not args.resume:
            raise FileExistsError(f"Existing run report; refusing to overwrite {report_path}")
        previous = read_json(report_path)
        if (previous.get("manifest_sha256") != manifest_hash
                or previous.get("checkpoint_sha256") != checkpoint_hash
                or previous.get("settings") != asdict(settings)
                or previous.get("resolved_device", resolved_device) != resolved_device):
            raise ValueError("Existing run report has different cohort, checkpoint or settings")
    geometries = {}
    for case in cases:
        if not case["case_id"] or Path(case["case_id"]).name != case["case_id"] or any(c in case["case_id"] for c in ("/", "\\", ":")):
            raise ValueError("Unsafe case ID")
        image = (manifest_path.parent / case["image"]).resolve()
        if sha256(image) != case["image_sha256"]:
            raise ValueError(f"Locked image changed: {case['case_id']}")
        geometries[case["case_id"]] = header_geometry(image)
        assert_geometry(geometries[case["case_id"]], case["geometry"], f"Locked image {case['case_id']}")
        output = output_dir / f"{case['case_id']}.nii.gz"
        sidecar = Path(str(output) + ".json")
        if output.exists() or sidecar.exists():
            if not args.resume:
                raise FileExistsError(f"Refusing to overwrite {output}; use --resume for hash-checked reuse")
            validate_existing(output, sidecar, case["image_sha256"], checkpoint_hash, settings,
                              geometries[case["case_id"]], resolved_device)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = None
    report = {"schema_version": 1, "cohort_id": manifest["cohort_id"], "manifest_sha256": manifest_hash,
              "settings": asdict(settings), "resolved_device": resolved_device, "checkpoint_sha256": checkpoint_hash,
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "cases": [],
              "reference_annotation_used": False, "candidate_profile": args.profile == "fast"}
    for index, case in enumerate(cases, start=1):
        image = (manifest_path.parent / case["image"]).resolve()
        output = output_dir / f"{case['case_id']}.nii.gz"
        sidecar = Path(str(output) + ".json")
        print(f"COHORT_CASE_START {index}/{len(cases)} {case['case_id']}", flush=True)
        if output.exists() or sidecar.exists():
            if not args.resume:
                raise FileExistsError(f"Refusing to overwrite {output}; use --resume for hash-checked reuse")
            metadata = validate_existing(output, sidecar, case["image_sha256"], checkpoint_hash, settings,
                                         geometries[case["case_id"]], resolved_device)
            reused = True
        else:
            start = time.perf_counter()
            if runtime is None:
                runtime = FullVolumeRuntime(args.model_dir, settings)
                if runtime.checkpoint_hash != checkpoint_hash or str(runtime.predictor.device) != resolved_device:
                    raise ValueError("Runtime checkpoint or resolved device changed after preflight")
            if sha256(image) != case["image_sha256"]:
                raise ValueError(f"Locked image changed before prediction: {case['case_id']}")
            prediction, metadata = runtime.predict(image)
            if metadata.get("input_sha256") != case["image_sha256"] or metadata.get("checkpoint_sha256") != checkpoint_hash:
                raise ValueError(f"Prediction input/checkpoint differs from locked case: {case['case_id']}")
            assert_geometry(geometry(prediction), geometries[case["case_id"]], f"Prediction {case['case_id']}")
            if sidecar.exists():
                raise FileExistsError(f"Result sidecar appeared during inference: {sidecar}")
            # Reserve exclusively before SimpleITK writes, preventing a second run from overwriting this result.
            with output.open("xb"):
                pass
            sitk.WriteImage(prediction, str(output), True)
            metadata["output_sha256"] = sha256(output)
            metadata["end_to_end_seconds"] = time.perf_counter() - start
            metadata["case_id"] = case["case_id"]
            write_json_new(sidecar, metadata)
            reused = False
        report["cases"].append({"case_id": case["case_id"], "output_sha256": metadata["output_sha256"],
                                "seconds": metadata["end_to_end_seconds"], "reused": reused})
        write_report(report_path, report)
        print(f"COHORT_CASE_OK {case['case_id']} seconds={metadata['end_to_end_seconds']:.2f} reused={reused}", flush=True)
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["completed"] = True
    write_report(report_path, report)
    print(f"COHORT_INFERENCE_OK {len(cases)} complete volumes", flush=True)


if __name__ == "__main__":
    main()
