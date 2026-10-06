"""Freeze full-volume case IDs, hashes and geometry before prediction/evaluation."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path

import SimpleITK as sitk

from evaluation_common import assert_geometry, geometry, read_json, sha256, write_json_new


def lock_cohort(args):
    case_ids = [value.strip() for value in args.cases.split(",") if value.strip()]
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError("Supply a nonempty list of unique case IDs")
    if any(Path(case_id).name != case_id or case_id in (".", "..") for case_id in case_ids):
        raise ValueError("Case IDs must be plain filename stems")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace locked cohort {output}")
    split_evidence = {"status": "unverified", "model": args.split_model,
                      "reason": "No checkpoint-linked training/validation case list provided"}
    excluded = set()
    if args.split_file:
        split_path = Path(args.split_file).resolve()
        split = read_json(split_path)
        if args.fold < 0 or args.fold >= len(split):
            raise ValueError("Fold index is outside the supplied split")
        fold = split[args.fold]
        train, val = set(fold["train"]), set(fold["val"])
        if train & val:
            raise ValueError("Supplied split contains overlapping train/validation IDs")
        excluded = train | val
        overlaps = sorted(set(case_ids) & excluded)
        split_evidence = {
            "status": "observed_overlap" if overlaps else "no_observed_id_overlap",
            "model": args.split_model,
            "file_sha256": sha256(split_path), "fold": args.fold,
            "training_case_count": len(train), "validation_case_count": len(val),
            "overlapping_case_ids": overlaps,
            "reason": "Filename-ID comparison only; checkpoint linkage and source/patient mapping must be independently verified",
        }
    cases = []
    for case_id in case_ids:
        image = Path(args.image_dir).resolve() / args.image_pattern.format(case_id=case_id)
        reference = Path(args.reference_dir).resolve() / args.reference_pattern.format(case_id=case_id)
        image_itk, reference_itk = sitk.ReadImage(str(image)), sitk.ReadImage(str(reference))
        image_geometry = geometry(image_itk)
        assert_geometry(geometry(reference_itk), image_geometry, f"{case_id} reference")
        cases.append({
            "case_id": case_id,
            "image": os.path.relpath(image, output.parent).replace("\\", "/"),
            "reference": os.path.relpath(reference, output.parent).replace("\\", "/"),
            "image_sha256": sha256(image), "reference_sha256": sha256(reference),
            "geometry": image_geometry,
            "observed_split_membership": "train_or_validation" if case_id in excluded else "unknown_or_outside_supplied_split",
        })
    manifest = {
        "schema_version": 1, "cohort_id": args.cohort_id,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": args.purpose,
        "input_extent": "crop" if args.purpose == "technical_crop" else "full_volume_as_supplied",
        "split_evidence": split_evidence,
        "held_out_certified": False,
        "limitation": "A locked manifest prevents silent case/file changes; it does not prove original scan completeness or absence of patient-level leakage",
        "cases": cases,
    }
    write_json_new(output, manifest)
    print(f"Locked {len(cases)} cases: {output}")
    print(f"Split evidence: {split_evidence['status']}; held-out certification: false")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--image-pattern", default="{case_id}_0000.nii.gz")
    parser.add_argument("--reference-pattern", default="{case_id}.nii.gz")
    parser.add_argument("--cases", required=True, help="Comma-separated IDs; explicit selection, no annotation-based filtering")
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--purpose", choices=("technical_full_volume", "retrospective_test", "technical_crop"), default="technical_full_volume")
    parser.add_argument("--split-file", help="Optional nnU-Net splits_final.json; retain correct checkpoint linkage separately")
    parser.add_argument("--split-model", default="model_b")
    parser.add_argument("--fold", type=int, default=0)
    lock_cohort(parser.parse_args())


if __name__ == "__main__":
    main()
