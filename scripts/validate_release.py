"""Check source/model staging and distinguish preparation from public release readiness."""

import argparse
import json
from pathlib import Path
import re
import sys

from model_artifacts import ROOT, load_artifacts, verify


def validate(source_root, model_root, publication=False):
    source_root = Path(source_root).resolve()
    model_root = Path(model_root).resolve()
    errors, blockers = [], []
    required = (
        ".gitignore", "README.md", "CITATION.cff", "THIRD_PARTY_NOTICES.md",
        "licenses/MONAI-APACHE-2.0.txt", "configs/model_manifest.yml",
        "configs/model_artifacts.lock.json", "configs/release_manifest.json",
        "requirements-cpu.txt", "requirements-gpu.txt", "requirements-release.txt",
        "requirements-evaluation.txt",
        "docs/USER_MANUAL.md", "docs/PUBLISHING_CHECKLIST.md",
        "scripts/download_models.ps1", "scripts/model_artifacts.py",
        "scripts/run_inference.py", "app/radiology/lib/infers/nnunet_runtime.py",
        "scripts/run_cohort_inference.py", "scripts/lock_cohort.py",
        "scripts/summarize_inference_run.py",
        "scripts/evaluation_common.py", "scripts/evaluate_predictions.py",
        "scripts/compare_evaluations.py", "scripts/export_public_evaluation.py",
        "scripts/create_failure_panels.py", "scripts/slicer_view_result.py",
        "scripts/slicer_inference_smoke.py",
    )
    errors.extend(f"Missing source file: {name}" for name in required
                  if not (source_root / name).is_file())
    try:
        artifacts = load_artifacts(source_root / "configs/model_artifacts.lock.json")
        errors.extend(verify(model_root, artifacts))
        allowed = {a["path"] for a in artifacts} | {"README.md", "SHA256SUMS.txt", "PROVENANCE.md", "LICENSE", ".gitattributes"}
        for path in model_root.rglob("*"):
            if path.is_file():
                name = path.relative_to(model_root).as_posix()
                if name not in allowed:
                    errors.append(f"Unexpected model repository file: {name}")
        expected_checksums = {a["path"]: a["sha256"].lower() for a in artifacts}
        recorded = {}
        for line in (model_root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
            if line.strip():
                digest, name = line.split(None, 1)
                recorded[name.strip().lstrip("*")] = digest.lower()
        if recorded != expected_checksums:
            errors.append("Model repository SHA256SUMS.txt differs from source artifact lock")
        manifest = json.loads((source_root / "configs/release_manifest.json").read_text(encoding="utf-8"))
        if not manifest.get("custom_source_license") or not (source_root / "LICENSE").is_file():
            blockers.append("Owner must choose and document the custom source license")
        if not manifest.get("checkpoint_license") or not (model_root / "LICENSE").is_file():
            blockers.append("Owner must choose and document checkpoint redistribution terms")
        if manifest.get("checkpoint_redistribution_confirmed") is not True:
            blockers.append("Checkpoint redistribution permission has not been confirmed")
        if not manifest.get("dataset_terms_review_reference"):
            blockers.append("Dataset terms/provenance review reference is missing")
        if not re.fullmatch(r"[a-fA-F0-9]{40}", manifest.get("source_commit") or ""):
            blockers.append("Exact clean source commit has not been recorded")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", manifest.get("huggingface_repository") or ""):
            blockers.append("Hugging Face destination repository has not been recorded")
        if not re.fullmatch(r"[a-fA-F0-9]{40}", manifest.get("huggingface_revision") or ""):
            blockers.append("Exact uploaded Hugging Face commit has not been recorded")
    except (ValueError, OSError, KeyError, IndexError) as exc:
        errors.append(f"Release manifest/artifact validation failed: {exc}")

    excluded = {".git", ".venv", "__pycache__", ".pytest_cache", "runtime", "outputs", "results", "logs", "tmp", "data"}
    for path in source_root.rglob("*"):
        relative = path.relative_to(source_root)
        if any(part in excluded for part in relative.parts):
            continue
        if relative.as_posix().startswith("app/radiology/lib/models/"):
            continue  # Downloaded runtime files are deliberately excluded from source Git.
        if path.is_file():
            name = relative.as_posix()
            if path.stat().st_size >= 100 * 1024 * 1024:
                errors.append(f"File exceeds GitHub regular-Git limit: {name}")
            if name.endswith((".pth", ".pt", ".nii", ".nii.gz", ".nrrd", ".dcm")):
                errors.append(f"Model or patient data in source staging: {name}")
            if path.suffix in {".py", ".md", ".json", ".yml", ".yaml", ".ps1", ".bat", ".sh", ".txt", ".cff"}:
                content = path.read_text(encoding="utf-8-sig", errors="replace")
                # Report only filenames; never echo potential secrets.
                if re.search(r"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}", content):
                    errors.append(f"Potential credential in source file: {name}")
                local_path_pattern = r"[A-Za-z]:[/\\](?:Users|Program Files)|" + "/content/" + "drive/"
                if re.search(local_path_pattern, content):
                    errors.append(f"Machine-specific path in source file: {name}")
    return {"preparation_passed": not errors, "publication_passed": not errors and not blockers,
            "errors": errors, "publication_blockers": blockers,
            "gate": "publication" if publication else "preparation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--publication", action="store_true")
    args = parser.parse_args()
    result = validate(args.source_root, args.model_root, args.publication)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["publication_passed" if args.publication else "preparation_passed"] else 1)


if __name__ == "__main__":
    main()
