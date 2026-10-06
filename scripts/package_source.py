"""Build a deterministic, allowlisted source candidate ZIP without runtime data."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

from validate_release import validate

ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL_FILES = {
    ".gitignore", ".gitattributes", "README.md", "CITATION.cff", "THIRD_PARTY_NOTICES.md", "LICENSE",
    "requirements-cpu.txt", "requirements-gpu.txt", "requirements-release.txt",
    "requirements-evaluation.txt",
    "environment-cpu.yml", "environment-gpu.yml",
}
# These exact files have been reviewed as anonymous public-LiTS metrics/docs.
# Add further evidence only after an explicit privacy/provenance review.
REVIEWED_EVALUATION_FILES = {
    "docs/evaluation/HISTORICAL_RESULTS.md", "docs/evaluation/FULL_VOLUME_PILOT.md",
    "docs/evaluation/historical_statistics.json",
    "docs/evaluation/historical_per_case_metrics.csv",
    "docs/evaluation/historical_failures.csv",
    "docs/evaluation/FULL_VOLUME_RESULTS.md",
    "docs/evaluation/full_volume_statistics.json",
    "docs/evaluation/full_volume_model_b_metrics.csv",
    "docs/evaluation/full_volume_model_c_metrics.csv",
    "docs/evaluation/full_volume_component_details.csv",
    "docs/evaluation/current_fast_model_b_runtime.json",
    "docs/evaluation/current_fast_model_b_latency.csv",
    "docs/evaluation/CURRENT_FAST_RESULTS.md",
    "docs/evaluation/current_fast_statistics.json",
    "docs/evaluation/current_fast_per_case_metrics.csv",
    "docs/evaluation/current_fast_failures.csv",
    "docs/evaluation/current_fast_component_details.csv",
}
REVIEWED_WORKFLOW_FILES = {".github/workflows/source-safety.yml"}
REVIEWED_LEGACY_POINTER_FILES = {
    "radiology_portable/README.md", "radiology_portable/RELEASE_NOTES_TEMPLATE.md",
}
FORBIDDEN_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", "tmp", "runtime",
                   "models", "model", "data", "outputs", "results", "logs"}
FIXED_TIME = (1980, 1, 1, 0, 0, 0)


def allowed(relative):
    parts = relative.parts
    name = relative.as_posix()
    if any(part in FORBIDDEN_PARTS for part in parts):
        return False
    if len(parts) == 1:
        return name in TOP_LEVEL_FILES
    if name in REVIEWED_WORKFLOW_FILES or name in REVIEWED_LEGACY_POINTER_FILES:
        return True
    if name in REVIEWED_EVALUATION_FILES:
        return True
    if parts[:2] == ("app", "radiology"):
        return relative.suffix == ".py" or name == "app/radiology/README.md"
    if parts[0] == "configs":
        return relative.suffix in {".json", ".yml", ".yaml"}
    if parts[0] == "docs" and len(parts) == 2:
        return relative.suffix == ".md"
    if parts[0] == "scripts" and len(parts) == 2:
        return relative.suffix in {".py", ".ps1", ".bat", ".sh"}
    if parts[0] == "tests" and len(parts) == 2:
        return relative.name.startswith("test_") and relative.suffix == ".py"
    return parts[0] == "licenses" and len(parts) == 2 and relative.suffix == ".txt"


def source_snapshot(root):
    root = Path(root).resolve()
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or not allowed(relative):
            continue
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Refusing file outside source root: {relative.as_posix()}")
        content = path.read_bytes()
        if len(content) >= 100 * 1024 * 1024 or b"\x00" in content:
            raise ValueError(f"Binary/oversized file rejected: {relative.as_posix()}")
        text = content.decode("utf-8-sig")
        if re.search(r"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}", text):
            raise ValueError(f"Potential credential rejected: {relative.as_posix()}")
        local_path_pattern = r"[A-Za-z]:[/\\](?:Users|Program Files)|" + "/content/" + "drive/"
        if re.search(local_path_pattern, text):
            raise ValueError(f"Machine-specific path rejected: {relative.as_posix()}")
        files.append((relative.as_posix(), content))
    if not files:
        raise ValueError("No source files selected")
    return sorted(files)


def build_archive(root, output, model_root):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.suffix.lower() != ".zip":
        raise ValueError("Output must be a .zip file")
    manifest_path = output.with_suffix(".manifest.json")
    checksum_path = output.with_suffix(".sha256.txt")
    if any(path.exists() for path in (output, manifest_path, checksum_path)):
        raise FileExistsError("Use a new output filename; existing archives/manifests are never replaced")
    verification = validate(root, model_root)
    if not verification["preparation_passed"]:
        raise ValueError("Release preparation failed: " + "; ".join(verification["errors"]))
    files = source_snapshot(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = "liver-tumor-segmentation-source/"
    with zipfile.ZipFile(output, mode="x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in files:
            info = zipfile.ZipInfo(prefix + name, date_time=FIXED_TIME)
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100755 if name.endswith(".sh") else 0o100644) << 16
            archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive CRC validation failed")
        if archive.namelist() != [prefix + name for name, _ in files]:
            raise ValueError("Archive file inventory differs from source snapshot")
        for name, content in files:
            if archive.read(prefix + name) != content:
                raise ValueError(f"Archive content mismatch: {name}")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {"schema_version": 1, "artifact_type": "private_source_candidate",
                "archive": output.name, "archive_sha256": digest,
                "publication_ready": verification["publication_passed"],
                "publication_blockers": verification["publication_blockers"],
                "deterministic_zip_timestamp": list(FIXED_TIME),
                "file_count": len(files),
                "files": [{"path": name, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                          for name, content in files]}
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    with checksum_path.open("x", encoding="utf-8") as stream:
        stream.write(f"{digest}  {output.name}\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--model-root", type=Path, required=True,
                        help="Verified model staging; artifacts are checked but never copied into the ZIP")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_archive(args.source_root, args.output, args.model_root)
    print(json.dumps({"archive": manifest["archive"], "files": manifest["file_count"],
                      "sha256": manifest["archive_sha256"], "publication_ready": manifest["publication_ready"]}, indent=2))


if __name__ == "__main__":
    main()
