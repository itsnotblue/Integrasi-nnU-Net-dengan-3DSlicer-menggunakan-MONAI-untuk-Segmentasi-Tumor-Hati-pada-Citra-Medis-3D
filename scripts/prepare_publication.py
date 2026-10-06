"""Plan/export a clean local handoff; never log in, initialize Git, or upload."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from model_artifacts import contained_path, load_artifacts, sha256, verify
from package_source import source_snapshot
from validate_release import validate

ROOT = Path(__file__).resolve().parents[1]
MODEL_DOCUMENTS = {"README.md", "PROVENANCE.md", "SHA256SUMS.txt", ".gitattributes", "LICENSE"}
REQUIRED_MODEL_DOCUMENTS = {"README.md", "PROVENANCE.md", "SHA256SUMS.txt"}
CREDENTIAL_PATTERN = r"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
LOCAL_PATH_PATTERN = r"[A-Za-z]:[/\\](?:Users|Program Files)|" + "/content/" + "drive/"


def checked_model_document(path, name):
    content = path.read_bytes()
    if len(content) >= 1024 * 1024 or b"\x00" in content:
        raise ValueError(f"Binary/oversized model document rejected: {name}")
    text = content.decode("utf-8-sig")
    if re.search(CREDENTIAL_PATTERN, text):
        raise ValueError(f"Potential credential in model document: {name}")
    if re.search(LOCAL_PATH_PATTERN, text):
        raise ValueError(f"Machine-specific path in model document: {name}")
    return content


def model_inventory(model_root, artifacts):
    """Only exact runtime artifacts and reviewed model documents may be uploaded."""
    model_root = Path(model_root).resolve()
    errors = verify(model_root, artifacts)
    if errors:
        raise ValueError("Model artifacts rejected: " + "; ".join(errors))
    runtime_names = {a["path"] for a in artifacts}
    allowed_names = runtime_names | MODEL_DOCUMENTS
    actual_names = set()
    for path in model_root.rglob("*"):
        if path.is_file():
            name = path.relative_to(model_root).as_posix()
            contained_path(model_root, name)
            actual_names.add(name)
    unexpected = actual_names - allowed_names
    if unexpected:
        raise ValueError("Unexpected file in model staging; remove it from the upload selection")
    if not REQUIRED_MODEL_DOCUMENTS.issubset(actual_names):
        raise ValueError("Required model card, provenance or checksums are missing")
    records = []
    expected = {a["path"]: a for a in artifacts}
    for name in sorted(actual_names):
        path = contained_path(model_root, name)
        if name in runtime_names:
            records.append({"path": name, "size": expected[name]["size"],
                            "sha256": expected[name]["sha256"].lower(), "kind": "runtime_artifact"})
        else:
            content = checked_model_document(path, name)
            records.append({"path": name, "size": len(content),
                            "sha256": hashlib.sha256(content).hexdigest(), "kind": "model_document"})
    return records


def source_inventory(snapshot):
    return [{"path": name, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            for name, content in snapshot]


def inventory_digest(records):
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def check_source(source_root):
    """Scan allowed text and, in Git, reject any extra tracked file."""
    source_root = Path(source_root).resolve()
    snapshot = source_snapshot(source_root)
    tracked_checked = False
    if (source_root / ".git").exists():
        top = subprocess.run(["git", "-C", str(source_root), "rev-parse", "--show-toplevel"],
                             check=True, capture_output=True, text=True)
        if Path(top.stdout.strip()).resolve() != source_root:
            raise ValueError("Source root must be the exact Git checkout root")
        tracked = subprocess.run(["git", "-C", str(source_root), "ls-files", "-z"],
                                 check=True, capture_output=True)
        selected = {name for name, _ in snapshot}
        if any(name.decode("utf-8") not in selected for name in tracked.stdout.split(b"\0") if name):
            raise ValueError("Git tracks an unapproved, missing or excluded source file")
        tracked_checked = True
    return {"selected_source_files": len(snapshot), "source_inventory_sha256": inventory_digest(source_inventory(snapshot)),
            "tracked_inventory_checked": tracked_checked, "model_artifacts_checked": False,
            "remote_operations": [], "scope": "source hygiene only; not inference or publication acceptance"}


def build_plan(source_root, model_root):
    source_root, model_root = Path(source_root).resolve(), Path(model_root).resolve()
    gate = validate(source_root, model_root)
    if not gate["preparation_passed"]:
        raise ValueError("Release preparation rejected: " + "; ".join(gate["errors"]))
    manifest = json.loads((source_root / "configs/release_manifest.json").read_text(encoding="utf-8"))
    repository = manifest.get("source_repository")
    if repository and not re.fullmatch(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?", repository):
        raise ValueError("Source repository must be a plain GitHub HTTPS URL without credentials")
    model_repository = manifest.get("huggingface_repository")
    if model_repository and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", model_repository):
        raise ValueError("Hugging Face repository must be owner/repository without credentials")
    artifacts = load_artifacts(source_root / "configs/model_artifacts.lock.json")
    snapshot = source_snapshot(source_root)
    source_files = source_inventory(snapshot)
    model_files = model_inventory(model_root, artifacts)
    plan = {
        "schema_version": 1, "status": "local_preparation_only", "remote_operations": [],
        "authentication_verified": False, "redistribution_granted_by_this_plan": False,
        "recorded_source_repository": repository, "recorded_huggingface_repository": model_repository,
        "publication_manifest_complete": gate["publication_passed"],
        "publication_blockers": gate["publication_blockers"],
        "owner_confirmation_needed": ["existing GitHub history or a new repository", "repository visibility",
                                      "source and checkpoint licenses", "institution/dataset redistribution terms",
                                      "Hugging Face destination and upload approval"],
        "source_file_count": len(source_files), "source_inventory_sha256": inventory_digest(source_files),
        "source_files": source_files,
        "model_upload_file_count": len(model_files), "model_upload_inventory_sha256": inventory_digest(model_files),
        "model_upload_bytes": sum(record["size"] for record in model_files), "model_upload_files": model_files,
        "runtime_artifact_count": len(artifacts), "runtime_artifact_bytes": sum(a["size"] for a in artifacts),
        "checkpoint_bytes": sum(a["size"] for a in artifacts if a["path"].endswith(".pth")),
        "release_sequence": [
            "Resolve owner rights/licenses, destination/history/visibility, and authentication.",
            "Prepare a clean source checkout; review/commit it locally and record its baseline commit.",
            "With explicit upload approval, upload only the exact model inventory; record the model commit.",
            "Verify a separate download of that immutable model commit against all locked artifacts.",
            "Record model/source provenance, pass publication checks, then commit the final source manifest.",
            "Run a fresh dependency/model installation and Slicer UAT; publish/tag only after acceptance.",
        ],
        "limits": ["Inventories are snapshots; regenerate and reverify after any edits.",
                   "Recorded repository IDs are not proof of access or permission.",
                   "A local export is not a Git commit or a hosted release.",
                   "Clinical accuracy, independent validation and full runtime UAT are separate gates."],
    }
    return plan, snapshot


def export_source(source_root, model_root, destination):
    """Write a new source-only folder and inventory; preserve all existing folders."""
    source_root, model_root = Path(source_root).resolve(), Path(model_root).resolve()
    destination = Path(destination).resolve()
    if (destination.is_relative_to(source_root) or source_root.is_relative_to(destination)
            or destination.is_relative_to(model_root) or model_root.is_relative_to(destination)):
        raise ValueError("Export destination must be separate from source and model staging")
    if destination.exists():
        raise FileExistsError("Use a new export directory; existing exports are never overwritten")
    plan, snapshot = build_plan(source_root, model_root)
    destination.mkdir(parents=True, exist_ok=False)
    exported = destination / "github_source"
    exported.mkdir()
    for name, content in snapshot:
        target = contained_path(exported, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(content)
    verify_export(exported, plan["source_files"])
    # No model copies, Git initialization, auth commands or remote calls happen here.
    with (destination / "publication_plan.json").open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return plan


def verify_export(exported, expected):
    exported = Path(exported).resolve()
    actual = {path.relative_to(exported).as_posix() for path in exported.rglob("*") if path.is_file()}
    if actual != {record["path"] for record in expected}:
        raise ValueError("Export inventory mismatch")
    for record in expected:
        path = contained_path(exported, record["path"])
        if path.stat().st_size != record["size"] or sha256(path) != record["sha256"]:
            raise ValueError("Export content/hash mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "export", "check-source"))
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "check-source":
            result = check_source(args.source_root)
        else:
            if args.model_root is None:
                raise ValueError("--model-root is required for plan/export")
            if args.action == "export":
                if args.output_dir is None:
                    raise ValueError("--output-dir is required for export")
                result = export_source(args.source_root, args.model_root, args.output_dir)
            else:
                if args.output_dir is not None:
                    raise ValueError("Use export to create files; plan is read-only")
                result, _ = build_plan(args.source_root, args.model_root)
        print(json.dumps(result, indent=2, sort_keys=True))
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Release preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
