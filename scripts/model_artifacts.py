"""Verify or download the exact nnU-Net artifacts in the release lock file."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ("model_a", "model_b", "model_c")


def load_artifacts(lock_path, models=MODEL_NAMES):
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1:
        raise ValueError("Unsupported model artifact lock schema")
    artifacts = []
    seen = set()
    for artifact in lock["artifacts"]:
        path = PurePosixPath(artifact["path"])
        if (not path.parts or path.is_absolute() or ".." in path.parts or "\\" in str(path)
                or ":" in str(path) or path.parts[0] not in MODEL_NAMES):
            raise ValueError("Unsafe artifact path in lock file")
        if str(path) in seen:
            raise ValueError("Duplicate artifact path in lock file")
        seen.add(str(path))
        if not re.fullmatch(r"[a-fA-F0-9]{64}", artifact["sha256"]):
            raise ValueError("Invalid SHA-256 in lock file")
        if not isinstance(artifact["size"], int) or artifact["size"] <= 0:
            raise ValueError("Invalid artifact size in lock file")
        if path.parts[0] in models:
            artifacts.append(artifact)
    for model in models:
        expected = {f"{model}/{name}" for name in (
            "dataset.json", "plans.json", "dataset_fingerprint.json",
            "fold_0/checkpoint_best.pth")}
        if {a["path"] for a in artifacts if a["path"].startswith(model + "/")} != expected:
            raise ValueError(f"The lock must contain the four runtime files for {model}")
    return artifacts


def contained_path(root, relative):
    root = Path(root).resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("Artifact path escapes its root")
    return candidate


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(model_root, artifacts):
    errors = []
    for artifact in artifacts:
        path = contained_path(model_root, artifact["path"])
        if not path.is_file():
            errors.append(f"Missing: {artifact['path']}")
        elif path.stat().st_size != artifact["size"]:
            errors.append(f"Size mismatch: {artifact['path']}")
        elif sha256(path) != artifact["sha256"].lower():
            errors.append(f"SHA-256 mismatch: {artifact['path']}")
    return errors


def download(repo_id, revision, destination, artifacts):
    if not re.fullmatch(r"[a-fA-F0-9]{40}", revision or ""):
        raise ValueError("Use the full 40-character Hugging Face commit hash, not main or a tag")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_id or ""):
        raise ValueError("Repository ID must have the form owner/repository")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ValueError("Install requirements-release.txt in the active environment") from exc
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    # Verify all downloaded files before replacing any installed artifact.
    with tempfile.TemporaryDirectory(prefix="verified-models-", dir=destination.parent) as temporary:
        snapshot_download(repo_id=repo_id, repo_type="model", revision=revision,
                          allow_patterns=[a["path"] for a in artifacts], local_dir=temporary)
        errors = verify(temporary, artifacts)
        if errors:
            raise ValueError("Downloaded artifacts rejected:\n" + "\n".join(errors))
        for artifact in artifacts:
            source = contained_path(temporary, artifact["path"])
            target = contained_path(destination, artifact["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, staged_name = tempfile.mkstemp(prefix=target.name + ".", dir=target.parent)
            os.close(descriptor)
            staged = Path(staged_name)
            try:
                shutil.copyfile(source, staged)
                os.replace(staged, target)
            finally:
                staged.unlink(missing_ok=True)
    errors = verify(destination, artifacts)
    if errors:
        raise ValueError("Installed artifacts rejected:\n" + "\n".join(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "download"))
    parser.add_argument("--lock", type=Path, default=ROOT / "configs/model_artifacts.lock.json")
    parser.add_argument("--model-root", type=Path, default=ROOT / "app/radiology/lib/models")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--repo-id")
    parser.add_argument("--revision")
    args = parser.parse_args()
    try:
        artifacts = load_artifacts(args.lock, args.models)
        if args.action == "download":
            download(args.repo_id, args.revision, args.model_root, artifacts)
        else:
            errors = verify(args.model_root, artifacts)
            if errors:
                raise ValueError("\n".join(errors))
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Artifact verification failed: {exc}\n")
    print(f"Verified {len(artifacts)} exact artifacts for {', '.join(args.models)}")


if __name__ == "__main__":
    main()
