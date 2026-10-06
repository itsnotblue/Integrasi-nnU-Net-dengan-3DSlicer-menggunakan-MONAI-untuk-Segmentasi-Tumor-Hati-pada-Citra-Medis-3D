"""Artifact integrity checks use tiny synthetic files, never medical data or real weights."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/model_artifacts.py"
SPEC = importlib.util.spec_from_file_location("release_artifacts", SCRIPT)
artifacts_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(artifacts_module)


class ArtifactIntegrityTests(unittest.TestCase):
    def setUp(self):
        temporary_root = SCRIPT.parents[1] / "tmp/release-tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.artifacts = []
        for name in ("dataset.json", "plans.json", "dataset_fingerprint.json", "fold_0/checkpoint_best.pth"):
            relative = "model_b/" + name
            content = ("synthetic " + name).encode()
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            self.artifacts.append({"path": relative, "size": len(content),
                                   "sha256": hashlib.sha256(content).hexdigest()})
        self.lock = self.root / "lock.json"
        self.lock.write_text(json.dumps({"schema_version": 1, "artifacts": self.artifacts}), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def test_verified_artifacts_pass(self):
        loaded = artifacts_module.load_artifacts(self.lock, ["model_b"])
        self.assertEqual([], artifacts_module.verify(self.source, loaded))

    def test_same_size_corruption_is_rejected(self):
        target = self.source / self.artifacts[0]["path"]
        target.write_bytes(b"X" * target.stat().st_size)
        self.assertEqual(["SHA-256 mismatch: model_b/dataset.json"],
                         artifacts_module.verify(self.source, self.artifacts))

    def test_missing_metadata_is_rejected(self):
        (self.source / "model_b/plans.json").unlink()
        self.assertEqual(["Missing: model_b/plans.json"], artifacts_module.verify(self.source, self.artifacts))

    def test_path_traversal_is_rejected(self):
        self.artifacts[0]["path"] = "model_b/../../escape"
        self.lock.write_text(json.dumps({"schema_version": 1, "artifacts": self.artifacts}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            artifacts_module.load_artifacts(self.lock, ["model_b"])

    def test_incomplete_model_lock_is_rejected(self):
        self.lock.write_text(json.dumps({"schema_version": 1, "artifacts": self.artifacts[:-1]}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "four runtime files"):
            artifacts_module.load_artifacts(self.lock, ["model_b"])

    def test_mutable_remote_revision_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "40-character"):
            artifacts_module.download("owner/models", "main", self.root / "destination", self.artifacts)

    def test_corrupt_download_preserves_installed_artifacts(self):
        destination = self.root / "installed"
        shutil.copytree(self.source, destination)

        def fake_download(**kwargs):
            shutil.copytree(self.source, kwargs["local_dir"], dirs_exist_ok=True)
            bad = Path(kwargs["local_dir"]) / "model_b/dataset.json"
            bad.write_bytes(b"X" * bad.stat().st_size)

        fake_hub = types.SimpleNamespace(snapshot_download=fake_download)
        with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            with self.assertRaisesRegex(ValueError, "Downloaded artifacts rejected"):
                artifacts_module.download("owner/models", "a" * 40, destination, self.artifacts)
        self.assertEqual([], artifacts_module.verify(destination, self.artifacts))

    def test_valid_download_installs_only_locked_files(self):
        destination = self.root / "installed"

        def fake_download(**kwargs):
            self.assertEqual("a" * 40, kwargs["revision"])
            self.assertEqual({a["path"] for a in self.artifacts}, set(kwargs["allow_patterns"]))
            shutil.copytree(self.source, kwargs["local_dir"], dirs_exist_ok=True)

        with patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(snapshot_download=fake_download)}):
            artifacts_module.download("owner/models", "a" * 40, destination, self.artifacts)
        self.assertEqual([], artifacts_module.verify(destination, self.artifacts))


if __name__ == "__main__":
    unittest.main()
