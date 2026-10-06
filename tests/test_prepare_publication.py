"""Local publication preparation uses synthetic weights and makes no remote calls."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_publication as publication
from validate_release import validate as real_validate


class PublicationPreparationTests(unittest.TestCase):
    def setUp(self):
        # Nested Git fixtures must not inherit a long Windows checkout path.
        self.temporary = tempfile.TemporaryDirectory(prefix="publication-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.models = self.root / "models"
        self.source.mkdir()
        self.models.mkdir()
        self.write_source("README.md", "Synthetic source\n")
        self.write_source(".gitignore", "tmp/\n__pycache__/\n")
        self.write_source("scripts/example.py", "print('synthetic')\n")
        self.artifacts = []
        for model in ("model_a", "model_b", "model_c"):
            for name in ("dataset.json", "plans.json", "dataset_fingerprint.json", "fold_0/checkpoint_best.pth"):
                relative = model + "/" + name
                content = ("synthetic " + relative).encode()
                path = self.models / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                self.artifacts.append({"path": relative, "size": len(content),
                                       "sha256": hashlib.sha256(content).hexdigest()})
        self.write_source("configs/model_artifacts.lock.json", json.dumps({"schema_version": 1, "artifacts": self.artifacts}))
        self.write_source("configs/release_manifest.json", json.dumps({"source_repository": "https://github.com/example/research",
                                                                      "huggingface_repository": None}))
        for name in publication.REQUIRED_MODEL_DOCUMENTS:
            (self.models / name).write_text("Synthetic model documentation\n", encoding="utf-8")
        self.gate = {"preparation_passed": True, "publication_passed": False,
                     "errors": [], "publication_blockers": ["Synthetic owner decision"]}
        self.gate_patch = patch.object(publication, "validate", return_value=self.gate)
        self.gate_patch.start()

    def tearDown(self):
        self.gate_patch.stop()
        self.temporary.cleanup()

    def write_source(self, name, text):
        path = self.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def test_plan_is_local_only_and_exact(self):
        with patch.object(subprocess, "run", side_effect=AssertionError("No commands expected")):
            plan, _ = publication.build_plan(self.source, self.models)
        self.assertEqual([], plan["remote_operations"])
        self.assertFalse(plan["authentication_verified"])
        self.assertFalse(plan["redistribution_granted_by_this_plan"])
        self.assertFalse(plan["publication_manifest_complete"])
        self.assertEqual(12, plan["runtime_artifact_count"])
        self.assertEqual(15, plan["model_upload_file_count"])
        self.assertEqual(sum(a["size"] for a in self.artifacts), plan["runtime_artifact_bytes"])
        self.assertIsNone(plan["recorded_huggingface_repository"])

    def test_source_license_does_not_grant_checkpoint_or_data_rights(self):
        # Copy reviewed source text only; all model bytes below remain synthetic.
        for name, content in publication.source_snapshot(ROOT):
            self.write_source(name, content.decode("utf-8-sig"))
        self.write_source("configs/model_artifacts.lock.json",
                          json.dumps({"schema_version": 1, "artifacts": self.artifacts}))
        self.write_source("configs/release_manifest.json",
                          json.dumps({"custom_source_license": "Apache-2.0"}))
        self.write_source("LICENSE", "Synthetic source license\n")
        (self.models / "SHA256SUMS.txt").write_text(
            "".join(f"{a['sha256']}  {a['path']}\n" for a in self.artifacts), encoding="utf-8")
        gate = real_validate(self.source, self.models)
        self.assertTrue(gate["preparation_passed"], gate["errors"])
        self.assertFalse(gate["publication_passed"])
        self.assertNotIn("Owner must choose and document the custom source license",
                         gate["publication_blockers"])
        for blocker in ("Owner must choose and document checkpoint redistribution terms",
                        "Checkpoint redistribution permission has not been confirmed",
                        "Dataset terms/provenance review reference is missing"):
            self.assertIn(blocker, gate["publication_blockers"])
        self.assertFalse((self.models / "LICENSE").exists())

    def test_export_contains_only_source_and_inventory(self):
        self.write_source("data/patient.nii.gz", "private synthetic data")
        self.write_source("outputs/prediction.nrrd", "synthetic prediction")
        self.write_source(".env", "private configuration")
        self.write_source("app/radiology/lib/models/model_b/fold_0/checkpoint_best.pth", "synthetic weight")
        destination = self.root / "export"
        with patch.object(subprocess, "run", side_effect=AssertionError("No commands expected")):
            plan = publication.export_source(self.source, self.models, destination)
        self.assertEqual({"github_source", "publication_plan.json"}, {p.name for p in destination.iterdir()})
        files = [p for p in (destination / "github_source").rglob("*") if p.is_file()]
        self.assertEqual(plan["source_file_count"], len(files))
        self.assertFalse(any(p.suffix in {".pth", ".nrrd", ".gz"} for p in files))
        self.assertFalse((destination / "github_source/.git").exists())
        publication.verify_export(destination / "github_source", plan["source_files"])

    def test_export_never_overwrites(self):
        destination = self.root / "export"
        destination.mkdir()
        marker = destination / "existing.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            publication.export_source(self.source, self.models, destination)
        self.assertEqual("keep", marker.read_text(encoding="utf-8"))

    def test_export_rejects_overlapping_source_and_models(self):
        for destination in (self.source / "new", self.models / "new", self.root, self.source):
            with self.assertRaisesRegex(ValueError, "separate"):
                publication.export_source(self.source, self.models, destination)

    def test_failed_gate_does_not_create_export(self):
        self.gate.update(preparation_passed=False, errors=["Synthetic corruption"])
        destination = self.root / "export"
        with self.assertRaisesRegex(ValueError, "rejected"):
            publication.export_source(self.source, self.models, destination)
        self.assertFalse(destination.exists())

    def test_unexpected_model_file_rejected_before_export(self):
        (self.models / "private_ct.nrrd").write_bytes(b"synthetic private")
        destination = self.root / "export"
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            publication.export_source(self.source, self.models, destination)
        self.assertFalse(destination.exists())

    def test_corrupt_weight_rejected(self):
        path = self.models / "model_b/fold_0/checkpoint_best.pth"
        path.write_bytes(b"X" * path.stat().st_size)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            publication.build_plan(self.source, self.models)

    def test_model_document_credential_rejected_without_echo(self):
        token = "hf_" + "X" * 25
        (self.models / "README.md").write_text("Synthetic " + token, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "credential") as caught:
            publication.build_plan(self.source, self.models)
        self.assertNotIn(token, str(caught.exception))

    def test_model_document_private_path_rejected(self):
        private = "C:" + "/Users/" + "Synthetic/private"
        (self.models / "PROVENANCE.md").write_text(private, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Machine-specific"):
            publication.build_plan(self.source, self.models)

    def test_repository_credentials_rejected(self):
        self.write_source("configs/release_manifest.json", json.dumps({"source_repository": "https://user:password@github.com/example/research"}))
        with self.assertRaisesRegex(ValueError, "plain GitHub"):
            publication.build_plan(self.source, self.models)

    def test_source_only_check_does_not_claim_model_acceptance(self):
        result = publication.check_source(self.source)
        self.assertFalse(result["model_artifacts_checked"])
        self.assertFalse(result["tracked_inventory_checked"])
        self.assertEqual([], result["remote_operations"])

    def test_source_only_check_rejects_tracked_excluded_files(self):
        if not shutil.which("git"):
            self.skipTest("Git not installed")
        subprocess.run(["git", "init", str(self.source)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(self.source), "add", "README.md", "scripts", "configs", ".gitignore"],
                       check=True, capture_output=True)
        self.assertTrue(publication.check_source(self.source)["tracked_inventory_checked"])
        self.write_source("private_ct.nrrd", "synthetic private")
        subprocess.run(["git", "-C", str(self.source), "add", "private_ct.nrrd"], check=True, capture_output=True)
        with self.assertRaisesRegex(ValueError, "unapproved"):
            publication.check_source(self.source)

    def test_export_tampering_rejected(self):
        destination = self.root / "export"
        plan = publication.export_source(self.source, self.models, destination)
        (destination / "github_source/README.md").write_text("tampered", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "content/hash"):
            publication.verify_export(destination / "github_source", plan["source_files"])


if __name__ == "__main__":
    unittest.main()
