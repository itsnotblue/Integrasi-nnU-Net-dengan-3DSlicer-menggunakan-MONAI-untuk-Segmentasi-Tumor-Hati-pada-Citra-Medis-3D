"""Privacy and determinism tests for the source-only archive allowlist."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("source_packager", ROOT / "scripts/package_source.py")
packager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packager)


class SourcePackageTests(unittest.TestCase):
    def setUp(self):
        test_root = ROOT / "tmp/packaging-tests"
        test_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=test_root)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.write("README.md", "Synthetic source package\n")
        self.write("scripts/example.py", "print('synthetic')\n")
        self.write("configs/model_manifest.yml", "schema_version: 1\n")

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, relative, content):
        path = self.source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def fake_gate(self):
        return {"preparation_passed": True, "publication_passed": False,
                "errors": [], "publication_blockers": ["Synthetic unresolved license"]}

    def test_medical_models_logs_results_and_caches_are_excluded(self):
        forbidden = ["app/radiology/lib/models/model_b/fold_0/checkpoint_best.pth",
                     "data/ct.nii.gz", "outputs/result.nrrd", "logs/runtime.txt", ".env",
                     "app/radiology/runtime/generated.py", "scripts/__pycache__/cache.py",
                     "tmp/scratch.py", "docs/evaluation/unreviewed.json", "tests/data/ct.nrrd",
                     "results/private_ct.nrrd", "results/metrics.json"]
        for relative in forbidden:
            self.write(relative, "synthetic forbidden content")
        self.assertEqual(["README.md", "configs/model_manifest.yml", "scripts/example.py"],
                         [name for name, _ in packager.source_snapshot(self.source)])

    def test_reviewed_metric_file_is_included(self):
        self.write("docs/evaluation/full_volume_model_b_metrics.csv", "case_id,dice\nLITS_121,0.1\n")
        self.assertIn("docs/evaluation/full_volume_model_b_metrics.csv",
                      [name for name, _ in packager.source_snapshot(self.source)])

    def test_portable_git_attributes_not_credential_dotfiles_are_included(self):
        self.write(".gitattributes", "* text=auto eol=lf\n*.bat text eol=crlf\n")
        self.write(".git-credentials", "synthetic private credential configuration\n")
        self.write(".env.production", "synthetic private environment\n")
        names = [name for name, _ in packager.source_snapshot(self.source)]
        self.assertIn(".gitattributes", names)
        self.assertNotIn(".git-credentials", names)
        self.assertNotIn(".env.production", names)

    def test_source_license_scope_and_upstream_notices_are_included(self):
        self.write("LICENSE", "Synthetic source license\n")
        self.write("docs/LICENSE_SCOPE.md", "Synthetic source-only license scope\n")
        self.write("THIRD_PARTY_NOTICES.md", "Synthetic retained attribution\n")
        self.write("licenses/MONAI-APACHE-2.0.txt", "Synthetic retained upstream license\n")
        self.write("models/LICENSE", "Synthetic separate checkpoint terms\n")
        names = [name for name, _ in packager.source_snapshot(self.source)]
        for name in ("LICENSE", "docs/LICENSE_SCOPE.md", "THIRD_PARTY_NOTICES.md",
                     "licenses/MONAI-APACHE-2.0.txt"):
            self.assertIn(name, names)
        self.assertNotIn("models/LICENSE", names)

    def test_only_reviewed_safety_workflow_is_included(self):
        self.write(".github/workflows/source-safety.yml", "name: Synthetic safety\n")
        self.write(".github/workflows/unreviewed-upload.yml", "name: Not approved\n")
        names = [name for name, _ in packager.source_snapshot(self.source)]
        self.assertIn(".github/workflows/source-safety.yml", names)
        self.assertNotIn(".github/workflows/unreviewed-upload.yml", names)

    def test_legacy_documentation_pointers_not_the_old_bundle_are_included(self):
        self.write("radiology_portable/README.md", "Synthetic current-manual pointer\n")
        self.write("radiology_portable/RELEASE_NOTES_TEMPLATE.md", "Synthetic draft pointer\n")
        self.write("radiology_portable/radiology/main.py", "Synthetic old application\n")
        self.write("radiology_portable/monai_studies_rad/private_ct.nii.gz", "Synthetic private data\n")
        names = [name for name, _ in packager.source_snapshot(self.source)]
        self.assertIn("radiology_portable/README.md", names)
        self.assertIn("radiology_portable/RELEASE_NOTES_TEMPLATE.md", names)
        self.assertNotIn("radiology_portable/radiology/main.py", names)
        self.assertNotIn("radiology_portable/monai_studies_rad/private_ct.nii.gz", names)

    def test_possible_credential_in_allowlisted_file_is_rejected(self):
        self.write("README.md", "synthetic credential hf_" + "X" * 25)
        with self.assertRaisesRegex(ValueError, "credential"):
            packager.source_snapshot(self.source)

    def test_existing_archive_is_preserved(self):
        archive = self.root / "source.zip"
        archive.write_bytes(b"existing")
        with self.assertRaises(FileExistsError):
            packager.build_archive(self.source, archive, self.root / "models")
        self.assertEqual(b"existing", archive.read_bytes())

    def test_equivalent_source_snapshots_produce_identical_archives(self):
        with patch.object(packager, "validate", return_value=self.fake_gate()):
            first = packager.build_archive(self.source, self.root / "first.zip", self.root / "models")
            second = packager.build_archive(self.source, self.root / "second.zip", self.root / "models")
        self.assertEqual(first["archive_sha256"], second["archive_sha256"])
        self.assertFalse(first["publication_ready"])
        with zipfile.ZipFile(self.root / "first.zip") as archive:
            self.assertTrue(all(info.date_time == packager.FIXED_TIME for info in archive.infolist()))
            self.assertIsNone(archive.testzip())

    def test_failed_preparation_creates_no_archive(self):
        gate = self.fake_gate()
        gate.update(preparation_passed=False, errors=["Synthetic failing gate"])
        archive = self.root / "source.zip"
        with patch.object(packager, "validate", return_value=gate), self.assertRaises(ValueError):
            packager.build_archive(self.source, archive, self.root / "models")
        self.assertFalse(archive.exists())


if __name__ == "__main__":
    unittest.main()
