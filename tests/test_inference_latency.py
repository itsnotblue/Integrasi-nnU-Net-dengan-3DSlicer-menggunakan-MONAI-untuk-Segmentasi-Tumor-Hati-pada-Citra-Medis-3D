"""Strict latency checks use synthetic bytes/metadata, never CTs or a network."""

import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import summarize_inference_run as latency


class InferenceLatencyTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / "tmp"
        scratch.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="latency_test_", dir=scratch)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.run = self.root / "run"
        self.run.mkdir()
        self.manifest_path = self.root / "cohort.json"
        self.report_path = self.run / "run_settings.json"
        self.geometry = {"size_xyz": [8, 7, 6], "spacing_xyz_mm": [0.8, 0.7, 1.5],
                         "origin_xyz_mm": [-10, 20, 30], "direction": [0, -1, 0, 1, 0, 0, 0, 0, 1]}
        self.settings = {"profile": "fast", "device": "cuda", "tile_step_size": 0.5, "threads": 2}
        self.manifest = {"schema_version": 1, "cohort_id": "synthetic_cohort", "input_extent": "full_volume_as_supplied", "cases": []}
        self.report = {"schema_version": 1, "cohort_id": "synthetic_cohort", "settings": self.settings,
                       "resolved_device": "cuda", "checkpoint_sha256": "a" * 64,
                       "started_at_utc": "2026-10-06T00:00:00+00:00", "completed_at_utc": "2026-10-06T00:01:00+00:00",
                       "completed": True, "reference_annotation_used": False, "candidate_profile": True, "cases": []}
        self.sidecars = {}
        for index, seconds in enumerate((20, 12, 10), 1):
            case_id = f"CASE_{index}"
            data = f"synthetic label bytes {case_id}".encode()
            output_hash = hashlib.sha256(data).hexdigest()
            (self.run / f"{case_id}.nii.gz").write_bytes(data)
            self.manifest["cases"].append({"case_id": case_id, "image": f"MISSING_CT_{case_id}.nii.gz",
                                           "reference": f"MISSING_REFERENCE_{case_id}.nii.gz", "image_sha256": str(index) * 64,
                                           "geometry": copy.deepcopy(self.geometry)})
            self.report["cases"].append({"case_id": case_id, "output_sha256": output_hash, "seconds": seconds, "reused": False})
            self.sidecars[case_id] = {"schema_version": 1, "case_id": case_id, "input_sha256": str(index) * 64,
                "checkpoint_sha256": "a" * 64, "output_sha256": output_hash,
                "checkpoint_name": "checkpoint_best.pth", "folds": [0], "configuration": {"patch_size": [4, 4, 4]},
                "settings": {**self.settings, "use_mirroring": False, "use_gaussian": True,
                             "network_precision": "float32", "aggregation": "nnunet_cpu_float16"},
                "candidate_profile": True, "manual_roi": False, "reference_annotation_used": False,
                "input_size_xyz": [8, 7, 6], "output_size_xyz": [8, 7, 6], "spacing_xyz": [0.8, 0.7, 1.5],
                "origin_xyz": [-10, 20, 30], "direction": [0, -1, 0, 1, 0, 0, 0, 0, 1],
                "preprocessing_seconds": 1, "prediction_seconds": 5, "reconstruction_seconds": 1,
                "runtime_seconds_before_file_save": 8, "end_to_end_seconds": seconds, "tiles": index * 2,
                "gpu": "Synthetic GPU", "peak_cuda_allocated_bytes": index * 1024**3,
                "software": {"torch": "synthetic", "nnunetv2": "synthetic", "numpy": "synthetic"}}
        self.save()

    def save(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.report["manifest_sha256"] = latency.sha256(self.manifest_path)
        self.report_path.write_text(json.dumps(self.report), encoding="utf-8")
        for case_id, sidecar in self.sidecars.items():
            (self.run / f"{case_id}.nii.gz.json").write_text(json.dumps(sidecar), encoding="utf-8")

    def summary(self):
        return latency.summarize(self.report_path, self.manifest_path)

    def test_complete_exact_statistics_cold_and_cached(self):
        result = self.summary()
        self.assertEqual(result["case_count"], 3)
        self.assertEqual(result["statistics_all_cases"]["end_to_end_seconds"],
                         {"count": 3, "mean": 14, "median": 12, "min": 10, "max": 20})
        self.assertEqual(result["cases"][0]["cache_state"], "cold_first_new_case")
        self.assertEqual(result["statistics_by_cache_state"]["cached_runtime_case"]["case_count"], 2)
        self.assertEqual(result["cases"][0]["outside_predict_seconds"], 12)
        self.assertEqual(result["cases"][0]["other_runtime_seconds"], 1)
        self.assertTrue(result["output_hashes_verified"])
        self.assertFalse(result["ct_or_reference_files_read"])
        self.assertEqual(result["sum_recorded_case_end_to_end_seconds"], 42)

    def test_no_ct_reference_or_checkpoint_reads(self):
        actual_open = Path.open
        permitted = {self.report_path, self.manifest_path} | set(self.run.glob("CASE_*"))
        def guarded_open(path, *args, **kwargs):
            self.assertIn(path, permitted)
            return actual_open(path, *args, **kwargs)
        with patch.object(Path, "open", guarded_open):
            self.summary()

    def test_incomplete_rejected_even_with_all_sidecars(self):
        for key in ("completed", "completed_at_utc"):
            old = self.report.pop(key)
            self.save()
            with self.assertRaises(ValueError):
                self.summary()
            self.report[key] = old

    def test_duplicate_missing_reordered_and_changed_cases_rejected(self):
        original = copy.deepcopy(self.report["cases"])
        for cases in (original + [original[0]], original[:-1], list(reversed(original)),
                      [{**original[0], "case_id": "WRONG"}] + original[1:]):
            self.report["cases"] = cases
            self.save()
            with self.assertRaises(ValueError):
                self.summary()

    def test_manifest_identity_rejected(self):
        self.report["manifest_sha256"] = "b" * 64
        self.report_path.write_text(json.dumps(self.report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "locked cohort"):
            self.summary()

    def test_missing_and_extra_sidecars_rejected(self):
        target = self.run / "CASE_3.nii.gz.json"
        target.rename(self.run / "EXTRA.nii.gz.json")
        with self.assertRaisesRegex(ValueError, "sidecars"):
            self.summary()

    def test_unreported_prediction_rejected(self):
        (self.run / "UNREPORTED.nii.gz").write_bytes(b"synthetic")
        with self.assertRaisesRegex(ValueError, "prediction files"):
            self.summary()

    def test_case_input_checkpoint_and_output_hashes_rejected(self):
        original = copy.deepcopy(self.sidecars["CASE_2"])
        for key, value in (("case_id", "WRONG"), ("input_sha256", "b" * 64),
                           ("checkpoint_sha256", "b" * 64), ("output_sha256", "b" * 64)):
            self.sidecars["CASE_2"] = {**original, key: value}
            self.save()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.summary()

    def test_actual_output_bytes_checked(self):
        (self.run / "CASE_1.nii.gz").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "output hash/content"):
            self.summary()

    def test_profile_device_threads_precision_and_mirroring_rejected(self):
        original = copy.deepcopy(self.sidecars["CASE_2"]["settings"])
        for key, value in (("profile", "reference"), ("device", "cpu"), ("threads", True),
                           ("threads", 4), ("use_mirroring", True), ("tile_step_size", 1),
                           ("network_precision", "cuda_autocast_float16"), ("use_gaussian", False)):
            self.sidecars["CASE_2"]["settings"] = {**original, key: value}
            self.save()
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.summary()

    def test_missing_dimensions_and_geometry_change_rejected(self):
        original = copy.deepcopy(self.sidecars["CASE_2"])
        for key, value in (("input_size_xyz", None), ("output_size_xyz", [8, 7, 5]),
                           ("spacing_xyz", [0.8, 0.7, 2]), ("origin_xyz", [0, 0, 0]), ("direction", None)):
            self.sidecars["CASE_2"] = {**original, key: value}
            self.save()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.summary()

    def test_nonfinite_negative_missing_and_inconsistent_timings_rejected(self):
        original = copy.deepcopy(self.sidecars["CASE_2"])
        for key, value in (("prediction_seconds", float("nan")), ("preprocessing_seconds", -1),
                           ("tiles", True), ("reconstruction_seconds", None),
                           ("runtime_seconds_before_file_save", 3), ("end_to_end_seconds", 13)):
            self.sidecars["CASE_2"] = {**original, key: value}
            self.save()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.summary()

    def test_different_software_hardware_configuration_rejected(self):
        original = copy.deepcopy(self.sidecars["CASE_2"])
        for key, value in (("software", {"torch": "other"}), ("gpu", "Different GPU"),
                           ("configuration", {"patch_size": [8, 8, 8]})):
            self.sidecars["CASE_2"] = {**original, key: value}
            self.save()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.summary()

    def test_roi_reference_and_profile_claims_required(self):
        original = copy.deepcopy(self.sidecars["CASE_2"])
        for key, value in (("manual_roi", None), ("reference_annotation_used", True),
                           ("candidate_profile", False), ("folds", [False]),
                           ("checkpoint_name", "different.pth")):
            self.sidecars["CASE_2"] = {**original, key: value}
            self.save()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.summary()

    def test_reused_original_cache_state_not_misclassified(self):
        self.report["cases"][0]["reused"] = True
        self.save()
        result = self.summary()
        self.assertEqual([row["cache_state"] for row in result["cases"]],
                         ["reused_original_cache_state_unknown", "cold_first_new_case", "cached_runtime_case"])
        for case in self.report["cases"]:
            case["reused"] = True
        self.save()
        self.assertEqual(self.summary()["statistics_by_cache_state"]["cold_first_new_case"]["case_count"], 0)

    def test_cpu_reference_has_no_gpu_statistics(self):
        self.settings.update({"profile": "reference", "device": "auto"})
        self.report.update({"resolved_device": "cpu", "candidate_profile": False})
        for sidecar in self.sidecars.values():
            sidecar.update({"candidate_profile": False, "gpu": None, "peak_cuda_allocated_bytes": None})
            sidecar["settings"].update({"profile": "reference", "device": "cpu", "use_mirroring": True})
        self.save()
        result = self.summary()
        self.assertEqual(result["statistics_all_cases"]["peak_cuda_allocated_bytes"]["count"], 0)

    def test_export_nonoverwrite_csv_and_no_local_paths(self):
        result = self.summary()
        output = self.root / "new_summary"
        latency.export_summary(result, output)
        text = (output / "inference_latency.json").read_text(encoding="utf-8")
        self.assertNotIn(str(self.root), text)
        self.assertNotIn("MISSING_CT", text)
        self.assertNotIn("MISSING_REFERENCE", text)
        with (output / "per_case_latency.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]["input_size_xyz"], "8x7x6")
        with self.assertRaises(FileExistsError):
            latency.export_summary(result, output)

    def test_duplicate_json_keys_rejected(self):
        self.report_path.write_text('{"completed": false, "completed": true}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            self.summary()


if __name__ == "__main__":
    unittest.main()
