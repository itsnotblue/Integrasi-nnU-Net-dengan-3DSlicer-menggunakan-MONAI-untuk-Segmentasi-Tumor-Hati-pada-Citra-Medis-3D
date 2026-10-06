"""Reject unproved checkpoint attribution and incomplete cohort evaluation."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest

import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluation_common import geometry, sha256, write_json_new
from evaluate_predictions import preflight_current_predictions, validate_cohort_run, validate_prediction_metadata


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.case = {"case_id": "one", "image_sha256": "input", "geometry": geometry(sitk.Image(5, 6, 7, sitk.sitkUInt8))}
        g = self.case["geometry"]
        self.metadata = {"input_sha256": "input", "checkpoint_sha256": "checkpoint", "output_sha256": "output",
                         "case_id": "one", "manual_roi": False, "reference_annotation_used": False,
                         "input_size_xyz": g["size_xyz"], "output_size_xyz": g["size_xyz"],
                         "spacing_xyz": g["spacing_xyz_mm"], "origin_xyz": g["origin_xyz_mm"], "direction": g["direction"],
                         "end_to_end_seconds": 12.0, "labels": [0, 1, 2],
                         "settings": {"profile": "fast", "device": "cpu", "use_mirroring": False, "tile_step_size": 0.5, "threads": 2}}
        self.manifest = {"cohort_id": "locked", "input_extent": "full_volume_as_supplied", "cases": [self.case]}
        self.run = {"completed": True, "completed_at_utc": "today", "manifest_sha256": "manifest", "cohort_id": "locked",
                    "checkpoint_sha256": "checkpoint", "reference_annotation_used": False, "resolved_device": "cpu",
                    "settings": {"profile": "fast", "device": "auto", "tile_step_size": 0.5, "threads": 2},
                    "cases": [{"case_id": "one", "output_sha256": "output", "seconds": 12.0}]}

    def validate(self, metadata):
        return validate_prediction_metadata(metadata, self.case, "checkpoint", "output")

    def test_valid_sidecar_and_completed_run(self):
        rows = validate_cohort_run(self.run, self.manifest, "manifest", "checkpoint")
        self.assertEqual(validate_prediction_metadata(self.metadata, self.case, "checkpoint", "output", self.run, rows["one"]), self.metadata)

    def test_supplied_checkpoint_cannot_invent_attribution(self):
        for key in ("input_sha256", "checkpoint_sha256", "output_sha256"):
            bad = copy.deepcopy(self.metadata)
            bad[key] = "wrong"
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(bad)

    def test_geometry_and_reference_claims_must_be_explicit(self):
        for key, value in (("manual_roi", True), ("reference_annotation_used", True), ("input_size_xyz", [5, 6, 6]), ("origin_xyz", [1, 0, 0])):
            bad = copy.deepcopy(self.metadata)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(bad)
        bad = copy.deepcopy(self.metadata)
        del bad["manual_roi"]
        with self.assertRaises(ValueError):
            self.validate(bad)

    def test_inconsistent_settings_and_nan_timing_are_rejected(self):
        for key, value in (("use_mirroring", True), ("threads", True), ("tile_step_size", 0)):
            bad = copy.deepcopy(self.metadata)
            bad["settings"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(bad)
        bad = copy.deepcopy(self.metadata)
        bad["end_to_end_seconds"] = float("nan")
        with self.assertRaises(ValueError):
            self.validate(bad)

    def test_partial_or_mismatched_batch_is_not_complete_evidence(self):
        for key, value in (("completed", False), ("manifest_sha256", "wrong"), ("checkpoint_sha256", "wrong"), ("cases", [])):
            bad = copy.deepcopy(self.run)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_cohort_run(bad, self.manifest, "manifest", "checkpoint")

    def test_per_case_output_and_settings_match_completed_run(self):
        for key, value in (("output_sha256", "wrong"), ("seconds", 5.0)):
            row = copy.deepcopy(self.run["cases"][0])
            row[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_prediction_metadata(self.metadata, self.case, "checkpoint", "output", self.run, row)

    def test_sidecar_is_required_and_actual_header_is_checked(self):
        with tempfile.TemporaryDirectory(prefix="provenance_test_", dir=Path.cwd()) as directory:
            root = Path(directory)
            prediction = root / "one.nii.gz"
            sitk.WriteImage(sitk.Image(5, 6, 7, sitk.sitkUInt8), str(prediction))
            manifest = root / "manifest.json"
            write_json_new(manifest, self.manifest)
            with self.assertRaisesRegex(ValueError, "requires.*sidecar"):
                preflight_current_predictions(self.manifest, manifest, {"one": prediction}, "checkpoint")
            metadata = copy.deepcopy(self.metadata)
            metadata["output_sha256"] = sha256(prediction)
            write_json_new(str(prediction) + ".json", metadata)
            self.assertEqual(set(preflight_current_predictions(self.manifest, manifest, {"one": prediction}, "checkpoint")), {"one"})
            wrong = sitk.Image(5, 6, 7, sitk.sitkUInt8)
            wrong.SetOrigin((1, 0, 0))
            sitk.WriteImage(wrong, str(prediction))
            # Even if a new byte hash is declared, the actual physical header
            # must still match the locked case's original coordinates.
            metadata["output_sha256"] = sha256(prediction)
            sidecar = Path(str(prediction) + ".json")
            sidecar.unlink()
            write_json_new(sidecar, metadata)
            with self.assertRaisesRegex(ValueError, "origin"):
                preflight_current_predictions(self.manifest, manifest, {"one": prediction}, "checkpoint")


if __name__ == "__main__":
    unittest.main()
