"""Public aggregates must retain uncertainty/provenance without local paths."""

import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_public_evaluation import COMPONENT_KEYS, METRIC_KEYS, public_fast_report


class PublicEvaluationTests(unittest.TestCase):
    def setUp(self):
        case = {"case_id": "LITS_121", "prediction_sha256": "prediction",
                "prediction_provenance": {"sidecar_sha256": "sidecar", "contents": {
                    "input": "C:" + "/Users/" + "private/raw_scan.nii.gz", "token": "secret", "end_to_end_seconds": 12.0,
                    "settings": {"profile": "fast", "device": "cuda", "use_mirroring": False,
                                 "tile_step_size": 0.5, "threads": 2, "private_path": "secret"}}},
                "classes": {name: dict.fromkeys(METRIC_KEYS, 0) for name in ("liver_label_1", "tumor_label_2", "liver_region_1_or_2")},
                "lesions": dict.fromkeys(COMPONENT_KEYS, 0), "failure_flags": []}
        self.current = {"model": "model_b", "checkpoint_sha256": "current", "checkpoint_attribution": "verified_prediction_sidecars",
                        "current_provenance_verified_all_cases": True, "completed_cohort_run_verified": True,
                        "manifest_sha256": "manifest", "protocol": {}, "cohort_id": "cohort", "software": {}, "summary": {},
                        "per_case": [case], "inference_settings": {"sha256": "run", "contents": {"private_path": "secret"}}}
        self.historical = {"model": "model_b", "checkpoint_sha256": None, "checkpoint_attribution": "historical_unpinned",
                           "manifest_sha256": "manifest", "protocol": {}, "summary": {}, "per_case": [case]}
        self.paired = {"case_ids": ["LITS_121"], "left_checkpoint_sha256": "current", "right_checkpoint_sha256": None,
                       "left_checkpoint_attribution": "verified_prediction_sidecars", "right_checkpoint_attribution": "historical_unpinned",
                       "left_model": "model_b", "right_model": "model_b", "difference_direction": "left minus right",
                       "cohort_id": "cohort", "held_out_certified": False, "bootstrap": {}, "paired_differences": {},
                       "left_evaluation_sha256": "left", "right_evaluation_sha256": "right"}

    def test_allowlist_omits_private_provenance(self):
        report = public_fast_report(self.current, self.historical, self.paired)
        encoded = json.dumps(report)
        for forbidden in ("secret", "C:" + "/Users", "raw_scan", "private_path", "token"):
            self.assertNotIn(forbidden, encoded)
        self.assertIsNone(report["historical_comparator"]["checkpoint_sha256"])
        self.assertFalse(report["held_out_certified"])

    def test_unverified_or_incomplete_current_report_rejected(self):
        for key, value in (("checkpoint_attribution", "supplied_checkpoint_hash"), ("completed_cohort_run_verified", False)):
            report = copy.deepcopy(self.current)
            report[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                public_fast_report(report, self.historical, self.paired)

    def test_misattributed_comparator_and_non_lits_ids_rejected(self):
        historical = copy.deepcopy(self.historical)
        historical["checkpoint_sha256"] = "invented"
        with self.assertRaises(ValueError):
            public_fast_report(self.current, historical, self.paired)
        current = copy.deepcopy(self.current)
        current["per_case"][0]["case_id"] = "Patient Name"
        with self.assertRaises(ValueError):
            public_fast_report(current, self.historical, self.paired)


if __name__ == "__main__":
    unittest.main()
