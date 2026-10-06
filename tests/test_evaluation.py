"""Numerical and integrity regression tests for publication evaluation."""

import argparse
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluation_common import assert_geometry, geometry, validate_labels
from evaluate_predictions import binary_metrics, describe, evaluate_case, lesion_metrics, verify_historical_counts
from lock_cohort import lock_cohort
from summarize_historical_metrics import extract_cases
from compare_evaluations import compare


class EvaluationTests(unittest.TestCase):
    def test_empty_policy_does_not_invent_sensitivity(self):
        empty = np.zeros((5, 5, 5), dtype=bool)
        positive = empty.copy()
        positive[2, 2, 2] = True
        both = binary_metrics(empty, empty, (1, 1, 1))
        self.assertEqual(both["dice"], 1)
        self.assertIsNone(both["recall"])
        missing = binary_metrics(empty, positive, (1, 1, 1))
        self.assertEqual(missing["recall"], 0)
        self.assertIsNone(missing["hd95_mm"])
        self.assertEqual(missing["surface_status"], "undefined_one_empty")

    def test_physical_surface_distance_uses_anisotropic_spacing(self):
        pred, ref = np.zeros((8, 8, 8), bool), np.zeros((8, 8, 8), bool)
        ref[2, 3, 3], pred[3, 3, 3] = True, True
        result = binary_metrics(pred, ref, (2.5, 0.7, 0.7))
        self.assertEqual(result["hd95_mm"], 2.5)
        self.assertEqual(result["assd_mm"], 2.5)
        self.assertEqual(result["dice"], 0)

    def test_sparse_surface_queries_equal_independent_dense_edt(self):
        rng = np.random.default_rng(33)
        pred, ref = rng.random((12, 13, 14)) > 0.9, rng.random((12, 13, 14)) > 0.85
        spacing = (2.3, 0.7, 1.1)
        result = binary_metrics(pred, ref, spacing)
        structure = ndimage.generate_binary_structure(3, 1)
        pred_surface = pred ^ ndimage.binary_erosion(pred, structure=structure, border_value=0)
        ref_surface = ref ^ ndimage.binary_erosion(ref, structure=structure, border_value=0)
        expected = np.concatenate((
            ndimage.distance_transform_edt(~ref_surface, sampling=spacing)[pred_surface],
            ndimage.distance_transform_edt(~pred_surface, sampling=spacing)[ref_surface]))
        self.assertAlmostEqual(result["hd95_mm"], float(np.percentile(expected, 95)), places=12)
        self.assertAlmostEqual(result["assd_mm"], float(expected.mean()), places=12)

    def test_one_merged_prediction_does_not_detect_two_lesions(self):
        ref, pred = np.zeros((5, 5, 10), bool), np.zeros((5, 5, 10), bool)
        ref[2, 2, 2], ref[2, 2, 6] = True, True
        pred[2, 2, 2:7] = True
        result = lesion_metrics(pred, ref, (1, 1, 1), iou_threshold=0.1)
        self.assertEqual(result["reference_lesions"], 2)
        self.assertEqual(result["predicted_lesions"], 1)
        self.assertEqual(result["matched_lesions"], 1)
        self.assertEqual(result["missed_lesions"], 1)

    def test_lesion_miss_and_false_positive(self):
        ref, pred = np.zeros((10, 10, 10), bool), np.zeros((10, 10, 10), bool)
        ref[1, 1, 1], ref[8, 8, 8] = True, True
        pred[1, 1, 1], pred[4, 4, 4] = True, True
        result = lesion_metrics(pred, ref, (2, 1, 1))
        self.assertEqual(result["lesion_recall"], 0.5)
        self.assertEqual(result["false_positive_lesions"], 1)
        self.assertEqual(result["reference_lesion_details"][0]["reference_volume_mm3"], 2)

    def test_bootstrap_is_deterministic_and_singleton_has_no_ci(self):
        self.assertEqual(describe([0, 0.5, 1, None], 100, 4), describe([0, 0.5, 1, None], 100, 4))
        self.assertIsNone(describe([0.5], 100)["mean_ci95"])
        self.assertEqual(describe([0.5, None])["n_undefined"], 1)

    def test_paired_comparison_preserves_membership_and_pairing(self):
        classes = {"tumor_label_2": {name: 0.7 for name in ("dice", "iou", "precision", "recall", "hd95_mm", "assd_mm")}}
        left = {"manifest_sha256": "locked", "protocol": {"definition": "same"}, "model": "b", "checkpoint_sha256": None,
                "checkpoint_attribution": "historical_unpinned", "cohort_id": "test",
                "per_case": [{"case_id": "one", "classes": copy.deepcopy(classes)}, {"case_id": "two", "classes": copy.deepcopy(classes)}]}
        right = copy.deepcopy(left)
        right["model"] = "c"
        right["per_case"][0]["classes"]["tumor_label_2"]["dice"] = 0.5
        right["per_case"].reverse()
        result = compare(left, right, 100, 4)
        self.assertAlmostEqual(result["paired_differences"]["tumor_label_2"]["dice"]["mean"], 0.1)
        right["manifest_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "same locked"):
            compare(left, right)

    def test_geometry_rejects_origin_and_spacing_changes(self):
        image = sitk.Image(3, 4, 5, sitk.sitkUInt8)
        expected = geometry(image)
        image.SetOrigin((1, 0, 0))
        with self.assertRaisesRegex(ValueError, "origin"):
            assert_geometry(geometry(image), expected)
        with self.assertRaises(ValueError):
            validate_labels(np.array([0, 1, 2, 3]), "test")

    def test_lock_and_evaluation_detect_changed_reference(self):
        with tempfile.TemporaryDirectory(prefix="evaluation_test_", dir=Path.cwd()) as directory:
            root = Path(directory)
            image = sitk.GetImageFromArray(np.zeros((6, 7, 8), dtype=np.int16))
            ref_array = np.zeros((6, 7, 8), dtype=np.uint8)
            ref_array[2:4, 2:4, 2:4] = 1
            ref_array[2, 2, 2] = 2
            reference = sitk.GetImageFromArray(ref_array)
            sitk.WriteImage(image, str(root / "case_0000.nii.gz"))
            sitk.WriteImage(reference, str(root / "case.nii.gz"))
            sitk.WriteImage(reference, str(root / "prediction.nii.gz"))
            args = argparse.Namespace(cases="case", output=str(root / "manifest.json"),
                split_file=None, split_model="model_b", image_dir=str(root), reference_dir=str(root),
                image_pattern="{case_id}_0000.nii.gz", reference_pattern="{case_id}.nii.gz",
                purpose="technical_full_volume", cohort_id="synthetic", fold=0)
            manifest = lock_cohort(args)
            self.assertFalse(manifest["held_out_certified"])
            result = evaluate_case(manifest["cases"][0], root / "manifest.json", root / "prediction.nii.gz", 0.1)
            self.assertEqual(result["classes"]["tumor_label_2"]["dice"], 1)
            self.assertFalse(result["failure_flags"])
            summary = {"metric_per_case": [{"prediction_file": "/old/pred/case.nii.gz",
                "reference_file": "/old/gt/case.nii.gz", "metrics": {
                    "1": {"TP": 7, "FP": 0, "FN": 0, "TN": ref_array.size - 7, "n_ref": 7, "n_pred": 7, "Dice": 1, "IoU": 1},
                    "2": {"TP": 1, "FP": 0, "FN": 0, "TN": ref_array.size - 1, "n_ref": 1, "n_pred": 1, "Dice": 1, "IoU": 1}}}]}
            recovered = extract_cases(summary, manifest, root / "manifest.json")
            self.assertEqual(recovered[0]["classes"]["tumor_label_2"]["recall"], 1)
            verify_historical_counts(result, summary["metric_per_case"][0])
            self.assertTrue(result["historical_confusion_counts_verified"])
            summary["metric_per_case"][0]["metrics"]["2"]["n_ref"] = 2
            with self.assertRaisesRegex(ValueError, "confusion counts differ"):
                verify_historical_counts(result, summary["metric_per_case"][0])
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                extract_cases(summary, manifest, root / "manifest.json")
            ref_array[2, 2, 2] = 0
            sitk.WriteImage(sitk.GetImageFromArray(ref_array), str(root / "case.nii.gz"))
            with self.assertRaisesRegex(ValueError, "changed after cohort lock"):
                evaluate_case(manifest["cases"][0], root / "manifest.json", root / "prediction.nii.gz", 0.1)


if __name__ == "__main__":
    unittest.main()
