"""Full-volume safety and geometry tests using tiny images and mocked predictors."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import asdict

import numpy as np
import SimpleITK as sitk
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app/radiology"))
from lib.infers import nnunet_runtime as runtime_module
from lib.configs import nnunet_liver_modelb as task_module

SPEC = importlib.util.spec_from_file_location("inference_cli", ROOT / "scripts/run_inference.py")
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)
sys.path.insert(0, str(ROOT / "scripts"))
BATCH_SPEC = importlib.util.spec_from_file_location("cohort_cli", ROOT / "scripts/run_cohort_inference.py")
batch = importlib.util.module_from_spec(BATCH_SPEC)
BATCH_SPEC.loader.exec_module(batch)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / "tmp/inference-tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.root = Path(self.temporary.name)
        self.input = self.root / "synthetic_ct.nii.gz"
        array = np.zeros((5, 6, 7), dtype=np.int16)
        self.original = sitk.GetImageFromArray(array)
        self.original.SetSpacing((0.7, 1.1, 2.3))
        self.original.SetOrigin((-10.0, 5.0, 42.0))
        self.original.SetDirection((0., -1., 0., 1., 0., 0., 0., 0., 1.))
        sitk.WriteImage(self.original, str(self.input))
        # Compare against read geometry, allowing the NIfTI writer's precision.
        self.original = sitk.ReadImage(str(self.input))
        self.segmentation = np.zeros((5, 6, 7), dtype=np.uint8)
        self.segmentation[1:4, 1:5, 1:6] = 1
        self.segmentation[2, 3, 4] = 2
        self.preprocessor = MagicMock()
        self.preprocessor.run_case.return_value = (np.ones((1, 5, 6, 7), dtype=np.float32), None, {"synthetic": True})
        self.predictor = SimpleNamespace(
            configuration_manager=SimpleNamespace(preprocessor_class=MagicMock(return_value=self.preprocessor),
                                                  patch_size=(4, 4, 4), configuration={"configuration": "synthetic"}),
            plans_manager=object(), dataset_json={"labels": {"background": 0, "liver": 1, "lesion": 2}},
            label_manager=object(), device=torch.device("cpu"), use_mirroring=True,
            _internal_get_sliding_window_slicers=MagicMock(return_value=[None, None]),
            predict_logits_from_preprocessed_data=MagicMock(return_value=torch.zeros((3, 5, 6, 7))),
        )
        self.runtime = runtime_module.FullVolumeRuntime.__new__(runtime_module.FullVolumeRuntime)
        self.runtime.settings = runtime_module.InferenceSettings(device="cpu")
        self.runtime.predictor = self.predictor
        self.runtime.checkpoint = self.root / "checkpoint_best.pth"
        self.runtime.checkpoint_hash = "a" * 64

    def tearDown(self):
        self.temporary.cleanup()

    def predict(self, segmentation=None):
        converted = self.segmentation if segmentation is None else segmentation
        with patch.object(runtime_module, "convert_predicted_logits_to_segmentation_with_correct_shape", return_value=converted) as converter:
            output, metadata = self.runtime.predict(self.input)
        return output, metadata, converter

    def test_complete_size_and_physical_geometry_are_preserved(self):
        output, metadata, _ = self.predict()
        self.assertEqual(self.original.GetSize(), output.GetSize())
        self.assertEqual(self.original.GetSpacing(), output.GetSpacing())
        self.assertEqual(self.original.GetOrigin(), output.GetOrigin())
        self.assertEqual(self.original.GetDirection(), output.GetDirection())
        np.testing.assert_array_equal(self.segmentation, sitk.GetArrayFromImage(output))
        self.assertEqual([0, 1, 2], metadata["labels"])
        self.assertFalse(metadata["manual_roi"])
        self.assertFalse(metadata["reference_annotation_used"])
        self.assertEqual(runtime_module.sha256(self.input), metadata["input_sha256"])
        self.assertEqual([7, 6, 5], metadata["output_size_xyz"])
        self.preprocessor.run_case.assert_called_once()
        self.assertEqual([str(self.input.resolve())], self.preprocessor.run_case.call_args.args[0])
        self.assertIsNone(self.preprocessor.run_case.call_args.args[1])
        json.dumps(metadata, allow_nan=False)

    def test_reference_settings_are_reported(self):
        _, metadata, _ = self.predict()
        self.assertEqual("reference", metadata["settings"]["profile"])
        self.assertEqual("cpu", metadata["settings"]["device"])
        self.assertTrue(metadata["settings"]["use_mirroring"])
        self.assertFalse(metadata["candidate_profile"])

    def test_export_honors_recorded_thread_count(self):
        _, _, converter = self.predict()
        self.assertEqual(2, converter.call_args.kwargs["num_threads_torch"])

    def test_no_tumor_is_a_valid_segmentation(self):
        segmentation = self.segmentation.copy()
        segmentation[segmentation == 2] = 1
        _, metadata, _ = self.predict(segmentation)
        self.assertEqual([0, 1], metadata["labels"])

    def test_wrong_prediction_shape_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "shape"):
            self.predict(np.zeros((4, 6, 7), dtype=np.uint8))

    def test_unknown_labels_are_rejected_before_cast(self):
        segmentation = self.segmentation.astype(np.int16)
        segmentation[0, 0, 0] = -254  # uint8 would wrap to the valid class 2.
        with self.assertRaises(RuntimeError):
            self.predict(segmentation)

    def test_fractional_labels_are_rejected_before_cast(self):
        segmentation = self.segmentation.astype(np.float32)
        segmentation[0, 0, 0] = 1.5
        with self.assertRaises(RuntimeError):
            self.predict(segmentation)

    def test_nan_logits_are_rejected_before_export(self):
        self.predictor.predict_logits_from_preprocessed_data.return_value[0, 0, 0, 0] = float("nan")
        with self.assertRaises(RuntimeError):
            self.predict()

    def test_infinite_logits_are_rejected_before_export(self):
        self.predictor.predict_logits_from_preprocessed_data.return_value[0, 0, 0, 0] = float("inf")
        with self.assertRaises(RuntimeError):
            self.predict()

    def test_non_scalar_input_is_rejected(self):
        vector = sitk.GetImageFromArray(np.zeros((5, 6, 7, 2), dtype=np.float32), isVector=True)
        self.input = self.root / "vector.nrrd"
        sitk.WriteImage(vector, str(self.input))
        with self.assertRaisesRegex(ValueError, "scalar 3D"):
            self.predict()
        self.preprocessor.run_case.assert_not_called()

    def test_2d_input_is_rejected(self):
        self.input = self.root / "slice.nrrd"
        sitk.WriteImage(sitk.GetImageFromArray(np.zeros((6, 7), dtype=np.int16)), str(self.input))
        with self.assertRaisesRegex(ValueError, "scalar 3D"):
            self.predict()
        self.preprocessor.run_case.assert_not_called()

    def test_cpu_and_fast_predictor_selection(self):
        model = self.root / "model"
        checkpoint = model / "fold_0/checkpoint_best.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"synthetic checkpoint")
        with patch.object(runtime_module, "nnUNetPredictor") as reference, patch.object(runtime_module, "Float32Predictor") as fast, patch.object(torch, "set_num_threads"):
            runtime_module.FullVolumeRuntime(model, runtime_module.InferenceSettings(device="cpu"))
            self.assertTrue(reference.call_args.kwargs["use_mirroring"])
            runtime_module.FullVolumeRuntime(model, runtime_module.InferenceSettings(profile="fast", device="cpu"))
            self.assertFalse(fast.call_args.kwargs["use_mirroring"])
            self.assertFalse(fast.call_args.kwargs["perform_everything_on_device"])

    def test_unavailable_cuda_gives_explicit_fallback_message(self):
        model = self.root / "model"
        checkpoint = model / "fold_0/checkpoint_best.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"synthetic checkpoint")
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "use --device cpu"):
                runtime_module.FullVolumeRuntime(model, runtime_module.InferenceSettings(device="cuda"))

    def test_cli_does_not_overwrite_prediction_or_sidecar(self):
        for existing in ("result.nrrd", "result.nrrd.json"):
            dest = self.root / "result.nrrd"
            path = self.root / existing
            path.write_bytes(b"original result")
            with patch.object(sys, "argv", ["run_inference.py", "--model-dir", "unused", "--input", str(self.input), "--output", str(dest)]), patch.object(cli, "FullVolumeRuntime") as constructor:
                with self.assertRaises(FileExistsError):
                    cli.main()
            constructor.assert_not_called()
            self.assertEqual(b"original result", path.read_bytes())
            path.unlink()

    def test_model_b_response_saves_geometry_and_unique_results(self):
        task = task_module._NnUNetLiverModelBInfer(str(self.root), {"device": "cpu"}, {"liver": 1, "lesion": 2}, str(self.root))
        fake_runtime = MagicMock()
        output = sitk.GetImageFromArray(self.segmentation)
        output.CopyInformation(self.original)
        fake_runtime.predict.side_effect = [(output, {"labels": [0, 1, 2]}), (output, {"labels": [0, 1, 2]})]
        with patch.object(task_module, "FullVolumeRuntime", return_value=fake_runtime) as constructor:
            first, first_response = task({"image": str(self.input), "output_dir": str(self.root / "outputs"), "result_extension": ".nrrd"})
            second, _ = task({"image": str(self.input), "output_dir": str(self.root / "outputs"), "result_extension": ".nii.gz"})
        constructor.assert_called_once()
        self.assertNotEqual(first, second)
        actual = sitk.ReadImage(first)
        self.assertEqual(self.original.GetSize(), actual.GetSize())
        self.assertEqual(self.original.GetDirection(), actual.GetDirection())
        self.assertTrue(Path(first + ".json").is_file())
        self.assertEqual(runtime_module.sha256(first), first_response["inference"]["output_sha256"])

    def test_invalid_task_output_extension_is_rejected_before_prediction(self):
        task = task_module._NnUNetLiverModelBInfer(str(self.root), {"device": "cpu"}, {"liver": 1, "lesion": 2}, str(self.root))
        task.runtime = MagicMock()
        with self.assertRaises(ValueError):
            task({"image": str(self.input), "output_dir": str(self.root / "outputs"), "result_extension": ".txt"})
        task.runtime.predict.assert_not_called()


class SettingsTests(unittest.TestCase):
    def test_invalid_settings_rejected(self):
        for options in ({"profile": "unknown"}, {"device": "mps"}, {"tile_step_size": 0},
                        {"tile_step_size": 1.1}, {"tile_step_size": float("nan")},
                        {"threads": 0}, {"threads": 1.5}, {"threads": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                runtime_module.InferenceSettings(**options)


class BatchTests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / "tmp/cohort-inference-tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.root = Path(self.temporary.name)
        self.image = self.root / "input_0000.nii.gz"
        self.prediction = sitk.GetImageFromArray(np.zeros((4, 5, 6), dtype=np.uint8))
        self.prediction.SetSpacing((0.8, 1.2, 2.5))
        self.prediction.SetOrigin((-4., 6., 8.))
        sitk.WriteImage(self.prediction, str(self.image))
        self.prediction = sitk.ReadImage(str(self.image))
        self.model = self.root / "model"
        checkpoint = self.model / "fold_0/checkpoint_best.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"synthetic checkpoint")
        self.checkpoint_hash = runtime_module.sha256(checkpoint)
        self.geometry = batch.header_geometry(self.image)
        self.manifest = self.root / "cohort.json"
        self.cohort = {"schema_version": 1, "cohort_id": "synthetic",
                       "input_extent": "full_volume_as_supplied", "cases": [{
                           "case_id": "case_a", "image": self.image.name,
                           "reference": "REFERENCE_MUST_NOT_BE_READ.nii.gz",
                           "image_sha256": runtime_module.sha256(self.image),
                           "geometry": self.geometry}]}
        self.manifest.write_text(json.dumps(self.cohort), encoding="utf-8")
        self.output_dir = self.root / "outputs"
        self.output = self.output_dir / "case_a.nii.gz"
        self.sidecar = Path(str(self.output) + ".json")
        self.settings = runtime_module.InferenceSettings("fast", "cpu", 0.5, 2)

    def tearDown(self):
        self.temporary.cleanup()

    def metadata(self):
        return {"input_sha256": self.cohort["cases"][0]["image_sha256"],
                "checkpoint_sha256": self.checkpoint_hash,
                "settings": {**asdict(self.settings), "use_mirroring": False},
                "manual_roi": False, "reference_annotation_used": False,
                "input_size_xyz": self.geometry["size_xyz"], "output_size_xyz": self.geometry["size_xyz"],
                "spacing_xyz": self.geometry["spacing_xyz_mm"], "origin_xyz": self.geometry["origin_xyz_mm"],
                "direction": self.geometry["direction"], "end_to_end_seconds": 0.1}

    def existing(self, change=None, output_image=None):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(output_image or self.prediction, str(self.output))
        metadata = self.metadata()
        metadata["output_sha256"] = runtime_module.sha256(self.output)
        if change:
            change(metadata)
        self.sidecar.write_text(json.dumps(metadata), encoding="utf-8")
        return metadata

    def validate(self):
        return batch.validate_existing(self.output, self.sidecar, self.cohort["cases"][0]["image_sha256"],
                                       self.checkpoint_hash, self.settings, self.geometry, "cpu")

    def run_main(self, resume=False, runtime=None):
        args = ["run_cohort_inference.py", "--manifest", str(self.manifest), "--model-dir", str(self.model),
                "--output-dir", str(self.output_dir), "--profile", "fast", "--device", "cpu"]
        if resume:
            args.append("--resume")
        fake = runtime or MagicMock(checkpoint_hash=self.checkpoint_hash,
                                    predictor=SimpleNamespace(device=torch.device("cpu")))
        fake.predict.return_value = (self.prediction, self.metadata())
        with patch.object(sys, "argv", args), patch.object(batch, "FullVolumeRuntime", return_value=fake) as constructor:
            batch.main()
        return constructor, fake

    def test_valid_resume_never_loads_network_or_reads_reference(self):
        self.existing()
        constructor, fake = self.run_main(resume=True)
        constructor.assert_not_called()
        fake.predict.assert_not_called()
        report = json.loads((self.output_dir / "run_settings.json").read_text())
        self.assertTrue(report["completed"])
        self.assertTrue(report["cases"][0]["reused"])
        self.assertFalse(report["reference_annotation_used"])

    def test_complete_new_inference_is_evaluator_filename_compatible(self):
        constructor, fake = self.run_main()
        constructor.assert_called_once()
        fake.predict.assert_called_once_with(self.image.resolve())
        self.assertTrue(self.output.is_file())
        self.assertTrue(self.sidecar.is_file())
        self.validate()

    def test_existing_outputs_are_not_overwritten_without_resume(self):
        self.existing()
        original = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_main()
        self.assertEqual(original, self.output.read_bytes())

    def test_missing_sidecar_is_rejected(self):
        self.existing()
        self.sidecar.unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.validate()

    def test_missing_dimensions_are_rejected(self):
        self.existing(lambda metadata: metadata.pop("output_size_xyz"))
        with self.assertRaisesRegex(ValueError, "geometry metadata"):
            self.validate()

    def test_output_geometry_is_checked_despite_matching_hash(self):
        wrong = sitk.Image(self.prediction)
        wrong.SetOrigin((50., 50., 50.))
        self.existing(output_image=wrong)
        with self.assertRaisesRegex(ValueError, "origin"):
            self.validate()

    def test_changed_output_bytes_are_rejected(self):
        self.existing()
        with self.output.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.validate()

    def test_checkpoint_input_settings_and_resolved_device_are_checked(self):
        changes = [lambda m: m.update(checkpoint_sha256="b" * 64),
                   lambda m: m.update(input_sha256="b" * 64),
                   lambda m: m["settings"].update(threads=3),
                   lambda m: m["settings"].update(profile="reference"),
                   lambda m: m["settings"].update(device="cuda"),
                   lambda m: m["settings"].update(use_mirroring=True)]
        for change in changes:
            with self.subTest(change=change):
                self.existing(change)
                with self.assertRaises(ValueError):
                    self.validate()

    def test_crop_or_reference_based_provenance_is_rejected(self):
        for key in ("manual_roi", "reference_annotation_used"):
            with self.subTest(key=key):
                self.existing(lambda metadata: metadata.update({key: True}))
                with self.assertRaisesRegex(ValueError, "provenance"):
                    self.validate()

    def test_conflicting_existing_report_is_preserved(self):
        self.existing()
        report = self.output_dir / "run_settings.json"
        report.write_text(json.dumps({"manifest_sha256": "different"}), encoding="utf-8")
        original = report.read_bytes()
        with self.assertRaisesRegex(ValueError, "different cohort"):
            self.run_main(resume=True)
        self.assertEqual(original, report.read_bytes())

    def test_locked_input_change_is_rejected_before_network(self):
        self.cohort["cases"][0]["image_sha256"] = "b" * 64
        self.manifest.write_text(json.dumps(self.cohort), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Locked image changed"):
            self.run_main()

    def test_locked_input_geometry_change_is_rejected(self):
        self.cohort["cases"][0]["geometry"]["origin_xyz_mm"] = [99., 99., 99.]
        self.manifest.write_text(json.dumps(self.cohort), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "origin"):
            self.run_main()

    def test_wrong_runtime_geometry_is_rejected_before_saving(self):
        self.prediction.SetOrigin((50., 50., 50.))
        with self.assertRaisesRegex(ValueError, "origin"):
            self.run_main()
        self.assertFalse(self.output.exists())

    def test_duplicate_or_unsafe_case_ids_are_rejected(self):
        for case_id in ("../escape", "case_a"):
            bad = dict(self.cohort["cases"][0], case_id=case_id)
            changed = dict(self.cohort, cases=[self.cohort["cases"][0], bad])
            self.manifest.write_text(json.dumps(changed), encoding="utf-8")
            with self.subTest(case_id=case_id), self.assertRaises(ValueError):
                self.run_main()


if __name__ == "__main__":
    unittest.main()
