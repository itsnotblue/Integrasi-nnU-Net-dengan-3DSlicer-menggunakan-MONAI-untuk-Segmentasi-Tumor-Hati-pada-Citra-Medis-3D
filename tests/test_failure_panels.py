"""Verify retrospective plane choice, image export and provenance on a phantom."""

import argparse
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from create_failure_panels import create_panels
from evaluation_common import read_json
from lock_cohort import lock_cohort


class FailurePanelTests(unittest.TestCase):
    def test_full_field_of_view_and_reference_only_selection(self):
        with tempfile.TemporaryDirectory(prefix="panel_test_", dir=Path.cwd()) as directory:
            root = Path(directory)
            ct = np.zeros((6, 20, 24), dtype=np.int16)
            reference = np.zeros_like(ct, dtype=np.uint8)
            reference[2, 4:9, 5:12] = 1
            reference[2, 5:7, 6:8] = 2
            reference[4, 5, 6] = 2
            sitk.WriteImage(sitk.GetImageFromArray(ct), str(root / "case_0000.nii.gz"))
            sitk.WriteImage(sitk.GetImageFromArray(reference), str(root / "case.nii.gz"))
            manifest = root / "manifest.json"
            lock_cohort(argparse.Namespace(cases="case", output=str(manifest), split_file=None,
                split_model="b", image_dir=str(root), reference_dir=str(root), image_pattern="{case_id}_0000.nii.gz",
                reference_pattern="{case_id}.nii.gz", purpose="technical_full_volume", cohort_id="phantom", fold=0))
            output = root / "panels.png"
            create_panels(argparse.Namespace(manifest=str(manifest), case_ids="case", prediction_dirs=[str(root)],
                model_labels=["Phantom"], evaluations=None, prediction_pattern="{case_id}.nii.gz", output=str(output),
                hu_min=-150, hu_max=250, dpi=50))
            metadata = read_json(str(output) + ".json")
            self.assertEqual(metadata["cases"][0]["slice_index_z"], 2)
            self.assertTrue(metadata["cases"][0]["reference_used_for_figure_only"])
            self.assertEqual(metadata["cases"][0]["source_geometry"]["size_xyz"], [24, 20, 6])
            with Image.open(output) as image:
                self.assertEqual(image.size, (600, 200))


if __name__ == "__main__":
    unittest.main()
