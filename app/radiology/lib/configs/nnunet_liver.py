# SPDX-License-Identifier: Apache-2.0
import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import SimpleITK as sitk
import torch

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType

# nnUNet v2 predictor
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

logger = logging.getLogger(__name__)


class NnUNetLiver(TaskConfig):
    """
    TaskConfig to expose nnUNetv2 Liver/Lesion segmentation inside Radiology app.

    Configuration keys (pass via --conf):
      - nnunet_results: path to nnUNet_results
      - nnunet_preprocessed: path to nnUNet_preprocessed
      - dataset_name: e.g., Dataset408_LTS
      - trainer: e.g., nnUNetTrainer
      - configuration: e.g., nnUNetPlans__3d_lowres
      - fold: e.g., 0
    """

    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner=None):
        self.name = name
        self.model_dir = model_dir
        self.conf = conf or {}
        self.labels = {"liver": 1, "lesion": 2}

    def infer(self) -> InferTask:
        return _NnUNetLiverInfer(self.model_dir, self.conf, self.labels)

    def trainer(self):
        return None

    def strategy(self):
        return None

    def scoring_method(self):
        return None


class _NnUNetLiverInfer(InferTask):
    def __init__(self, model_dir: str, conf: Dict[str, str], labels: Dict[str, int], local_model_path: str = None):
        # Use local model path from app folder (model_a)
        if local_model_path is None:
            local_model_path = os.path.join(
                os.path.dirname(__file__), "..", "models", "model_a"
            )

        self.local_model_path = local_model_path

        # Optional: fallback to nnUNet_results if local model not found
        release_runtime = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "runtime"))
        self.nnunet_results = conf.get(
            "nnunet_results", os.environ.get("nnUNet_results", os.path.join(release_runtime, "nnUNet_results"))
        )
        self.nnunet_preprocessed = conf.get(
            "nnunet_preprocessed",
            os.environ.get("nnUNet_preprocessed", os.path.join(release_runtime, "nnUNet_preprocessed")),
        )

        # Model metadata (for logging only now)
        self.dataset_name = conf.get("dataset_name", "Dataset408_LTS")
        self.trainer = conf.get("trainer", "nnUNetTrainer")
        self.configuration = conf.get("configuration", "nnUNetPlans__3d_lowres")
        self.fold = int(conf.get("fold", 0))

        # Export to env for nnUNet (minimal, for preprocessing only)
        os.environ["nnUNet_results"] = self.nnunet_results
        os.environ["nnUNet_preprocessed"] = self.nnunet_preprocessed

        super().__init__(
            type=InferType.SEGMENTATION,
            labels=labels,
            dimension=3,
            description="Liver and lesion segmentation using nnUNetv2",
        )

        self.model_dir = model_dir
        self.labels = labels
        self.predictor: Optional[nnUNetPredictor] = None

    def is_valid(self) -> bool:
        # Check local model path first (root level or fold_0)
        ck_best = os.path.join(self.local_model_path, "checkpoint_best.pth")
        ck_final = os.path.join(self.local_model_path, "checkpoint_final.pth")
        ck_best_fold = os.path.join(self.local_model_path, "fold_0", "checkpoint_best.pth")
        ck_final_fold = os.path.join(self.local_model_path, "fold_0", "checkpoint_final.pth")

        valid = os.path.exists(ck_best) or os.path.exists(ck_final) or os.path.exists(ck_best_fold) or os.path.exists(ck_final_fold)

        if valid:
            logger.info(f"✓ Model checkpoint found: {self.local_model_path}")
        else:
            logger.warning(f"✗ Checkpoint not found in {self.local_model_path}")

        return valid

    def info(self) -> Dict[str, Any]:
        return {
            "type": InferType.SEGMENTATION,
            "labels": self.labels,
            "dimension": 3,
            "description": "nnUNetv2 liver+lesion",
            "config": {
                "dataset": self.dataset_name,
                "configuration": self.configuration,
                "fold": self.fold,
            },
        }

    def pre_transforms(self, data=None):
        return []

    def inferer(self, data=None):
        return None

    def inverse_transforms(self, data=None):
        return []

    def post_transforms(self, data=None):
        return []

    def __call__(self, request):
        image_path = request.get("image")
        if not image_path or not os.path.exists(image_path):
            raise ValueError(f"Image not found: {image_path}")

        if self.predictor is None:
            self._init_predictor()

        import tempfile, shutil
        temp_dir = tempfile.mkdtemp()
        try:
            # Predict; nnUNet saves .nii.gz files
            _ = self.predictor.predict_from_files(
                [[image_path]],
                output_folder_or_list_of_truncated_output_files=temp_dir,
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1,
            )

            pred_files = [f for f in os.listdir(temp_dir) if f.endswith(".nii.gz")]
            if not pred_files:
                raise RuntimeError("No nnUNet prediction generated")
            pred_path = os.path.join(temp_dir, pred_files[0])
            pred_img = sitk.ReadImage(pred_path)
            pred = sitk.GetArrayFromImage(pred_img)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        # Preserve original image metadata
        orig = sitk.ReadImage(image_path)
        out_img = sitk.GetImageFromArray(pred.astype(np.uint8))
        out_img.SetSpacing(orig.GetSpacing())
        out_img.SetOrigin(orig.GetOrigin())
        out_img.SetDirection(orig.GetDirection())

        # Decide output path under studies
        session_id = request.get("session_id", os.path.splitext(os.path.basename(image_path))[0])
        ext = request.get("result_extension", ".nrrd")
        compress = bool(request.get("result_compress", False))

        # Radiology app writes labels back into studies folder next to image
        studies_root = request.get("studies", os.path.dirname(os.path.dirname(image_path)))
        os.makedirs(studies_root, exist_ok=True)
        out_file = os.path.join(studies_root, f"{session_id}{ext}")
        sitk.WriteImage(out_img, out_file, compress)

        return out_file, {"label_names": self.labels, "file": out_file}

    def _init_predictor(self):
        """Initialize the nnUNet predictor from local model path"""
        logger.info(f"Initializing nnUNet predictor from: {self.local_model_path}")

        # Verify checkpoint exists (check both root and fold_0)
        ck_best = os.path.join(self.local_model_path, "checkpoint_best.pth")
        ck_final = os.path.join(self.local_model_path, "checkpoint_final.pth")
        ck_best_fold = os.path.join(self.local_model_path, "fold_0", "checkpoint_best.pth")
        ck_final_fold = os.path.join(self.local_model_path, "fold_0", "checkpoint_final.pth")

        # Determine which checkpoint exists and use appropriate folder
        if os.path.exists(ck_best):
            ck_name = "checkpoint_best.pth"
            model_folder = self.local_model_path
        elif os.path.exists(ck_final):
            ck_name = "checkpoint_final.pth"
            model_folder = self.local_model_path
        elif os.path.exists(ck_best_fold):
            ck_name = "checkpoint_best.pth"
            model_folder = self.local_model_path
        elif os.path.exists(ck_final_fold):
            ck_name = "checkpoint_final.pth"
            model_folder = self.local_model_path
        else:
            raise FileNotFoundError(f"No checkpoint found in {self.local_model_path} or {os.path.join(self.local_model_path, 'fold_0')}")

        # Initialize predictor from local folder
        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,
        )

        self.predictor.initialize_from_trained_model_folder(
            model_folder,
            use_folds=(self.fold,),
            checkpoint_name=ck_name,
        )
        logger.info("✓ nnUNet predictor initialized successfully")
