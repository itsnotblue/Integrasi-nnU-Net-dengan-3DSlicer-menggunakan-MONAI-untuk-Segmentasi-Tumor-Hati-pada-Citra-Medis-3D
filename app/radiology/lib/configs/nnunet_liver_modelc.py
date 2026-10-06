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


class NnUNetLiverModelC(TaskConfig):
    """
    nnUNetv2 Liver/Lesion segmentation - Model C (Fine-tuned v2).
    Loads checkpoint from lib/models/model_c folder.
    """

    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner=None):
        self.name = name
        self.model_dir = model_dir
        self.conf = conf or {}
        self.labels = {"liver": 1, "lesion": 2}

    def infer(self) -> InferTask:
        local_model_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "model_c"
        )
        return _NnUNetLiverModelCInfer(self.model_dir, self.conf, self.labels, local_model_path)

    def trainer(self):
        return None

    def strategy(self):
        return None

    def scoring_method(self):
        return None


class _NnUNetLiverModelCInfer(InferTask):
    def __init__(self, model_dir: str, conf: Dict[str, str], labels: Dict[str, int], local_model_path: str = None):
        # Use local model path from app folder (model_c)
        if local_model_path is None:
            local_model_path = os.path.join(
                os.path.dirname(__file__), "..", "models", "model_c"
            )

        self.local_model_path = local_model_path

        # Minimal env setup for preprocessing
        release_runtime = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "runtime"))
        self.nnunet_preprocessed = conf.get(
            "nnunet_preprocessed",
            os.environ.get("nnUNet_preprocessed", os.path.join(release_runtime, "nnUNet_preprocessed")),
        )
        os.environ["nnUNet_preprocessed"] = self.nnunet_preprocessed

        super().__init__(
            type=InferType.SEGMENTATION,
            labels=labels,
            dimension=3,
            description="nnUNetv2 liver+lesion (Model C - Fine-tuned v2)",
        )
        self.model_dir = model_dir
        self.labels = labels
        self.predictor: Optional[nnUNetPredictor] = None

    def is_valid(self) -> bool:
        # Check local model path (root level or fold_0)
        ck_best = os.path.join(self.local_model_path, "checkpoint_best.pth")
        ck_final = os.path.join(self.local_model_path, "checkpoint_final.pth")
        ck_best_fold = os.path.join(self.local_model_path, "fold_0", "checkpoint_best.pth")
        ck_final_fold = os.path.join(self.local_model_path, "fold_0", "checkpoint_final.pth")

        valid = os.path.exists(ck_best) or os.path.exists(ck_final) or os.path.exists(ck_best_fold) or os.path.exists(ck_final_fold)

        if valid:
            logger.info(f"✓ Model C checkpoint found: {self.local_model_path}")
        else:
            logger.warning(f"✗ Model C checkpoint not found in {self.local_model_path}")

        return valid

    def info(self) -> Dict[str, Any]:
        return {
            "type": InferType.SEGMENTATION,
            "labels": self.labels,
            "dimension": 3,
            "description": "nnUNetv2 liver+lesion (Model C - Fine-tuned v2)",
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
        """
        Override main inference call to use nnUNet predictor directly.
        """
        logger.info("Starting nnUNet inference (Model C)")

        # Get input image path
        image_path = request.get("image")
        if not image_path or not os.path.exists(image_path):
            raise ValueError(f"Image not found: {image_path}")

        logger.info(f"Input image: {image_path}")

        # Initialize predictor if not already done
        if self.predictor is None:
            self._init_predictor()

        # Run nnUNet prediction
        try:
            logger.info("Running nnUNet prediction...")

            import tempfile
            temp_dir = tempfile.mkdtemp()
            logger.info(f"Using temporary directory: {temp_dir}")

            predicted_arrays = self.predictor.predict_from_files(
                [[image_path]],
                output_folder_or_list_of_truncated_output_files=temp_dir,
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1
            )

            # Load the prediction that was saved
            pred_files = [f for f in os.listdir(temp_dir) if f.endswith('.nii.gz')]
            if pred_files:
                pred_file = os.path.join(temp_dir, pred_files[0])
                pred_img = sitk.ReadImage(pred_file)
                predicted_array = sitk.GetArrayFromImage(pred_img)
                logger.info(f"Loaded prediction from {pred_file}")
            else:
                raise RuntimeError("No prediction files generated")

            # Clean up temp directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

            logger.info(f"Prediction shape: {predicted_array.shape}")
            logger.info(f"Unique values: {np.unique(predicted_array)}")

        except Exception as e:
            logger.error(f"nnUNet prediction failed: {e}", exc_info=True)
            raise

        # Create result dictionary in MONAILabel format
        result = {
            "image": image_path,
            "pred": predicted_array,
            "label_names": self.labels,
        }

        # Save the result
        result_file_name, result_json = self.writer(result, request)

        return result_file_name, result_json

    def _init_predictor(self):
        """Initialize the nnUNet predictor from local model path"""
        logger.info(f"Initializing nnUNet predictor (Model C) from: {self.local_model_path}")

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
            use_folds=(0,),
            checkpoint_name=ck_name,
        )
        logger.info("✓ nnUNet predictor (Model C) initialized successfully")

    def writer(self, data, request):
        """
        Save the segmentation result.
        """
        logger.info("Saving segmentation result...")

        # Get prediction array
        pred = data.get("pred")
        if pred is None:
            raise ValueError("No prediction found in data")

        # Get image path to extract metadata
        image_path = data.get("image")

        # Read original image to get metadata (spacing, origin, direction)
        try:
            original_image = sitk.ReadImage(image_path)
            spacing = original_image.GetSpacing()
            origin = original_image.GetOrigin()
            direction = original_image.GetDirection()
        except Exception as e:
            logger.warning(f"Could not read original image metadata: {e}")
            spacing = None
            origin = None
            direction = None

        # Convert numpy array to SimpleITK image
        pred_image = sitk.GetImageFromArray(pred.astype(np.uint8))

        # Set metadata if available
        if spacing is not None:
            pred_image.SetSpacing(spacing)
        if origin is not None:
            pred_image.SetOrigin(origin)
        if direction is not None:
            pred_image.SetDirection(direction)

        # Get output path from request
        result_extension = request.get("result_extension", ".nrrd")
        result_dtype = request.get("result_dtype", "uint8")
        result_compress = request.get("result_compress", False)

        # Generate output filename
        session_id = request.get("session_id", "session")
        output_dir = request.get("output_dir") or request.get("studies") or os.environ.get("MONAILABEL_STUDIES")
        if not output_dir:
            release_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
            output_dir = os.path.join(release_root, "data", "studies")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{session_id}{result_extension}")

        # Save the image
        logger.info(f"Saving segmentation to: {output_file}")
        sitk.WriteImage(pred_image, output_file, result_compress)

        # Create result JSON
        result_json = {
            "label_names": self.labels,
            "file": output_file,
            "size": os.path.getsize(output_file) if os.path.exists(output_file) else 0,
        }

        logger.info(f"Segmentation saved successfully: {output_file}")

        return output_file, result_json
