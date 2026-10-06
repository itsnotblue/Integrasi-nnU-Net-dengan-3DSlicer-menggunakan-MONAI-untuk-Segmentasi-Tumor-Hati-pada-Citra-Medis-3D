"""Model B baseline: complete-volume inference with geometry and provenance."""
import json
import os
from pathlib import Path
import threading
import time
import uuid

import SimpleITK as sitk
from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
from lib.infers.nnunet_runtime import FullVolumeRuntime, InferenceSettings, sha256


class NnUNetLiverModelB(TaskConfig):
    def init(self, name, model_dir, conf, planner=None):
        self.name, self.model_dir, self.conf = name, model_dir, conf or {}
        self.labels = {"liver": 1, "lesion": 2}

    def infer(self):
        path = self.conf.get("model_b_dir") or Path(__file__).resolve().parent.parent / "models" / "model_b"
        return _NnUNetLiverModelBInfer(self.model_dir, self.conf, self.labels, str(path))

    def trainer(self):
        return None

    def strategy(self):
        return None

    def scoring_method(self):
        return None


class _NnUNetLiverModelBInfer(InferTask):
    def __init__(self, model_dir, conf, labels, local_model_path=None):
        super().__init__(type=InferType.SEGMENTATION, labels=labels, dimension=3,
                         description="nnU-Net liver+tumor (Model B baseline)")
        self.labels = labels
        self.model_dir = model_dir
        self.local_model_path = local_model_path or str(Path(__file__).resolve().parent.parent / "models" / "model_b")
        self.settings = InferenceSettings(
            profile=conf.get("inference_profile", "reference"),
            device=conf.get("device", "auto"),
            tile_step_size=float(conf.get("tile_step_size", "0.5")),
            threads=int(conf.get("inference_threads", "2")),
        )
        self.runtime = None
        self.lock = threading.Lock()

    def is_valid(self):
        root = Path(self.local_model_path)
        return all((root / f).is_file() for f in ("fold_0/checkpoint_best.pth", "plans.json", "dataset.json"))

    def info(self):
        return {"type": InferType.SEGMENTATION, "labels": self.labels, "dimension": 3,
                "description": f"nnU-Net Model B baseline ({self.settings.profile} profile)",
                "inference_profile": self.settings.profile}

    def pre_transforms(self, data=None):
        return []

    def inferer(self, data=None):
        return None

    def inverse_transforms(self, data=None):
        return []

    def post_transforms(self, data=None):
        return []

    def __call__(self, request):
        image = request.get("image")
        if not image or not Path(image).is_file():
            raise ValueError(f"Image not found: {image}")
        extension = request.get("result_extension", ".nrrd")
        if extension not in (".nrrd", ".nii.gz"):
            raise ValueError("result_extension must be .nrrd or .nii.gz")
        with self.lock:
            started = time.perf_counter()
            if self.runtime is None:
                self.runtime = FullVolumeRuntime(self.local_model_path, self.settings)
            prediction, metadata = self.runtime.predict(image)
            root = Path(__file__).resolve().parents[4]
            output_dir = Path(request.get("output_dir") or request.get("studies") or os.environ.get("MONAILABEL_STUDIES") or root / "data" / "studies")
            output_dir.mkdir(parents=True, exist_ok=True)
            output = output_dir / f"model_b_{uuid.uuid4().hex}{extension}"
            sitk.WriteImage(prediction, str(output), True)
            metadata["output_sha256"] = sha256(output)
            metadata["end_to_end_seconds"] = time.perf_counter() - started
            Path(str(output) + ".json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
            return str(output), {"label_names": self.labels, "file": str(output), "size": output.stat().st_size,
                                 "inference": metadata}
