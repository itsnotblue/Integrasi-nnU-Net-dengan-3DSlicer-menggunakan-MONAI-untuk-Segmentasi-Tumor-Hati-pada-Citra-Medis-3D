"""Full-volume nnU-Net inference, with an opt-in accelerated research profile."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from pathlib import Path
import time
import numpy as np
import SimpleITK as sitk
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class InferenceSettings:
    profile: str = "reference"
    device: str = "auto"
    tile_step_size: float = 0.5
    threads: int = 2

    def __post_init__(self):
        if self.profile not in ("reference", "fast"):
            raise ValueError("profile must be reference or fast")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")
        if isinstance(self.threads, bool) or not isinstance(self.threads, int):
            raise ValueError("threads must be a positive integer")
        if not 0 < self.tile_step_size <= 1 or self.threads < 1:
            raise ValueError("Invalid tile_step_size or thread count")


class Float32Predictor(nnUNetPredictor):
    """Float32 network calls; retain nnU-Net's normal Gaussian aggregation."""
    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict(self, x):
        with torch.autocast(self.device.type, enabled=False):
            return super()._internal_maybe_mirror_and_predict(x.float())


class FullVolumeRuntime:
    def __init__(self, model_dir, settings=None):
        self.settings = settings or InferenceSettings()
        self.model_dir = Path(model_dir).resolve()
        self.checkpoint = self.model_dir / "fold_0" / "checkpoint_best.pth"
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        device = self.settings.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --device cpu")
        torch.set_num_threads(self.settings.threads)
        cls = Float32Predictor if self.settings.profile == "fast" else nnUNetPredictor
        self.predictor = cls(tile_step_size=self.settings.tile_step_size, use_gaussian=True,
            use_mirroring=self.settings.profile == "reference", perform_everything_on_device=False,
            device=torch.device(device), verbose=False, verbose_preprocessing=False, allow_tqdm=True)
        self.predictor.initialize_from_trained_model_folder(str(self.model_dir), use_folds=(0,), checkpoint_name=self.checkpoint.name)
        self.checkpoint_hash = sha256(self.checkpoint)

    def predict(self, image_path):
        start = time.perf_counter()
        image_path = Path(image_path).resolve()
        original = sitk.ReadImage(str(image_path))
        if original.GetDimension() != 3 or original.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError("Expected a scalar 3D CT volume")
        p = self.predictor
        preprocessor = p.configuration_manager.preprocessor_class(verbose=False)
        data, _, properties = preprocessor.run_case([str(image_path)], None, p.plans_manager, p.configuration_manager, p.dataset_json)
        preprocessing_seconds = time.perf_counter() - start
        padded_shape = [max(a, b) for a, b in zip(data.shape[1:], p.configuration_manager.patch_size)]
        tiles = len(p._internal_get_sliding_window_slicers(padded_shape))
        print(f"PREPROCESS_OK shape={data.shape} tiles={tiles} profile={self.settings.profile} device={p.device}", flush=True)
        if p.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        predict_start = time.perf_counter()
        logits = p.predict_logits_from_preprocessed_data(torch.from_numpy(np.ascontiguousarray(data)))
        if p.device.type == "cuda":
            torch.cuda.synchronize()
        prediction_seconds = time.perf_counter() - predict_start
        del data
        export_start = time.perf_counter()
        if not all(bool(torch.isfinite(channel).all()) for channel in logits):
            raise RuntimeError("Prediction logits contain non-finite values")
        segmentation = convert_predicted_logits_to_segmentation_with_correct_shape(logits, p.plans_manager,
            p.configuration_manager, p.label_manager, properties, return_probabilities=False,
            num_threads_torch=self.settings.threads)
        del logits
        raw = np.asarray(segmentation)
        if not np.isfinite(raw).all() or not np.isin(raw, [0, 1, 2]).all():
            raise RuntimeError("Prediction contains non-finite or invalid segmentation labels")
        segmentation = np.asarray(segmentation, dtype=np.uint8)
        expected = tuple(reversed(original.GetSize()))
        if segmentation.shape != expected:
            raise RuntimeError(f"Output shape {segmentation.shape} differs from input {expected}")
        labels = [int(v) for v in np.unique(segmentation)]
        if not set(labels).issubset({0, 1, 2}):
            raise RuntimeError(f"Unexpected prediction labels: {labels}")
        output = sitk.GetImageFromArray(segmentation)
        output.CopyInformation(original)
        versions = {}
        for package in ("torch", "nnunetv2", "monailabel", "SimpleITK", "numpy"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        metadata = {"schema_version": 1, "time_utc": datetime.now(timezone.utc).isoformat(),
            "input_name": image_path.name, "input_sha256": sha256(image_path),
            "checkpoint_name": self.checkpoint.name, "checkpoint_sha256": self.checkpoint_hash,
            "folds": [0], "configuration": p.configuration_manager.configuration,
            "settings": {**asdict(self.settings), "device": str(p.device), "use_mirroring": p.use_mirroring,
                "network_precision": "float32" if self.settings.profile == "fast" or p.device.type == "cpu" else "cuda_autocast_float16",
                "aggregation": "nnunet_cpu_float16", "use_gaussian": True},
            "input_size_xyz": list(original.GetSize()), "output_size_xyz": list(output.GetSize()),
            "spacing_xyz": list(original.GetSpacing()), "origin_xyz": list(original.GetOrigin()), "direction": list(original.GetDirection()),
            "labels": labels, "label_definition": {"0": "background", "1": "liver", "2": "tumor"},
            "tiles": tiles, "manual_roi": False, "reference_annotation_used": False,
            "preprocessing_seconds": preprocessing_seconds, "prediction_seconds": prediction_seconds,
            "reconstruction_seconds": time.perf_counter() - export_start,
            "runtime_seconds_before_file_save": time.perf_counter() - start, "software": versions,
            "gpu": torch.cuda.get_device_name(p.device) if p.device.type == "cuda" else None,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if p.device.type == "cuda" else None,
            "candidate_profile": self.settings.profile == "fast"}
        return output, metadata
