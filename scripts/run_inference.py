"""Run a pinned checkpoint on a complete image and save geometry + provenance."""
import argparse
import json
from pathlib import Path
import sys
import time
import SimpleITK as sitk
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "radiology"))
from lib.infers.nnunet_runtime import FullVolumeRuntime, InferenceSettings, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", choices=["reference", "fast"], default="reference")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--tile-step-size", type=float, default=0.5)
    args = parser.parse_args()
    dest = Path(args.output).resolve()
    if not str(dest).lower().endswith((".nrrd", ".nii.gz")):
        raise ValueError("output must end in .nrrd or .nii.gz")
    sidecar = Path(str(dest) + ".json")
    if dest.exists() or sidecar.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    runtime = FullVolumeRuntime(args.model_dir, InferenceSettings(args.profile, args.device, args.tile_step_size, args.threads))
    image, metadata = runtime.predict(args.input)
    sitk.WriteImage(image, str(dest), True)
    metadata["output_sha256"] = sha256(dest)
    metadata["end_to_end_seconds"] = time.perf_counter() - start
    sidecar.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    print("FULL_VOLUME_INFERENCE_OK " + json.dumps({"output": str(dest), "dimensions": image.GetSize(), "labels": metadata["labels"], "seconds": metadata["end_to_end_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
