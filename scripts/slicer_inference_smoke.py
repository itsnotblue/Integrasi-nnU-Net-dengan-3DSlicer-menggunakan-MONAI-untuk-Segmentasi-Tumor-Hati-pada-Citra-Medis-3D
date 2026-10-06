"""Run Model B through Slicer's MONAI Label extension, then save Four-Up.

Launch a NEW Slicer process. Set LIVER_AUDIT_IMAGE, LIVER_AUDIT_OUTPUT_DIR,
and optionally LIVER_AUDIT_SERVER (default http://127.0.0.1:8002).
The existing viewer script checks geometry and saves the editable segmentation.
This script never reads a ground-truth annotation or initiates training.
"""
import json
import os
from pathlib import Path
import runpy
import time
import traceback
from urllib.parse import urlparse
import slicer


try:
    image = Path(os.environ["LIVER_AUDIT_IMAGE"])
    output = Path(os.environ["LIVER_AUDIT_OUTPUT_DIR"])
    server = os.environ.get("LIVER_AUDIT_SERVER", "http://127.0.0.1:8002")
    if urlparse(server).hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("This smoke test only permits a local server; do not upload patient scans")
    output.mkdir(parents=True, exist_ok=True)
    prediction = output / "model_b_slicer_prediction.nrrd"
    response_path = output / "slicer_inference.json"
    if prediction.exists() or response_path.exists():
        raise FileExistsError("Use a new output directory; refusing to overwrite inference")
    if not image.is_file():
        raise FileNotFoundError(image)
    slicer.util.selectModule("MONAILabel")
    widget = slicer.modules.monailabel.widgetRepresentation().self()
    widget.ui.serverComboBox.setCurrentText(server)
    widget.updateServerSettings()
    info = widget.logic.info()
    model = "nnunet_liver_modelb"
    if model not in info["models"]:
        raise ValueError("Server does not register Model B")
    started = time.perf_counter()
    result, response = widget.logic.infer(model, image.name, params={}, file=str(image))
    seconds = time.perf_counter() - started
    if not result or not Path(result).is_file():
        raise RuntimeError("MONAI Label did not return a result file")
    labelmap = slicer.util.loadLabelVolume(result)
    if not slicer.util.saveNode(labelmap, str(prediction)):
        raise RuntimeError("Could not save the returned labelmap")
    slicer.mrmlScene.RemoveNode(labelmap)
    report = {"server": server, "model": model, "input_name": image.name,
              "input_extent": os.environ.get("LIVER_AUDIT_INPUT_EXTENT", "unknown"),
              "prediction": prediction.name, "slicer_client_end_to_end_seconds": seconds,
              "response": response, "reference_annotation_used": False}
    with response_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    os.environ["LIVER_AUDIT_PREDICTION"] = str(prediction)
    print("SLICER_INFERENCE_OK seconds=" + str(seconds), flush=True)
    runpy.run_path(str(Path(__file__).with_name("slicer_view_result.py")), run_name="__main__")
except Exception:
    print("SLICER_INFERENCE_FAILED " + traceback.format_exc(), flush=True)
    slicer.app.exit(1)
