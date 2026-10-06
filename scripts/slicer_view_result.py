"""Load a complete inference result in a NEW Slicer process and save four views.

Launch Slicer with --python-script and environment variables LIVER_AUDIT_IMAGE,
LIVER_AUDIT_PREDICTION, LIVER_AUDIT_OUTPUT_DIR. No reference mask is needed.
"""
import json
import os
from pathlib import Path
import traceback
from itertools import product
import numpy as np
import qt
import slicer
import vtk


def run():
    image_path = Path(os.environ["LIVER_AUDIT_IMAGE"])
    prediction_path = Path(os.environ["LIVER_AUDIT_PREDICTION"])
    output = Path(os.environ["LIVER_AUDIT_OUTPUT_DIR"])
    extent = os.environ.get("LIVER_AUDIT_INPUT_EXTENT", "unknown")
    if extent not in ("full_volume", "technical_crop", "unknown"):
        raise ValueError("LIVER_AUDIT_INPUT_EXTENT must be full_volume, technical_crop or unknown")
    output.mkdir(parents=True, exist_ok=True)
    screenshot = output / "slicer_full_volume_four_up.png"
    segmentation_path = output / "model_b_full_volume.seg.nrrd"
    if screenshot.exists() or segmentation_path.exists():
        raise FileExistsError("Use a new output directory; refusing to overwrite visualization")
    if slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLScalarVolumeNode"):
        raise RuntimeError("Run this script in a new Slicer instance to preserve existing scenes")
    volume = slicer.util.loadVolume(str(image_path))
    prediction = slicer.util.loadLabelVolume(str(prediction_path))
    if volume.GetImageData().GetDimensions() != prediction.GetImageData().GetDimensions():
        raise RuntimeError("Volume dimensions differ")
    matrices = []
    for node in (volume, prediction):
        matrix = vtk.vtkMatrix4x4()
        node.GetIJKToRASMatrix(matrix)
        matrices.append(np.array([[matrix.GetElement(i, j) for j in range(4)] for i in range(4)]))
    if not np.allclose(*matrices, rtol=0, atol=1e-5):
        raise RuntimeError("Volume physical geometry differs")
    manager = slicer.app.layoutManager()
    manager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
    segmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Model B complete-input prediction")
    segmentation.SetReferenceImageGeometryParameterFromVolumeNode(volume)
    segmentation.CreateDefaultDisplayNodes()
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(prediction, segmentation)
    segments = segmentation.GetSegmentation()
    array = slicer.util.arrayFromVolume(prediction)
    foreground_labels = [int(v) for v in np.unique(array) if int(v) != 0]
    if segments.GetNumberOfSegments() != len(foreground_labels):
        raise RuntimeError("Imported segment count differs from prediction labels")
    for index in range(segments.GetNumberOfSegments()):
        segment = segments.GetNthSegment(index)
        label = foreground_labels[index]
        segment.SetName("Liver" if label == 1 else "Tumor")
        segment.SetColor(*( (0.22, 0.75, 0.35) if label == 1 else (1.0, 0.82, 0.12) ))
    segmentation.CreateClosedSurfaceRepresentation()
    display = segmentation.GetDisplayNode()
    display.SetVisibility3D(True)
    display.SetOpacity3D(0.65)
    display.SetOpacity2DFill(0.25)
    display.SetOpacity2DOutline(0.8)
    # Use segmentation overlays so slice/3D colors remain consistent.
    slicer.util.setSliceViewerLayers(background=volume)
    ct_display = volume.GetDisplayNode()
    ct_display.AutoWindowLevelOff()
    ct_display.SetWindowLevel(400, 60)
    target = 2 if np.any(array == 2) else 1
    counts = np.count_nonzero(array == target, axis=(1, 2))
    k = int(np.argmax(counts))
    coords = np.argwhere(array[k] == target)
    if len(coords):
        j, i = np.median(coords, axis=0)
    else:
        i, j = (dim / 2 for dim in volume.GetImageData().GetDimensions()[:2])
    ras = matrices[0] @ np.array([i, j, k, 1])
    slicer.util.selectModule("Segmentations")
    slicer.util.mainWindow().resize(1600, 1000)
    slicer.app.processEvents()
    slicer.util.resetSliceViews()
    for name in ("Red", "Green", "Yellow"):
        # Change only the plane's normal offset, preserving the CT-centered
        # in-plane fit. Centering on a peripheral tumor can clip the scan.
        manager.sliceWidget(name).mrmlSliceNode().JumpSliceByOffsetting(*ras[:3])
    slicer.app.processEvents()
    # Fit after the window/layout has its final aspect ratio. Include all
    # predicted islands, including false positives: do not hide model errors.
    view = manager.threeDWidget(0).threeDView()
    renderer = view.renderWindow().GetRenderers().GetFirstRenderer()
    renderer.ResetCamera()
    renderer.GetActiveCamera().Zoom(0.85)
    renderer.ResetCameraClippingRange()
    view.forceRender()
    if not slicer.util.saveNode(segmentation, str(segmentation_path)):
        raise RuntimeError("Could not save editable segmentation")

    def capture_impl():
        coverage = {}
        corners = [matrices[0] @ np.array([i, j, k, 1])
                   for i, j, k in product(*[(0, d - 1) for d in volume.GetImageData().GetDimensions()])]
        for name in ("Red", "Green", "Yellow"):
            node = manager.sliceWidget(name).mrmlSliceNode()
            transform = vtk.vtkMatrix4x4()
            vtk.vtkMatrix4x4.Invert(node.GetXYToRAS(), transform)
            xy = np.array([transform.MultiplyPoint(point)[:2] for point in corners])
            dimensions = list(node.GetDimensions()[:2])
            lower, upper = xy.min(axis=0), xy.max(axis=0)
            inside = bool(np.all(lower >= -1) and np.all(upper <= np.array(dimensions) + 1))
            coverage[name] = {"full_ct_projection_inside_view": inside,
                              "ct_bounds_xy": [lower.tolist(), upper.tolist()], "view_size_xy": dimensions}
            if not inside:
                raise RuntimeError(f"Full CT projection extends outside {name} viewport: {coverage[name]}")
        if not slicer.util.mainWindow().grab().save(str(screenshot)):
            raise RuntimeError("Could not save screenshot")
        report = {"input_name": image_path.name, "prediction_name": prediction_path.name,
                  "size_xyz": list(volume.GetImageData().GetDimensions()), "geometry_match": True,
                  "labels": [int(v) for v in np.unique(array)], "segments": segments.GetNumberOfSegments(),
                  "views": ["axial", "coronal", "sagittal", "3D"], "input_extent": extent,
                  "manual_roi_applied_by_viewer": False,
                  "slice_coverage": coverage,
                  "slicer_version": slicer.app.applicationVersion, "screenshot": screenshot.name,
                  "segmentation": segmentation_path.name,
                  "note": "All voxels of supplied input file (which may itself be a crop); no retraining or new inference in this visualization step"}
        (output / "slicer_visualization.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("SLICER_FULL_VOLUME_VIEW_OK " + json.dumps(report), flush=True)
        slicer.app.exit(0)

    def capture():
        try:
            capture_impl()
        except Exception:
            print("SLICER_FULL_VOLUME_VIEW_FAILED " + traceback.format_exc(), flush=True)
            slicer.app.exit(1)
    qt.QTimer.singleShot(2500, capture)


try:
    run()
except Exception:
    print("SLICER_FULL_VOLUME_VIEW_FAILED " + traceback.format_exc(), flush=True)
    slicer.app.exit(1)
