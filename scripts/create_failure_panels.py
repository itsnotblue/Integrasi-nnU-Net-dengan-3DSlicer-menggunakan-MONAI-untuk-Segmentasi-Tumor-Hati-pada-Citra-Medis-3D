"""Render full-CT axial slices for retrospective reference/prediction review."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import SimpleITK as sitk

from evaluation_common import assert_geometry, geometry, read_json, resolve_path, sha256, validate_labels, write_json_new


def create_panels(args):
    manifest = read_json(args.manifest)
    cases = {case["case_id"]: case for case in manifest["cases"]}
    selected = [value.strip() for value in args.case_ids.split(",") if value.strip()]
    if not selected or len(selected) != len(set(selected)) or not set(selected) <= set(cases):
        raise ValueError("Choose unique case IDs present in the locked manifest")
    if len(args.prediction_dirs) != len(args.model_labels):
        raise ValueError("Supply one model label per prediction directory")
    if args.evaluations and len(args.evaluations) != len(args.prediction_dirs):
        raise ValueError("Supply one evaluation report per prediction directory")
    reports = []
    for path in args.evaluations or []:
        report = read_json(path)
        if report["manifest_sha256"] != sha256(args.manifest):
            raise ValueError("Evaluation report uses a different locked manifest")
        reports.append({case["case_id"]: case for case in report["per_case"]})
    output = Path(args.output).resolve()
    sidecar = Path(str(output) + ".json")
    if output.suffix.lower() != ".png" or output.exists() or sidecar.exists():
        raise ValueError("Supply a new .png output path")
    if args.hu_max <= args.hu_min:
        raise ValueError("HU maximum must exceed minimum")
    columns = 2 + len(args.prediction_dirs)
    fig, axes = plt.subplots(len(selected), columns, figsize=(columns * 4, len(selected) * 4), squeeze=False)
    overlay = ListedColormap([(0, 0, 0, 0), (0.15, 0.9, 0.2, 0.25), (1, 0.7, 0, 0.8)])
    evidence = []
    for row, case_id in enumerate(selected):
        case = cases[case_id]
        image_path, reference_path = resolve_path(args.manifest, case["image"]), resolve_path(args.manifest, case["reference"])
        if sha256(image_path) != case["image_sha256"] or sha256(reference_path) != case["reference_sha256"]:
            raise ValueError(f"{case_id}: locked source changed")
        reference_itk = sitk.ReadImage(str(reference_path))
        assert_geometry(geometry(reference_itk), case["geometry"], "reference")
        reference = sitk.GetArrayViewFromImage(reference_itk)
        validate_labels(reference, "reference")
        if not np.allclose(reference_itk.GetDirection(), np.eye(3).ravel(), atol=1e-5, rtol=0):
            raise ValueError("This native axial panel expects LPS-aligned images; reorient all inputs consistently first")
        # This uses reference annotations only to select a retrospective figure
        # plane. It must never feed into model input selection or inference.
        counts = np.count_nonzero(reference == 2, axis=(1, 2))
        selection = "reference_tumor_max"
        if not np.any(counts):
            counts = np.count_nonzero(reference > 0, axis=(1, 2))
            selection = "reference_liver_max_tumor_empty"
        index = int(np.argmax(counts)) if np.any(counts) else reference.shape[0] // 2
        reference_slice = np.array(reference[index], copy=True)
        del reference, reference_itk
        image_itk = sitk.ReadImage(str(image_path))
        assert_geometry(geometry(image_itk), case["geometry"], "CT")
        image_slice = np.array(sitk.GetArrayViewFromImage(image_itk)[index], copy=True)
        del image_itk
        if not np.isfinite(image_slice).all():
            raise ValueError("CT slice contains nonfinite values")
        for column, axis in enumerate(axes[row]):
            axis.imshow(image_slice, cmap="gray", vmin=args.hu_min, vmax=args.hu_max, origin="upper", interpolation="nearest")
            axis.set_axis_off()
            axis.set_xlim(-0.5, image_slice.shape[1] - 0.5)
            axis.set_ylim(image_slice.shape[0] - 0.5, -0.5)
        axes[row, 0].set_title(f"{case_id} | full CT slice z={index}", fontsize=10)
        axes[row, 1].imshow(reference_slice, cmap=overlay, vmin=0, vmax=2, origin="upper", interpolation="nearest")
        axes[row, 1].set_title("Reference", fontsize=10)
        details = {"case_id": case_id, "source_geometry": case["geometry"], "slice_index_z": index,
                   "slice_selection": selection, "reference_used_for_figure_only": True, "predictions": []}
        for number, (directory, label) in enumerate(zip(args.prediction_dirs, args.model_labels)):
            prediction_path = Path(directory) / args.prediction_pattern.format(case_id=case_id)
            prediction_itk = sitk.ReadImage(str(prediction_path))
            assert_geometry(geometry(prediction_itk), case["geometry"], f"{label} prediction")
            prediction_slice = np.array(sitk.GetArrayViewFromImage(prediction_itk)[index], copy=True)
            del prediction_itk
            validate_labels(prediction_slice, "prediction slice")
            digest = sha256(prediction_path)
            title = label
            prediction_detail = {"label": label, "prediction_sha256": digest}
            if reports:
                metric = reports[number][case_id]
                if metric["prediction_sha256"] != digest:
                    raise ValueError("Prediction differs from evaluated artifact")
                dice = metric["classes"]["tumor_label_2"]["dice"]
                title += f" | full-volume tumor Dice={dice:.3f}"
                prediction_detail["whole_volume_tumor_dice"] = dice
            axes[row, number + 2].imshow(prediction_slice, cmap=overlay, vmin=0, vmax=2, origin="upper", interpolation="nearest")
            axes[row, number + 2].set_title(title, fontsize=9)
            details["predictions"].append(prediction_detail)
        evidence.append(details)
    fig.legend(handles=[Patch(facecolor=(0.15, 0.9, 0.2, 0.4), label="Liver label 1"),
                        Patch(facecolor=(1, 0.7, 0, 0.8), label="Tumor label 2")],
               loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Retrospective failure review | complete axial CT field of view", fontsize=13)
    fig.tight_layout(rect=(0, 0.035, 1, 0.965), w_pad=0.5, h_pad=0.6)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, facecolor="white")
    plt.close(fig)
    write_json_new(sidecar, {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha256(args.manifest), "figure_sha256": sha256(output), "cases": evidence,
        "hu_window": [args.hu_min, args.hu_max],
        "caption": "Each full axial CT slice is selected by maximum reference tumor voxel count for retrospective illustration only. Reference annotations never select inference input or ROI. Scores are whole-volume metrics, not slice metrics. Green is liver label 1, orange is tumor label 2. No CT field-of-view crop or image zoom is applied.",
        "publication_note": "Derived CT figures stay local until dataset/image redistribution rights are confirmed; do not include in source ZIP by default."})
    print(f"Saved retrospective panel and provenance: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prediction-dirs", nargs="+", required=True)
    parser.add_argument("--model-labels", nargs="+", required=True)
    parser.add_argument("--evaluations", nargs="+")
    parser.add_argument("--prediction-pattern", default="{case_id}.nii.gz")
    parser.add_argument("--case-ids", default="LITS_121,LITS_122,LITS_125,LITS_128")
    parser.add_argument("--output", required=True)
    parser.add_argument("--hu-min", type=float, default=-150)
    parser.add_argument("--hu-max", type=float, default=250)
    parser.add_argument("--dpi", type=int, default=140)
    create_panels(parser.parse_args())


if __name__ == "__main__":
    main()
