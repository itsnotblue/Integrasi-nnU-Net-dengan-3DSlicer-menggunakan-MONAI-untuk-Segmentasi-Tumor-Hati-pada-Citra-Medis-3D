"""Shared validation for locked segmentation cohorts and prediction evaluation."""

import hashlib
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def geometry(image):
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("Expected a scalar, three-dimensional image")
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    if not np.allclose(direction.T @ direction, np.eye(3), rtol=0, atol=1e-5):
        raise ValueError("Surface metrics require orthonormal image axes")
    return {
        "size_xyz": list(image.GetSize()),
        "spacing_xyz_mm": list(image.GetSpacing()),
        "origin_xyz_mm": list(image.GetOrigin()),
        "direction": list(image.GetDirection()),
    }


def assert_geometry(actual, expected, context="image"):
    if actual["size_xyz"] != expected["size_xyz"]:
        raise ValueError(f"{context}: size mismatch")
    for name in ("spacing_xyz_mm", "origin_xyz_mm", "direction"):
        if not np.allclose(actual[name], expected[name], rtol=0, atol=1e-5):
            raise ValueError(f"{context}: {name} mismatch; resampling is not automatic")


def validate_labels(array, context):
    values = np.unique(array)
    if not np.isfinite(values).all() or not np.isin(values, [0, 1, 2]).all():
        raise ValueError(f"{context}: expected integer labels 0, 1, 2; found {values}")
    return [int(value) for value in values]


def resolve_path(manifest_path, value):
    path = Path(value)
    return path if path.is_absolute() else (Path(manifest_path).parent / path).resolve()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
