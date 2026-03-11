# Release Notes Template

## Title
Portable MONAI Label Radiology + nnUNet (Models A/B/C)

## Summary
This release packages a portable MONAI Label Radiology app integrated with three nnUNet models for 3D liver/tumor segmentation in 3D Slicer.

## Included
- Radiology app code in `radiology/`
- Model checkpoints in `radiology/lib/models/model_a|model_b|model_c`
- Example studies in `monai_studies_rad/`
- Launch scripts:
  - Windows: `start_server.bat`
  - macOS/Linux: `start_server.sh`
- Environment files:
  - `environment.yml`
  - `environment_macos.yml`

## Canonical Model Keys
- `nnunet_liver`
- `nnunet_liver_modelb`
- `nnunet_liver_modelc`

## Quick Start
1. Clone repository.
2. Go to `radiology_portable/`.
3. Create and activate conda environment.
4. Start server using launcher script.
5. Connect 3D Slicer MONAI Label to `http://localhost:8002`.

## Known Notes
- macOS runs CPU-only in this setup and is slower than NVIDIA GPU systems.
- If `[Errno 10048]` appears, port `8002` is already in use; stop conflicting process or use another port.
