# Third-party provenance and notices

The owner's original source additions, scripts, configuration and documentation are licensed under Apache-2.0 in the root `LICENSE`. Project-specific contributions are attributed to Muhammad Hamdi Kurnia Rahman. See [license scope](docs/LICENSE_SCOPE.md). This source decision does not license or include trained checkpoints or medical data.

The application is derived from the MONAI Label radiology sample application. Files carrying `Copyright (c) MONAI Consortium` and Apache License 2.0 headers retain those headers. The supplied upstream Apache license text is also preserved in `licenses/MONAI-APACHE-2.0.txt`; no upstream ownership or attribution is replaced.

| Component | Evidence in this release | Upstream |
|---|---|---|
| MONAI Label radiology sample | Apache 2.0 headers in `app/radiology/main.py` and sample task/configuration/transform files; installed dependency metadata for 0.8.5 | https://github.com/Project-MONAI/MONAILabel |
| MONAI | Runtime dependency `monai==1.5.1`; installed separately | https://github.com/Project-MONAI/MONAI |
| nnU-Net v2 | Runtime dependency `nnunetv2==2.6.2`; installed dependency metadata identifies Apache 2.0 | https://github.com/MIC-DKFZ/nnUNet |
| PyTorch | CPU/CUDA runtime dependency, installed separately; see environment requirements | https://pytorch.org/ |
| SimpleITK | Runtime dependency, installed separately | https://simpleitk.org/ |
| SciPy | Evaluation dependency `scipy==1.15.3`, installed separately | https://scipy.org/ |
| Matplotlib | Static scientific figure dependency `matplotlib==3.10.8`, installed separately | https://matplotlib.org/ |
| 3D Slicer | External application; Slicer is not included in this source repository | https://www.slicer.org/ |

Project-specific adaptations include nnU-Net task registration for Models A/B/C, model loading and geometry restoration, launch scripts, release and evaluation tools, and project documentation. Modified upstream files retain their attribution and identify project changes. `configs/release_manifest.json` records Apache-2.0 for original source only; checkpoint terms remain separate.

LiTS-derived CT data and annotations are not included. The source/framework licenses do not establish permission to redistribute training data, derived checkpoints, or university-owned work. Record the dataset acquisition terms and any institutional ownership requirements before publishing artifacts.
