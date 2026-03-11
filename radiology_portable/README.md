# Portable MONAI Label Radiology + nnUNet (3 Models)

This folder is a self-contained package for running MONAI Label Radiology with 3 nnUNet models:

GitHub repository:

- https://github.com/itsnotblue/Integrasi-nnU-Net-dengan-3DSlicer-menggunakan-MONAI-untuk-Segmentasi-Tumor-Hati-pada-Citra-Medis-3D

- `nnunet_liver` (Model A)
- `nnunet_liver_modelb` (Model B)
- `nnunet_liver_modelc` (Model C)

## Package Contents

- `radiology/` -> app code + model weights (`lib/models/model_a|b|c`)
- `monai_studies_rad/` -> sample studies
- `environment.yml` -> Windows-oriented environment export
- `environment_macos.yml` -> macOS (Apple Silicon) environment
- `start_server.bat` -> Windows launcher
- `start_server.sh` -> macOS/Linux launcher

## 1) Windows Run

```powershell
cd <path-to-cloned-repo>\radiology_portable
conda env create -f environment.yml
conda activate monai
.\start_server.bat
```

Direct command (without batch):

```powershell
python -m monailabel.main start_server --app ".\radiology" --studies ".\monai_studies_rad" --port 8002 --conf models "nnunet_liver,nnunet_liver_modelb,nnunet_liver_modelc"
```

## 2) macOS (M1/M2) Run

```bash
cd ~/radiology_portable
conda env create -f environment_macos.yml
conda activate monai
chmod +x start_server.sh
./start_server.sh
```

## 3) Verify Server

```bash
curl http://localhost:8002/info
```

Expected model keys in response:

- `nnunet_liver`
- `nnunet_liver_modelb`
- `nnunet_liver_modelc`

## 4) Connect from 3D Slicer

- Open MONAI Label extension.
- Server URL: `http://localhost:8002`
- If model list is stale, disconnect/reconnect or restart Slicer.

## 5) Troubleshooting

### A) Port bind error (`[Errno 10048] ... bind ... 8002`)

Meaning: port `8002` is already used by another process.

Check process using port:

```powershell
Get-NetTCPConnection -LocalPort 8002 -State Listen | Select-Object LocalAddress,LocalPort,OwningProcess
Get-Process -Id (Get-NetTCPConnection -LocalPort 8002 -State Listen).OwningProcess
```

Stop conflicting process:

```powershell
Stop-Process -Id <PID> -Force
```

Or start MONAI Label on another port:

```powershell
python -m monailabel.main start_server --app ".\radiology" --studies ".\monai_studies_rad" --port 8003 --conf models "nnunet_liver,nnunet_liver_modelb,nnunet_liver_modelc"
```

### B) Invalid model names

Use only canonical keys:

- `nnunet_liver`
- `nnunet_liver_modelb`
- `nnunet_liver_modelc`

Do **not** use aliases like `nnunet_liver_model_a` unless you explicitly add alias mapping in app code.

### C) Slow inference on Mac

macOS runs CPU-only for this setup; inference is expected to be slower than Windows + NVIDIA GPU.

## 6) Portable Zip

Create zip from inside `radiology_portable` parent folder:

```powershell
Compress-Archive -Path ".\radiology_portable\*" -DestinationPath ".\radiology_portable_macos.zip" -Force
```

Default output filename:

`radiology_portable_macos.zip`
