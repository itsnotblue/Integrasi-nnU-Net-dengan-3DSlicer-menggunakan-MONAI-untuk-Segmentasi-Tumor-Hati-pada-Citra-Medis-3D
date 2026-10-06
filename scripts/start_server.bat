@echo off
setlocal
set "ROOT=%~dp0.."
set "APP=%ROOT%\app\radiology"
set "STUDIES=%ROOT%\data\studies"
if not exist "%STUDIES%" mkdir "%STUDIES%"
set "MONAILABEL_STUDIES=%STUDIES%"
set "SEG_ENV=%~1"
if not defined SEG_ENV set "SEG_ENV=liver-seg-cpu"
set "SEG_PROFILE=%~2"
if not defined SEG_PROFILE set "SEG_PROFILE=reference"
set "SEG_PORT=%~3"
if not defined SEG_PORT set "SEG_PORT=8002"
if not "%SEG_PROFILE%"=="reference" if not "%SEG_PROFILE%"=="fast" (
  echo Profile must be reference or fast.
  exit /b 1
)
set "SEG_CONDA=%CONDA_EXE%"
if not defined SEG_CONDA for /f "delims=" %%I in ('where conda.exe 2^>nul') do if not defined SEG_CONDA set "SEG_CONDA=%%I"
if not defined SEG_CONDA if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "SEG_CONDA=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not defined SEG_CONDA (
  echo Conda not found. Set CONDA_EXE to your conda.exe path.
  exit /b 1
)
"%SEG_CONDA%" run --no-capture-output -n "%SEG_ENV%" python -m monailabel.main start_server ^
  --app "%APP%" ^
  --studies "%STUDIES%" ^
  --host 127.0.0.1 ^
  --port %SEG_PORT% ^
  --conf models "nnunet_liver,nnunet_liver_modelb,nnunet_liver_modelc" ^
  --conf sam2 false ^
  --conf scribbles false ^
  --conf skip_trainers true ^
  --conf inference_profile "%SEG_PROFILE%" ^
  --conf inference_threads 2
exit /b %errorlevel%
