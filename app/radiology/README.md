<!--
Copyright (c) MONAI Consortium
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# MONAI Label app

Modified from the MONAI Label radiology sample for this project's nnU-Net/Slicer integration. The upstream license notice above is retained.

The supported launchers enable `nnunet_liver`, `nnunet_liver_modelb` and `nnunet_liver_modelc`, with training, SAM2 and scribbles disabled. Use the [project README](../../README.md) and [user manual](../../docs/USER_MANUAL.md).

## Structure

- `main.py`: application setup and task registration.
- `lib/configs/nnunet_liver*.py`: checkpoint loading and Models A/B/C configuration.
- `lib/infers/nnunet_runtime.py`: Model B full-volume inference.
- `lib/models/model_{a,b,c}/`: downloaded checkpoints/metadata; excluded from source Git.
- Other task modules: retained MONAI sample code. The app's configuration discovery imports these modules during startup; the project launchers do not enable their models.

Do not use `--conf models all` for this project. The extra MONAI example checkpoints are not included.
