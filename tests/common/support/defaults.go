/*
Copyright 2025.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package support

const (
	RayVersion        = "2.35.0"
	RayImage          = "quay.io/modh/ray:2.35.0-py311-cu121"
	RayROCmImage      = "quay.io/modh/ray:2.35.0-py311-rocm62"
	RayTorchCudaImage = "quay.io/rhoai/ray:2.35.0-py311-cu121-torch24-fa26"
	RayTorchROCmImage = "quay.io/rhoai/ray:2.35.0-py311-rocm61-torch24-fa26"
	// RHOAIENG-93167: py311 training images offboarded from rhoai-2.25; 3.5 replacements below.
	// Names mirror RELATED_IMAGE_ODH_TRAINING_* keys in manifests-config.yaml.
	TrainingCuda128Torch28Py312Rhel9Image = "quay.io/rhoai/odh-training-cuda128-torch28-py312-rhel9:rhoai-3.5"
	TrainingCuda128Torch29Py312Image      = "quay.io/rhoai/odh-training-cuda128-torch29-py312-rhel9:rhoai-3.5"
	TrainingRocm64Torch28Py312Image       = "quay.io/rhoai/odh-training-rocm64-torch28-py312-rhel9:rhoai-3.5"
	TrainingRocm64Torch29Py312Image       = "quay.io/rhoai/odh-training-rocm64-torch29-py312-rhel9:rhoai-3.5"
)
