# Copyright 2023-present, Argilla, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ruff: noqa: E402

import warnings

deprecation_message = (
    "Importing from 'distilabel.llms' is deprecated and will be removed in a version 1.7.0. "
    "Import from 'distilabel.models' instead."
)

warnings.warn(deprecation_message, DeprecationWarning, stacklevel=2)

from distilabel.models.llms.base import LLM, AsyncLLM
from distilabel.models.llms.huggingface import InferenceEndpointsLLM, TransformersLLM
from distilabel.models.llms.openai import OpenAILLM
from distilabel.models.llms.vllm import ClientvLLM, vLLM
from distilabel.models.mixins.cuda_device_placement import CudaDevicePlacementMixin
from distilabel.typing import GenerateOutput, HiddenState

__all__ = [
    "LLM",
    "AsyncLLM",
    "ClientvLLM",
    "CudaDevicePlacementMixin",
    "GenerateOutput",
    "HiddenState",
    "InferenceEndpointsLLM",
    "OpenAILLM",
    "TransformersLLM",
    "vLLM",
]
