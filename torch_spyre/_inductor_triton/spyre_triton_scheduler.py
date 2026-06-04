# Copyright 2025 The Torch-Spyre Authors.
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

from typing import Any

from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
from torch._inductor.codegen.triton import TritonScheduling

from .spyre_triton_kernel import SpyreTritonKernel


class SpyreTritonScheduling(TritonScheduling):
    """
    Spyre-specific Triton scheduling that uses SpyreTritonKernel.
    """

    def create_kernel_choices(  # type: ignore[override]
        self,
        kernel_features: SIMDKernelFeatures,
        kernel_args: list[Any],
        kernel_kwargs: dict[str, Any],
    ) -> list[Any]:
        self.kernel_type = SpyreTritonKernel  # type: ignore[assignment]
        return super().create_kernel_choices(
            kernel_features, kernel_args, kernel_kwargs
        )
