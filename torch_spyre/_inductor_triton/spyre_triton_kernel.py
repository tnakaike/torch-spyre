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

from typing import Optional

import sympy

from torch._inductor.codegen.common import CSEVariable
from torch._inductor.codegen.triton import (
    FixedTritonConfig,
    TritonKernel,
)
from torch._inductor.virtualized import StoreMode


class SpyreTritonKernel(TritonKernel):
    def __init__(
        self,
        tiling: dict[str, sympy.Expr],
        min_elem_per_thread=0,
        optimize_mask=True,
        fixed_config: Optional[FixedTritonConfig] = None,
        hint_override: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            tiling,
            min_elem_per_thread,
            optimize_mask,
            fixed_config,
            hint_override,
            **kwargs,
        )

    def __enter__(self):
        super(TritonKernel, self).__enter__()
        return self

    def codegen_kernel(self, name=None) -> str:
        return super().codegen_kernel(name)

    def codegen_body(self):
        return super().codegen_body()

    def load(self, name: str, index: sympy.Expr):
        return super().load(name, index)

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> None:
        return super().store(name, index, value, mode)

    def store_reduction(self, name: str, index: sympy.Expr, value: CSEVariable) -> None:
        return super().store_reduction(name, index, value)
