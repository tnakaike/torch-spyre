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

from torch_spyre._inductor.ktir_cpu_wrapper import KtirCpuWrapperCodegen


class SpyreTritonPythonWrapperCodegen(KtirCpuWrapperCodegen):
    """Wrapper codegen for the Spyre Triton path.

    Inherits the device-free ktir-cpu host-layout overrides (stickify inputs,
    ``ktir_empty_with_layout`` allocations, destickify outputs) and the
    ``create()`` classmethod from ``KtirCpuWrapperCodegen``; its only addition is
    rebinding the generated ``async_compile`` to ``SpyreTritonAsyncCompile`` so
    Triton kernels compile through the Triton async-compile path.
    """

    def write_header(self) -> None:
        super().write_header()
        self.header.writeline(
            "from torch_spyre._triton_kernel.async_compile"
            " import SpyreTritonAsyncCompile"
        )
        self.header.writeline("async_compile = SpyreTritonAsyncCompile()")
