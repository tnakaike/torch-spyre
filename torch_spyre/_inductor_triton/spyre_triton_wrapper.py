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

from torch._inductor.codegen.wrapper import SubgraphPythonWrapperCodegen
from torch._inductor.ir import GraphPartitionSignature

from torch_spyre._inductor.wrapper import (
    PythonWrapperCodegen,
    SpyrePythonWrapperCodegen,
)


class SpyreTritonPythonWrapperCodegen(SpyrePythonWrapperCodegen):
    """Wrapper codegen for the Spyre Triton path.

    Overrides create() so Inductor instantiates this class (not the base),
    and injects a SpyreTritonAsyncCompile import into the generated wrapper.
    """

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: Optional[str],
        parent_wrapper: Optional[PythonWrapperCodegen],
        partition_signatures: Optional[GraphPartitionSignature] = None,
    ):
        if is_subgraph:
            assert subgraph_name is not None
            assert parent_wrapper is not None
            return SubgraphPythonWrapperCodegen(
                subgraph_name, parent_wrapper, partition_signatures
            )
        return SpyreTritonPythonWrapperCodegen()

    def write_header(self) -> None:
        super().write_header()
        self.header.writeline(
            "from torch_spyre._inductor_triton.async_compile import SpyreTritonAsyncCompile"
        )
        self.header.writeline("async_compile = SpyreTritonAsyncCompile()")
