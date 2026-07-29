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

import os
from typing import Optional

import sympy
from torch._inductor.codegen.wrapper import SubgraphPythonWrapperCodegen
from torch._inductor.ir import GraphPartitionSignature
from torch._inductor.virtualized import V

from torch_spyre._inductor.ir import FixedTiledLayout
from torch_spyre._inductor.wrapper import (
    PythonWrapperCodegen,
    SpyrePythonWrapperCodegen,
)


def _ktir_cpu_mode() -> bool:
    """Whether buffers should be allocated for the device-free ktir-cpu path."""
    return os.getenv("TORCH_SPYRE_KTIR_CPU", "0") != "0"


def _tiled_layout(node) -> Optional[FixedTiledLayout]:
    """Return the node's FixedTiledLayout, or None if it has no tiled layout."""
    try:
        layout = node.get_layout()
    except (AttributeError, NotImplementedError):
        return None
    return layout if isinstance(layout, FixedTiledLayout) else None


class SpyreTritonPythonWrapperCodegen(SpyrePythonWrapperCodegen):
    """Wrapper codegen for the Spyre Triton path.

    Overrides create() so Inductor instantiates this class (not the base),
    and injects a SpyreTritonAsyncCompile import into the generated wrapper.

    In the device-free ktir-cpu path (``TORCH_SPYRE_KTIR_CPU``), buffer
    allocations are emitted with ``ktir_empty_with_layout`` (a host NumPy buffer
    in physical layout) instead of ``spyre_empty_with_layout`` (a Spyre-device
    tensor), tensor inputs are stickified to physical layout, and tiled outputs
    are destickified back to logical layout before return.
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
            "from torch_spyre._triton_kernel.async_compile"
            " import SpyreTritonAsyncCompile"
        )
        self.header.writeline("async_compile = SpyreTritonAsyncCompile()")
        if _ktir_cpu_mode():
            self.header.writeline(
                "from torch_spyre.execution.ktir_cpu_runner import "
                "ktir_empty_with_layout, ktir_stickify, ktir_destickify"
            )
            self.header.writeline("from torch_spyre._C import get_spyre_tensor_layout")

    def codegen_input_size_and_nan_asserts(self) -> None:
        """Emit input asserts, then ktir_stickify tensor inputs to physical layout.

        Inputs arrive as Spyre tensors, so their IR layout is logical (not a
        FixedTiledLayout); the physical device layout is carried on the tensor at
        runtime. Stickify each tensor input in place using its runtime layout so
        the kernel sees physical-layout data. Emitting here (after the
        ``assert_size_stride`` checks, which expect the logical shape, and while
        the prefix buffer is still inside its indent context) keeps both the
        ordering and indentation correct.
        """
        super().codegen_input_size_and_nan_asserts()
        if not _ktir_cpu_mode():
            return
        for name, box in V.graph.graph_inputs.items():
            if isinstance(box, sympy.Expr):
                continue  # skip symbolic-size (non-tensor) inputs
            self.prefix.writeline(
                f"{name} = ktir_stickify({name}, get_spyre_tensor_layout({name}))"
            )

    def generate_return(self, output_refs) -> None:
        """Destickify tiled outputs (physical -> logical) before returning."""
        if _ktir_cpu_mode():
            wrapped = []
            for ref in output_refs:
                layout = _tiled_layout(V.graph.get_buffer(ref))
                if layout is not None:
                    size = tuple(int(s) for s in layout.size)
                    wrapped.append(
                        f"ktir_destickify({ref}, {size}, {layout.device_layout!r})"
                    )
                else:
                    wrapped.append(ref)
            output_refs = wrapped
        return super().generate_return(output_refs)

    def make_buffer_allocation(self, buffer):
        """Allocate a buffer, routing to ``ktir_empty_with_layout`` in ktir-cpu
        mode.

        Reuses the parent's emission (which computes the physical
        ``SpyreTensorLayout`` and emits ``spyre_empty_with_layout(...)``) and
        just swaps the allocator name, so the layout logic stays in one place.
        """
        line = super().make_buffer_allocation(buffer)
        if _ktir_cpu_mode() and isinstance(line, str):
            line = line.replace("spyre_empty_with_layout(", "ktir_empty_with_layout(")
        return line
