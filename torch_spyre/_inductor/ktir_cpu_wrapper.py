# Copyright 2025-2026 The Torch-Spyre Authors.
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

"""Wrapper codegen for the device-free ktir-cpu host path.

Both OpSpec backends that can run their emitted KTIR on ktir-cpu -- the
OpSpec->Triton source generator and the OpSpec->KTIR emitter -- need the same
host-layout wrapper overrides: stickify tensor inputs to physical layout,
allocate host buffers with ``ktir_empty_with_layout`` instead of a Spyre-device
tensor, and destickify tiled outputs back to logical layout before returning.

Those overrides live here (on ``KtirCpuWrapperCodegen``) so the two backends
share them; ``SpyreTritonPythonWrapperCodegen`` subclasses this and adds only its
``SpyreTritonAsyncCompile`` rebind.  The KTIR path uses this class directly (its
inherited ``write_header`` binds ``SpyreAsyncCompile``, which carries the
``ktir`` method).
"""

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


class KtirCpuWrapperCodegen(SpyrePythonWrapperCodegen):
    """Wrapper codegen for the device-free ktir-cpu host path.

    In the device-free ktir-cpu path (``TORCH_SPYRE_KTIR_CPU``), buffer
    allocations are emitted with ``ktir_empty_with_layout`` (a host NumPy buffer
    in physical layout) instead of ``spyre_empty_with_layout`` (a Spyre-device
    tensor), tensor inputs are stickified to physical layout, and tiled outputs
    are destickified back to logical layout before return.
    """

    @classmethod
    def create(
        cls,
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
        return cls()

    def write_header(self) -> None:
        super().write_header()
        if _ktir_cpu_mode():
            self.header.writeline(
                "from torch_spyre.execution.ktir_cpu_runner import "
                "ktir_empty_with_layout, ktir_constant_tensor, ktir_stickify, "
                "ktir_destickify"
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

    def codegen_input_size_asserts(self) -> None:
        """Suppress logical-shape ``assert_size_stride`` checks on the ktir-cpu path.

        Upstream defers these asserts (queued here, emitted by the scheduler via
        ``codegen_deferred_input_asserts``) to just before the first kernel that
        uses each input -- i.e. *after* ``codegen_input_size_and_nan_asserts``
        has stickified inputs to their physical (tiled, higher-rank) layout. A
        logical-shape assert against a stickified tensor always fails on rank
        ("wrong number of dimensions"), and it validates nothing about the
        physical layout the kernel actually reads, so skip queuing it here. The
        device path (no stickify) keeps the base behavior.
        """
        if _ktir_cpu_mode():
            return
        super().codegen_input_size_asserts()

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

    def generate_const_tensor_fallback(self, node):
        """Emit a physical-layout constant fill on the ktir-cpu path.

        The base path emits ``spyre_constant_tensor`` -- a rank-0 Spyre tensor
        whose ``.to(device)`` copy stickifies it into a full replicated stick.
        The device-free ktir-cpu path has no device copy, so that rank-0 tensor
        would under-fill the ``device_size`` descriptor the kernel reads (its
        trailing within-stick lanes would be garbage).  Emit
        ``ktir_constant_tensor`` instead, which materializes the whole physical
        ``device_size`` buffer with the value replicated across every lane.
        """
        layout = _tiled_layout(node)
        if not _ktir_cpu_mode() or layout is None:
            return super().generate_const_tensor_fallback(node)
        value = node.constant_args[0]
        self.writeline(
            f"{node.get_name()} = ktir_constant_tensor("
            f"{value}, {layout.device_layout!r}, {layout.dtype})"
        )

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
