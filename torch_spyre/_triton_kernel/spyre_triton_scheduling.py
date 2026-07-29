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

"""Scheduling for the OpSpec -> Triton source generator path.

``SpyreOpSpecTritonScheduling`` reuses the whole SDSC frontend
(``SuperDSCScheduling``) and only swaps two things:

1. the kernel class instantiated in ``codegen_node`` /
   ``_codegen_counted_loop`` — ``SpyreOpSpecTritonKernel`` instead of
   ``SpyreKernel`` — so the finalized op_specs are projected to Triton source;
2. ``define_kernel`` — emits ``name = async_compile.triton('name', '''src''',
   device_str='spyre')`` instead of ``async_compile.sdsc('name', <flex>)``.

There is no upstream factory hook for the kernel class, so ``codegen_node`` /
``_codegen_counted_loop`` are copied from ``SuperDSCScheduling`` with the one
constructor line changed.  Everything else (buffer freeing, layout restores,
mark_run) is inherited unchanged.
"""

from typing import Union

from torch._inductor.codecache import code_hash
from torch._inductor.scheduler import (
    FusedSchedulerNode,
    SchedulerNode,
)
from torch._inductor.utils import (
    IndentedBuffer,
    get_fused_kernel_name,
    get_kernel_metadata,
)
from torch._inductor.virtualized import V

from torch_spyre._inductor.constants import DEVICE_NAME
from torch_spyre._inductor.scheduler import (
    CountedLoopSchedulerNode,
    SuperDSCScheduling,
)

from .spyre_triton_kernel import KERNEL_NAME_PLACEHOLDER, SpyreOpSpecTritonKernel


class SpyreOpSpecTritonScheduling(SuperDSCScheduling):
    """SDSC scheduling that emits Triton source via SpyreOpSpecTritonKernel."""

    def codegen_node(
        self, node: Union[FusedSchedulerNode, SchedulerNode, CountedLoopSchedulerNode]
    ) -> None:
        """Generate a kernel given a list of pre-fused nodes.

        Copy of ``SuperDSCScheduling.codegen_node`` with the kernel class swapped
        to ``SpyreOpSpecTritonKernel``.
        """
        if isinstance(node, CountedLoopSchedulerNode):
            self._codegen_counted_loop(node)
            return

        assert self.scheduler
        nodes = [
            n
            for n in node.get_nodes()
            if n.get_name() not in self.scheduler.removed_ops
        ]
        if len(nodes) == 0:
            return

        kernel = SpyreOpSpecTritonKernel()
        all_schedule_nodes: list[SchedulerNode] = []
        with kernel:
            self._codegen_into_kernel(nodes, kernel, all_schedule_nodes)

        self._emit_kernels(kernel, all_schedule_nodes)

    def _codegen_counted_loop(self, node: CountedLoopSchedulerNode) -> None:
        """Generate a kernel for a counted loop group.

        Copy of ``SuperDSCScheduling._codegen_counted_loop`` with the kernel
        class swapped to ``SpyreOpSpecTritonKernel``.
        """
        assert self.scheduler
        inner_nodes = [
            n
            for n in node.get_nodes()
            if n.get_name() not in self.scheduler.removed_ops
        ]
        if len(inner_nodes) == 0:
            return

        kernel = SpyreOpSpecTritonKernel()
        all_schedule_nodes: list[SchedulerNode] = []
        with kernel:
            self._codegen_into_kernel(inner_nodes, kernel, all_schedule_nodes)

        kernel.wrap_op_specs_in_loop(node.loop_count)

        self._emit_kernels(kernel, all_schedule_nodes)

    def _emit_kernels(
        self,
        kernel: SpyreOpSpecTritonKernel,
        all_schedule_nodes: list[SchedulerNode],
    ) -> None:
        """Project the finalized op_specs to one or more Triton kernels.

        ``kernel.codegen_kernels()`` partitions the op_specs into fusion groups
        (same iteration space + work division) and returns one plan per group.
        Each plan becomes its own kernel definition + ``.run()`` call, so ops
        with different work divisions land in separate kernels while ops that
        share one fuse.  ``mark_run`` / layout restores / buffer freeing stay at
        the node level, exactly as in the single-kernel path.
        """
        with V.set_kernel_handler(kernel):
            plans = kernel.codegen_kernels()
            for snode in all_schedule_nodes:
                snode.mark_run()

        for plan in plans:
            kernel_name = self.define_kernel(plan.source, all_schedule_nodes, kernel)
            self.codegen_comment(all_schedule_nodes, kernel_name)
            call_args = ", ".join(plan.call_args)
            V.graph.wrapper_code.writeline(f"{kernel_name}.run({call_args})")

        # kernel_name / code_hash are per-node metadata; record the last group's
        # so downstream consumers that read them see a valid value.
        kernel.kernel_name = kernel_name
        kernel.code_hash = code_hash("".join(p.source for p in plans))
        kernel.emit_layout_restores(self._collect_layout_restores(all_schedule_nodes))

        V.graph.removed_buffers |= kernel.removed_buffers
        V.graph.inplaced_to_remove |= kernel.inplaced_to_remove

        self.free_buffers_in_scheduler()

    def define_kernel(self, src_code, node_schedule, kernel):
        """Emit ``name = async_compile.triton('name', '''src''', device_str=...)``.

        The generated source carries a ``__KERNEL_NAME__`` placeholder (the name
        is only known here, after fusion); substitute it so the source defines a
        function ``async_compile.triton`` can look up by name.
        """
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            return wrapper.src_to_kernel[src_code]

        fused_name = get_fused_kernel_name(node_schedule, "original_aten")
        kernel_name = "_".join(["triton", fused_name, wrapper.next_kernel_suffix()])
        wrapper.src_to_kernel[src_code] = kernel_name

        kernel_src = src_code.replace(KERNEL_NAME_PLACEHOLDER, kernel_name)
        buf = IndentedBuffer()
        buf.writeline(f"async_compile.triton('{kernel_name}', '''")
        # The source must sit at column 0 inside the triple-quoted string, so
        # write it raw rather than through buf.indent().
        buf.writeline(kernel_src)
        buf.writeline(f"''', device_str='{DEVICE_NAME}')")

        origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
        metadata_comment = f"{origins}\n{detailed_origins}"
        wrapper.define_kernel(kernel_name, buf.getvalue(), metadata_comment)

        return kernel_name
