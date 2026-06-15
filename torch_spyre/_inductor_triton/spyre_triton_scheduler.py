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

from typing import Any, Optional

from torch._inductor.codegen.simd import SIMDKernelFeatures
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.dependencies import MemoryDep
from torch._inductor.utils import IndentedBuffer

from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.pass_utils import (
    apply_splits_from_index_coeff,
    iteration_space,
)
from torch_spyre._inductor.scheduler import (
    CountedLoopSchedulerNode,
    _find_leaf_sched_node,
    _tiled_syms_for_sched_node_at_depth,
)

from .spyre_triton_kernel import SpyreTritonKernel

logger = get_inductor_logger("spyre_triton_scheduler")


class SpyreTritonScheduling(TritonScheduling):
    """
    Spyre-specific Triton scheduling that uses SpyreTritonKernel.

    Extends TritonScheduling to handle CountedLoopSchedulerNode (coarse tiling)
    by emitting a ``for tile_idx in tl.range(N):`` loop in the generated kernel.
    """

    def __init__(self, scheduler) -> None:
        super().__init__(scheduler)
        # Tiling info passed from codegen_node → create_kernel_choices.
        # Tuple (loop_count, tiled_syms, tile_steps) or None.
        self._pending_tiling: Optional[tuple] = None

    def codegen_node(self, node) -> None:
        if isinstance(node, CountedLoopSchedulerNode):
            self._codegen_counted_loop_triton(node)
            return
        return super().codegen_node(node)

    def _codegen_counted_loop_triton(self, node: CountedLoopSchedulerNode) -> None:
        """Generate a Triton kernel for a CountedLoopSchedulerNode.

        Flattens the inner SchedulerNodes and runs them through the normal
        Triton codegen path, but with the kernel pre-configured to emit a
        ``for tile_idx in tl.range(loop_count):`` loop that advances tiled
        device offsets per iteration.
        """
        assert self.scheduler

        # Collect inner leaf SchedulerNodes (skipping removed ops).
        inner_nodes: list = []
        self._collect_inner_sched_nodes(node, inner_nodes)
        if not inner_nodes:
            return

        # Get tiled symbols and per-tile ranges from the first leaf node.
        leaf = _find_leaf_sched_node(node)
        if leaf is None:
            logger.debug(
                "SpyreTritonScheduling: no leaf node in %s, falling back",
                node.get_name(),
            )
            return super().codegen_node(node)

        tiled_syms = _tiled_syms_for_sched_node_at_depth(leaf, 0)
        it_space = iteration_space(leaf)

        # Compute per-symbol core division from op_it_space_splits so we can
        # derive the per-core per-tile step.  Mutation buffers from
        # insert_tiling_propagation may appear as StarDep (no index) in the
        # write or read set — skip them and use the first MemoryDep pair.
        ir_node = leaf.node
        write_dep = next(
            (d for d in leaf.read_writes.writes if isinstance(d, MemoryDep)), None
        )
        read_dep = next(
            (d for d in leaf.read_writes.reads if isinstance(d, MemoryDep)), None
        )
        if (
            hasattr(ir_node, "op_it_space_splits")
            and write_dep is not None
            and read_dep is not None
        ):
            core_div = apply_splits_from_index_coeff(
                ir_node.op_it_space_splits,
                write_dep.index,
                read_dep.index,
                it_space,
            )
        else:
            core_div = {}

        # tile_step = per-core per-tile range = per_tile_range / core_divisor.
        # This is how far the device offset advances per tile iteration for a
        # single core (each core holds a contiguous block of rows).
        tile_steps = {
            sym: max(1, int(it_space[sym]) // max(1, core_div.get(sym, 1)))
            for sym in tiled_syms
            if sym in it_space
        }

        logger.debug(
            "SpyreTritonScheduling: counted loop %s, loop_count=%s, "
            "tiled_syms=%s, tile_steps=%s",
            node.get_name(),
            node.loop_count,
            tiled_syms,
            tile_steps,
        )

        self._pending_tiling = (int(node.loop_count), tiled_syms, tile_steps)
        try:
            self._codegen_nodes(inner_nodes)
        finally:
            self._pending_tiling = None

    def _collect_inner_sched_nodes(
        self, node: CountedLoopSchedulerNode, result: list
    ) -> None:
        """Recursively flatten inner SchedulerNodes, skipping removed ops."""
        assert self.scheduler
        for inner in node.get_nodes():
            if inner.get_name() in self.scheduler.removed_ops:
                continue
            if isinstance(inner, CountedLoopSchedulerNode):
                # Nested loop: flatten for now (outer loop_count already captured).
                self._collect_inner_sched_nodes(inner, result)
            else:
                result.append(inner)

    def create_kernel_choices(
        self,
        kernel_features: SIMDKernelFeatures,
        kernel_args: list[Any],
        kernel_kwargs: dict[str, Any],
    ) -> list[Any]:
        self.kernel_type = SpyreTritonKernel  # type: ignore[assignment]
        kernels = super().create_kernel_choices(
            kernel_features, kernel_args, kernel_kwargs
        )

        # If a CountedLoopSchedulerNode triggered this call, configure each
        # candidate kernel with the tiling loop parameters.
        if self._pending_tiling is not None:
            loop_count, tiled_syms, tile_steps = self._pending_tiling
            for kernel in kernels:
                kernel._tiling_loop_count = loop_count
                kernel._tiling_loop_tiled_syms = tiled_syms
                kernel._tiling_tile_steps = tile_steps
                kernel._loop_offset_code = IndentedBuffer()
                # Scale spatial numels by tile_count so XBLOCK and xnumel
                # reflect the full per-core range (tile_count × per-tile range)
                # rather than the per-tile range of the inner SchedulerNode.
                # With XBLOCK = per_core_total, xoffset = pid * XBLOCK gives
                # c0 = pid * per_core_rows, enabling contiguous per-core tiling.
                for prefix in list(kernel.numels.keys()):
                    if not prefix.startswith("r"):
                        kernel.numels[prefix] = kernel.numels[prefix] * loop_count
                logger.debug(
                    "SpyreTritonScheduling: configured kernel %s with "
                    "loop_count=%d, tiled_syms=%s",
                    type(kernel).__name__,
                    loop_count,
                    tiled_syms,
                )

        return kernels
