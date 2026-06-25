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

import sympy

from torch._inductor import config
from torch._inductor.codegen.simd import SIMDKernelFeatures
from torch._inductor.codegen.triton import FixedTritonConfig, TritonScheduling
from torch._inductor.dependencies import MemoryDep
from torch._inductor.runtime import triton_heuristics
from torch._inductor.scheduler import FusedSchedulerNode, SchedulerNode
from torch._inductor.tiling_utils import analyze_memory_coalescing
from torch._inductor.utils import IndentedBuffer, Placeholder
from torch._inductor.virtualized import V

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
        # Counter for naming SpyreTritonKernelBundle entry functions (bundle_0, …).
        self._bundle_counter: int = 0
        # Counter for naming standalone (non-bundle) kernels' opspec dumps
        # (kernel_0, kernel_1, …); bundle members get their bundled name instead.
        self._kernel_counter: int = 0
        # True while _codegen_bundle builds member kernels, so create_kernel_choices
        # leaves their _opspec_name for the bundle to assign.
        self._building_bundle: bool = False

    def codegen_node(self, node) -> None:
        if isinstance(node, CountedLoopSchedulerNode):
            self._codegen_counted_loop_triton(node)
            return

        # torch-spyre fuses ops with *different* iteration spaces into one
        # FusedSchedulerNode (spyre_fuse_nodes in fusion.py, driven by the
        # 6-tensor budget -- not iteration-space compatibility; Inductor's own
        # fusion is disabled).  A single upstream TritonKernel requires one
        # (numel, rnumel), so a heterogeneous group trips "unexpected group" in
        # generate_node_schedule.  Codegen each member SchedulerNode as its own
        # SpyreTritonKernel -- each node has exactly one iteration space by
        # construction -- and pack the members into a single SpyreTritonKernel-
        # Bundle: one async_compile.triton('triton_bundle_N', ...) block of
        # noinline bundled kernels + one entry fn + one .run() launch (one launch
        # keeps any LX intermediate live across the bundled kernels).  Same-
        # iteration-space merging of adjacent bundled kernels is a later
        # milestone.  CountedLoopSchedulerNode is a FusedSchedulerNode subclass
        # but is handled above, so it never reaches this split.
        if isinstance(node, FusedSchedulerNode):
            assert self.scheduler
            members = [
                n
                for n in node.get_nodes()
                if isinstance(n, SchedulerNode)
                and n.get_name() not in self.scheduler.removed_ops
            ]
            if len(members) > 1:
                logger.debug(
                    "SpyreTritonScheduling: bundling %s into %d per-node "
                    "SpyreTritonKernels",
                    node.get_name(),
                    len(members),
                )
                self._codegen_bundle(members)
                return

        return super().codegen_node(node)

    def _codegen_bundle(self, members: list) -> None:
        """Build one SpyreTritonKernel per member, then emit them as a bundle.

        Mirrors the per-member half of ``SIMDScheduling._codegen_nodes`` /
        ``codegen_node_schedule`` (tiling + kernel + body) but skips the per-
        member ``define_kernel`` / ``call_kernel``; the built bundled kernels are
        handed to ``_emit_bundle`` which synthesizes one entry kernel that calls
        them, emitted as a single
        ``async_compile.triton('triton_bundle_N', ...)`` block + one ``.run()``.
        """
        assert self.scheduler
        # Allocate the bundle id up front so each kernel can be tagged with its
        # final bundled-kernel name (triton_bundle_<id>_kernel_<i>) *before*
        # codegen_node_schedule_with_kernel runs -- the opspec dump (in
        # store/store_reduction) reads that name, and it must match the kernel
        # name _emit_bundle emits.
        bundle_id = self._bundle_counter
        self._bundle_counter += 1
        entry_name = f"triton_bundle_{bundle_id}"
        bundled_kernels: list = []
        # Suppress create_kernel_choices's kernel_<N> auto-naming while building
        # bundle members; their name is assigned explicitly below.
        self._building_bundle = True
        for i, member in enumerate(members):
            coalesce = (
                analyze_memory_coalescing(member)
                if config.triton.coalesce_tiling_analysis
                else None
            )
            _, (numel, rnumel) = member.group
            node_schedule = self.generate_node_schedule([member], numel, rnumel)
            features = SIMDKernelFeatures(node_schedule, numel, rnumel, coalesce)
            tiling, tiling_score = self.get_tiling_and_scores(
                node_schedule,
                features.numel,
                features.reduction_numel,
                features.coalesce_analysis,
            )
            (kernel,) = self.create_kernel_choices(
                features,
                [tiling],
                {"features": features, "tiling_scores": tiling_score},
            )
            kernel._opspec_name = f"{entry_name}_kernel_{i}"
            self.codegen_node_schedule_with_kernel(node_schedule, kernel)
            # Buffer/liveness bookkeeping normally done by codegen_node_schedule.
            with V.set_kernel_handler(kernel):
                for snode in features.scheduler_nodes():
                    snode.mark_run()
            V.graph.removed_buffers |= kernel.removed_buffers
            V.graph.inplaced_to_remove |= kernel.inplaced_to_remove
            bundled_kernels.append(kernel)
        self._building_bundle = False

        self._emit_bundle(bundled_kernels, entry_name)
        self.free_buffers_in_scheduler()

    def _bundled_kernel_constants(self, kernel) -> list[str]:
        """Literal numel + block-size args a bundled kernel takes after tensors.

        Order matches ``TritonKernel.codegen_kernel``: per-active-tree
        ``{prefix}numel`` then per-tree ``{PREFIX}BLOCK`` constexpr.  These are
        compile-time constants for the bundle, so they are passed as literals.
        """
        constants = [
            str(int(V.graph.sizevars.size_hint(tree.numel)))
            for tree in kernel.active_range_trees()
        ]
        cfg = kernel.fixed_config.config if kernel.fixed_config else {}
        for tree in kernel.range_trees:
            if tree.tensor_dim is None:
                continue
            key = f"{tree.prefix.upper()}BLOCK"
            if key in cfg:
                constants.append(str(cfg[key]))
        return constants

    def _emit_bundle(self, bundled_kernels: list, entry_name: str) -> None:
        """Synthesize one entry kernel that calls the bundled kernels.

        The entry has no IR node, so it is built by hand: the bundled kernels are
        added as ``noinline`` helper functions (emitted before the entry by
        ``codegen_kernel``), the entry's body is the sequence of their calls, and
        its args are the union of their tensors.  ``codegen_kernel`` /
        ``define_kernel`` / ``call_kernel`` then emit the decorator, signature,
        ``triton_meta`` and ``.run()`` -- reused from the normal path.

        ``entry_name`` (``triton_bundle_<id>``) is allocated by the caller so the
        per-kernel names (``<entry_name>_kernel_<i>``) match the ``_opspec_name``
        tags set before codegen.
        """
        # Per bundled kernel: source text, ordered tensor outer names, constants.
        bundled_kernel_texts: list[str] = []
        bundled_kernel_tensors: list[list[str]] = []
        bundled_kernel_constants: list[list[str]] = []
        written: list[str] = []  # outer names any bundled kernel writes (outputs)
        read: list[str] = []  # outer names any bundled kernel reads
        spyre_grids: dict[str, tuple] = {}
        for i, kernel in enumerate(bundled_kernels):
            bundled_kernel_name = f"{entry_name}_kernel_{i}"
            with V.set_kernel_handler(kernel):
                bundled_kernel_texts.append(
                    kernel.codegen_kernel(
                        name=bundled_kernel_name, as_bundled_kernel=True
                    )
                )
            _argdefs, call_args, _sig, _types = kernel.args.python_argdefs()
            bundled_kernel_tensors.append(list(call_args))
            bundled_kernel_constants.append(self._bundled_kernel_constants(kernel))
            for outer in kernel.args.output_buffers:
                if outer not in written:
                    written.append(outer)
            for outer in kernel.args.input_buffers:
                if outer not in read:
                    read.append(outer)
            spyre_grids[bundled_kernel_name] = kernel._compute_spyre_grid()

        # Entry args: inputs = read-but-never-written; outputs = written.  Build
        # a synthetic SpyreTritonKernel, drop its range trees (no iteration space
        # of its own -> only tensor params), and register the union of args.
        entry_inputs = [o for o in read if o not in written]
        entry_outputs = written
        entry_features = SIMDKernelFeatures([], sympy.S.One, sympy.S.One, None)
        (entry,) = self.create_kernel_choices(
            entry_features,
            [{"x": sympy.S.One}],
            {"features": entry_features, "tiling_scores": None},
        )
        # The entry only sequences bundled-kernel calls -- it has no iteration
        # space.  Drop its range trees so codegen_kernel emits no xnumel/XBLOCK
        # params, and clear the body of the xoffset/xindex/xmask prologue that
        # TritonKernel.__init__ emitted via codegen_range_tree() (the entry reads
        # no program_id; that prologue would reference an undefined XBLOCK).
        # Force Grid1D since _get_grid_type rejects 0 dims -- DistributeWork then
        # stamps grid=[1] for the pid-less entry; each bundled kernel carries its
        # own real grid.
        entry.range_trees = []
        entry.body.clear()
        entry._get_grid_type = lambda: triton_heuristics.Grid1D  # type: ignore[method-assign]
        entry.fixed_config = FixedTritonConfig(config={})
        outer_to_inner: dict[str, str] = {}
        for outer in entry_inputs:
            outer_to_inner[outer] = entry.args.input(outer)
        for outer in entry_outputs:
            outer_to_inner[outer] = entry.args.output(outer)

        # Bundled kernels are emitted before the entry via helper_functions; the
        # entry body calls each with the entry's own (inner) param names + consts.
        for text in bundled_kernel_texts:
            entry.helper_functions.finalized_helpers.append(text)
        for i in range(len(bundled_kernels)):
            bundled_kernel_name = f"{entry_name}_kernel_{i}"
            call_args = [outer_to_inner[o] for o in bundled_kernel_tensors[i]]
            call_args += bundled_kernel_constants[i]
            entry.body.writeline(f"{bundled_kernel_name}({', '.join(call_args)})")

        # Per-bundled-kernel grids for the backend; a single spyre_grid also
        # suffices today since every bundled kernel shares (32,).
        entry._bundle_meta = {
            "spyre_grids": spyre_grids,
            "spyre_entry": entry_name,
            "spyre_grid": next(iter(spyre_grids.values())),
        }

        # Reuse codegen_kernel (imports + bundled kernels + entry) but name the
        # entry ourselves (triton_bundle_<id>) instead of the auto-generated from
        # the standard define_kernel.  codegen_kernel(name=None) emits the
        # KERNEL_NAME / DESCRIPTIVE_NAME placeholders that define_kernel would
        # substitute; we substitute them with entry_name and emit one block.
        with V.set_kernel_handler(entry):
            src_code = entry.codegen_kernel()
        src_code = src_code.replace(str(Placeholder.KERNEL_NAME), entry_name)
        src_code = src_code.replace(str(Placeholder.DESCRIPTIVE_NAME), entry_name)

        wrapper = V.graph.wrapper_code
        compile_wrapper = IndentedBuffer()
        compile_wrapper.writeline(f"async_compile.triton({entry_name!r}, '''")
        compile_wrapper.splice(src_code, strip=True)
        compile_wrapper.writeline("''', device_str='spyre')")
        wrapper.define_kernel(
            entry_name, compile_wrapper.getvalue(), f"# bundle: {entry_name}"
        )
        entry.kernel_name = entry_name
        entry.call_kernel(entry_name)

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

        # Tag standalone (non-bundle) kernels with a kernel_<N> opspec name, so
        # their opspec dump reads kernel_0/kernel_1/… (consistent with the
        # bundle's _kernel_<i> suffix) instead of the scheduler node name.
        # Bundle members are skipped here; _codegen_bundle assigns their name.
        if not self._building_bundle:
            for kernel in kernels:
                kernel._opspec_name = f"kernel_{self._kernel_counter}"
                self._kernel_counter += 1

        return kernels
