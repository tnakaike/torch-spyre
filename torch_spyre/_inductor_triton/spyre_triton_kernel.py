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

import contextlib
from typing import Any, Optional

import sympy
import torch

from torch._inductor.codegen.common import (
    CSEProxy,
    CSEVariable,
    DeferredLine,
    triton_type,
)
from torch._inductor.codegen.triton import (
    FixedTritonConfig,
    TritonKernel,
    TritonSymbols,
    get_triton_reduction_function,
)
from torch._inductor.dependencies import MemoryDep
from torch._inductor.utils import IndentedBuffer, sympy_subs
from torch._inductor.virtualized import ReductionType, StoreMode, V
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._sympy.symbol import SymT, symbol_is_type

from torch_spyre._inductor.ir import FixedTiledLayout
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.op_spec import IndirectAccess, LoopSpec, OpSpec, TensorArg
from torch_spyre._inductor.pass_utils import (
    apply_splits_from_index_coeff,
    concretize_index,
    iteration_space,
)
from torch_spyre._inductor.spyre_kernel import (
    _codegen_op_spec_list,
    _iter_op_specs,
    simplify_op_spec,
)
from torch_spyre._inductor.views import compute_coordinates


def _normalize_floor_div(expr: sympy.Expr) -> sympy.Expr:
    """Replace sympy.floor(a/n) with FloorDiv(a, n) for integer floor division.

    compute_coordinates uses Python's ``//`` operator, which sympy evaluates to
    ``floor(a / n)`` — a rational floor.  TritonPrinter renders this as
    ``libdevice.floor((1/n)*a)`` (floating-point), which is wrong for integer
    offsets.  This walks the expression tree and replaces every such subexpression
    with the Inductor-native FloorDiv form, which prints as ``a // n``.
    """

    def _rewrite(e: sympy.Expr) -> sympy.Expr:
        if isinstance(e, sympy.floor):
            inner = e.args[0]
            coeff, base = inner.as_coeff_Mul()
            if isinstance(coeff, sympy.Rational) and coeff.p > 0 and coeff.q > 1:
                return FloorDiv(_normalize_floor_div(base) * coeff.p, coeff.q)
            # floor(expr) with no rational factor: floor(expr/1)
            return FloorDiv(_normalize_floor_div(inner), 1)
        if e.args:
            new_args = [_rewrite(a) for a in e.args]
            if any(na is not oa for na, oa in zip(new_args, e.args)):
                return e.func(*new_args)
        return e

    return _rewrite(expr)


logger = get_inductor_logger("spyre_triton_kernel")

# TritonKernelOverrides is not publicly exported; access it via the class attr.
_TritonKernelOverrides: type[Any] = TritonKernel.overrides


class SpyreTritonOverrides(_TritonKernelOverrides):  # type: ignore[misc]
    """TritonKernelOverrides subclass for SpyreTritonKernel.

    Overrides ops.dot to handle 3D descriptor blocks: reshapes the loaded
    [M_tile, K-sticks, K-elems] and [K, N-sticks, N-elems] blocks to 2D
    matrices before calling tl.dot.

    Overrides the elementwise math ops that upstream emits via ``libdevice``
    (a GPU/CUDA-specific module the Spyre Triton backend does not provide, so
    ``libdevice.*`` resolves to ``None`` at ttir generation time).  Each is
    re-emitted as the equivalent ``tl.*`` op, which lowers to the MLIR
    ``math`` dialect that the Spyre backend accepts.

    Only ops that have a ``tl.*`` equivalent are overridden: ``exp``, ``exp2``,
    ``log2``, ``erf``, ``rsqrt``, ``floor``, ``ceil``.  Upstream ops without a
    ``tl.*`` counterpart (e.g. ``expm1``, ``log10``, ``log1p``, ``tan``,
    ``sinh``/``cosh``, the inverse-trig family, ``pow``, ``fmod``, ``erfc``,
    ``lgamma``, ``trunc``, ``round``) keep their libdevice mapping and will
    still fail until the Spyre backend grows native support.  ``abs``, ``cos``,
    ``sin``, ``log`` and ``sqrt`` already use ``tl_math.*``/``tl.*`` upstream
    and need no override.
    """

    @staticmethod
    def _tl_unary(fn: str, x) -> str:
        """Emit a Spyre ``tl.*`` unary math op, upcasting fp16/bf16 to fp32.

        Spyre's ``tl.*`` math ops accept only fp32/fp64, so fp16/bf16 inputs
        are upcast to fp32 and the result is downcast back to the input dtype.
        (Upstream relies on ``libdevice``, which accepts fp16 on GPU, so it
        does not upcast under the default ``codegen_upcast_to_fp32=True``.)
        """
        dtype = getattr(x, "dtype", None)
        if dtype in (torch.float16, torch.bfloat16):
            return f"{fn}({x}.to(tl.float32)).to({triton_type(dtype)})"
        return f"{fn}({x})"

    @staticmethod
    def exp(x):  # type: ignore[override]
        return SpyreTritonOverrides._tl_unary("tl.exp", x)

    @staticmethod
    def exp2(x):  # type: ignore[override]
        return SpyreTritonOverrides._tl_unary("tl.exp2", x)

    @staticmethod
    def log2(x):  # type: ignore[override]
        return SpyreTritonOverrides._tl_unary("tl.log2", x)

    @staticmethod
    def erf(x):  # type: ignore[override]
        return SpyreTritonOverrides._tl_unary("tl.erf", x)

    @staticmethod
    def rsqrt(x):  # type: ignore[override]
        return SpyreTritonOverrides._tl_unary("tl.rsqrt", x)

    @staticmethod
    def floor(x):  # type: ignore[override]
        return SpyreTritonOverrides._tl_unary("tl.floor", x)

    @staticmethod
    def ceil(x):  # type: ignore[override]
        return SpyreTritonOverrides._tl_unary("tl.ceil", x)

    @staticmethod
    def dot(a, b):  # type: ignore[override]
        """Collapse descriptor blocks to matrices (or batched matrices) for tl.dot.

        A descriptor block's two innermost dims are the stick split of the
        contraction/free dim (``[..., sticks, elems]``); collapsing them yields
        the matrix dim.  The leading dims are kept as-is, so:

        - plain matmul: A ``[M, Ksticks, Kelems]`` -> ``[M, K]`` and
          B ``[K, Nsticks, Nelems]`` -> ``[K, N]`` (2D tl.dot).
        - batched matmul (bmm): A ``[B, M, Ksticks, Kelems]`` -> ``[B, M, K]``
          and B ``[B, K, Nsticks, Nelems]`` -> ``[B, K, N]`` (batched tl.dot;
          the leading batch dim is preserved).  The batch dim is placed first by
          ``_matmul_operand_permutation`` when the descriptor is built.

        A linear-derived bmm carries a batch dim on the activation but a
        broadcast (un-batched) weight, so the two collapsed operands differ in
        rank (e.g. A ``[batch, M, K]`` vs B ``[K, N]``).  ``tl.dot`` requires
        equal ranks; because the weight is shared across both batch and M, those
        leading dims are all just matmul rows, so they are folded into a single
        row dim on the higher-rank operand (``[batch, M, K] -> [batch*M, K]``).
        The size-1 batch case (``[1, M, K] -> [M, K]``) is the degenerate
        instance.  The store side reshapes the ``[rows, N]`` result back to the
        output block shape.
        """
        kernel = V.kernel

        def _collapse(operand):
            shape = getattr(operand, "shape", None)
            if not shape or len(shape) < 3:
                return operand  # already a (batched) matrix
            lead = [int(s) for s in shape[:-2]]
            inner = int(shape[-2]) * int(shape[-1])
            new_shape = lead + [inner]
            return kernel.cse.generate(
                kernel.compute,
                f"tl.reshape({operand}, {new_shape})",
                dtype=operand.dtype,
                shape=tuple(str(s) for s in new_shape),
            )

        def _fold_leading_dims(operand, target_rank):
            # Fold the leading dims (all but the last, collapsed matrix dim)
            # into a single row dim so the operand reaches target_rank.  For a
            # broadcast-weight bmm the leading dims (batch, M) are all matmul
            # rows sharing the same weight, so [batch, M, K] -> [batch*M, K]
            # (and the size-1 [1, M, K] -> [M, K]) is exact.  Only used to match
            # a broadcast operand's lower rank; a real bmm (both batched, equal
            # rank) never reaches here.
            shape = getattr(operand, "shape", None)
            if shape is None or len(shape) <= target_rank or target_rank < 2:
                return operand
            dims = [int(s) for s in shape]
            n_fold = len(dims) - target_rank + 1  # leading dims merged into one
            row = 1
            for s in dims[:n_fold]:
                row *= s
            new_shape = [row] + dims[n_fold:]
            return kernel.cse.generate(
                kernel.compute,
                f"tl.reshape({operand}, {new_shape})",
                dtype=operand.dtype,
                shape=tuple(str(s) for s in new_shape),
            )

        a_mat = _collapse(a)
        b_mat = _collapse(b)

        # Reconcile operand ranks for tl.dot (linear-derived bmm: batched
        # activation vs broadcast weight).  Fold the higher-rank operand's
        # leading dims into its row dim.
        ra = len(getattr(a_mat, "shape", ()) or ())
        rb = len(getattr(b_mat, "shape", ()) or ())
        if ra > rb > 0:
            a_mat = _fold_leading_dims(a_mat, rb)
        elif rb > ra > 0:
            b_mat = _fold_leading_dims(b_mat, ra)

        return f'tl.dot({a_mat}, {b_mat}, input_precision="ieee")'


class _SpyreGatherCSEProxy(CSEProxy):
    """CSEProxy that skips the upstream negative-index wrap + bounds-check for
    ``indirect_indexing``.

    Upstream ``CSEProxy.indirect_indexing`` emits
    ``ops.add(index, index_expr(size))`` (plus ``where`` / ``device_assert``)
    and propagates shapes by broadcasting.  That broadcast is incompatible with
    SpyreTritonKernel's device-tile-shaped descriptor loads: the loaded index
    carries its device block shape (e.g. ``(1, 32)``), not the kernel's
    iteration shape (e.g. ``(XBLOCK,)``), so the broadcast asserts whenever they
    do not match (it happens to broadcast for some tilings, which is why a 2-D
    source slips through but a 1-D-flattened one does not).

    The Spyre gather addresses with the raw index buffer, so the wrap result and
    the bounds assert are unused for addressing.  We mint the index symbol
    directly (matching the SDSC ``SpyreKernel`` path) and skip both.
    """

    def indirect_indexing(
        self,
        var: CSEVariable,
        size: Any,
        check: bool = True,
        wrap_neg: bool = True,
    ) -> sympy.Symbol:
        if isinstance(size, int):
            size = sympy.Integer(size)
        return self.parent_handler.indirect_indexing(var, size, check)


class SpyreTritonKernel(TritonKernel):
    overrides = SpyreTritonOverrides  # type: ignore[assignment]

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
        # Mapping from Triton prefixes ("x", "r0_", ...) to OpSpec symbols (c0, c1, ...)
        self.triton_opspec_map: dict[str, list[sympy.Symbol]] = {}
        # Captured from the most recent split_and_set_ranges call; consumed by
        # set_current_node to build triton_opspec_map before the first load/store.
        self._pending_split_result: list[list[sympy.Expr]] = []
        # Maps each Triton entry symbol to its corresponding OpSpec symbol.
        # Populated by _build_triton_opspec_map; used in load/store for offset computation.
        self._triton_to_opspec: dict[sympy.Symbol, sympy.Symbol] = {}
        # Core divisors per OpSpec symbol (how many cores split each dimension).
        self._core_division: dict[sympy.Symbol, int] = {}
        # Caches emitted Level-2 variable names: opspec_sym -> var_name (e.g. c0 -> "c0").
        # Populated by _emit_scalar_offsets on first call; shared across all buffers.
        self._logical_offset_vars: dict[sympy.Symbol, str] = {}
        # Caches emitted Level-3 variable names keyed by device_coords tuple.
        # Each entry maps coords_key -> [dim0, dim1, ...] var name list.
        self._device_offset_vars: dict[tuple, list[str]] = {}
        # Guard: dump the per-kernel opspec file only once per kernel instance.
        self._opspec_dumped: bool = False
        # Tiling loop state (set by SpyreTritonScheduling for CountedLoopSchedulerNode).
        self._tiling_loop_count: Optional[int] = None
        self._tiling_loop_tiled_syms: list[sympy.Symbol] = []
        # Maps each tiled symbol to its per-tile range (tile_step to add per iteration).
        self._tiling_tile_steps: dict[sympy.Symbol, int] = {}
        # Buffer for Logical→Device assignments emitted inside the tile loop.
        self._loop_offset_code: Optional[IndentedBuffer] = None
        # Set True in load() when an indirect (gather) load is emitted; read by
        # store() to permute the output descriptor (gathered/output-row axis to
        # dim 0) so the gather result block stores directly.
        self._is_gather: bool = False
        # The gathered output-row iteration symbol (the index dep's symbol, e.g.
        # c0).  Set during the gather load; used by store() to permute the
        # output-row device dim to dim 0 so the row-first gather result stores
        # directly (matching shape with the gather result).
        self._gather_row_sym: Optional[sympy.Symbol] = None
        # When this kernel is a SpyreTritonKernelBundle entry, extra triton_meta
        # fields (spyre_grids, spyre_entry, spyre_grid) to merge in codegen_body
        # so they land in the emitted @fixed_config decorator.  None otherwise.
        self._bundle_meta: Optional[dict] = None
        # Kernel name for the opspec dump's op name and file name, assigned by
        # the scheduler before codegen: triton_bundle_<id>_kernel_<i> for bundle
        # members, kernel_<N> for standalone kernels.  None only if unset (the
        # dump then falls back to the scheduler node name).
        self._opspec_name: Optional[str] = None

    def __enter__(self):
        super(TritonKernel, self).__enter__()
        # Stack a CSEProxy that skips the upstream indirect_indexing wrap /
        # bounds-check codegen (incompatible with device-tile-shaped descriptor
        # loads).  This shadows the standard CSEProxy installed by the base
        # __enter__; all other ops behave identically.
        self.exit_stack.enter_context(
            V.set_ops_handler(_SpyreGatherCSEProxy(self, self.overrides()))
        )
        return self

    def split_and_set_ranges(self, lengths):
        result = super().split_and_set_ranges(lengths)
        self._pending_split_result = result
        return result

    @contextlib.contextmanager
    def set_current_node(self, node):
        with super().set_current_node(node):
            # Build the map once, before the body (and thus any load/store) runs.
            # Fused nodes share the same iteration space, so the map from the
            # first node is reused for all subsequent nodes in the kernel.
            if self._pending_split_result and not self.triton_opspec_map:
                self.triton_opspec_map = self._build_triton_opspec_map()
                self._core_division = self._compute_core_division()
                block_size = self._get_triton_block_size()
                config = {
                    f"{prefix.upper()}BLOCK": size
                    for prefix, size in block_size.items()
                }
                self.fixed_config = FixedTritonConfig(config=config)
                logger.debug(
                    "SpyreTritonKernel: fixed_config=%s", self.fixed_config.config
                )
            yield

    def codegen_kernel(self, name=None, as_bundled_kernel=False) -> str:
        src = super().codegen_kernel(name)
        if not as_bundled_kernel:
            return src
        # Bundled-kernel flavor for a SpyreTritonKernelBundle: keep only the
        # function (from `def {name}(` onward) and make it a device-callable
        # noinline jit function.  The host-side @triton_heuristics.fixed_config
        # decorator is dropped -- it returns a launcher, which the bundle entry
        # cannot call; the entry carries the config/grids instead.
        # super().codegen_kernel (name set) already omits the imports block, and
        # still populates self.triton_meta / self.fixed_config as a side effect,
        # which the bundle emitter reads back for the entry's signature and grids.
        lines = src.splitlines()
        def_idx = next(
            i
            for i, line in enumerate(lines)
            if line.lstrip().startswith(f"def {name}(")
        )
        return "@triton.jit(noinline=True)\n" + "\n".join(lines[def_idx:])

    def codegen_body(self):
        if self._tiling_loop_count is not None:
            # Scale spatial range-tree numels by tile_count so that
            # codegen_static_numels() (called by codegen_kernel() after this
            # method) writes xnumel = full_tensor_numel instead of the
            # per-tile numel of the inner SchedulerNode.
            for tree in self.range_trees:
                if not tree.prefix.startswith("r"):
                    tree.numel = tree.numel * int(self._tiling_loop_count)

        # Inject spyre_grid into triton_meta so async_compile.py can pass
        # the correct grid shape to SpyreOptions (and thus DistributeWork).
        # triton_meta is set on self just before codegen_body() is called,
        # and the same dict object is serialized into the generated Python —
        # mutations here ARE reflected in the @fixed_config decorator call.
        if self.triton_meta is not None and self.triton_opspec_map:
            self.triton_meta["spyre_grid"] = self._compute_spyre_grid()

        # Bundle entry: merge spyre_grids / spyre_entry / spyre_grid so they
        # serialize into the entry's @fixed_config decorator (triton_meta is set
        # on self just before codegen_body() runs).
        if self._bundle_meta is not None and self.triton_meta is not None:
            self.triton_meta.update(self._bundle_meta)

        if self._tiling_loop_count is None:
            return super().codegen_body()
        return self._codegen_tile_loop_body()

    def _codegen_tile_loop_body(self) -> None:
        """Emit the tiling-loop body: indexing_code outside, loop wrapping the rest.

        Called when _tiling_loop_count is set (CountedLoopSchedulerNode path).
        Only pointwise (non-reduction) kernels are supported; reduction kernels
        fall through to the standard TritonKernel codegen_body path unchanged.
        """
        if self.inside_reduction:
            # Reduction tiling is not yet supported — fall back to the standard
            # TritonKernel body (no tile loop).
            TritonKernel.codegen_body(self)
            return

        if not (
            self.indexing_code
            or self.loads
            or self.stores
            or self.compute
            or self.post_loop_combine
            or self.post_loop_store
        ):
            return

        # Pointwise: indexing_code is stable per program invocation and goes
        # OUTSIDE the tile loop.
        self.body.splice(self.indexing_code)
        self.indexing_code.clear()

        # Emit the tile loop.
        assert self._tiling_loop_count is not None
        self.body.writeline(
            f"for tile_idx in tl.range({int(self._tiling_loop_count)}):"
        )
        with self.body.indent():
            # Logical→Device offset variables (with tile_idx adjustment for
            # tiled dims) go inside the loop so tile_idx is in scope.
            if self._loop_offset_code:
                self.body.splice(self._loop_offset_code)
                self._loop_offset_code.clear()
            self.body.splice(self.loads)
            self.body.splice(self.compute)
            self.body.splice(self.stores)

        self.loads.clear()
        self.compute.clear()
        self.stores.clear()
        if hasattr(self, "post_loop_combine"):
            self.post_loop_combine.clear()
        if hasattr(self, "post_loop_store"):
            self.post_loop_store.clear()

    def _compute_spyre_grid(self) -> tuple:
        """Compute the per-axis program count for SpyreOptions.grid.

        Returns a tuple (num_x_programs [, num_y_programs]) where:
          grid[0] = programs on axis 0 (x = tl.program_id(0))
          grid[1] = programs on axis 1 (y = tl.program_id(1)), if present

        For a 1D pointwise kernel with XBLOCK=16384 and xnumel=524288 the
        result is (32,).  For a 2D matmul kernel with XBLOCK=512/xnumel=512
        and YBLOCK=8/ynumel=256 the result is (1, 32).
        """
        config = self.fixed_config.config if self.fixed_config else {}
        grid = []
        for prefix in ["x", "y", "z"]:
            numel = self.numels.get(prefix)
            if numel is None:
                break
            block = config.get(f"{prefix.upper()}BLOCK", 1)
            numel_hint = V.graph.sizevars.size_hint(numel)
            grid.append(max(1, (numel_hint + block - 1) // block))
        assert grid, "_compute_spyre_grid: no x-range numel found in self.numels"
        return tuple(grid)

    def load(self, name: str, index: sympy.Expr):
        if not self.triton_opspec_map:
            return super().load(name, index)

        buf = V.graph.get_buffer(name)
        layout = buf.get_layout() if buf is not None else None
        if not isinstance(layout, FixedTiledLayout):
            return super().load(name, index)

        # LX and pool (HBM intermediate-pool) buffers are both routed through the
        # descriptor path.  After the per-node split a fused intermediate is a
        # real buffer threaded between separate bundled kernels (not a removed SSA
        # temp), so it must load/store like any other device tensor.  A pool
        # tensor gets a plain HBM descriptor (LowerDescriptorMemory defaults the
        # memory space to HBM); full LX-as-baked-offset support is a later
        # milestone.

        # Find the MemoryDep for this buffer in the read set.
        dep = next(
            (d for d in self.current_node.read_writes.reads if d.name == name),
            None,
        )
        if dep is None:
            return super().load(name, index)

        # Resolve mutation aliases for arg registration.
        scheduler = getattr(V.graph, "scheduler", None)
        resolved = (
            name if scheduler is None else scheduler.mutation_real_name.get(name, name)
        )
        var = self.args.input(resolved)

        # Indirect (gather) load: a device coordinate references an indirect
        # SymT.TMP symbol (the loaded index). Route to the gather primitive
        # instead of a plain descriptor offset load (which the Spyre Triton
        # backend cannot lower).
        k_star, device_coords = self._gather_indirect_dim(dep, layout)
        if k_star is not None:
            return self._emit_gather_load(name, var, dep, layout, k_star, device_coords)

        if self.is_native_matmul:
            desc_var, offset_var_names, block_shape = (
                self._emit_symbol_first_tensor_descriptor(name, var, dep, layout)
            )
        else:
            desc_var, offset_var_names, block_shape = self._emit_tensor_descriptor(
                name, var, dep, layout
            )
        return self._emit_descriptor_load(
            name, desc_var, offset_var_names, block_shape, layout
        )

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> None:
        if not self.triton_opspec_map or mode is not None:
            return super().store(name, index, value, mode)

        buf = V.graph.get_buffer(name)
        layout = buf.get_layout() if buf is not None else None
        if not isinstance(layout, FixedTiledLayout):
            return super().store(name, index, value, mode)

        # LX and pool (HBM intermediate-pool) buffers are both routed through the
        # descriptor path (mirror load()): a bundled intermediate is a real buffer
        # threaded between separate kernels, so its store must be emitted (not
        # elided as a removed SSA temp).  pool tensors get a plain HBM descriptor.

        # Find the MemoryDep for this buffer in the write set.
        dep = next(
            (d for d in self.current_node.read_writes.writes if d.name == name),
            None,
        )
        if dep is None:
            return super().store(name, index, value, mode)

        var = self.args.output(name)

        if not self._opspec_dumped:
            self._opspec_dumped = True
            self._dump_opspec(name, dep)

        if self._is_gather:
            # Gather output: the gather result block is [num_rows, *rest], with
            # the output-row axis at dim 0.  Permute the output device dim whose
            # coordinate is the index's iteration symbol (self._gather_row_sym)
            # to dim 0, so the row-first gather result stores with matching shape
            # (not just the first bare symbol, which is wrong for >=3D sources).
            desc_var, offset_var_names, block_shape = (
                self._emit_symbol_first_tensor_descriptor(
                    name, var, dep, layout, row_sym=self._gather_row_sym
                )
            )
        else:
            desc_var, offset_var_names, block_shape = self._emit_tensor_descriptor(
                name, var, dep, layout
            )
        self._emit_descriptor_store(name, desc_var, offset_var_names, layout, value)

    def _get_reduction_axis(self) -> int:
        """Return the device axis to pass to tl.sum for Spyre descriptor kernels.

        The reduction symbol (e.g. c1 = 4096 elements) maps to two device
        dimensions for stick layouts:
          - NS  (outer stick): coord = c1 // stick_size  (FloorDiv) → reduce here
          - NE  (inner stick): coord = c1 %  stick_size  (Mod)       → skip

        Reducing over NS while leaving NE intact is correct: the Spyre KTIR
        backend implicitly reduces NE in hardware when it sees a tt.reduce on
        the outer stick axis.

        For non-stick reductions (coord == c1 directly), returns the device
        dimension that contains the reduction symbol without a Mod wrapper.
        """
        it_space = iteration_space(self.current_node)
        write_dep = next(iter(self.current_node.read_writes.writes))
        spatial_syms = set(write_dep.ranges.keys())
        reduction_syms = [s for s in it_space if s not in spatial_syms]

        if not reduction_syms:
            return 0

        for read_dep in self.current_node.read_writes.reads:
            buf = V.graph.get_buffer(read_dep.name)
            if buf is None:
                continue
            layout = buf.get_layout()
            if not isinstance(layout, FixedTiledLayout):
                continue

            device_size = [int(s) for s in layout.device_layout.device_size]
            dep_idx = sympy_subs(
                read_dep.index, V.graph.sizevars.precomputed_replacements
            )
            dep_idx = concretize_index(dep_idx, set(it_space.keys()))
            device_coords = compute_coordinates(
                device_size, layout.device_layout.stride_map, it_space, dep_idx
            )

            for r_sym in reduction_syms:
                for k, coord in enumerate(device_coords):
                    if r_sym not in coord.free_symbols:
                        continue
                    # Skip Mod / ModularIndexing dims (inner stick = NE).
                    # Only the FloorDiv or direct-symbol dim is the outer
                    # stick (NS) that we want to reduce.
                    if isinstance(coord, (sympy.Mod, ModularIndexing)):
                        continue
                    logger.debug(
                        "SpyreTritonKernel: reduction axis=%d (sym=%s, coord=%s)",
                        k,
                        r_sym,
                        coord,
                    )
                    return k
            break  # only inspect the first FixedTiledLayout read dep

        return 0  # fallback

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: CSEVariable,
    ) -> CSEVariable:
        """For Spyre descriptor kernels: reduce over the outer stick dimension.

        Bypasses TritonKernel's broadcast_to pattern (incompatible with 3D
        descriptor blocks) and emits the appropriate tl.* or triton_helpers.*
        reduction call directly.

        "dot" (matmul) and indexed reductions ("argmin", "argmax") are not
        handled here — they are deferred to the parent which uses a different
        code path.
        """
        if not self.triton_opspec_map or not self.current_node.is_reduction():
            return super().reduction(dtype, src_dtype, reduction_type, value)

        # Matmul: the value is the 2D tl.dot result from SpyreTritonOverrides.dot().
        # Pass it through to store_reduction() without any reshaping.
        if reduction_type == "dot" and self.is_native_matmul:
            return value

        # Indexed reductions are structurally different; defer to TritonKernel.
        if reduction_type in ("dot", "argmin", "argmax"):
            return super().reduction(dtype, src_dtype, reduction_type, value)

        # Upcast fp16/bf16 → float32 for all reduction types.  TritonKernel
        # does this unconditionally because max/min don't support fp16/bf16.
        if hasattr(value, "dtype") and value.dtype in (torch.float16, torch.bfloat16):
            value = self.cse.generate(
                self.compute,
                f"{value}.to(tl.float32)",
                dtype=torch.float32,
                shape=getattr(value, "shape", None),
            )

        triton_fn = get_triton_reduction_function(reduction_type)
        axis = self._get_reduction_axis()
        val_shape = getattr(value, "shape", None)
        result_shape = (
            val_shape[:axis] + val_shape[axis + 1 :]
            if (val_shape and axis < len(val_shape))
            else None
        )
        result_dtype = getattr(value, "dtype", dtype)
        return self.cse.generate(
            self.compute,
            f"{triton_fn}({value}, {axis})",
            dtype=result_dtype,
            shape=result_shape,
        )

    def store_reduction(self, name: str, index: sympy.Expr, value: CSEVariable) -> None:
        if not self.triton_opspec_map:
            return super().store_reduction(name, index, value)

        buf = V.graph.get_buffer(name)
        layout = buf.get_layout() if buf is not None else None
        if not isinstance(layout, FixedTiledLayout):
            return super().store_reduction(name, index, value)

        # LX and pool (HBM intermediate-pool) buffers are both routed through the
        # descriptor path (mirror load()/store()) so a reduction output on LX or
        # in the HBM pool is materialized between split kernels.  pool tensors get
        # a plain HBM descriptor.

        dep = next(
            (d for d in self.current_node.read_writes.writes if d.name == name),
            None,
        )
        if dep is None:
            return super().store_reduction(name, index, value)

        var = self.args.output(name)

        if not self._opspec_dumped:
            self._opspec_dumped = True
            self._dump_opspec(name, dep)

        if self.is_native_matmul:
            desc_var, offset_var_names, block_shape = (
                self._emit_symbol_first_tensor_descriptor(name, var, dep, layout)
            )
        else:
            desc_var, offset_var_names, block_shape = self._emit_tensor_descriptor(
                name, var, dep, layout
            )

        # Downcast from float32 (reduction accumulation dtype) to output dtype.
        out_dtype = V.graph.get_dtype(name)
        if out_dtype in (torch.float16, torch.bfloat16):
            triton_dtype = "tl.float16" if out_dtype == torch.float16 else "tl.bfloat16"
            value = self.cse.generate(
                self.stores,
                f"{value}.to({triton_dtype})",
                dtype=out_dtype,
                shape=getattr(value, "shape", None),
            )

        # Reshape to the declared block_shape so the descriptor store receives
        # a matching 3D block.
        # For matmul: [M_tile, N] → [M_tile, N-sticks, N-elems]
        # For sum:    [M, NE]     → [1, M, NE]
        reshaped = self.cse.generate(
            self.stores,
            f"tl.reshape({value}, {list(block_shape)})",
            dtype=out_dtype,
            shape=tuple(str(s) for s in block_shape),
        )

        self._emit_descriptor_store(name, desc_var, offset_var_names, layout, reshaped)

    def _emit_tensor_descriptor(
        self,
        name: str,
        var: str,
        dep,
        layout: "FixedTiledLayout",
    ) -> tuple[str, list[str], list]:
        """Emit tl.make_tensor_descriptor to prologue (hoisted, loop-invariant).

        Scalar offset variables are emitted before the descriptor so the kernel
        body reads:

            # Triton -> Logical layouts
            c0 = xoffset // N
            c1 = xoffset % N
            # Logical layouts -> Device layouts
            dim0 = c1 // 64
            ...
            desc_0 = tl.make_tensor_descriptor(...)

        Returns (descriptor_variable_name, offset_var_names, block_shape) where
        offset_var_names is the list of Level-3 dim variable name strings (e.g.
        ["dim0", "dim1", "dim2"]) reused by _emit_descriptor_load /
        _emit_descriptor_store.
        """
        it_space = iteration_space(self.current_node)
        device_size = [int(s) for s in layout.device_layout.device_size]

        dep_index = sympy_subs(dep.index, V.graph.sizevars.precomputed_replacements)
        dep_index = concretize_index(dep_index, set(it_space.keys()))

        device_coords = compute_coordinates(
            device_size,
            layout.device_layout.stride_map,
            it_space,
            dep_index,
        )

        block_shape = self._device_block_shape(device_size, device_coords)
        strides = self._row_major_strides(device_size)

        # Emit the two-section scalar offset block before the descriptor so that
        # dim0/dim1/... variables are defined when the descriptor line is reached.
        offset_var_names = self._emit_scalar_offsets(name, device_coords)

        f = self.index_to_str
        desc_line = (
            f"tl.make_tensor_descriptor("
            f"{var}, "
            f"shape={f(device_size)}, "
            f"strides={f(strides)}, "
            f"block_shape={f(block_shape)})"
        )

        existing = self.cse.try_get(desc_line)
        if existing:
            return str(existing), offset_var_names, block_shape

        block_ptr_id = next(self.block_ptr_id)
        desc_name = f"desc_{block_ptr_id}"
        named_var = self.cse.namedvar(desc_name, dtype=torch.uint64, shape=[])
        self.cse.put(desc_line, named_var)
        self.prologue.writeline(DeferredLine(name, f"{desc_name} = {desc_line}"))
        logger.debug("SpyreTritonKernel: emitted %s = %s", desc_name, desc_line)
        return desc_name, offset_var_names, block_shape

    @staticmethod
    def _symbol_first_permutation(device_coords: list) -> list:
        """Return the permutation that places the plain-symbol dim first.

        For standard Spyre 3D tensor device layouts the structure is always
        [FloorDiv_expr, plain_symbol, Mod_expr].  This moves the device
        dimension whose coordinate is a bare iteration symbol (not a
        FloorDiv/Mod stick split) to position 0.  Callers want that dim
        leading: matmul reshapes the loaded 3D block to a 2D matrix for
        tl.dot via a single tl.reshape; the gather output store puts the
        output-row axis first so the gather result stores directly.

        Returns the permutation as a list of indices, e.g. [1, 0, 2].
        """
        for k, coord in enumerate(device_coords):
            if isinstance(coord, sympy.Symbol):
                if k == 0:
                    return list(range(len(device_coords)))
                return [k] + [i for i in range(len(device_coords)) if i != k]
        return list(range(len(device_coords)))

    @staticmethod
    def _matmul_operand_permutation(
        device_coords: list, batch_sym: Optional[sympy.Symbol] = None
    ) -> list:
        """Permutation placing the sticked matrix dim's stick pair innermost.

        A Spyre matmul operand's non-leading matrix dim (K for A, N for B) is
        stored as a stick split: an outer-stick dim (``FloorDiv(sym, stick)``)
        and the within-stick dim (``Mod(sym, stick)``), the latter always the
        last device dim.  ``SpyreTritonOverrides.dot()`` collapses the two
        innermost dims into that matrix dim, so the outer-stick and within-stick
        dims must be adjacent and innermost, with the remaining dims (the leading
        matrix dim(s) — batch/M for A, batch/K for B) kept ahead of them.  When
        a batch dim is present (bmm) it must lead so the block reshapes to a
        batched matrix ``[B, M, K]`` / ``[B, K, N]`` for a batched tl.dot.

        Anchoring on the stick pair — the two dims that share the within-stick
        dim's iteration symbol — keeps this correct even when the row dim M is
        size 1 and its coordinate degenerates to a constant ``0`` (the
        decode-phase / GEMV case, where the old bare-symbol search found nothing
        and left the K stick dims non-adjacent).  For a non-degenerate operand
        it yields the same order the bare-symbol permutation did, so the working
        matmul/bmm paths are unchanged.
        """
        rank = len(device_coords)
        if rank < 3:
            return list(range(rank))  # already a (batched) matrix; nothing to move
        within = rank - 1  # within-stick dim is always the last device dim
        within_syms = device_coords[within].free_symbols
        outer = None
        if within_syms:
            outer = next(
                (
                    k
                    for k in range(rank - 1)
                    if device_coords[k].free_symbols & within_syms
                ),
                None,
            )
        if outer is None:
            return list(range(rank))  # not stick-split; leave as-is
        leading = [k for k in range(rank) if k not in (outer, within)]
        # A batched-matmul operand must lead with its batch dim.
        if batch_sym is not None:
            b = next((k for k in leading if device_coords[k] == batch_sym), None)
            if b is not None:
                leading = [b] + [k for k in leading if k != b]
        return leading + [outer, within]

    def _batch_symbol(self) -> Optional[sympy.Symbol]:
        """OpSpec symbol of the bmm batch dim (the ``z`` Triton prefix), or None.

        A batched matmul has a 3D grid (z=batch, y=M, x=N); a plain 2D matmul
        has no ``z`` prefix.  The batch symbol is used to keep the batch device
        dim leading so the loaded block reshapes to a batched matrix for tl.dot.
        """
        syms = self.triton_opspec_map.get("z")
        return syms[0] if syms else None

    def _batch_symbol_first_permutation(
        self, device_coords: list, batch_sym: sympy.Symbol
    ) -> list:
        """Permutation that leads with the batch dim, then the plain-symbol dim.

        For a bmm operand the device layout carries both the batch symbol and
        the matmul row/contraction symbol as bare iteration symbols.  This
        orders them ``[batch, plain_symbol, <remaining stick dims>]`` so the
        loaded block reshapes cleanly to ``[B, M, K]`` / ``[B, K, N]`` (batch
        leading) for a batched tl.dot.  Falls back to ``_symbol_first_permutation``
        when the batch dim is absent from this tensor.
        """
        batch_idx = next(
            (k for k, c in enumerate(device_coords) if c == batch_sym), None
        )
        if batch_idx is None:
            return self._symbol_first_permutation(device_coords)
        rest = [i for i in range(len(device_coords)) if i != batch_idx]
        # The plain-symbol (matmul row/contraction) dim among the non-batch dims.
        sym_idx = next(
            (i for i in rest if isinstance(device_coords[i], sympy.Symbol)),
            rest[0],
        )
        ordered_rest = [sym_idx] + [i for i in rest if i != sym_idx]
        return [batch_idx] + ordered_rest

    def _gather_output_permutation(
        self, device_coords: list, row_sym: Optional[sympy.Symbol]
    ) -> list:
        """Permutation that places the gathered output-row axis first.

        The gather result is row-first (``[num_rows, ...]``), so the output
        store must put the dense output-row device dim — the dim whose
        coordinate is the index's iteration symbol ``row_sym`` (e.g. c0) — at
        position 0, matching the gather result's shape without a transpose.

        Unlike ``_symbol_first_permutation`` (which takes the *first* bare
        symbol — wrong when an outer dim such as c1 precedes the row axis, as in
        a >=3D source), this targets ``row_sym`` specifically.  Falls back to
        ``_symbol_first_permutation`` when ``row_sym`` is unknown or absent.
        """
        if row_sym is not None:
            for k, coord in enumerate(device_coords):
                if coord == row_sym:
                    if k == 0:
                        return list(range(len(device_coords)))
                    return [k] + [i for i in range(len(device_coords)) if i != k]
        return self._symbol_first_permutation(device_coords)

    def _emit_symbol_first_tensor_descriptor(
        self,
        name: str,
        var: str,
        dep,
        layout: "FixedTiledLayout",
        row_sym: Optional[sympy.Symbol] = None,
    ) -> tuple[str, list[str], list]:
        """Emit tl.make_tensor_descriptor with a chosen device dim permuted first.

        The descriptor dimensions are reordered so a chosen device dim leads at
        position 0.  Used where that dim must be outermost:

        - matmul (``row_sym=None``): the plain-symbol (tiling) dim — M for A/C,
          K for B — so the loaded 3D block reshapes to 2D for tl.dot without a
          transpose (``_symbol_first_permutation``).
        - gather output store (``row_sym`` set): the output-row dim whose
          coordinate is the index's iteration symbol, so the row-first gather
          result stores with matching shape (``_gather_output_permutation``).

        Returns (desc_var, offset_var_names, permuted_block_shape).
        """
        it_space = iteration_space(self.current_node)
        device_size = [int(s) for s in layout.device_layout.device_size]

        dep_index = sympy_subs(dep.index, V.graph.sizevars.precomputed_replacements)
        dep_index = concretize_index(dep_index, set(it_space.keys()))

        device_coords = compute_coordinates(
            device_size,
            layout.device_layout.stride_map,
            it_space,
            dep_index,
        )

        if row_sym is not None:
            perm = self._gather_output_permutation(device_coords, row_sym)
        else:
            # Matmul operand: place the sticked matrix dim's (outer-stick,
            # within-stick) pair adjacent and innermost so dot() collapses them
            # into the matrix dim, with the batch dim (bmm) kept leading.
            # Anchoring on the stick pair (not a bare row symbol) stays correct
            # when M == 1 collapses the row coordinate to a constant.
            perm = self._matmul_operand_permutation(device_coords, self._batch_symbol())

        phys_strides = self._row_major_strides(device_size)
        phys_block_shape = self._device_block_shape(device_size, device_coords)

        perm_size = [device_size[p] for p in perm]
        perm_strides = [phys_strides[p] for p in perm]
        perm_block_shape = [phys_block_shape[p] for p in perm]
        perm_coords = [device_coords[p] for p in perm]

        offset_var_names = self._emit_scalar_offsets(name, perm_coords)

        f = self.index_to_str
        desc_line = (
            f"tl.make_tensor_descriptor("
            f"{var}, "
            f"shape={f(perm_size)}, "
            f"strides={f(perm_strides)}, "
            f"block_shape={f(perm_block_shape)})"
        )

        existing = self.cse.try_get(desc_line)
        if existing:
            return str(existing), offset_var_names, perm_block_shape

        block_ptr_id = next(self.block_ptr_id)
        desc_name = f"desc_{block_ptr_id}"
        named_var = self.cse.namedvar(desc_name, dtype=torch.uint64, shape=[])
        self.cse.put(desc_line, named_var)
        self.prologue.writeline(DeferredLine(name, f"{desc_name} = {desc_line}"))
        logger.debug(
            "SpyreTritonKernel: symbol-first desc %s = %s (perm=%s, row_sym=%s)",
            desc_name,
            desc_line,
            perm,
            row_sym,
        )
        return desc_name, offset_var_names, perm_block_shape

    def _gather_indirect_dim(
        self, dep, layout: "FixedTiledLayout"
    ) -> tuple[Optional[int], list]:
        """Detect an indirect (gather) load.

        Returns ``(k, device_coords)`` where ``k`` is the device dimension whose
        coordinate references an indirect ``SymT.TMP`` symbol (the loaded index,
        i.e. the gathered axis), or ``(None, device_coords)`` for a plain load.
        """
        it_space = iteration_space(self.current_node)
        device_size = [int(s) for s in layout.device_layout.device_size]
        dep_index = sympy_subs(dep.index, V.graph.sizevars.precomputed_replacements)
        dep_index = concretize_index(dep_index, set(it_space.keys()))
        device_coords = compute_coordinates(
            device_size,
            layout.device_layout.stride_map,
            it_space,
            dep_index,
        )
        for k, coord in enumerate(device_coords):
            if any(symbol_is_type(s, SymT.TMP) for s in coord.free_symbols):
                return k, device_coords
        return None, device_coords

    def _emit_gather_load(
        self,
        name: str,
        var: str,
        dep,
        layout: "FixedTiledLayout",
        k_star: int,
        device_coords: list,
    ) -> CSEVariable:
        """Emit a gather for an indirect value load.

        Uses the index buffer's multi-D device-layout load directly as
        ``x_offsets`` (no flatten — preserves layout), permutes the indirect
        axis to descriptor dim 0 via permuted strides, and emits
        ``desc.gather(x_offsets, y_offset)``.
        """
        self._is_gather = True
        indirect_syms = {
            s
            for coord in device_coords
            for s in coord.free_symbols
            if symbol_is_type(s, SymT.TMP)
        }
        if len(indirect_syms) != 1:
            raise NotImplementedError(
                "gather supports exactly one indirect index symbol, got "
                f"{sorted(map(str, indirect_syms))}"
            )
        indirect_sym = next(iter(indirect_syms))

        x_offsets, idx_shape, num_rows = self._emit_index_xoffsets(indirect_sym)
        desc_var, y_offset, block_shape = self._emit_gather_descriptor(
            name, var, dep, layout, k_star, device_coords
        )
        return self._emit_descriptor_gather(
            name, desc_var, x_offsets, y_offset, block_shape, idx_shape, num_rows
        )

    def _emit_index_xoffsets(self, indirect_sym: sympy.Symbol) -> tuple[str, list, int]:
        """Resolve ``x_offsets`` to the index buffer's multi-D device load.

        The indirect ``SymT.TMP`` symbol's name *is* the CSE variable produced by
        the upstream index descriptor load (emitted earlier in ``self.loads`` by
        the normal ``_emit_tensor_descriptor`` path, because the index load runs
        before the value load).  We use that multi-D load **directly** as
        ``x_offsets`` — no flatten, so the index's device layout is preserved
        into the gather.  This requires the relaxed Triton verifier
        that accepts a >1D ``x_offsets`` on the Spyre target.

        Returns ``(x_offsets_var_name, idx_block_shape, num_rows)`` where
        ``idx_block_shape`` is the per-core device block of the index load and
        ``num_rows`` is its element count (the number of gathered rows).
        """
        it_space = iteration_space(self.current_node)
        # The index buffer is the lone int32 FixedTiledLayout read dep (the
        # current implementation supports a single 1-D index tensor only).
        idx_dep = None
        idx_layout = None
        for d in self.current_node.read_writes.reads:
            b = V.graph.get_buffer(d.name)
            if b is None:
                continue
            lay = b.get_layout()
            if isinstance(lay, FixedTiledLayout) and (
                V.graph.get_dtype(d.name) == torch.int32
            ):
                idx_dep, idx_layout = d, lay
                break
        if idx_dep is None or idx_layout is None:
            raise NotImplementedError(
                "gather: could not locate the int32 index buffer read dep"
            )

        idx_size = [int(s) for s in idx_layout.device_layout.device_size]
        idx_index = sympy_subs(idx_dep.index, V.graph.sizevars.precomputed_replacements)
        idx_index = concretize_index(idx_index, set(it_space.keys()))
        idx_coords = compute_coordinates(
            idx_size, idx_layout.device_layout.stride_map, it_space, idx_index
        )

        # The gathered output-row axis is the index dep's iteration symbol (e.g.
        # c0).  store() permutes the output device dim with this coordinate to
        # dim 0 so the row-first gather result stores with matching shape.
        row_syms = idx_index.free_symbols & set(it_space.keys())
        self._gather_row_sym = next(iter(row_syms)) if len(row_syms) == 1 else None
        # Per-core device block of the index load == the x_offsets tensor shape
        # (the upstream _emit_tensor_descriptor used the same _device_block_shape).
        idx_block = self._device_block_shape(idx_size, idx_coords)
        num_rows = 1
        for b in idx_block:
            num_rows *= int(b)
        if num_rows < 8:
            raise NotImplementedError(
                f"gather: x_offsets must have >= 8 rows, got {num_rows}"
            )

        # x_offsets is the upstream multi-D index load (its CSE var name is the
        # indirect symbol's name); no extra descriptor or load is emitted.
        x_offsets = str(indirect_sym)
        logger.debug(
            "SpyreTritonKernel: gather x_offsets=%s (idx_block=%s, num_rows=%d)",
            x_offsets,
            idx_block,
            num_rows,
        )
        return x_offsets, idx_block, num_rows

    def _emit_gather_descriptor(
        self,
        name: str,
        var: str,
        dep,
        layout: "FixedTiledLayout",
        k_star: int,
        device_coords: list,
    ) -> tuple[str, str, list]:
        """Emit the value descriptor for a gather.

        The indirect axis (device dim ``k_star``) is permuted to dim 0 and the
        true physical layout is expressed via permuted strides;
        ``block_shape[0]`` is forced to 1.  Only dim 1 may carry a runtime
        scalar offset (``y_offset``); dims >= 2 read their full block extent.

        Returns ``(desc_var, y_offset_str, permuted_block_shape)``.
        """
        device_size = [int(s) for s in layout.device_layout.device_size]
        rank = len(device_size)
        perm = [k_star] + [i for i in range(rank) if i != k_star]

        phys_strides = self._row_major_strides(device_size)
        phys_block_shape = self._device_block_shape(device_size, device_coords)

        perm_size = [device_size[p] for p in perm]
        perm_strides = [phys_strides[p] for p in perm]
        perm_block_shape = [phys_block_shape[p] for p in perm]
        perm_coords = [device_coords[p] for p in perm]
        perm_block_shape[0] = 1  # descriptor block must have exactly 1 row

        # Representability: only dim 1 carries an offset; dims >= 2 read the
        # full block extent.  A residual indirect symbol outside dim 0 would
        # require a second indirect axis — not expressible as one gather.
        for c in perm_coords[1:]:
            if any(symbol_is_type(s, SymT.TMP) for s in c.free_symbols):
                raise NotImplementedError(
                    "gather: more than one indirect axis is not expressible as "
                    "a single desc.gather"
                )

        y_offset = self._emit_gather_y_offset(name, perm_coords)

        f = self.index_to_str
        desc_line = (
            f"tl.make_tensor_descriptor("
            f"{var}, "
            f"shape={f(perm_size)}, "
            f"strides={f(perm_strides)}, "
            f"block_shape={f(perm_block_shape)})"
        )

        existing = self.cse.try_get(desc_line)
        if existing:
            return str(existing), y_offset, perm_block_shape

        block_ptr_id = next(self.block_ptr_id)
        desc_name = f"desc_{block_ptr_id}"
        named_var = self.cse.namedvar(desc_name, dtype=torch.uint64, shape=[])
        self.cse.put(desc_line, named_var)
        self.prologue.writeline(DeferredLine(name, f"{desc_name} = {desc_line}"))
        logger.debug(
            "SpyreTritonKernel: gather desc %s = %s (perm=%s, y_offset=%s)",
            desc_name,
            desc_line,
            perm,
            y_offset,
        )
        return desc_name, y_offset, perm_block_shape

    def _emit_gather_y_offset(self, name: str, perm_coords: list) -> str:
        """Emit the single direct scalar offset (dim 1) for a gather.

        dim 0 (indirect) is addressed by ``x_offsets`` — no scalar offset; dims
        >= 2 read the full block extent.  The Triton->Logical section (``c0 =
        ...``, ``c1 = ...``) is already emitted by the index load's
        ``_emit_scalar_offsets`` call, so those symbols are in scope here.
        """
        if len(perm_coords) < 2:
            return "0"
        coord = perm_coords[1]
        if coord == 0:
            return "0"
        fixed = _normalize_floor_div(coord)
        self.prologue.writeline(
            DeferredLine(name, f"y_off = {self.index_to_str(fixed)}")
        )
        return "y_off"

    def _emit_descriptor_gather(
        self,
        name: str,
        desc_var: str,
        x_offsets: str,
        y_offset: str,
        block_shape: list,
        idx_shape: list,
        num_rows: int,
    ) -> CSEVariable:
        """Emit ``val = desc.gather(x_offsets, y_offset)`` into self.loads.

        With a multi-D ``x_offsets`` of shape ``idx_shape`` the gather result is
        ``[*idx_shape, *block_shape[1:]]`` (the index dims lead; trailing dims
        read at full block extent).  The index dims are then collapsed via
        ``tl.reshape`` to the output's single row dim so the store path receives
        the expected ``[num_rows, *block_shape[1:]]`` block.  Row-major reshape
        maps index element ``[i0, i1, ...]`` to row ``flatten(i0, i1, ...)``,
        which matches the output-row order.
        """
        gather_line = f"{desc_var}.gather({x_offsets}, {y_offset})"
        dtype = V.graph.get_dtype(name)
        gather_shape = (
            *(str(s) for s in idx_shape),
            *(str(b) for b in block_shape[1:]),
        )
        result_var = self.cse.generate(
            self.loads, gather_line, dtype=dtype, shape=gather_shape
        )

        # Collapse multi-D index dims to the output's single row dim.
        if list(idx_shape) != [num_rows]:
            out_shape = [num_rows, *block_shape[1:]]
            result_var = self.cse.generate(
                self.loads,
                f"tl.reshape({result_var}, {out_shape})",
                dtype=dtype,
                shape=tuple(str(s) for s in out_shape),
            )

        if not self.inside_reduction:
            self.outside_loop_vars.add(result_var)
        logger.debug("SpyreTritonKernel: gather %s -> %s", name, gather_line)
        return result_var

    def _emit_descriptor_load(
        self,
        name: str,
        desc_var: str,
        offset_var_names: list[str],
        block_shape: list,
        layout: "FixedTiledLayout",
    ) -> CSEVariable:
        """Emit desc.load([dim0, dim1, ...]) and return the result CSE variable."""
        offset_str = ", ".join(offset_var_names)
        load_line = f"{desc_var}.load([{offset_str}])"

        dtype = V.graph.get_dtype(name)
        # Shape must be a tuple of strings/sympy exprs for TritonCSEVariable.
        shape = tuple(str(s) for s in block_shape)
        result_var = self.cse.generate(self.loads, load_line, dtype=dtype, shape=shape)
        if not self.inside_reduction:
            self.outside_loop_vars.add(result_var)
        logger.debug("SpyreTritonKernel: load %s -> %s", name, load_line)
        return result_var

    def _emit_descriptor_store(
        self,
        name: str,
        desc_var: str,
        offset_var_names: list[str],
        layout: "FixedTiledLayout",
        value: CSEVariable,
    ) -> None:
        """Emit desc.store([dim0, dim1, ...], value)."""
        offset_str = ", ".join(offset_var_names)
        store_line = f"{desc_var}.store([{offset_str}], {value})"

        self.stores.writeline(DeferredLine(name, store_line))
        if not self.inside_reduction:
            self.outside_loop_vars.add(value)
        logger.debug("SpyreTritonKernel: store %s -> %s", name, store_line)

    def _emit_split_logical_offsets(self, name: str) -> None:
        """Emit Level-2 logical offsets honoring the device-space work division.

        Each split spatial symbol's per-program tile base is taken from
        ``tl.program_id(0)``: the split dims are decomposed from the flat
        program id (outermost dim — largest range-tree divisor — varies
        slowest), unsplit spatial dims start at base 0 (the descriptor
        ``block_shape`` covers their full range), and reduction symbols keep
        their range-tree offset (``r0_offset``).

        This mirrors the SDSC device-space split (``spyre_div``, returned by
        ``_compute_core_division``) rather than the flat row-major XBLOCK cut:
        each program owns one ``range_s // div_s`` slice of every split dim.
        The total tile size (XBLOCK) is unchanged — only the per-program tile
        *shape* differs (e.g. for ``{c1:32}`` the tile is full-c0 × 1-c1 instead
        of the flat ``{c0:16, c1:2}`` cut's 1-c0 × 16-c1).
        """
        it_space = iteration_space(self.current_node)
        f = self.index_to_str

        # Partition symbols, preserving the range-tree (outer -> inner) order.
        spatial: list[tuple[sympy.Symbol, sympy.Symbol]] = []
        reduction: list[tuple[sympy.Symbol, sympy.Symbol]] = []
        for triton_sym, opspec_sym in self._triton_to_opspec.items():
            entry = self.range_tree_nodes[triton_sym]
            if entry.prefix.startswith("r"):
                reduction.append((triton_sym, opspec_sym))
            else:
                spatial.append((triton_sym, opspec_sym))

        # Split dims, outermost (largest range-tree divisor) first.
        split = [(ts, os) for ts, os in spatial if self._core_division.get(os, 1) > 1]
        split.sort(
            key=lambda p: V.graph.sizevars.size_hint(
                self.range_tree_nodes[p[0]].divisor
            ),
            reverse=True,
        )
        total_split_cores = 1
        for _, opspec_sym in split:
            total_split_cores *= self._core_division[opspec_sym]

        # program_id(0) -> per-split-dim tile index -> tile base offset.
        # With the innermost split dim varying fastest:
        #   idx_s = (program_id(0) // inner_cores) % div_s
        #   base  = idx_s * (range_s // div_s)
        # The "// inner_cores" is dropped for the innermost dim and the
        # "% div_s" for the outermost (it is then the identity over [0, ncores)).
        inner_cores = 1
        bases: dict[sympy.Symbol, str] = {}
        for _, opspec_sym in reversed(split):  # innermost first
            div = self._core_division[opspec_sym]
            s_range = V.graph.sizevars.size_hint(it_space[opspec_sym])
            extent = max(1, s_range // div)
            idx_expr = "tl.program_id(0)"
            if inner_cores > 1:
                idx_expr = f"({idx_expr} // {inner_cores})"
            if inner_cores * div != total_split_cores:
                idx_expr = f"({idx_expr} % {div})"
            bases[opspec_sym] = idx_expr if extent == 1 else f"({idx_expr}) * {extent}"
            inner_cores *= div

        # Spatial symbols in range-tree order: split dim -> program-id base,
        # unsplit dim -> 0 (the descriptor block covers its full range).
        for _, opspec_sym in spatial:
            var_name = str(opspec_sym)
            expr = bases.get(opspec_sym, "0")
            self.prologue.writeline(DeferredLine(name, f"{var_name} = {expr}"))
            self._logical_offset_vars[opspec_sym] = var_name

        # Reduction symbols keep their range-tree offset (r0_offset).
        for triton_sym, opspec_sym in reduction:
            entry = self.range_tree_nodes[triton_sym]
            root = entry.root
            xoffset = TritonSymbols.block_offsets[root.symt]
            index_sym = root.index_sym()
            scalar_expr = sympy_subs(entry.expr, {index_sym: xoffset})
            var_name = str(opspec_sym)
            self.prologue.writeline(
                DeferredLine(name, f"{var_name} = {f(scalar_expr)}")
            )
            self._logical_offset_vars[opspec_sym] = var_name

    def _emit_scalar_offsets(self, name: str, device_coords: list) -> list[str]:
        """Emit named scalar offset variables into prologue; return Level-3 names.

        Emits two comment sections (each at most once per kernel):

            # Triton -> Logical layouts
            c0 = xoffset // N   # one line per OpSpec symbol
            c1 = xoffset % N
            # Logical layouts -> Device layouts
            dim0 = c1 // 64     # one line per device dimension
            dim1 = c0
            dim2 = c1 % 64

        Results are cached so identical device_coords share the same variable
        names without re-emitting.  When two buffers have different layouts, a
        second group is emitted with names dim_1_0, dim_1_1, etc.
        """
        coords_key = tuple(str(c) for c in device_coords)
        if coords_key in self._device_offset_vars:
            return self._device_offset_vars[coords_key]

        f = self.index_to_str

        # --- Triton -> Logical layouts ---
        # Emit once: one assignment per OpSpec symbol (c0, c1, ...).
        if not self._logical_offset_vars:
            self.prologue.writeline("# Triton -> Logical layouts")
            # M5: outside native matmul (own 2D grid) and the tiling-loop path
            # (flat per-core tiling set up in the scheduler), drive the split
            # dim(s) from program_id so the per-symbol work division matches the
            # SDSC device-space split (spyre_div).  Otherwise fall back to the
            # flat row-major xoffset decomposition.
            if not self.is_native_matmul and self._tiling_loop_count is None:
                self._emit_split_logical_offsets(name)
            else:
                for triton_sym, opspec_sym in self._triton_to_opspec.items():
                    entry = self.range_tree_nodes[triton_sym]
                    root = entry.root
                    xoffset = TritonSymbols.block_offsets[root.symt]
                    index_sym = root.index_sym()
                    scalar_expr = sympy_subs(entry.expr, {index_sym: xoffset})
                    var_name = str(opspec_sym)  # "c0", "c1", etc.
                    self.prologue.writeline(
                        DeferredLine(name, f"{var_name} = {f(scalar_expr)}")
                    )
                    self._logical_offset_vars[opspec_sym] = var_name

        # --- Logical layouts -> Device layouts ---
        # device_coords already reference c0, c1, ... (OpSpec symbols).
        # After _normalize_floor_div they print as integer // using those names.
        #
        # In tiling-loop mode: emit to _loop_offset_code (inside the tile loop)
        # so tile_idx is in scope. Tiled dims get "+tile_idx * tile_step" added.
        # In non-tiling mode: emit to prologue as before.
        tiling = (
            self._tiling_loop_count is not None and self._loop_offset_code is not None
        )
        target_buf = self._loop_offset_code if tiling else self.prologue
        group = len(self._device_offset_vars)
        target_buf.writeline("# Logical layouts -> Device layouts")
        offset_var_names: list[str] = []
        for k, coord in enumerate(device_coords):
            fixed_coord = _normalize_floor_div(coord)
            var_name = f"dim{k}" if group == 0 else f"dim_{group}_{k}"
            if tiling:
                tiled_syms_in_coord = coord.free_symbols & set(
                    self._tiling_loop_tiled_syms
                )
                tile_offset_parts = []
                for sym in tiled_syms_in_coord:
                    step = self._tiling_tile_steps.get(sym, 0)
                    if step > 0:
                        tile_offset_parts.append(f"tile_idx * {step}")
                if tile_offset_parts:
                    offset_str = " + ".join(tile_offset_parts)
                    target_buf.writeline(
                        DeferredLine(
                            name, f"{var_name} = {f(fixed_coord)} + {offset_str}"
                        )
                    )
                else:
                    target_buf.writeline(
                        DeferredLine(name, f"{var_name} = {f(fixed_coord)}")
                    )
            else:
                target_buf.writeline(
                    DeferredLine(name, f"{var_name} = {f(fixed_coord)}")
                )
            offset_var_names.append(var_name)

        self._device_offset_vars[coords_key] = offset_var_names
        return offset_var_names

    def _dump_opspec(self, write_name: str, write_dep) -> None:
        """Dump this kernel's OpSpec to a per-kernel file in the debug directory.

        The dump matches the SDSC path's serialized form (``OpSpec`` / ``TensorArg``
        with ``sympify('...')``-wrapped exprs -- see ``spyre_kernel.codegen_kernel``)
        by building real OpSpec/TensorArg objects and reusing the same
        ``_codegen_op_spec_list`` serializer, so the two paths' op-specs are
        directly comparable.  One file per kernel (named by the scheduler node) so
        a bundle's kernels do not overwrite each other.  Lands next to
        ir_post_fusion.txt; only written when TORCH_COMPILE_DEBUG=1.
        """
        debug = getattr(V, "debug", None)
        if debug is None or not getattr(debug, "_path", None):
            return

        it_space = iteration_space(self.current_node)
        # arg_index = position in this kernel's runtime args (mirrors SDSC's
        # actuals.index(name)); -1 if the buffer is not a kernel argument.
        actuals = self.args.python_argdefs()[1]

        # Build {indirect_sym -> IndirectAccess(index_buffer_name)} so a gather's
        # value coordinates print IndirectAccess(...) instead of the raw tmpN,
        # matching the
        # SDSC op-spec.  The index buffer is the int32 FixedTiledLayout read dep;
        # the indirect symbols are the SymT.TMP atoms in the read indices.
        indirect_subs: dict = {}
        idx_name = next(
            (
                d.name
                for d in self.current_node.read_writes.reads
                if V.graph.get_dtype(d.name) == torch.int32
                and isinstance(
                    getattr(V.graph.get_buffer(d.name), "get_layout", lambda: None)(),
                    FixedTiledLayout,
                )
            ),
            None,
        )
        if idx_name is not None:
            for d in self.current_node.read_writes.reads:
                d_idx = sympy_subs(d.index, V.graph.sizevars.precomputed_replacements)
                for s in d_idx.free_symbols:
                    if symbol_is_type(s, SymT.TMP):
                        indirect_subs[s] = IndirectAccess(sympy.Symbol(idx_name))

        def _tensor_arg(name: str, dep, is_input: bool) -> TensorArg | None:
            if not isinstance(dep, MemoryDep):
                return None
            buf = V.graph.get_buffer(name)
            if buf is None:
                return None
            layout = buf.get_layout()
            if not isinstance(layout, FixedTiledLayout):
                return None
            idx = sympy_subs(dep.index, V.graph.sizevars.precomputed_replacements)
            idx = concretize_index(idx, set(it_space.keys()))
            coords = compute_coordinates(
                list(layout.device_layout.device_size),
                layout.device_layout.stride_map,
                it_space,
                idx,
                indirect_subs or None,
            )
            # Strip the redundant floor the Triton-path coordinate decomposition
            # adds around the (integer-valued) IndirectAccess on the indirect dim
            # (FloorDiv-by-1).  Genuine floors (e.g. floor(c1/64)) are untouched.
            coords = [
                c.replace(
                    lambda x: isinstance(x, sympy.floor)
                    and isinstance(x.args[0], IndirectAccess),
                    lambda x: x.args[0],
                )
                for c in coords
            ]
            return TensorArg(
                is_input=is_input,
                arg_index=actuals.index(name) if name in actuals else -1,
                device_dtype=layout.device_layout.device_dtype,
                device_size=[int(s) for s in layout.device_layout.device_size],
                device_coordinates=coords,
                allocation=dict(layout.allocation),
                per_tile_fixed=getattr(layout, "per_tile_fixed", False),
                name=self._opspec_name or self.current_node.get_name(),
            )

        args = []
        for dep in self.current_node.read_writes.reads:
            ta = _tensor_arg(dep.name, dep, is_input=True)
            if ta:
                args.append(ta)
        ta = _tensor_arg(write_name, write_dep, is_input=False)
        if ta:
            args.append(ta)

        # op name: the reduction type ('sum', ...) when this is a reduction (to
        # match the SDSC op field), else the scheduler node name.
        # Identify the opspec by the kernel name (triton_bundle_N_kernel_M for
        # bundle members, kernel_N for standalone kernels) so the dump's op field
        # and file name correlate with the kernel.  Fall back to the scheduler
        # node name only if the scheduler left _opspec_name unset.
        op_name = self._opspec_name or self.current_node.get_name()

        opspec: Any = OpSpec(
            op=op_name,
            is_reduction=self.current_node.is_reduction(),
            iteration_space={
                sym: (rng, self._core_division.get(sym, 1))
                for sym, rng in it_space.items()
            },
            args=args,
            op_info={},
            tiled_symbols=list(self._tiling_loop_tiled_syms),
        )

        # Under a tiling loop (CountedLoopSchedulerNode path) the op is wrapped in
        # a LoopSpec; mirror that so the dump matches the runtime structure.
        if self._tiling_loop_count is not None:
            opspec = LoopSpec(
                count=sympy.Integer(int(self._tiling_loop_count)),
                body=[opspec],
                tiled_symbols=list(self._tiling_loop_tiled_syms),
            )

        # Serialize exactly like the SDSC path (sympify('...')-wrapped exprs).
        def sympy_str(x: Any) -> str:
            if isinstance(x, IndirectAccess):
                return f"IndirectAccess('{x.args[0]}')"
            return "sympify('" + str(x) + "')"

        specs = [opspec]
        for s in _iter_op_specs(specs):
            simplify_op_spec(s)
        out = IndentedBuffer()
        out.writeline("[")
        with out.indent():
            _codegen_op_spec_list(specs, out, sympy_str)
        out.writeline("]")

        fname = f"opspec_{op_name}.py"
        with debug.fopen_context(fname) as f:
            f.write(out.getvalue())
        logger.debug("SpyreTritonKernel: dumped opspec to %s/%s", debug._path, fname)
        logger.debug("SpyreTritonKernel: dumped opspec to %s/opspec.json", debug._path)

    def _device_block_shape(self, device_size: list, device_coords: list) -> list:
        """Per-core block shape for tl.make_tensor_descriptor block_shape parameter.

        For each device dimension k, divides device_size[k] by the product of
        core divisors for all OpSpec symbols that appear in device_coords[k].

        The stick dimension (device_rank - 1, the innermost device dimension) is
        always 128 bytes / dtype element size (e.g. 64 for fp16/bf16).  It must
        never be divided across cores — block_shape for that dim must equal
        device_size[-1].  Work division only applies to dims 0 through
        device_rank - 2; splits are always stick-aligned.
        """
        it_space = iteration_space(self.current_node)
        last_dim = len(device_size) - 1
        loop_count = self._tiling_loop_count
        tiled_syms = (
            set(self._tiling_loop_tiled_syms) if loop_count is not None else set()
        )
        result = []
        for k, coord in enumerate(device_coords):
            # Stick dimension: always full size — never divide.
            if k == last_dim:
                result.append(int(device_size[k]))
                continue
            syms = coord.free_symbols & set(it_space.keys())
            divisor = 1
            for s in syms:
                divisor *= self._core_division.get(s, 1)
                # block_shape is used per tile-loop iteration, so also divide
                # by tile_count for tiled symbols.
                if s in tiled_syms:
                    assert loop_count is not None
                    divisor *= loop_count
            size_hint = V.graph.sizevars.size_hint(device_size[k])
            result.append(max(1, size_hint // max(1, divisor)))
        return result

    @staticmethod
    def _row_major_strides(device_size: list) -> list:
        n = len(device_size)
        strides = [1] * n
        for i in range(n - 2, -1, -1):
            strides[i] = strides[i + 1] * int(device_size[i + 1])
        return strides

    def _build_triton_opspec_map(self) -> dict[str, list[sympy.Symbol]]:
        """Map each Triton prefix to the OpSpec symbols it covers.

        Uses the structural/positional correspondence from split_and_set_ranges:
        result[group_idx][dim_idx] is the Triton symbol for the j-th dimension
        of the i-th range group, in the same order as the node's get_ranges().

        Spatial OpSpec symbols (from write dep ranges) correspond to the spatial
        range-tree group; reduction symbols correspond to the reduction group.
        This avoids coefficient matching and handles equal strides, broadcasts,
        and any pattern where the positional order is unambiguous.

        Also populates self._triton_to_opspec for use in load/store offset
        computation.
        """
        assert self.current_node is not None
        assert self._pending_split_result, "_pending_split_result not set"

        node = self.current_node

        # Spatial OpSpec symbols: write dep ranges in insertion order.
        write_dep = next(iter(node.read_writes.writes))
        spatial_syms = list(write_dep.ranges.keys())

        # Reduction OpSpec symbols: in the full iteration space but not in write dep.
        it_space = iteration_space(node)
        spatial_set = set(spatial_syms)
        reduction_syms = [s for s in it_space if s not in spatial_set]

        opspec_groups = [spatial_syms]
        if reduction_syms:
            opspec_groups.append(reduction_syms)

        mapping: dict[str, list[sympy.Symbol]] = {}
        self._triton_to_opspec.clear()
        for triton_group, opspec_group in zip(
            self._pending_split_result, opspec_groups
        ):
            for triton_expr, opspec_sym in zip(triton_group, opspec_group):
                if not isinstance(triton_expr, sympy.Symbol):
                    raise RuntimeError(
                        f"Expected a Triton symbol, got compound expression "
                        f"{triton_expr!r}. Fusion across incompatible tilings is "
                        "not supported."
                    )
                entry = self.range_tree_nodes.get(triton_expr)
                if entry is None:
                    raise RuntimeError(
                        f"Triton symbol {triton_expr!r} not found in range_tree_nodes"
                    )
                mapping.setdefault(entry.prefix, []).append(opspec_sym)
                self._triton_to_opspec[triton_expr] = opspec_sym

        logger.debug("triton_opspec_map=%s", mapping)
        logger.debug("triton_to_opspec=%s", self._triton_to_opspec)
        return mapping

    def _compute_core_division(self) -> dict[sympy.Symbol, int]:
        """Compute how many cores divide each OpSpec symbol's dimension.

        ``apply_splits_from_index_coeff`` reflects Spyre's *device-space* work
        division and may assign cores to a different logical symbol than the
        one the flat Triton xoffset actually splits.  For a 3D tensor
        [128, 256, 512] Spyre may split c1 (the device dim with 256 elements)
        while the flat Triton index splits c0 (highest logical stride) —
        producing wrong block_shape and descriptor offsets.

        Instead, derive the per-symbol division from the Triton range tree:
        each ``IterationRangesEntry`` carries a ``divisor`` equal to the
        product of all *inner* (lower-stride) symbol ranges.  The number of
        elements of symbol s that fall in one XBLOCK-wide tile is
        ``min(s_range, XBLOCK // entry.divisor)``, and the core divisor is
        ``s_range // s_per_core``.

        XBLOCK = total_size // n_cores.  n_cores is taken as the product of
        the Spyre work-division values (always correct in total even when the
        per-symbol assignment differs).
        """
        assert self.current_node is not None
        assert self._triton_to_opspec, "_build_triton_opspec_map must run first"

        it_space = iteration_space(self.current_node)
        ir_node = self.current_node.node
        if not hasattr(ir_node, "op_it_space_splits"):
            return {}

        write_index = next(iter(self.current_node.read_writes.writes)).index
        read_index = next(iter(self.current_node.read_writes.reads)).index
        spyre_div = apply_splits_from_index_coeff(
            ir_node.op_it_space_splits,
            write_index,
            read_index,
            it_space,
        )

        # For native matmul, Triton uses an independent 2D grid (program_id(1)
        # for y=M, program_id(0) for x=N) — the range-tree re-derivation below
        # assumes a flat 1D xblock layout and produces wrong results.
        # apply_splits_from_index_coeff already returns the correct per-symbol
        # division directly from op_it_space_splits, so return it as-is.
        if self.is_native_matmul:
            return spyre_div

        # M5: honor the SDSC device-space split directly.  The Level-2 logical
        # offsets (see _emit_split_logical_offsets) drive the split dim(s) from
        # program_id and leave unsplit dims at base 0 (the descriptor block
        # covers their full range), so the per-symbol division must equal
        # spyre_div.  This aligns the Triton and SDSC iteration-space dumps and
        # makes the per-core tile shape match SDSC (same XBLOCK, transposed
        # tile).  The tiling-loop path (CountedLoopSchedulerNode) keeps the flat
        # row-major re-derivation below, since its Level-2 offsets are still the
        # flat xoffset decomposition (per_core tiling set up in the scheduler).
        if self._tiling_loop_count is None:
            return spyre_div

        # Total core count is always correct even when the per-symbol
        # assignment is wrong (the product is always n_cores = 32).
        n_cores = 1
        for v in spyre_div.values():
            n_cores *= v
        if n_cores <= 1:
            return spyre_div

        # For reductions, xblock is based on SPATIAL dimensions only.
        # Including the reduction range in total_size causes cores to be
        # under-divided (core_div collapses to 1 for all spatial symbols).
        spatial_total = 1
        for prefix, syms in self.triton_opspec_map.items():
            if prefix.startswith("r"):  # reduction prefix (r0_, r1_, ...)
                continue
            for sym in syms:
                if sym in it_space:
                    spatial_total *= V.graph.sizevars.size_hint(it_space[sym])
        xblock = spatial_total // n_cores

        # Re-derive per-symbol core division from Triton range tree entries.
        # entry.divisor is the number of flat indices per unit of this symbol
        # (the "inner product" of all ranges that are inside / lower-stride).
        # Elements of symbol s per core = min(s_range, xblock // divisor).
        # Reduction symbols are not divided across cores: core_div = 1.
        result: dict[sympy.Symbol, int] = {}
        for triton_sym, opspec_sym in self._triton_to_opspec.items():
            entry = self.range_tree_nodes[triton_sym]
            if entry.prefix.startswith("r"):  # reduction symbol
                result[opspec_sym] = 1
                continue
            s_range = V.graph.sizevars.size_hint(it_space[opspec_sym])
            inner = V.graph.sizevars.size_hint(entry.divisor)
            if inner > 0 and xblock > 0:
                s_per_core = min(s_range, xblock // inner)
                result[opspec_sym] = max(1, s_range // max(1, s_per_core))
            else:
                result[opspec_sym] = 1
        return result

    def _get_triton_block_size(self) -> dict[str, int]:
        """Compute block size per Triton prefix from OpSpec core divisions.

        For each prefix, the block size is:
            product(range / core_divisor) for each OpSpec symbol mapped to it
        """
        assert self.triton_opspec_map, "_build_triton_opspec_map must run first"
        assert self.current_node is not None

        it_space = iteration_space(self.current_node)

        result: dict[str, int] = {}
        for prefix, syms in self.triton_opspec_map.items():
            if not syms:
                continue
            # R0_BLOCK / R1_BLOCK etc. are body constants in Triton reduction
            # kernels, not constexpr parameters — omit them from fixed_config.
            if prefix.startswith("r"):
                continue
            total_cores = 1
            total_size = 1
            loop_count = self._tiling_loop_count
            tiled_syms = (
                set(self._tiling_loop_tiled_syms) if loop_count is not None else set()
            )
            for sym in syms:
                total_cores *= self._core_division.get(sym, 1)
                if sym in it_space:
                    size = V.graph.sizevars.size_hint(it_space[sym])
                    # For tiled symbols, multiply by tile_count so XBLOCK
                    # covers the full per-core range (tile_count tiles × per-tile
                    # per-core range). This makes xoffset = pid * per_core_total.
                    if sym in tiled_syms:
                        assert loop_count is not None
                        size *= loop_count
                    total_size *= size
            result[prefix] = max(1, total_size // total_cores)

        logger.debug("triton_block_size=%s", result)
        return result
