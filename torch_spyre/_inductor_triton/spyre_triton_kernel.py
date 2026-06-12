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
import json
from typing import Any, Optional

import sympy
import torch

from torch._inductor.codegen.common import CSEVariable, DeferredLine
from torch._inductor.codegen.triton import (
    FixedTritonConfig,
    TritonKernel,
    TritonSymbols,
    get_triton_reduction_function,
)
from torch._inductor.utils import sympy_subs
from torch._inductor.virtualized import ReductionType, StoreMode, V
from torch.utils._sympy.functions import FloorDiv, ModularIndexing

from torch_spyre._inductor.ir import FixedTiledLayout
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.pass_utils import (
    apply_splits_from_index_coeff,
    concretize_index,
    iteration_space,
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
    """

    @staticmethod
    def dot(a, b):  # type: ignore[override]
        """Reshape 3D descriptor blocks to 2D and emit tl.dot."""
        kernel = V.kernel
        a_shape = getattr(a, "shape", None)
        b_shape = getattr(b, "shape", None)

        if a_shape and len(a_shape) == 3:
            # A: [M_tile, K-sticks, K-elems] → [M_tile, K]
            m_tile = int(a_shape[0])
            k = int(a_shape[1]) * int(a_shape[2])
            a_mat = kernel.cse.generate(
                kernel.compute,
                f"tl.reshape({a}, [{m_tile}, {k}])",
                dtype=a.dtype,
                shape=(str(m_tile), str(k)),
            )
        else:
            a_mat = a

        if b_shape and len(b_shape) == 3:
            # B: [K, N-sticks, N-elems] → [K, N]
            k2 = int(b_shape[0])
            n = int(b_shape[1]) * int(b_shape[2])
            b_mat = kernel.cse.generate(
                kernel.compute,
                f"tl.reshape({b}, [{k2}, {n}])",
                dtype=b.dtype,
                shape=(str(k2), str(n)),
            )
        else:
            b_mat = b

        return f'tl.dot({a_mat}, {b_mat}, input_precision="ieee")'


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
        # Guard: dump opspec.json only once per kernel instance.
        self._opspec_dumped: bool = False

    def __enter__(self):
        super(TritonKernel, self).__enter__()
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

    def codegen_kernel(self, name=None) -> str:
        return super().codegen_kernel(name)

    def codegen_body(self):
        # Inject spyre_grid into triton_meta so async_compile.py can pass
        # the correct grid shape to SpyreOptions (and thus DistributeWork).
        # triton_meta is set on self just before codegen_body() is called,
        # and the same dict object is serialized into the generated Python —
        # mutations here ARE reflected in the @fixed_config decorator call.
        if self.triton_meta is not None and self.triton_opspec_map:
            self.triton_meta["spyre_grid"] = self._compute_spyre_grid()
        return super().codegen_body()

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
        return tuple(grid) if grid else (32,)

    def load(self, name: str, index: sympy.Expr):
        if not self.triton_opspec_map:
            return super().load(name, index)

        buf = V.graph.get_buffer(name)
        layout = buf.get_layout() if buf is not None else None
        if not isinstance(layout, FixedTiledLayout):
            return super().load(name, index)

        if "lx" in layout.allocation or "pool" in layout.allocation:
            return super().load(name, index)

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

        if self.is_native_matmul:
            desc_var, offset_var_names, block_shape = (
                self._emit_matmul_tensor_descriptor(name, var, dep, layout)
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

        if "pool" in layout.allocation:
            return super().store(name, index, value, mode)

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
            self._dump_opspec_json(name, dep)

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

        if "pool" in layout.allocation:
            return super().store_reduction(name, index, value)

        dep = next(
            (d for d in self.current_node.read_writes.writes if d.name == name),
            None,
        )
        if dep is None:
            return super().store_reduction(name, index, value)

        var = self.args.output(name)

        if not self._opspec_dumped:
            self._opspec_dumped = True
            self._dump_opspec_json(name, dep)

        if self.is_native_matmul:
            desc_var, offset_var_names, block_shape = (
                self._emit_matmul_tensor_descriptor(name, var, dep, layout)
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
    def _matmul_dim_permutation(device_coords: list) -> list:
        """Return the permutation that places the direct-symbol dim first.

        For standard Spyre 3D tensor device layouts the structure is always
        [FloorDiv_expr, plain_symbol, Mod_expr].  Matmul descriptors need the
        plain-symbol ("tiling") dimension first so that a single tl.reshape
        converts the loaded 3D block into a 2D matrix for tl.dot without any
        permute.

        Returns the permutation as a list of indices, e.g. [1, 0, 2].
        """
        for k, coord in enumerate(device_coords):
            if isinstance(coord, sympy.Symbol):
                if k == 0:
                    return list(range(len(device_coords)))
                return [k] + [i for i in range(len(device_coords)) if i != k]
        return list(range(len(device_coords)))

    def _emit_matmul_tensor_descriptor(
        self,
        name: str,
        var: str,
        dep,
        layout: "FixedTiledLayout",
    ) -> tuple[str, list[str], list]:
        """Emit tl.make_tensor_descriptor with matmul dimension reordering.

        For matmul, the descriptor dimensions are permuted so the "tiling"
        dimension (M for A/C, K for B) is first.  This allows the loaded 3D
        block to be reshaped to 2D via a single tl.reshape before tl.dot,
        without any transpose.

        The permutation is [1, 0, 2] for standard Spyre 3D layouts.

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

        perm = self._matmul_dim_permutation(device_coords)

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
            "SpyreTritonKernel: matmul desc %s = %s (perm=%s)",
            desc_name,
            desc_line,
            perm,
        )
        return desc_name, offset_var_names, perm_block_shape

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
        group = len(self._device_offset_vars)
        self.prologue.writeline("# Logical layouts -> Device layouts")
        offset_var_names: list[str] = []
        for k, coord in enumerate(device_coords):
            fixed_coord = _normalize_floor_div(coord)
            var_name = f"dim{k}" if group == 0 else f"dim_{group}_{k}"
            self.prologue.writeline(
                DeferredLine(name, f"{var_name} = {f(fixed_coord)}")
            )
            offset_var_names.append(var_name)

        self._device_offset_vars[coords_key] = offset_var_names
        return offset_var_names

    def _dump_opspec_json(self, write_name: str, write_dep) -> None:
        """Dump a debug OpSpec dict to opspec.json in the inductor debug directory.

        The file lands next to ir_post_fusion.txt; only written when
        TORCH_COMPILE_DEBUG=1 (V.debug._path is set).
        """
        debug = getattr(V, "debug", None)
        if debug is None or not getattr(debug, "_path", None):
            return

        it_space = iteration_space(self.current_node)

        def _tensor_arg_dict(name: str, dep, is_input: bool) -> dict | None:
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
            )
            return {
                "is_input": is_input,
                "name": name,
                "device_dtype": str(layout.device_layout.device_dtype),
                "device_size": [int(s) for s in layout.device_layout.device_size],
                "device_coordinates": [str(c) for c in coords],
                "allocation": {k: str(v) for k, v in layout.allocation.items()},
            }

        args = []
        for dep in self.current_node.read_writes.reads:
            ta = _tensor_arg_dict(dep.name, dep, is_input=True)
            if ta:
                args.append(ta)
        ta = _tensor_arg_dict(write_name, write_dep, is_input=False)
        if ta:
            args.append(ta)

        opspec = {
            "op": self.current_node.get_name(),
            "is_reduction": self.current_node.is_reduction(),
            "iteration_space": {
                str(sym): {
                    "range": str(rng),
                    "work_division": self._core_division.get(sym, 1),
                }
                for sym, rng in it_space.items()
            },
            "args": args,
        }

        with debug.fopen_context("opspec.json") as f:
            json.dump(opspec, f, indent=2)
        logger.debug("SpyreTritonKernel: dumped opspec to %s/opspec.json", debug._path)

    def _device_block_shape(self, device_size: list, device_coords: list) -> list:
        """Per-core block shape for tl.make_tensor_descriptor block_shape parameter.

        For each device dimension k, divides device_size[k] by the product of
        core divisors for all OpSpec symbols that appear in device_coords[k].

        The stick dimension (device_rank - 1, the innermost device dimension) is
        always 128 bytes / dtype element size (e.g. 64 for fp16/bf16).  It must
        never be divided across cores — block_shape for that dim must equal
        device_size[-1].  (See docs/source/user_guide/tensors_and_layouts.md and
        docs/source/compiler/work_division_planning.md: work division only applies
        to dims 0 through device_rank - 2; splits are always stick-aligned.)
        """
        it_space = iteration_space(self.current_node)
        last_dim = len(device_size) - 1
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
            for sym in syms:
                total_cores *= self._core_division.get(sym, 1)
                if sym in it_space:
                    total_size *= V.graph.sizevars.size_hint(it_space[sym])
            result[prefix] = max(1, total_size // total_cores)

        logger.debug("triton_block_size=%s", result)
        return result
