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

"""OpSpec -> Triton source generator (see DESIGN-OpSpecToTriton.md).

``SpyreOpSpecTritonKernel`` is **not** a ``TritonKernel`` subclass.  It is a
``SpyreKernel`` (the SDSC frontend) that, after the op_specs are finalized,
*generates* Triton ``tl.*`` source by walking the finished ``OpSpec`` list.

The whole three-level dimension mapping that makes ``SpyreTritonKernel``
(~2555 lines, a ``TritonKernel`` subclass) large is unnecessary here: Inductor
never mints ``x/y/r0_`` symbols for us, so there is nothing to reconcile back to
the OpSpec symbols ``c0/c1/...``.  We *choose* the Triton program axes from the
OpSpec symbols by construction:

    one program axis per work-divided OpSpec symbol   (c0 -> x, c1 -> y, ...)
    per-core block extent   = device_size[k] // core_divisor
    grid                    = prod(core_divisors)
    program offsets         = mixed-radix decomposition of program_id(0)
    tensor descriptor       = make_tensor_descriptor(shape=device_size,
                                                     strides=row_major(device_size),
                                                     block_shape=per_core_shape)

This cut covers pointwise ops (``add.py``), counted loops over a pointwise body
(``add_mul_coarse.py``), and non-stick ``sum`` reductions (``sum.py`` with
``dim=0``).  A reduction whose reduced dim is the within-stick axis (``dim=1``),
spans multiple device axes, or is work-divided across cores raises
``NotImplementedError`` (needs a ``sum_stick`` primitive or the inter-core reduce
ring, respectively).  ``tl.dot`` (matmul) still raises ``NotImplementedError`` —
it is the next bring-up step.

Counted loops
-------------
A coarse-tiled node arrives as ``[LoopSpec(count, body)]``: the body's ops run
``count`` times, each iteration processing one tile of the tiled dimension
(``tiled_symbols[0]``).  We emit the body's fusion group inside a
``for loop0 in range(count):`` wrapper.  The tiled-dimension offset is threaded
by the loop variable: a full-size operand (``device_size`` on the tiled dim ==
``count * per_tile_range``) advances by ``loop0 * per_tile_range`` each
iteration, and its per-core block extent on that dim is
``device_size // count // work_division`` (one tile's worth, not the whole
tensor).  A ``per_tile_fixed`` operand (an LX scratch tile whose ``device_size``
is already one tile) does not move.  In the common case the loop-carried
intermediate is a register-threaded LX buffer (``arg_index == -1``) that never
touches HBM, so only the full-size operands get descriptors.

Fuse vs. split
--------------
``SpyreKernel`` (the SDSC frontend) builds one op_spec per single op and, because
SDSC drives a whole kernel from one grid, forces every op in a scheduler node to
share one work division.  The Triton path is not bound by that: we partition a
node's op_specs into *groups* that share an identical iteration space (same
symbols, ranges, and work divisions) and emit **one Triton kernel per group**.
Ops with matching iteration space fuse (tiles thread through registers); ops with
a different iteration space or work division are split into separate kernels,
each with its own grid and ``.run()`` call.  A value produced by one group and
consumed by another crosses the group boundary through its HBM buffer (every op
output already carries an HBM allocation + descriptor), since register threading
only holds within a single kernel.
"""

import dataclasses
import os

import sympy
from torch._inductor.codegen.triton import TritonKernel, texpr
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import V
from torch.utils._sympy.functions import FloorDiv

from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, TensorArg
from torch_spyre._inductor.spyre_kernel import SpyreKernel

logger = get_inductor_logger("opspec_triton_kernel")

# Placeholder substituted with the real kernel name in the scheduler's
# define_kernel (codegen_kernel runs before the name is assigned).
KERNEL_NAME_PLACEHOLDER = "__KERNEL_NAME__"

# Pointwise binary op name -> infix tl expression operator.
_BINARY_OPS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
}


@dataclasses.dataclass
class _KernelPlan:
    """One emitted Triton kernel: its source plus the ``.run`` call arguments.

    ``call_args`` are the caller-side buffer names (in signature order, deduped,
    with a leading ``_pool`` if the group touches pool memory) that the scheduler
    passes to ``<name>.run(...)`` for this group's kernel.
    """

    source: str
    call_args: list[str]


@dataclasses.dataclass
class _LoopCtx:
    """Loop-emission context for a ``LoopSpec`` body group.

    ``var`` is the Triton loop-variable name; ``count`` the trip count; ``tiled``
    the set of iteration-space symbols advanced by this loop (from the body ops'
    ``tiled_symbols[0]``); ``subs`` maps each tiled symbol ``s`` to
    ``s + var * per_tile_range`` for offsetting full-size operands' coordinates.
    """

    var: str
    count: int
    tiled: set
    subs: dict


def _normalize_floor_div(expr: sympy.Expr) -> sympy.Expr:
    """Replace ``sympy.floor(a / n)`` with ``FloorDiv(a, n)`` for integer floordiv.

    ``compute_coordinates`` builds coordinates with Python ``//``, which sympy
    stores as ``floor(a / n)``.  ``texpr`` renders that as a floating-point
    ``libdevice.floor(...)``, which is wrong for integer device offsets.  This
    rewrites it to the Inductor-native ``FloorDiv`` form so it prints as integer
    floor division.  (Lifted from ``spyre_triton_kernel.py`` on ``dev/triton``.)
    """

    def _rewrite(e: sympy.Expr) -> sympy.Expr:
        if isinstance(e, sympy.floor):
            inner = e.args[0]
            coeff, base = inner.as_coeff_Mul()
            if isinstance(coeff, sympy.Rational) and coeff.p > 0 and coeff.q > 1:
                return FloorDiv(_normalize_floor_div(base) * coeff.p, coeff.q)
            return FloorDiv(_normalize_floor_div(inner), 1)
        if e.args:
            new_args = [_rewrite(a) for a in e.args]
            if any(na is not oa for na, oa in zip(new_args, e.args)):
                return e.func(*new_args)
        return e

    return _rewrite(expr)


def _buf_id(arg: TensorArg) -> object:
    """Stable identity of the buffer an op arg refers to, for register threading.

    A fused-away intermediate carries ``arg_index == -1`` (the unassigned
    sentinel), so distinct intermediates collide on ``arg_index``.  The op-spec
    ``name`` is the buffer name, unique per buffer and identical whether the
    buffer appears as an input or an output, so it is the reliable key; fall back
    to ``arg_index`` only when a name is absent.
    """
    return arg.name if arg.name is not None else ("idx", arg.arg_index)


def _coord_str(coord: sympy.Expr) -> str:
    """Render a device coordinate expression as Triton scalar-index source."""
    return texpr(_normalize_floor_div(coord))


def _row_major_strides(device_size: list[int]) -> list[int]:
    """Row-major (C-contiguous) strides for a device-size list."""
    n = len(device_size)
    strides = [1] * n
    for i in range(n - 2, -1, -1):
        strides[i] = strides[i + 1] * int(device_size[i + 1])
    return strides


class SpyreOpSpecTritonKernel(SpyreKernel):
    """SpyreKernel that emits Triton source instead of the flex OpSpec literal.

    All op_spec building (``load``/``store``/``store_reduction``/
    ``create_op_spec``/``create_tensor_arg``) is inherited unchanged from
    ``SpyreKernel`` — the mature layout frontend comes for free.  Only the final
    projection to source is replaced.
    """

    # Triton loop-variable name for a counted-loop (LoopSpec) body.  Only one
    # loop nesting level is supported, so a single fixed name suffices.
    LOOP_VAR = "loop0"

    def codegen_kernels(self) -> list[_KernelPlan]:
        """Finalize the op_specs and emit one Triton kernel per fusion group.

        ``super().codegen_kernel()`` runs ``simplify_op_spec`` on each op and
        assigns ``tensor_arg.arg_index`` + ``allocation["hbm"]`` (we need
        ``arg_index`` to map each ``TensorArg`` to its kernel parameter; the flex
        literal it returns is discarded).  The finalized op_specs are then
        partitioned by iteration space and each group is projected to its own
        Triton kernel — see the module docstring's "Fuse vs. split".
        """
        super().codegen_kernel()
        plans = [
            self._emit_group(group, loop_ctx)
            for group, loop_ctx in self._partition_specs()
        ]
        for plan in plans:
            self._dump_source(plan.source)
        return plans

    # -- fusion grouping ----------------------------------------------------

    def _partition_specs(self) -> list[tuple[list[OpSpec], "_LoopCtx | None"]]:
        """Partition the op_specs into (fusion group, loop context) pairs.

        A plain node's op_specs partition into contiguous same-iteration-space
        groups, each with ``loop_ctx=None``.  A coarse-tiled node arrives as a
        single ``LoopSpec``; its body partitions the same way, but every group
        carries a ``_LoopCtx`` so ``_emit_group`` wraps it in a counted loop.
        """
        specs = self.op_specs
        if len(specs) == 1 and isinstance(specs[0], LoopSpec):
            loop = specs[0]
            groups = self._group_body(loop.body)
            if len(groups) != 1:
                raise NotImplementedError(
                    "OpSpec->Triton: counted loop whose body spans more than one "
                    "iteration space is not supported yet"
                )
            return [(groups[0], self._make_loop_ctx(loop, groups[0]))]
        return [(group, None) for group in self._group_body(specs)]

    def _group_body(self, specs: list) -> list[list[OpSpec]]:
        """Split a flat op list into maximal contiguous same-iteration-space runs.

        Ops that share an iteration space (symbols + ranges + work divisions)
        fuse into one kernel; a change of iteration space starts a new group.
        Grouping is *contiguous* so the kernels run in op order — a later op that
        reads an earlier op's (now HBM) output is always sequenced after it.
        """
        groups: list[list[OpSpec]] = []
        prev_key: object = None
        for spec in specs:
            self._validate_spec(spec)
            assert isinstance(spec, OpSpec)  # narrowed by _validate_spec
            key = self._iteration_space_key(spec)
            if not groups or key != prev_key:
                groups.append([spec])
            else:
                groups[-1].append(spec)
            prev_key = key
        if not groups:
            raise NotImplementedError("OpSpec->Triton: no ops to emit")
        return groups

    def _make_loop_ctx(self, loop: LoopSpec, group: list[OpSpec]) -> "_LoopCtx":
        """Build the loop-emission context for a ``LoopSpec`` body group.

        The tiled symbols are the innermost-level ``tiled_symbols[0]`` of the
        body ops (they agree within a fusion group); each advances by its
        per-tile iteration-space range times the loop variable.
        """
        it_space = group[0].iteration_space
        tiled: set = set()
        for spec in group:
            if spec.tiled_symbols:
                tiled.update(spec.tiled_symbols[0])
        loop_sym = sympy.Symbol(self.LOOP_VAR)
        subs = {sym: sym + loop_sym * it_space[sym][0] for sym in tiled}
        return _LoopCtx(
            var=self.LOOP_VAR,
            count=int(self._size_hint(loop.count)),
            tiled=tiled,
            subs=subs,
        )

    @staticmethod
    def _validate_spec(spec) -> None:
        """Assert ``spec`` is one of the ops this cut supports.

        A top-level ``LoopSpec`` is handled by ``_partition_specs`` before this
        runs; a ``LoopSpec`` reaching here is a *nested* loop inside a body,
        which is not supported yet.
        """
        if isinstance(spec, LoopSpec):
            raise NotImplementedError(
                "OpSpec->Triton: nested counted loops are not supported yet"
            )
        if not isinstance(spec, OpSpec):
            raise NotImplementedError(
                f"OpSpec->Triton: unexpected spec {type(spec).__name__}"
            )
        if spec.is_reduction:
            SpyreOpSpecTritonKernel._validate_reduction(spec)
            return
        if spec.op not in _BINARY_OPS:
            raise NotImplementedError(
                f"OpSpec->Triton: op '{spec.op}' not supported yet"
            )

    @staticmethod
    def _reduction_axes(in_arg: TensorArg, out_arg: TensorArg) -> tuple[set, list[int]]:
        """Reduced symbols and the input device axes that carry them.

        A reduction collapses one iteration-space symbol (e.g. ``torch.sum``'s
        reduced dim): it appears in the input's ``device_coordinates`` but not in
        the output's (the user-confirmed rule — see ``sum`` SDSC artifacts).  The
        reduced symbols are therefore ``input_free_syms - output_free_syms``; the
        axes are the input device dimensions whose coordinate references one.

        A non-stick reduction (``dim=0`` on ``(128, 256)``) puts the reduced
        symbol on exactly one input axis -> a single ``tl.sum``.  A stick-dim
        reduction (``dim=1``) spreads it across the outer-stick and within-stick
        axes (two axes) -> not yet supported (needs a ``sum_stick`` primitive).
        """
        out_syms: set = set()
        for coord in out_arg.device_coordinates:
            out_syms |= coord.free_symbols
        reduced: set = set()
        for coord in in_arg.device_coordinates:
            reduced |= coord.free_symbols - out_syms
        axes = [
            k
            for k, coord in enumerate(in_arg.device_coordinates)
            if coord.free_symbols & reduced
        ]
        return reduced, axes

    @staticmethod
    def _validate_reduction(spec: OpSpec) -> None:
        """Guard the reduction cases this cut supports; raise loudly otherwise.

        Supported: a single-input ``sum`` whose reduced symbol lands on exactly
        one non-within-stick input axis and is not work-divided across cores.
        The unsupported cases each need machinery that does not exist yet:

        - stick-dim / multi-axis reduce -> a within-stick ``sum_stick`` primitive;
        - reduction dim split across cores -> the HW inter-core reduce ring
          (``tl.inter_tile``; see ``2607-InterCoreReduction.md``).
        """
        if spec.op != "sum":
            raise NotImplementedError(
                f"OpSpec->Triton: reduction '{spec.op}' not supported yet (only sum)"
            )
        inputs = [a for a in spec.args if a.is_input]
        outputs = [a for a in spec.args if not a.is_input]
        if len(inputs) != 1 or len(outputs) != 1:
            raise NotImplementedError(
                "OpSpec->Triton: reduction must have exactly one input and output"
            )
        in_arg = inputs[0]
        reduced, axes = SpyreOpSpecTritonKernel._reduction_axes(in_arg, outputs[0])
        if len(axes) != 1:
            raise NotImplementedError(
                "OpSpec->Triton: within-stick / multi-axis reduction not supported "
                f"yet (reduced symbol spans input device axes {axes}); needs a "
                "sum_stick primitive"
            )
        if axes[0] == len(in_arg.device_coordinates) - 1:
            raise NotImplementedError(
                "OpSpec->Triton: within-stick reduction (reduced symbol on the "
                "innermost stick axis) not supported yet; needs a sum_stick "
                "primitive"
            )
        for sym in reduced:
            if int(spec.iteration_space.get(sym, (0, 1))[1]) != 1:
                raise NotImplementedError(
                    "OpSpec->Triton: reduction dim is work-divided across cores "
                    "(inter-core reduce ring not implemented; see "
                    "2607-InterCoreReduction.md). Retry with fewer SENCORES."
                )

    @staticmethod
    def _iteration_space_key(spec: OpSpec) -> tuple:
        """Hashable canonical form of ``spec.iteration_space`` for grouping.

        Two ops fuse iff this key matches: same symbols, same ranges, and same
        work divisions.  Symbols/ranges are compared by their string form so the
        key is order-independent and hashable.
        """
        return tuple(
            sorted(
                (str(sym), str(rng), int(div))
                for sym, (rng, div) in spec.iteration_space.items()
            )
        )

    # -- generator core -----------------------------------------------------

    def _emit_group(
        self, group: list[OpSpec], loop_ctx: "_LoopCtx | None"
    ) -> _KernelPlan:
        """Emit one Triton kernel (source + ``.run`` args) for a fusion group.

        ``loop_ctx`` is ``None`` for a plain group; for a counted-loop body it
        carries the trip count and per-tile offsets, so the device-dim offsets
        and ops are emitted inside a ``for loop0 in range(count):`` block while
        the program-id bases and descriptors (loop-invariant) stay above it.
        """
        # Kernel parameter names / caller buffer names, in arg_index order
        # (in_ptr0, in_ptr1, ..., out_ptr0).  TensorArg.arg_index indexes into
        # these parallel lists (assigned by super().codegen_kernel()).
        argdefs, actuals, _sig, _types = self.args.python_argdefs()
        param_names = [a.name for a in argdefs]

        # Unique *materialized* tensor args of this group, first occurrence per
        # arg_index.  A negative arg_index marks a buffer that was never made a
        # kernel parameter (a fused-away intermediate): it is threaded through
        # registers by _emit_ops, so it gets no descriptor and no signature slot.
        tensor_args: dict[int, TensorArg] = {}
        for spec in group:
            for a in spec.args:
                if (
                    isinstance(a, TensorArg)
                    and a.arg_index >= 0
                    and a.arg_index not in tensor_args
                ):
                    tensor_args[a.arg_index] = a
        used = sorted(tensor_args)

        # Every op in a group shares one iteration space (grouping invariant).
        it_space = group[0].iteration_space
        grid = 1
        for _rng, div in it_space.values():
            grid *= int(div)

        # Per-core block_shape per tensor arg, computed once and shared by the
        # descriptor emission and the reduction reshape (which must target the
        # output's block_shape).  Depends only on arg + work division + loop_ctx,
        # not on the loop variable, so it is loop-invariant.
        divisor_of = {sym: int(div) for sym, (_rng, div) in it_space.items()}
        block_of = {
            i: self._device_block_shape(tensor_args[i], divisor_of, loop_ctx)
            for i in used
        }

        # Build the body at column 0; buf.indent() re-indents it under `def`.
        # Program-id bases and descriptors are loop-invariant (they address the
        # full HBM buffer), so they stay above the loop; only the device-dim
        # offsets (which carry loop0) and the ops go inside it.
        body = IndentedBuffer()
        self._emit_logical_offsets(body, it_space)
        if loop_ctx is None:
            dims_of = self._emit_dim_vars(body, tensor_args, None)
            desc_of = self._emit_descriptors(body, tensor_args, param_names, block_of)
            self._emit_ops(body, group, desc_of, dims_of, block_of)
        else:
            desc_of = self._emit_descriptors(body, tensor_args, param_names, block_of)
            body.writeline(f"for {loop_ctx.var} in range({loop_ctx.count}):")
            with body.indent():
                dims_of = self._emit_dim_vars(body, tensor_args, loop_ctx)
                self._emit_ops(body, group, desc_of, dims_of, block_of)

        signature = ", ".join(param_names[i] for i in used)
        header = self._emit_header(grid, used)
        buf = IndentedBuffer()
        buf.splice(header)
        buf.writeline(f"def {KERNEL_NAME_PLACEHOLDER}({signature}):")
        with buf.indent():
            buf.splice(body.getvalue())

        return _KernelPlan(
            source=buf.getvalue(),
            call_args=self._group_call_args(tensor_args, used, actuals),
        )

    @staticmethod
    def _group_call_args(
        tensor_args: dict[int, TensorArg],
        used: list[int],
        actuals: list[str],
    ) -> list[str]:
        """Caller-side buffer names for this group's ``.run`` call.

        Mirrors ``SpyreKernel.call_kernel``: a leading ``_pool`` when the group
        touches pool memory, then the used arg buffers in arg_index order,
        deduplicated (an in-place op lists the same buffer as input and output).
        """
        call_args: list[str] = []
        if any("pool" in a.allocation for a in tensor_args.values()):
            call_args.append("_pool")
        seen: set[str] = set()
        for i in used:
            name = actuals[i]
            if name not in seen:
                seen.add(name)
                call_args.append(name)
        return call_args

    def _emit_logical_offsets(
        self,
        body: IndentedBuffer,
        it_space: dict[sympy.Symbol, tuple[sympy.Expr, int]],
    ) -> None:
        """Emit ``# Triton -> Logical layouts``: c0/c1/... program-id bases.

        The Spyre grid is 1D (``program_id(0)`` = flattened core index).  Each
        work-divided symbol owns one ``range // div`` slice; the flat program id
        is decomposed mixed-radix with the innermost split symbol varying
        fastest, matching the SDSC device-space split.
        """
        body.writeline("# Triton -> Logical layouts")

        # Split symbols (div > 1) in iteration order (outermost first).
        split = [(s, rng, div) for s, (rng, div) in it_space.items() if div > 1]
        total_cores = 1
        for _s, _rng, div in split:
            total_cores *= div

        bases: dict[sympy.Symbol, str] = {}
        inner_cores = 1
        for sym, rng, div in reversed(split):  # innermost first
            extent = max(1, int(self._size_hint(rng)) // div)
            idx = "tl.program_id(0)"
            if inner_cores > 1:
                idx = f"({idx} // {inner_cores})"
            if inner_cores * div != total_cores:
                idx = f"({idx} % {div})"
            bases[sym] = idx if extent == 1 else f"({idx}) * {extent}"
            inner_cores *= div

        for sym in it_space:
            body.writeline(f"{sym} = {bases.get(sym, '0')}")

    def _emit_dim_vars(
        self,
        body: IndentedBuffer,
        tensor_args: dict[int, TensorArg],
        loop_ctx: "_LoopCtx | None",
    ) -> dict[int, list[str]]:
        """Emit device-dim offset vars (``dim0``/``dim1``/...) per tensor arg.

        Returns ``dims_of`` keyed by ``arg_index``: the list of device-dim
        offset variable names.  For a counted-loop body, a full-size operand's
        tiled-dim coordinate is advanced by ``loop0 * per_tile_range`` (via
        ``loop_ctx.subs``); a ``per_tile_fixed`` operand is left untouched.
        Because the offset changes the coordinate string, full-size and
        per-tile operands that shared a layout naturally get distinct dim vars.
        """
        dims_of: dict[int, list[str]] = {}
        # Cache device-dim var names by coordinate signature so operands with an
        # identical device layout share dim0/dim1/... (as SDSC/dev-triton do).
        coords_seen: dict[tuple[str, ...], list[str]] = {}

        for arg_index in sorted(tensor_args):
            arg = tensor_args[arg_index]
            coords = list(arg.device_coordinates)
            if loop_ctx is not None and not arg.per_tile_fixed:
                coords = [c.subs(loop_ctx.subs) for c in coords]
            key = tuple(str(c) for c in coords)
            if key not in coords_seen:
                group = len(coords_seen)
                dim_names = []
                body.writeline("# Logical layouts -> Device layouts")
                for k, coord in enumerate(coords):
                    name = f"dim{k}" if group == 0 else f"dim_{group}_{k}"
                    body.writeline(f"{name} = {_coord_str(coord)}")
                    dim_names.append(name)
                coords_seen[key] = dim_names
            dims_of[arg_index] = coords_seen[key]

        return dims_of

    def _emit_descriptors(
        self,
        body: IndentedBuffer,
        tensor_args: dict[int, TensorArg],
        param_names: list[str],
        block_of: dict[int, list[int]],
    ) -> dict[int, str]:
        """Emit one ``tl.make_tensor_descriptor`` per tensor arg.

        Returns ``desc_of`` keyed by ``arg_index``.  Descriptors address the full
        HBM buffer (``shape=device_size``) so they are loop-invariant; only the
        per-core ``block_shape`` (from the shared ``block_of``) reflects the work
        division (and, in a loop, one tile's worth of the tiled dim).
        """
        desc_of: dict[int, str] = {}
        for arg_index in sorted(tensor_args):
            arg = tensor_args[arg_index]
            device_size = [int(s) for s in arg.device_size]
            strides = _row_major_strides(device_size)
            block_shape = block_of[arg_index]
            desc = f"desc_{arg_index}"
            body.writeline(
                f"{desc} = tl.make_tensor_descriptor("
                f"{param_names[arg_index]}, "
                f"shape={device_size}, "
                f"strides={strides}, "
                f"block_shape={block_shape})"
            )
            desc_of[arg_index] = desc

        return desc_of

    def _device_block_shape(
        self,
        arg: TensorArg,
        divisor_of: dict[sympy.Symbol, int],
        loop_ctx: "_LoopCtx | None",
    ) -> list[int]:
        """Per-core ``block_shape`` for ``tl.make_tensor_descriptor``.

        Divides each non-stick device dim by the product of core divisors
        (``divisor_of``, this group's iteration-space work divisions) of the
        OpSpec symbols appearing in that dim's coordinate.  The last device dim
        is the inner-stick dim: always the full ``device_size[-1]`` (64 fp16 /
        32 fp32 / 128 int8), never divided across cores.

        In a counted loop, a full-size operand's ``device_size`` on the tiled dim
        spans the whole tensor (``count`` tiles), but each iteration loads only
        one tile, so that dim is first divided by ``count``.  A ``per_tile_fixed``
        operand already holds one tile, so it is left alone.
        """
        device_size = [int(s) for s in arg.device_size]
        coords = arg.device_coordinates
        last = len(device_size) - 1
        tile_this_arg = loop_ctx is not None and not arg.per_tile_fixed

        block = []
        for k, coord in enumerate(coords):
            if k == last:
                block.append(device_size[k])
                continue
            size = device_size[k]
            if (
                tile_this_arg
                and loop_ctx is not None
                and (coord.free_symbols & loop_ctx.tiled)
            ):
                size //= loop_ctx.count
            divisor = 1
            for sym in coord.free_symbols:
                divisor *= divisor_of.get(sym, 1)
            block.append(max(1, size // max(1, divisor)))
        return block

    def _emit_ops(
        self,
        body: IndentedBuffer,
        specs: list[OpSpec],
        desc_of: dict[int, str],
        dims_of: dict[int, list[str]],
        block_of: dict[int, list[int]],
    ) -> None:
        """Emit load / compute / store for each op (pointwise or reduction).

        Values produced by an earlier op (its output buffer) are reused by a
        later op that reads the same buffer, so a fused chain threads tiles
        through registers rather than reloading.  A fused-away intermediate
        (``arg_index < 0``) is never materialized: its value stays in a register
        and is not stored to HBM.  Buffers are identified by name so distinct
        intermediates (all sharing the ``-1`` sentinel) do not alias.
        """
        tmp_counter = 0
        produced: dict[object, str] = {}  # buffer id -> tmp var holding its value

        def _fresh() -> str:
            nonlocal tmp_counter
            var = f"tmp{tmp_counter}"
            tmp_counter += 1
            return var

        def _load(arg: TensorArg) -> str:
            key = _buf_id(arg)
            if key in produced:
                return produced[key]
            # A register-threaded intermediate must have been produced earlier in
            # this kernel; only materialized args (arg_index >= 0) load from HBM.
            offsets = ", ".join(dims_of[arg.arg_index])
            var = _fresh()
            body.writeline(f"{var} = {desc_of[arg.arg_index]}.load([{offsets}])")
            produced[key] = var
            return var

        def _store(out: TensorArg, var: str) -> None:
            # Only write HBM for a materialized output; a fused-away intermediate
            # (arg_index < 0) is consumed from the register in a later op.
            if out.arg_index >= 0:
                offsets = ", ".join(dims_of[out.arg_index])
                body.writeline(f"{desc_of[out.arg_index]}.store([{offsets}], {var})")
            produced[_buf_id(out)] = var

        for spec in specs:
            inputs = [a for a in spec.args if a.is_input]
            outputs = [a for a in spec.args if not a.is_input]
            assert len(outputs) == 1, "op must have exactly one output"

            if spec.is_reduction:
                self._emit_reduction(
                    body, spec, inputs, outputs[0], _load, _fresh, _store, block_of
                )
                continue

            assert len(inputs) == 2, f"binary op '{spec.op}' needs two inputs"
            lhs = _load(inputs[0])
            rhs = _load(inputs[1])
            out_var = _fresh()
            body.writeline(f"{out_var} = {lhs} {_BINARY_OPS[spec.op]} {rhs}")
            _store(outputs[0], out_var)

    def _emit_reduction(
        self,
        body: IndentedBuffer,
        spec: OpSpec,
        inputs: list[TensorArg],
        out: TensorArg,
        _load,
        _fresh,
        _store,
        block_of: dict[int, list[int]],
    ) -> None:
        """Emit ``tl.sum`` over the reduced axis, reshaped to the output block.

        The reduced axis is the single input device axis carrying the reduced
        symbol (``_validate_reduction`` has guaranteed exactly one, non-within-
        stick, work-division 1).  ``tl.sum`` drops that axis; the surviving tile
        is then reshaped to the output's ``block_shape`` (the output layout may
        add/remove unit axes relative to the reduced input — e.g. ``dim=0`` on
        ``(128, 256)`` reduces input ``[4, 128, 64]`` on axis 1 to ``[4, 64]``,
        which reshapes to output ``[1, 4, 64]``).
        """
        in_arg = inputs[0]
        # A fused-away reduction input has no descriptor/block; not supported yet
        # (sum.py reads an HBM buffer).  Raise loudly rather than emit wrong code.
        if in_arg.arg_index < 0:
            raise NotImplementedError(
                "OpSpec->Triton: reduction with a register-threaded (fused) input "
                "not supported yet"
            )
        _reduced, axes = self._reduction_axes(in_arg, out)
        axis = axes[0]

        in_var = _load(in_arg)
        red_var = _fresh()
        body.writeline(f"{red_var} = tl.sum({in_var}, {axis})")
        out_var = red_var

        # Shape of the tile after tl.sum drops `axis`, vs. the output block shape.
        in_block = block_of[in_arg.arg_index]
        reduced_shape = [s for k, s in enumerate(in_block) if k != axis]
        out_block = block_of.get(out.arg_index) if out.arg_index >= 0 else None
        if out_block is not None and out_block != reduced_shape:
            self._check_reshape_is_order_preserving(
                in_arg, out, axis, in_block, out_block
            )
            out_var = _fresh()
            body.writeline(f"{out_var} = tl.reshape({red_var}, {out_block})")
        _store(out, out_var)

    @staticmethod
    def _check_reshape_is_order_preserving(
        in_arg: TensorArg,
        out: TensorArg,
        axis: int,
        in_block: list[int],
        out_block: list[int],
    ) -> None:
        """Raise unless reshaping the reduced tile to the output block is a no-op
        on element order (i.e. only unit axes are added/removed, no permute).

        A bare ``tl.reshape`` is correct only when the row-major enumeration of
        the surviving input coordinates equals that of the output coordinates.
        We approximate that by requiring the non-unit-*block* coordinates (in
        axis order) to match on both sides — a block-size-1 axis holds a single
        element and so does not affect ordering.  A genuine permute would need
        ``tl.permute`` and is not supported yet.
        """
        surviving = [
            str(coord)
            for k, (coord, size) in enumerate(zip(in_arg.device_coordinates, in_block))
            if k != axis and size != 1
        ]
        produced_out = [
            str(coord)
            for coord, size in zip(out.device_coordinates, out_block)
            if size != 1
        ]
        if surviving != produced_out:
            raise NotImplementedError(
                "OpSpec->Triton: reduction output layout requires a permute "
                f"({surviving} -> {produced_out}); tl.permute not supported yet"
            )

    # -- preamble -----------------------------------------------------------

    def _emit_header(self, grid: int, used: list[int]) -> str:
        """Imports + ``@triton.jit`` decorator preamble.

        A ``@triton_heuristics.fixed_config`` decorator (with real
        ``triton_meta`` restricted to this group's ``used`` args) is attempted so
        ``output_code.py`` is structurally a Spyre Triton kernel; if any metadata
        helper is unavailable it falls back to a plain ``@triton.jit`` so source
        is always emitted (execution may still fail at ``.run`` — expected for
        this source-only cut).
        """
        buf = IndentedBuffer()
        buf.splice(TritonKernel.gen_common_triton_imports())
        buf.writeline("")
        decorator = self._fixed_config_decorator(grid, used)
        if decorator is not None:
            buf.splice(decorator)
        buf.writeline("@triton.jit")
        return buf.getvalue()

    def _fixed_config_decorator(self, grid: int, used: list[int]):
        """Build a ``@triton_heuristics.fixed_config(...)`` block, or None.

        ``triton_meta`` is built from only this group's ``used`` args (a subset
        of ``python_argdefs``), so the signature matches the emitted kernel.
        """
        try:
            from torch._inductor.codegen.triton import config_of, signature_to_meta
            from torch._inductor.runtime.hints import DeviceProperties

            argdefs, _call, signature, _types = self.args.python_argdefs()
            argdefs = [argdefs[i] for i in used]
            signature = [signature[i] for i in used]
            triton_meta = {
                "signature": signature_to_meta(
                    signature, size_dtype=None, argdefs=argdefs
                ),
                "device": DeviceProperties.create(
                    V.graph.get_current_device_or_throw()
                ),
                "constants": {},
                "configs": [config_of(signature)],
                "native_matmul": False,
                "spyre_grid": [grid],
            }
            inductor_meta = {
                "grid_type": "Grid1D",
                "kernel_name": KERNEL_NAME_PLACEHOLDER,
                "mutated_arg_names": [],
                "no_x_dim": False,
            }
            buf = IndentedBuffer()
            buf.writeline("@triton_heuristics.fixed_config(")
            with buf.indent():
                buf.writeline("config={},")
                buf.writeline("filename=__file__,")
                buf.writeline(f"triton_meta={triton_meta!r},")
                buf.writeline(f"inductor_meta={inductor_meta!r},")
            buf.writeline(")")
            return buf.getvalue()
        except Exception as exc:  # pragma: no cover - fidelity best-effort
            logger.debug("OpSpec->Triton: fixed_config unavailable (%s)", exc)
            return None

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _size_hint(expr) -> int:
        """Concrete size hint for an iteration-space range expression."""
        if isinstance(expr, (int, sympy.Integer)):
            return int(expr)
        return int(V.graph.sizevars.size_hint(expr))

    def _dump_source(self, src: str) -> None:
        """Write the emitted source to ./opspec-triton-dump/ for inspection."""
        try:
            out_dir = os.path.join(os.getcwd(), "opspec-triton-dump")
            os.makedirs(out_dir, exist_ok=True)
            idx = len(os.listdir(out_dir))
            path = os.path.join(out_dir, f"opspec_kernel_{idx}.py")
            with open(path, "w") as fh:
                fh.write(src)
            logger.debug("OpSpec->Triton: dumped kernel to %s", path)
        except OSError as exc:  # pragma: no cover
            logger.debug("OpSpec->Triton: dump failed (%s)", exc)
