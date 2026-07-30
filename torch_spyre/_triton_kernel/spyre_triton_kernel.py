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
(``add_mul_coarse.py``), non-stick ``sum`` reductions (``sum.py`` with
``dim=0``), and 2D / batched matmul (``matmul.py`` / ``bmm.py``, via ``tl.dot``).
A reduction whose reduced dim is the within-stick axis (``dim=1``), spans
multiple device axes, or is work-divided across cores raises
``NotImplementedError`` (needs a ``sum_stick`` primitive or the inter-core
reduce ring, respectively).  Matmul supports a single (non-work-divided)
contraction dim and at most one batch dim; more than one batch dim, a
work-divided K, degenerate ``K == 1``, and fp8 matmul each raise
``NotImplementedError`` (see ``_validate_matmul``).

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
import torch
from torch._inductor.codegen.triton import TritonKernel, texpr
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import V
from torch.utils._sympy.functions import FloorDiv

from torch_spyre._inductor import constants
from torch_spyre._inductor.codegen.opspec_utils import (
    _buf_id,
    _check_reshape_is_order_preserving,
    _device_block_shape,
    _gather_operands,
    _is_gather_spec,
    _is_restickify_spec,
    _iteration_space_key,
    _LoopCtx,
    _matmul_operand_permutation,
    _reduction_axes,
    _restickify_operands,
    _restickify_plan,
    _row_major_strides,
    _size_hint,
)
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

# Reduction op names lowered via ``tl.dot`` (matmul) rather than ``tl.sum``.
# Only plain fp16 matmul is emitted so far; ``batchmatmulfp8`` is deferred.
_MATMUL_OPS = {constants.BATCH_MATMUL_OP}


def _consumes_registers(spec: OpSpec) -> bool:
    """Whether ``spec`` can consume a register-threaded (fused-away) operand.

    Only a pointwise binary op reads its inputs straight from the register a
    prior op left them in.  Every other op (matmul, reduction, restickify,
    gather) loads its operands through a ``tl.make_tensor_descriptor`` and needs
    per-arg ``block_shape`` / permutation metadata, so its operands must be
    HBM-resident -- see ``_group_body``, which splits the producer off.
    """
    return spec.op in _BINARY_OPS and not spec.is_reduction


# torch dtype -> Triton pointer-type string, for synthesizing the fixed_config
# signature of a materialized cross-group pool buffer (which has no entry in
# ``python_argdefs``; see ``_assign_pool_slots``).
_TORCH_PTR_TYPE = {
    torch.float16: "*fp16",
    torch.float32: "*fp32",
    torch.bfloat16: "*bf16",
    torch.int64: "*i64",
    torch.int32: "*i32",
    torch.int8: "*i8",
    torch.uint8: "*u8",
    torch.bool: "*i1",
}


@dataclasses.dataclass
class _KernelPlan:
    """One emitted Triton kernel: its source plus the ``.run`` call arguments.

    ``call_args`` are the caller-side buffer names (in signature order, deduped)
    that the scheduler passes to ``<name>.run(...)`` for this group's kernel.  A
    cross-group pool intermediate appears here as its own materialized HBM tensor
    (see ``_assign_pool_slots``); the shared ``_pool`` region is never passed.
    """

    source: str
    call_args: list[str]


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


def _arg_free_symbols(arg: TensorArg) -> set:
    """Union of free symbols over a tensor arg's device coordinates."""
    syms: set = set()
    for coord in arg.device_coordinates:
        syms |= coord.free_symbols
    return syms


def _coord_str(coord: sympy.Expr) -> str:
    """Render a device coordinate expression as Triton scalar-index source."""
    return texpr(_normalize_floor_div(coord))


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

    def create_tensor_arg(self, is_input, name, tensor, opspec_name=None):
        """Populate ``TensorArg.name`` with the buffer name for every arg.

        The base ``SpyreKernel`` leaves ``name`` unset outside the gather path,
        so ``_buf_id`` falls back to the ``arg_index`` sentinel and distinct
        fused-away intermediates (all ``arg_index == -1``) collide.  The Triton
        projection needs a stable per-buffer identity -- to thread register
        values without aliasing, and to identify a cross-group pool buffer by
        name (see ``_assign_pool_slots``) -- so default ``opspec_name`` to the
        buffer name.  This is a projection detail local to the Triton path; the
        SDSC frontend is untouched.
        """
        return super().create_tensor_arg(
            is_input, name, tensor, opspec_name=opspec_name or name
        )

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
        partitions = self._partition_specs()
        self._assign_pool_slots([group for group, _ in partitions])
        plans = [self._emit_group(group, loop_ctx) for group, loop_ctx in partitions]
        for plan in plans:
            self._dump_source(plan.source)
        return plans

    # -- cross-group pool buffers -------------------------------------------

    def _assign_pool_slots(self, groups: list[list[OpSpec]]) -> None:
        """Assign a kernel-parameter slot to each cross-group pool buffer.

        A pool buffer (``arg_index < 0``, ``allocation={'pool': ...}``) that is
        referenced in more than one fusion group is a fused-away intermediate
        that must actually cross the kernel boundary through HBM (e.g.
        ``sum_sum``'s ``buf0``, produced by group 0's reduction and consumed by
        group 1's).  The OpSpec keeps it as a pool buffer -- identical to the
        SDSC contract -- so the *generator* materializes it: it gets a synthetic
        parameter slot appended after the real ``python_argdefs`` args (its
        OpSpec ``arg_index`` stays ``< 0``), threaded through every group that
        uses it, plus its own HBM allocation emitted into the wrapper by the
        scheduler.  A pool buffer confined to a single group is register-threaded
        as before (``add_mul_coarse``'s loop-carried intermediate), so it gets no
        slot here.
        """
        self._pool_slot_of: dict[str, int] = {}
        self._pool_param_of: dict[int, str] = {}
        self.materialized_pool_names: list[str] = []

        groups_of: dict[str, set[int]] = {}
        for gi, group in enumerate(groups):
            for spec in group:
                for a in spec.args:
                    if (
                        isinstance(a, TensorArg)
                        and a.arg_index < 0
                        and "pool" in a.allocation
                    ):
                        groups_of.setdefault(str(_buf_id(a)), set()).add(gi)

        base = len(self.args.python_argdefs()[0])
        for bid, gids in groups_of.items():
            if len(gids) <= 1:
                continue  # register-threaded within a single group
            slot = base + len(self._pool_slot_of)
            self._pool_slot_of[bid] = slot
            self._pool_param_of[slot] = bid
            self.materialized_pool_names.append(bid)
            # Own the buffer entirely: keep Inductor from allocating/freeing it
            # (it is a pool buffer, so it is kernel-local to Inductor anyway); the
            # scheduler emits its allocation and threads it through both kernels.
            self.removed_buffers.add(bid)

    def _slot(self, arg: TensorArg) -> int:
        """Kernel-parameter slot for a tensor arg, or ``-1`` if register-threaded.

        A real arg uses its assigned ``arg_index``; a materialized cross-group
        pool buffer uses its synthetic slot (see ``_assign_pool_slots``);
        anything else (a within-group fused-away intermediate) has no slot and is
        threaded through registers.
        """
        if arg.arg_index >= 0:
            return arg.arg_index
        return self._pool_slot_of.get(str(_buf_id(arg)), -1)

    def _pool_ptr_type(self, slot: int) -> str:
        """Triton pointer-type string for a materialized pool buffer's slot."""
        dtype = V.graph.get_buffer(self._pool_param_of[slot]).get_dtype()
        return _TORCH_PTR_TYPE.get(dtype, "*fp16")

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

        A second boundary is forced independently of the iteration space: an op
        that loads its operands through descriptors (matmul, reduction,
        restickify, gather -- anything but a pointwise binary op) cannot consume
        a register-threaded operand, so it must not fuse with an op *within the
        same group* that produces one of its inputs.  Splitting there pushes the
        producer's output across the kernel boundary (it becomes a materialized
        HBM pool buffer, see ``_assign_pool_slots``) so the consumer reads it via
        a descriptor.  ``linear.py`` is the canonical case: a restickify feeds
        the matmul weight, and both share ``{c0:(512,1), c1:(512,1)}``.
        """
        groups: list[list[OpSpec]] = []
        prev_key: object = None
        produced: set[object] = set()  # buffer ids produced in the current group
        for spec in specs:
            self._validate_spec(spec)
            assert isinstance(spec, OpSpec)  # narrowed by _validate_spec
            key = _iteration_space_key(spec)
            reads_produced = any(
                a.is_input and _buf_id(a) in produced for a in spec.args
            )
            needs_split = reads_produced and not _consumes_registers(spec)
            if not groups or key != prev_key or needs_split:
                groups.append([spec])
                produced = set()
            else:
                groups[-1].append(spec)
            for a in spec.args:
                if not a.is_input:
                    produced.add(_buf_id(a))
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
            count=int(_size_hint(loop.count)),
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
            if spec.op in _MATMUL_OPS:
                SpyreOpSpecTritonKernel._validate_matmul(spec)
            else:
                SpyreOpSpecTritonKernel._validate_reduction(spec)
            return
        if _is_gather_spec(spec):
            # Structural validation; the layout guards (>= 8 rows, no
            # work-divided trailing axis) run in _emit_group where the per-core
            # block shapes and work divisions are known.
            _gather_operands(spec)
            return
        if _is_restickify_spec(spec):
            # Cross-stick transpose copy; the reshape/permute/reshape plan (which
            # needs the per-core block shapes) is built in _emit_restickify.
            _restickify_operands(spec)
            return
        if spec.op not in _BINARY_OPS:
            raise NotImplementedError(
                f"OpSpec->Triton: op '{spec.op}' not supported yet"
            )

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
        reduced, axes = _reduction_axes(in_arg, outputs[0])
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
    def _matmul_operands(spec: OpSpec):
        """Identify ``(x, y, out)`` and the K/N/M/batch symbol sets of a matmul.

        Uses the BatchMatmul semantic-dimension definitions over the operands'
        ``device_coordinates`` free symbols (the OpSpec analogue of
        ``pass_utils.identify_matmul_inputs``):

          reduction_dim K: in x and y, NOT out
          generated_dim N: in y and out, NOT x
          preserved_dim  M: in x and out, NOT y
          noreuse_dim batch: in x, y, and out

        ``y`` is identified by its generated dim (present in y & out, absent
        from x) -- robust even when ``M == 1`` folds M out of both x and out.
        Returns ``(x, y, out, k_syms, n_syms, m_syms, batch_syms)``; raises if
        the operand arity is wrong or y cannot be identified.
        """
        inputs = [a for a in spec.args if a.is_input]
        outputs = [a for a in spec.args if not a.is_input]
        if len(inputs) != 2 or len(outputs) != 1:
            raise NotImplementedError(
                "OpSpec->Triton: matmul must have exactly two inputs and one output"
            )
        out = outputs[0]
        out_syms = _arg_free_symbols(out)
        a_syms = _arg_free_symbols(inputs[0])
        b_syms = _arg_free_symbols(inputs[1])
        # y carries the generated dim (in y & out, not x); a carries it otherwise.
        if (b_syms & out_syms) - a_syms:
            x, y, x_syms, y_syms = inputs[0], inputs[1], a_syms, b_syms
        elif (a_syms & out_syms) - b_syms:
            x, y, x_syms, y_syms = inputs[1], inputs[0], b_syms, a_syms
        else:
            raise NotImplementedError(
                "OpSpec->Triton: could not identify matmul y (generated dim)"
            )
        k_syms = (x_syms & y_syms) - out_syms
        n_syms = (y_syms & out_syms) - x_syms
        m_syms = (x_syms & out_syms) - y_syms
        batch_syms = x_syms & y_syms & out_syms
        return x, y, out, k_syms, n_syms, m_syms, batch_syms

    @staticmethod
    def _validate_matmul(spec: OpSpec) -> None:
        """Guard the matmul cases this cut supports; raise loudly otherwise.

        Supported: a plain 2D ``batchmatmul`` (no batch dim) or a batched
        ``batchmatmul`` with a single noreuse/batch dim -- both with a single
        contraction dim K that is not work-divided across cores.  The batch dim
        leads the operand permutation so each operand reshapes to a batched
        matrix (``[B, M, K]`` / ``[B, K, N]``) for a batched ``tl.dot`` (the
        backend lowers a rank-3 ``tt.dot`` to ``linalg.batch_matmul``).  Deferred
        cases each need machinery that does not exist yet:

        - more than one batch dim -> multi-dim batched ``tl.dot``;
        - K work-divided across cores -> the HW inter-core reduce ring;
        - degenerate ``K == 1`` -> pointwise-mul lowering (retired to patches);
        - fp8 matmul (``batchmatmulfp8``).
        """
        _x, _y, _out, k_syms, _n, _m, batch_syms = (
            SpyreOpSpecTritonKernel._matmul_operands(spec)
        )
        if len(k_syms) != 1:
            raise NotImplementedError(
                "OpSpec->Triton: matmul must have exactly one contraction dim "
                f"(got K symbols {k_syms})"
            )
        if len(batch_syms) > 1:
            raise NotImplementedError(
                "OpSpec->Triton: matmul with more than one batch dim not "
                f"supported yet (batch symbols {batch_syms}); needs a "
                "multi-dim batched tl.dot"
            )
        k_sym = next(iter(k_syms))
        k_range, k_div = spec.iteration_space.get(k_sym, (0, 1))
        if int(k_div) != 1:
            raise NotImplementedError(
                "OpSpec->Triton: matmul contraction dim is work-divided across "
                "cores (inter-core reduce ring not implemented; see "
                "2607-InterCoreReduction.md). Retry with fewer SENCORES."
            )
        if _size_hint(k_range) <= 1:
            raise NotImplementedError(
                "OpSpec->Triton: degenerate K==1 matmul not supported yet "
                "(pointwise-mul lowering retired to patches)"
            )

    def _prepare_gather(
        self,
        spec: OpSpec,
        tensor_args: dict[int, TensorArg],
        divisor_of: dict[sympy.Symbol, int],
        block_of: dict[int, list[int]],
        perm_of: dict[int, list[int]],
    ) -> None:
        """Set gather permutations / block shapes and guard unsupported layouts.

        Mutates ``perm_of`` and ``block_of`` in place: permutes the value arg's
        indirect axis and the output's row axis to descriptor dim 0, and forces
        the value arg's dim-0 block to 1 (the gather "descriptor block must have
        exactly 1 row" rule).  Raises for layouts outside this cut (< 8 gathered
        rows, or a work-divided trailing axis the single ``y_offset`` cannot
        address).
        """
        index_arg, value_arg, out_arg, k_star, row_axis = _gather_operands(spec)
        vi, oi = value_arg.arg_index, out_arg.arg_index

        vrank = len(value_arg.device_size)
        perm_of[vi] = [k_star] + [d for d in range(vrank) if d != k_star]
        orank = len(out_arg.device_size)
        perm_of[oi] = [row_axis] + [d for d in range(orank) if d != row_axis]

        # x_offsets rows == the index buffer's per-core element count (>= 8, C5).
        num_rows = 1
        for b in block_of[index_arg.arg_index]:
            num_rows *= int(b)
        if num_rows < 8:
            raise NotImplementedError(
                f"OpSpec->Triton: gather x_offsets must have >= 8 rows, got {num_rows}"
            )

        # dims >= 2 after the permute read their full block extent with no offset
        # (C4); a work-divided trailing axis would need a per-core offset the
        # gather cannot express (only dim 1 carries y_offset).
        for k in range(2, vrank):
            coord = value_arg.device_coordinates[perm_of[vi][k]]
            if any(divisor_of.get(s, 1) != 1 for s in coord.free_symbols):
                raise NotImplementedError(
                    "OpSpec->Triton: gather with a work-divided trailing axis "
                    "(dims >= 2 read the full block extent; no per-core offset) "
                    "not supported yet. Retry with fewer SENCORES."
                )

        # Descriptor block on the indirect axis (permuted dim 0) must be 1 (C2).
        block_of[vi] = list(block_of[vi])
        block_of[vi][k_star] = 1

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
        actuals = list(actuals)
        # Materialized cross-group pool buffers have no python_argdefs entry;
        # append a synthetic parameter slot for each (its name is both the Triton
        # parameter and the caller-side buffer name).  Slots are global (indexed
        # after the real args, identically for every group), so ``used`` and the
        # signature/descriptor lookups below stay a single flat int keyspace.
        for slot in sorted(self._pool_param_of):
            assert slot == len(param_names)
            param_names.append(self._pool_param_of[slot])
            actuals.append(self._pool_param_of[slot])

        # Unique tensor args of this group, keyed by their kernel-parameter slot
        # (== arg_index for a real arg; the synthetic slot for a materialized
        # pool buffer).  A slot of -1 marks a register-threaded intermediate: it
        # is threaded through registers by _emit_ops, so it gets no descriptor
        # and no signature slot.
        tensor_args: dict[int, TensorArg] = {}
        for spec in group:
            for a in spec.args:
                if not isinstance(a, TensorArg):
                    continue
                slot = self._slot(a)
                if slot >= 0 and slot not in tensor_args:
                    tensor_args[slot] = a
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
            i: _device_block_shape(tensor_args[i], divisor_of, loop_ctx) for i in used
        }

        # Matmul operands are addressed through a *permuted* tensor descriptor so
        # the sticked matrix dim's [outer_stick, inner_stick] pair is innermost
        # and adjacent, ready to collapse into one matrix dim for tl.dot (the
        # "weight transpose", expressed as a permuted descriptor rather than a
        # tl.trans).  A gather likewise permutes the value arg's indirect axis to
        # descriptor dim 0 and the output's row axis to dim 0.  Non-permuted
        # groups use identity permutations, so their descriptor / offset source
        # is byte-identical to before.  ``dim_skip`` holds arg_indices whose
        # scalar offsets are *not* emitted -- the gather value arg, addressed by
        # ``desc.gather(x_offsets, y_offset)`` and whose indirect coordinate is
        # not renderable as a scalar offset.
        is_matmul = any(s.op in _MATMUL_OPS for s in group)
        is_gather = not is_matmul and any(_is_gather_spec(s) for s in group)
        perm_of = {i: list(range(len(tensor_args[i].device_size))) for i in used}
        dim_skip: set[int] = set()
        if is_matmul:
            # A batched matmul carries its (single) noreuse batch symbol into the
            # permutation so it leads each operand's descriptor axes, giving a
            # batched matrix for tl.dot; None for a 2D matmul.
            mm_spec = next(s for s in group if s.op in _MATMUL_OPS)
            _batch = self._matmul_operands(mm_spec)[6]
            batch_sym = next(iter(_batch)) if _batch else None
            perm_of = {
                i: _matmul_operand_permutation(
                    tensor_args[i].device_coordinates, batch_sym
                )
                for i in used
            }
        elif is_gather:
            self._prepare_gather(group[0], tensor_args, divisor_of, block_of, perm_of)
            _idx, value_arg, _out, _k, _row = _gather_operands(group[0])
            dim_skip = {value_arg.arg_index}

        # Build the body at column 0; buf.indent() re-indents it under `def`.
        # Program-id bases and descriptors are loop-invariant (they address the
        # full HBM buffer), so they stay above the loop; only the device-dim
        # offsets (which carry loop0) and the ops go inside it.
        body = IndentedBuffer()
        self._emit_logical_offsets(body, it_space)
        if loop_ctx is None:
            dims_of = self._emit_dim_vars(body, tensor_args, None, perm_of, dim_skip)
            desc_of = self._emit_descriptors(
                body, tensor_args, param_names, block_of, perm_of
            )
            self._emit_ops(body, group, desc_of, dims_of, block_of, perm_of)
        else:
            desc_of = self._emit_descriptors(
                body, tensor_args, param_names, block_of, perm_of
            )
            body.writeline(f"for {loop_ctx.var} in range({loop_ctx.count}):")
            with body.indent():
                dims_of = self._emit_dim_vars(
                    body, tensor_args, loop_ctx, perm_of, dim_skip
                )
                self._emit_ops(body, group, desc_of, dims_of, block_of, perm_of)

        signature = ", ".join(param_names[i] for i in used)
        header = self._emit_header(grid, used)
        buf = IndentedBuffer()
        buf.splice(header)
        buf.writeline(f"def {KERNEL_NAME_PLACEHOLDER}({signature}):")
        with buf.indent():
            buf.splice(body.getvalue())

        # Caller-side buffer names in signature order, deduped.  No leading
        # ``_pool``: within-group intermediates are register-threaded and
        # cross-group ones are materialized HBM tensors (real actuals here), so
        # the Triton path never passes the shared pool region.
        seen: set[str] = set()
        call_args: list[str] = []
        for i in used:
            name = actuals[i]
            if name not in seen:
                seen.add(name)
                call_args.append(name)

        return _KernelPlan(source=buf.getvalue(), call_args=call_args)

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
            extent = max(1, int(_size_hint(rng)) // div)
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
        perm_of: dict[int, list[int]],
        skip: "set[int] | None" = None,
    ) -> dict[int, list[str]]:
        """Emit device-dim offset vars (``dim0``/``dim1``/...) per tensor arg.

        Returns ``dims_of`` keyed by ``arg_index``: the list of device-dim
        offset variable names, in the arg's (possibly permuted) descriptor-axis
        order.  For a counted-loop body, a full-size operand's tiled-dim
        coordinate is advanced by ``loop0 * per_tile_range`` (via
        ``loop_ctx.subs``); a ``per_tile_fixed`` operand is left untouched.
        Because the offset changes the coordinate string, full-size and
        per-tile operands that shared a layout naturally get distinct dim vars.
        ``perm_of`` reorders each arg's coordinates so they line up with its
        permuted descriptor axes (identity for non-matmul args).  ``skip`` names
        arg_indices with no scalar offsets (the gather value arg, addressed by
        ``desc.gather`` and whose indirect coordinate is not renderable).
        """
        dims_of: dict[int, list[str]] = {}
        # Cache device-dim var names by coordinate signature so operands with an
        # identical device layout share dim0/dim1/... (as SDSC/dev-triton do).
        coords_seen: dict[tuple[str, ...], list[str]] = {}

        for arg_index in sorted(tensor_args):
            if skip and arg_index in skip:
                continue
            arg = tensor_args[arg_index]
            coords = list(arg.device_coordinates)
            if loop_ctx is not None and not arg.per_tile_fixed:
                coords = [c.subs(loop_ctx.subs) for c in coords]
            coords = [coords[p] for p in perm_of[arg_index]]
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
        perm_of: dict[int, list[int]],
    ) -> dict[int, str]:
        """Emit one ``tl.make_tensor_descriptor`` per tensor arg.

        Returns ``desc_of`` keyed by ``arg_index``.  Descriptors address the full
        HBM buffer (``shape=device_size``) so they are loop-invariant; only the
        per-core ``block_shape`` (from the shared ``block_of``) reflects the work
        division (and, in a loop, one tile's worth of the tiled dim).

        ``perm_of`` reorders each arg's descriptor axes (identity for non-matmul
        args).  The strides are the *natural* row-major strides permuted the same
        way -- i.e. the physical HBM tensor viewed under a reordered axis basis --
        so a permuted descriptor is a genuine transpose of the same buffer, not a
        re-layout.
        """
        desc_of: dict[int, str] = {}
        for arg_index in sorted(tensor_args):
            arg = tensor_args[arg_index]
            perm = perm_of[arg_index]
            device_size_nat = [int(s) for s in arg.device_size]
            strides_nat = _row_major_strides(device_size_nat)
            device_size = [device_size_nat[p] for p in perm]
            strides = [strides_nat[p] for p in perm]
            block_shape = [block_of[arg_index][p] for p in perm]
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

    def _emit_ops(
        self,
        body: IndentedBuffer,
        specs: list[OpSpec],
        desc_of: dict[int, str],
        dims_of: dict[int, list[str]],
        block_of: dict[int, list[int]],
        perm_of: dict[int, list[int]],
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
            # this kernel; only args with a slot (real, or a materialized
            # cross-group pool buffer) load from HBM.
            slot = self._slot(arg)
            offsets = ", ".join(dims_of[slot])
            var = _fresh()
            body.writeline(f"{var} = {desc_of[slot]}.load([{offsets}])")
            produced[key] = var
            return var

        def _store(out: TensorArg, var: str) -> None:
            # Write HBM for any output with a slot (a real output or a
            # materialized cross-group pool buffer); a register-threaded
            # intermediate (no slot) is consumed from the register in a later op.
            slot = self._slot(out)
            if slot >= 0:
                offsets = ", ".join(dims_of[slot])
                body.writeline(f"{desc_of[slot]}.store([{offsets}], {var})")
            produced[_buf_id(out)] = var

        for spec in specs:
            inputs = [a for a in spec.args if a.is_input]
            outputs = [a for a in spec.args if not a.is_input]
            assert len(outputs) == 1, "op must have exactly one output"

            if _is_gather_spec(spec):
                self._emit_gather(
                    body, spec, _load, _fresh, _store, desc_of, block_of, perm_of
                )
                continue

            if _is_restickify_spec(spec):
                self._emit_restickify(body, spec, _load, _fresh, _store, block_of)
                continue

            if spec.op in _MATMUL_OPS:
                self._emit_matmul(body, spec, _load, _fresh, _store, block_of, perm_of)
                continue

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
        # A reduction input with no slot is register-threaded (fused away) and has
        # no descriptor/block; not supported yet.  A materialized cross-group pool
        # buffer (sum_sum's buf0) does have a slot and loads from HBM below.
        in_slot = self._slot(in_arg)
        if in_slot < 0:
            raise NotImplementedError(
                "OpSpec->Triton: reduction with a register-threaded (fused) input "
                "not supported yet"
            )
        _reduced, axes = _reduction_axes(in_arg, out)
        axis = axes[0]

        in_var = _load(in_arg)
        red_var = _fresh()
        body.writeline(f"{red_var} = tl.sum({in_var}, {axis})")
        out_var = red_var

        # Shape of the tile after tl.sum drops `axis`, vs. the output block shape.
        in_block = block_of[in_slot]
        reduced_shape = [s for k, s in enumerate(in_block) if k != axis]
        out_slot = self._slot(out)
        out_block = block_of.get(out_slot) if out_slot >= 0 else None
        if out_block is not None and out_block != reduced_shape:
            _check_reshape_is_order_preserving(in_arg, out, axis, in_block, out_block)
            out_var = _fresh()
            body.writeline(f"{out_var} = tl.reshape({red_var}, {out_block})")
        _store(out, out_var)

    def _emit_matmul(
        self,
        body: IndentedBuffer,
        spec: OpSpec,
        _load,
        _fresh,
        _store,
        block_of: dict[int, list[int]],
        perm_of: dict[int, list[int]],
    ) -> None:
        """Emit ``tl.dot`` for a 2D or batched matmul (``batchmatmul``).

        ``_validate_matmul`` has guaranteed a single, non-work-divided
        contraction dim K and at most one batch dim.  Each operand was loaded
        through a permuted descriptor placing its (single) batch dim first and
        its sticked matrix dim's ``[outer_stick, inner_stick]`` pair innermost,
        so its per-core block is ``[batch?, row, ..., outer_stick, inner_stick]``.
        Collapsing everything after the batch+row dims into one column yields the
        canonical (batched) matrix the Spyre ``tt.dot`` -> ``linalg.matmul`` /
        ``linalg.batch_matmul`` lowering expects::

            A block [B?, M, K_out, K_in] -> [B?, M, K]   (K = K_out * K_in)
            B block [B?, K, N_out, N_in] -> [B?, K, N]   (N = N_out * N_in)
            tl.dot(A, B)                 -> [B?, M, N]
            reshape                      -> out block [B?, M, N_out, N_in], store

        The collapse is order-preserving: the within-stick index plus
        ``outer * stick`` reproduces the matrix-dim iteration symbol exactly
        (``c = FloorDiv(c, stick) * stick + Mod(c, stick)``).  For a plain 2D
        matmul ``n_batch == 0`` and this reduces to ``[M, K]`` / ``[K, N]``.
        """
        x, y, out, _k, _n, _m, batch_syms = self._matmul_operands(spec)
        n_batch = 1 if batch_syms else 0

        def _collapse(arg: TensorArg, var: str) -> str:
            # Index block_of/perm_of by the kernel-parameter slot, not the raw
            # arg_index: a matmul operand may be a materialized cross-group pool
            # buffer (arg_index == -1, synthetic slot), e.g. linear.py's
            # restickified weight buf1 produced by a prior kernel.
            slot = self._slot(arg)
            perm_block = [block_of[slot][p] for p in perm_of[slot]]
            batch = perm_block[:n_batch]
            rows = perm_block[n_batch]
            cols = 1
            for s in perm_block[n_batch + 1 :]:
                cols *= s
            flat = _fresh()
            body.writeline(f"{flat} = tl.reshape({var}, {batch + [rows, cols]})")
            return flat

        a2d = _collapse(x, _load(x))
        b2d = _collapse(y, _load(y))
        dot_var = _fresh()
        body.writeline(f'{dot_var} = tl.dot({a2d}, {b2d}, input_precision="ieee")')

        out_slot = self._slot(out)
        out_block = [block_of[out_slot][p] for p in perm_of[out_slot]]
        out_var = _fresh()
        body.writeline(f"{out_var} = tl.reshape({dot_var}, {out_block})")
        _store(out, out_var)

    def _emit_restickify(
        self,
        body: IndentedBuffer,
        spec: OpSpec,
        _load,
        _fresh,
        _store,
        block_of: dict[int, list[int]],
    ) -> None:
        """Emit reshape -> permute -> reshape for a cross-stick restickify.

        A restickify moves which logical dim is the within-stick (last) axis, so
        the 64 within-stick elements physically cross sticks -- not expressible
        as a plain strided copy.  ``_restickify_plan`` splits the input tile at
        the stick boundary of the axis that becomes stick-split on the output,
        permutes the atoms into the output device-axis order, then merges the
        output's now-full former-stick axis back from its pair -- reproducing the
        reference ``clone/transpose`` kernel.  The descriptors are the natural
        (unpermuted) ones, so the tile arrives in ``block_of`` device order.
        """
        in_arg = next(a for a in spec.args if a.is_input)
        out = next(a for a in spec.args if not a.is_input)
        in_block = [int(b) for b in block_of[in_arg.arg_index]]
        out_block = (
            [int(b) for b in block_of[out.arg_index]]
            if out.arg_index >= 0
            else [int(s) for s in out.device_size]
        )
        reshape1, permute, reshape2 = _restickify_plan(in_arg, out, in_block, out_block)

        var = _load(in_arg)
        cur_shape = in_block
        if reshape1 != cur_shape:
            nv = _fresh()
            body.writeline(f"{nv} = tl.reshape({var}, {reshape1})")
            var, cur_shape = nv, reshape1
        if permute != list(range(len(permute))):
            nv = _fresh()
            body.writeline(f"{nv} = tl.permute({var}, {permute})")
            var = nv
            cur_shape = [cur_shape[p] for p in permute]
        if reshape2 != cur_shape:
            nv = _fresh()
            body.writeline(f"{nv} = tl.reshape({var}, {reshape2})")
            var = nv
        _store(out, var)

    def _emit_gather(
        self,
        body: IndentedBuffer,
        spec: OpSpec,
        _load,
        _fresh,
        _store,
        desc_of: dict[int, str],
        block_of: dict[int, list[int]],
        perm_of: dict[int, list[int]],
    ) -> None:
        """Emit ``desc.gather(x_offsets, y_offset)`` for an indirect (gather) load.

        The index buffer's device tile is loaded normally and used directly as
        the multi-D ``x_offsets`` (no flatten -- the Spyre verifier accepts a
        >1D ``x_offsets``); the value arg is addressed through its permuted
        descriptor (indirect axis at dim 0, block 1) with the single direct
        ``y_offset`` on dim 1 and full block extent on dims >= 2.  The gather
        result ``[*idx_block, *block[1:]]`` collapses to the output's single row
        dim so the row-first result stores directly through the row-permuted
        output descriptor (``_prepare_gather`` set both permutations).
        """
        index_arg, value_arg, out, _k, _row = _gather_operands(spec)
        vi = value_arg.arg_index

        # x_offsets: the index buffer's device tile (loaded via its descriptor).
        x_offsets = _load(index_arg)

        # y_offset: the single direct scalar offset (permuted dim 1 of the value
        # arg).  dim 0 is indirect (x_offsets, no offset); dims >= 2 read full.
        perm = perm_of[vi]
        value_coords = [value_arg.device_coordinates[p] for p in perm]
        y_offset = _coord_str(value_coords[1]) if len(value_coords) > 1 else "0"

        result = _fresh()
        body.writeline(f"{result} = {desc_of[vi]}.gather({x_offsets}, {y_offset})")

        # Collapse the multi-D index dims to the output's single row dim.
        perm_block = [block_of[vi][p] for p in perm]
        idx_block = block_of[index_arg.arg_index]
        num_rows = 1
        for b in idx_block:
            num_rows *= int(b)
        if list(idx_block) != [num_rows]:
            out_shape = [num_rows, *perm_block[1:]]
            reshaped = _fresh()
            body.writeline(f"{reshaped} = tl.reshape({result}, {out_shape})")
            result = reshaped

        _store(out, result)

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
            base = len(argdefs)
            # Real args come first in ``used`` (slots < base), materialized pool
            # buffers last (slots >= base, no python_argdefs entry).  Build the
            # signature/divisibility from the real args, then append a pointer
            # type for each pool buffer (its position in the emitted signature is
            # after every real arg, so the divisibility config indices still line
            # up); a missing divisibility hint is merely conservative.
            real_used = [i for i in used if i < base]
            pool_used = [i for i in used if i >= base]
            argdefs_r = [argdefs[i] for i in real_used]
            signature_r = [signature[i] for i in real_used]
            sig_meta = signature_to_meta(
                signature_r, size_dtype=None, argdefs=argdefs_r
            )
            for slot in pool_used:
                sig_meta[self._pool_param_of[slot]] = self._pool_ptr_type(slot)
            triton_meta = {
                "signature": sig_meta,
                "device": DeviceProperties.create(
                    V.graph.get_current_device_or_throw()
                ),
                "constants": {},
                "configs": [config_of(signature_r)],
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
