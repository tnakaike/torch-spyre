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

"""OpSpec -> KTIR emitter.

``generate_ktir`` is an OpSpec consumer: it consumes the finished
``list[OpSpec | LoopSpec]`` kernel contract (the same contract the SDSC bundle
emitter ``generate_bundle`` consumes) and emits **KTDP-dialect MLIR** directly.
The module is built with the ``mlir_ktdp`` Python builders, so the returned
``str(module)`` is canonical, verifier-checked MLIR that the golden snapshot
test consumes without drift.

It uses the OpSpec-reading helpers from ``opspec_utils`` to adapt the OpSpec
information to generate_ktir.
"""

from __future__ import annotations

from collections.abc import Sequence

from torch_spyre._C import DataFormats
from torch_spyre._inductor.codegen.opspec_utils import (
    _align_reshape_plan,
    _buf_id,
    _decompose_work_divisions,
    _device_block_shape,
    _iteration_space_key,
    _row_major_strides,
)
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, TensorArg, UnimplementedOp

# Pointwise op name -> the ``linalg`` named op that implements it.
# TENTATIVE (dbo workaround): dbo-opt's construct-three-stage-pipeline
# requires linalg named ops (linalg.add / linalg.mul) rather than
# arith scalar-on-tensor ops (arith.addf / arith.mulf).  The test
# fixtures under dataflow-scheduler/test/.../ConstructThreeStagePipeline/
# all use linalg ops; arith.addf on tensors is not handled.
_LINALG_OP = {"add": "add", "mul": "mul"}


def _val(x):
    """The SSA ``Value`` of a builder result.

    Return the result of x if x is not a SSA value (e.g., ``OpView``).
    """
    return x.result if hasattr(x, "result") else x


def _mlir_elt_type(ir, device_dtype: DataFormats):
    """The ``mlir_ktdp.ir`` element type for a Spyre device dtype.

    ``ir`` is the lazily-imported ``mlir_ktdp.ir`` module (the file never
    imports it at top level, so it stays importable where the dialect build is
    absent).  The two fp16 device formats both map to ``f16``; extend this map
    (never fall through silently) as new dtypes are supported.
    """
    # Direct type-builder references (not name strings) resolved here, where
    # ``ir`` is in scope.
    mapping = {
        DataFormats.IEEE_FP16: ir.F16Type,
        DataFormats.SEN169_FP16: ir.F16Type,
        DataFormats.IEEE_FP32: ir.F32Type,
        DataFormats.BFLOAT16: ir.BF16Type,
    }
    builder = mapping.get(device_dtype)
    if builder is None:
        raise NotImplementedError(
            f"OpSpec->KTIR: unsupported device dtype {device_dtype!r}"
        )
    return builder.get()


def generate_ktir(
    kernel_name: str,
    specs: Sequence[OpSpec | LoopSpec | UnimplementedOp],
) -> str:
    """Build a KTDP-dialect MLIR module for ``specs`` and return ``str(module)``.

    ``specs`` is the finished OpSpec kernel contract (the same value
    ``call_kernel`` passes positionally to ``.run(...)``).  Func parameters are
    the unique operand buffers in ascending ``arg_index`` order so the emitted
    signature matches that positional binding.
    """
    # Validate scope before the mlir_ktdp import: these checks need no dialect
    # build, so an unsupported request fails fast (and is testable) whether or
    # not mlir_ktdp is installed.
    op_specs = _collect_pointwise_op_specs(specs)

    # ``mlir_ktdp`` is imported lazily so the module stays importable (and the
    # golden test can skip) where the dialect-packaged mlir_ktdp is not built.
    from mlir_ktdp import ir
    from mlir_ktdp.dialects import arith, func, ktdp, linalg, tensor

    # Fused ops must share one iteration space (same grid / work-division); the
    # register-threaded intermediate between them has the same extents as the
    # output.  Differing spaces (mixed work-division within one node) are not
    # supported yet.
    it_space = op_specs[0].iteration_space
    it_key = _iteration_space_key(op_specs[0])
    for spec in op_specs[1:]:
        if _iteration_space_key(spec) != it_key:
            raise NotImplementedError(
                "OpSpec->KTIR: fused ops with differing iteration spaces "
                "(mixed work-division) are not supported yet"
            )
    work_divisions, total_cores = _decompose_work_divisions(it_space)
    divisor_of = {sym: div for sym, div, _inner in work_divisions}

    # Ordered unique operand buffers -> func parameter position.  Ascending
    # arg_index matches the positional order call_kernel passes to .run(...),
    # so the emitted func signature lines up with that binding.  Only real
    # external buffers (arg_index >= 0) become func parameters; register-threaded
    # fused intermediates carry the -1 sentinel and are threaded as SSA values,
    # never bound positionally.
    ordered_args: dict[object, TensorArg] = {}
    for spec in op_specs:
        for arg in spec.args:
            ordered_args.setdefault(_buf_id(arg), arg)
    param_args = sorted(
        (a for a in ordered_args.values() if a.arg_index >= 0),
        key=lambda a: a.arg_index,
    )
    param_index = {_buf_id(a): i for i, a in enumerate(param_args)}

    with ir.Context() as ctx, ir.Location.unknown():
        ktdp.register_dialects(ctx)
        index_t = ir.IndexType.get()

        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            fn_type = ir.FunctionType.get([index_t] * len(param_args), [])
            fn = func.FuncOp(kernel_name, fn_type)
            i64 = ir.IntegerType.get_signless(64)
            fn.attributes["grid"] = ir.ArrayAttr.get(
                [ir.IntegerAttr.get(i64, total_cores)]
            )
            block = fn.add_entry_block()
            block_args = list(block.arguments)

            with ir.InsertionPoint(block):
                c0 = arith.ConstantOp(index_t, 0)

                # One memory view per unique buffer, in param order.
                memory_views: dict[object, ir.Value] = {}
                for arg in param_args:
                    bid = _buf_id(arg)
                    memory_views[bid] = _emit_memory_view(
                        ir, ktdp, arg, block_args[param_index[bid]]
                    )

                # Per-core offset from the flat grid id for each work-divided
                # symbol.  Nothing is emitted when total_cores == 1, so the
                # single-core path stays byte-identical to the pointwise PR.
                core_offset: dict[object, ir.Value] = {}
                if total_cores > 1:
                    tile_id = ktdp.get_compute_tile_id([index_t])
                    for sym, div, inner_cores in work_divisions:
                        idx = tile_id
                        if inner_cores > 1:
                            idx = _val(
                                arith.DivUIOp(
                                    idx, arith.ConstantOp(index_t, inner_cores)
                                )
                            )
                        if inner_cores * div != total_cores:
                            idx = _val(
                                arith.RemUIOp(idx, arith.ConstantOp(index_t, div))
                            )
                        core_offset[sym] = idx

                # SSA value threaded from each producer to its consumers; a
                # fused-away intermediate is recorded here instead of stored.
                produced: dict[object, ir.Value] = {}
                for spec in op_specs:
                    _emit_pointwise_op(
                        ir,
                        ktdp,
                        arith,
                        linalg,
                        tensor,
                        spec,
                        memory_views,
                        produced,
                        core_offset,
                        divisor_of,
                        c0,
                        index_t,
                    )

                func.ReturnOp([])

        return str(module)


def _collect_pointwise_op_specs(
    specs: Sequence[OpSpec | LoopSpec | UnimplementedOp],
) -> list[OpSpec]:
    """Validate ``specs`` and return the flat list of pointwise ``OpSpec``s.

    Rejects everything outside the supported scope with an explicit
    ``NotImplementedError``.
    """
    op_specs: list[OpSpec] = []
    for entry in specs:
        if isinstance(entry, UnimplementedOp):
            raise NotImplementedError(f"OpSpec->KTIR: unimplemented op {entry.op!r}")
        if isinstance(entry, LoopSpec):
            raise NotImplementedError(
                "OpSpec->KTIR: counted loops (LoopSpec) are not supported yet"
            )
        if not isinstance(entry, OpSpec):
            raise NotImplementedError(
                f"OpSpec->KTIR: unexpected spec entry {type(entry).__name__}"
            )
        if entry.is_reduction:
            raise NotImplementedError("OpSpec->KTIR: reductions are not supported yet")
        if entry.op not in _LINALG_OP:
            raise NotImplementedError(
                f"OpSpec->KTIR: op {entry.op!r} is not supported yet "
                f"(only pointwise {sorted(_LINALG_OP)})"
            )
        op_specs.append(entry)
    if not op_specs:
        raise NotImplementedError("OpSpec->KTIR: no OpSpec to emit")
    return op_specs


def _emit_memory_view(ir, ktdp, arg: TensorArg, offset):
    """Emit ``ktdp.construct_memory_view`` for one buffer, return its SSA value."""
    sizes = [int(s) for s in arg.device_size]
    strides = _row_major_strides(sizes)
    memref_t = ir.MemRefType.get(sizes, _mlir_elt_type(ir, arg.device_dtype))
    coord_set = _coordinate_set_attr(ir, sizes)
    # No Python builder is exposed for the ``spyre_memory_space`` enum attribute
    # (only the ktdp *types* have getters), so this small enum literal is the one
    # unavoidable textual attribute.
    memory_space = ir.Attribute.parse("#ktdp.spyre_memory_space<HBM>")
    # All extents are static -> empty dynamic size/stride operand lists.
    return ktdp.construct_memory_view(
        memref_t,
        offset,
        [],
        [],
        sizes,
        strides,
        memory_space,
        coord_set,
    )


def _emit_pointwise_op(
    ir,
    ktdp,
    arith,
    linalg,
    tensor,
    spec: OpSpec,
    memory_views,
    produced,
    core_offset,
    divisor_of,
    c0,
    index_t,
):
    """Emit the load / compute / store sequence for one pointwise ``OpSpec``.

    An input whose buffer was produced earlier in this kernel is
    register-threaded (its SSA value is reused, no load is emitted); an output
    that is a fused-away intermediate (``arg_index < 0``, no memory view) is
    only recorded in ``produced`` and never stored.
    """
    inputs = [a for a in spec.args if a.is_input]
    outputs = [a for a in spec.args if not a.is_input]
    if len(outputs) != 1:
        raise NotImplementedError(
            f"OpSpec->KTIR: expected exactly one output, got {len(outputs)}"
        )
    if len(inputs) != 2:
        raise NotImplementedError(
            f"OpSpec->KTIR: {spec.op!r} expects two inputs, got {len(inputs)}"
        )
    out = outputs[0]

    out_extents = [int(s) for s in out.device_size]
    for arg in inputs:
        # In-place (input buffer aliases the output) is not supported yet.
        if _buf_id(arg) == _buf_id(out):
            raise NotImplementedError(
                "OpSpec->KTIR: in-place ops (input aliases output) not supported"
            )
        # Reject broadcast / transpose operands: only operands whose device
        # axes already match the output tile exactly are supported.
        plan = _align_reshape_plan(
            list(arg.device_coordinates),
            [int(s) for s in arg.device_size],
            list(out.device_coordinates),
            out_extents,
        )
        if plan is not None:
            raise NotImplementedError(
                "OpSpec->KTIR: broadcast / reshape operands not supported yet"
            )

    loaded = []
    for arg in inputs:
        bid = _buf_id(arg)
        if bid in produced:
            # Register-threaded fused intermediate: reuse the producer's value.
            loaded.append(produced[bid])
        elif bid in memory_views:
            loaded.append(
                _emit_load(
                    ir,
                    ktdp,
                    arith,
                    arg,
                    memory_views[bid],
                    core_offset,
                    divisor_of,
                    c0,
                    index_t,
                )
            )
        else:
            # Neither a func parameter nor produced earlier: a cross-group HBM
            # intermediate that a single KTIR kernel cannot thread.  Fail loud.
            raise NotImplementedError(
                "OpSpec->KTIR: operand buffer is neither a func parameter nor a "
                "register-threaded intermediate produced earlier in this kernel "
                "(cross-group HBM intermediates are not supported yet)"
            )

    # TENTATIVE: emit linalg named op + tensor.empty() outs, matching the
    # pattern the dataflow-scheduler test fixtures use.
    block = _device_block_shape(out, divisor_of)
    elt_t = _mlir_elt_type(ir, out.device_dtype)
    tensor_t = ir.RankedTensorType.get(block, elt_t)
    empty = _val(tensor.EmptyOp(block, elt_t))
    linalg_op_name = _LINALG_OP[spec.op]
    linalg_builder = getattr(linalg, linalg_op_name)
    result = _val(linalg_builder(*loaded, outs=[empty], result_tensors=[tensor_t]))

    out_bid = _buf_id(out)
    if out_bid in memory_views:
        _emit_store(
            ir,
            ktdp,
            arith,
            out,
            memory_views[out_bid],
            result,
            core_offset,
            divisor_of,
            c0,
            index_t,
        )
    # Record for any downstream consumer (a real output may be read back too).
    produced[out_bid] = result


def _emit_tile_offsets(arith, arg: TensorArg, block, core_offset, c0, index_t):
    """Per-device-axis base offset for ``arg``'s per-core access-tile slice.

    A device axis divided across cores starts at
    ``core_offset(sym) * block[k]`` (the per-core extent on that axis); an
    undivided axis -- and the within-stick (last) axis, which is never divided
    -- starts at 0.
    """
    coords = arg.device_coordinates
    last = len(block) - 1
    offsets = []
    for k, coord in enumerate(coords):
        syms = set() if k == last else coord.free_symbols
        sym = next(iter(syms)) if len(syms) == 1 else None
        if sym is not None and sym in core_offset:
            extent = arith.ConstantOp(index_t, block[k])
            offsets.append(arith.MulIOp(core_offset[sym], extent))
        else:
            offsets.append(c0)
    return offsets


def _emit_access_tile(
    ir, ktdp, arith, arg: TensorArg, memory_view, core_offset, divisor_of, c0, index_t
):
    """Emit ``ktdp.construct_access_tile`` for ``arg``'s per-core slice."""
    block = _device_block_shape(arg, divisor_of)
    rank = len(block)
    at_t = ktdp.AccessTileType.get(block, ir.IndexType.get())
    identity = ir.AffineMapAttr.get(ir.AffineMap.get_identity(rank))
    tile_set = _coordinate_set_attr(ir, block)
    offsets = _emit_tile_offsets(arith, arg, block, core_offset, c0, index_t)
    return ktdp.construct_access_tile(
        at_t,
        memory_view,
        identity,
        offsets,
        [],
        tile_set,
        identity,
    )


def _emit_load(
    ir, ktdp, arith, arg: TensorArg, memory_view, core_offset, divisor_of, c0, index_t
):
    """Emit a per-core access tile + ``ktdp.load`` for an input ``arg``."""
    block = _device_block_shape(arg, divisor_of)
    tensor_t = ir.RankedTensorType.get(block, _mlir_elt_type(ir, arg.device_dtype))
    tile = _emit_access_tile(
        ir, ktdp, arith, arg, memory_view, core_offset, divisor_of, c0, index_t
    )
    return ktdp.load(tensor_t, tile)


def _emit_store(
    ir,
    ktdp,
    arith,
    arg: TensorArg,
    memory_view,
    value,
    core_offset,
    divisor_of,
    c0,
    index_t,
):
    """Emit a per-core access tile + ``ktdp.store`` of ``value`` into ``arg``."""
    tile = _emit_access_tile(
        ir, ktdp, arith, arg, memory_view, core_offset, divisor_of, c0, index_t
    )
    ktdp.store(value, tile)


# ---------------------------------------------------------------------------
# Attribute builders
# ---------------------------------------------------------------------------


def _coordinate_set_attr(ir, sizes: list[int]):
    """Per-dim bounding integer set ``(0 <= d_i <= size_i - 1)`` as an attribute.

    Built with ``ir.IntegerSet`` from ``AffineExpr`` constraints (no textual
    round-trip): for each dim ``i`` two inequalities ``d_i >= 0`` and
    ``-d_i + (size_i - 1) >= 0``, matching the ``affine_set`` MLIR prints.
    """
    exprs = []
    eq_flags: list[bool] = []
    for i, s in enumerate(sizes):
        dim = ir.AffineExpr.get_dim(i)
        # d_i >= 0
        exprs.append(dim)
        eq_flags.append(False)
        # -d_i + (size_i - 1) >= 0
        neg_dim = ir.AffineExpr.get_mul(ir.AffineExpr.get_constant(-1), dim)
        upper = ir.AffineExpr.get_add(neg_dim, ir.AffineExpr.get_constant(int(s) - 1))
        exprs.append(upper)
        eq_flags.append(False)
    integer_set = ir.IntegerSet.get(len(sizes), 0, exprs, eq_flags)
    return ir.IntegerSetAttr.get(integer_set)
