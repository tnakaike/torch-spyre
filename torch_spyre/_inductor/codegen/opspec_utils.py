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

"""Backend-agnostic spec-reading helpers shared by the OpSpec backends.

Both OpSpec backends -- the Triton source generator
(``SpyreOpSpecTritonKernel``, on the fork) and the planned KTIR emitter
(``generate_ktir``) -- consume the same *finished* ``OpSpec``/``LoopSpec`` list
and must agree on the "decision / arithmetic" it implies: grouping, per-core
tile/block shape, reduction-axis selection, loop offsets, call arguments, and
reshape order-preservation.  Those computations are **pure** (sympy / int over
the ``op_specs``) -- no ``tl.*``, no MLIR builder, no live Inductor kernel state
-- so they live here as plain functions rather than as base-class methods (the
two backends deliberately share *functions*, not a base class; see
``OPSPEC_BACKEND_FUNCTIONS.md`` -- there is no ``SpyreOpSpecKernel``).

**This module must stay Triton-free** (guard:
``grep -n "triton" opspec_utils.py`` is empty) so the KTIR path can import it
without pulling Triton in.  Emission primitives (``texpr``/``tl.*`` on the
Triton side, ``ktdp.*``/``linalg.*`` on the KTIR side) stay in the respective
backends.
"""

from __future__ import annotations

import dataclasses

import sympy
from torch._inductor.virtualized import V

from torch_spyre._inductor.constants import RESTICKIFY_OP
from torch_spyre._inductor.op_spec import IndirectAccess, OpSpec, TensorArg


@dataclasses.dataclass
class _LoopCtx:
    """Loop-emission context for a ``LoopSpec`` body group.

    ``var`` is the loop-variable name; ``count`` the trip count; ``tiled`` the
    set of iteration-space symbols advanced by this loop (from the body ops'
    ``tiled_symbols[0]``); ``subs`` maps each tiled symbol ``s`` to
    ``s + var * per_tile_range`` for offsetting full-size operands' coordinates.
    """

    var: str
    count: int
    tiled: set
    subs: dict


def _size_hint(expr) -> int:
    """Concrete size hint for an iteration-space range expression."""
    if isinstance(expr, (int, sympy.Integer)):
        return int(expr)
    return int(V.graph.sizevars.size_hint(expr))


def _row_major_strides(device_size: list[int]) -> list[int]:
    """Row-major (C-contiguous) strides for a device-size list."""
    n = len(device_size)
    strides = [1] * n
    for i in range(n - 2, -1, -1):
        strides[i] = strides[i + 1] * int(device_size[i + 1])
    return strides


def _buf_id(arg: TensorArg) -> object:
    """Stable identity of the buffer an op arg refers to, for register threading.

    A fused-away intermediate carries ``arg_index == -1`` (the unassigned
    sentinel), so distinct intermediates collide on ``arg_index``.  The op-spec
    ``name`` is the buffer name, unique per buffer and identical whether the
    buffer appears as an input or an output, so it is the reliable key; fall back
    to ``arg_index`` only when a name is absent.
    """
    return arg.name if arg.name is not None else ("idx", arg.arg_index)


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


def _reduction_axes(in_arg: TensorArg, out_arg: TensorArg) -> tuple[set, list[int]]:
    """Reduced symbols and the input device axes that carry them.

    A reduction collapses one iteration-space symbol (e.g. ``torch.sum``'s
    reduced dim): it appears in the input's ``device_coordinates`` but not in
    the output's (the user-confirmed rule -- see ``sum`` SDSC artifacts).  The
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


def _device_block_shape(
    arg: TensorArg,
    divisor_of: dict[sympy.Symbol, int],
    loop_ctx: _LoopCtx | None,
) -> list[int]:
    """Per-core ``block_shape`` for the access tile.

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


def _matmul_operand_permutation(
    device_coords: list, batch_sym: sympy.Symbol | None = None
) -> list[int]:
    """Permutation placing the sticked matrix dim's stick pair innermost.

    A Spyre matmul operand's non-leading matrix dim (K for A, N for B) is
    stored as a stick split: an outer-stick dim (``FloorDiv(sym, stick)``) and
    the inner-stick dim (``Mod(sym, stick)``), the latter always the last
    device dim.  ``_emit_matmul`` collapses the two innermost dims back into
    that matrix dim, so the outer-stick and inner-stick dims must be adjacent
    and innermost, with the remaining dims (the leading matrix dim(s) --
    batch/M for A, batch/K for B) kept ahead of them.  When a batch dim is
    present (bmm) it must lead so the block reshapes to a batched matrix
    ``[B, M, K]`` / ``[B, K, N]`` for a batched ``tl.dot``.

    Anchoring on the stick pair -- the two dims that share the inner-stick
    dim's iteration symbol -- keeps this correct even when the row dim M is
    size 1 and its coordinate degenerates to a constant ``0`` (the decode-phase
    / GEMV case).  For a non-degenerate operand it yields the natural order, so
    a canonical row-major operand is left unchanged.
    """
    rank = len(device_coords)
    if rank < 3:
        return list(range(rank))  # already a (batched) matrix; nothing to move
    inner_stick = rank - 1  # inner-stick dim is always the last device dim
    inner_stick_syms = device_coords[inner_stick].free_symbols
    outer_stick = None
    if inner_stick_syms:
        outer_stick = next(
            (
                k
                for k in range(rank - 1)
                if device_coords[k].free_symbols & inner_stick_syms
            ),
            None,
        )
    if outer_stick is None:
        return list(range(rank))  # not stick-split; leave as-is
    leading = [k for k in range(rank) if k not in (outer_stick, inner_stick)]
    # A batched-matmul operand must lead with its batch dim.
    if batch_sym is not None:
        b = next((k for k in leading if device_coords[k] == batch_sym), None)
        if b is not None:
            leading = [b] + [k for k in leading if k != b]
    return leading + [outer_stick, inner_stick]


def _coord_has_indirect(coord: sympy.Expr) -> bool:
    """True if a device coordinate contains an ``IndirectAccess`` (gather/scatter)."""
    return bool(coord.atoms(IndirectAccess))


def _is_gather_spec(spec: OpSpec) -> bool:
    """True if any *input* arg has an indirect device coordinate (a gather load).

    The frontend (``SpyreKernel``) already resolves ``aten.index`` to an
    ``identity`` op whose value input carries an ``IndirectAccess(idx_name)``
    coordinate on the gathered axis; this recognizes that finished shape.
    """
    return any(
        a.is_input and any(_coord_has_indirect(c) for c in a.device_coordinates)
        for a in spec.args
    )


def _gather_operands(
    spec: OpSpec,
) -> tuple[TensorArg, TensorArg, TensorArg, int, int]:
    """Identify ``(index_arg, value_arg, out_arg, k_star, row_axis)`` of a gather.

    - ``value_arg`` is the input whose ``device_coordinates`` carry an
      ``IndirectAccess`` (the gathered tensor); ``k_star`` is the device dim
      holding it -- the gathered axis, permuted to descriptor dim 0.
    - ``index_arg`` is the other input: the int32 index buffer whose device
      tile is loaded directly as ``x_offsets``.
    - ``row_axis`` is the output device dim whose coordinate is the index
      buffer's (single) iteration symbol -- the gathered output-row axis, moved
      to dim 0 on the store so the row-first gather result stores directly.

    Raises ``NotImplementedError`` for shapes outside this cut (arity != 2+1,
    zero/multiple indirect axes, an index buffer without a single iteration
    symbol, or an output not carrying that symbol on exactly one dim).
    """
    inputs = [a for a in spec.args if a.is_input]
    outputs = [a for a in spec.args if not a.is_input]
    if len(inputs) != 2 or len(outputs) != 1:
        raise NotImplementedError(
            "OpSpec->Triton: gather must have exactly two inputs (index + value) "
            f"and one output (got {len(inputs)} inputs, {len(outputs)} outputs)"
        )
    out = outputs[0]

    value_arg: TensorArg | None = None
    k_star = -1
    for a in inputs:
        indirect_dims = [
            k for k, c in enumerate(a.device_coordinates) if _coord_has_indirect(c)
        ]
        if not indirect_dims:
            continue
        if value_arg is not None or len(indirect_dims) != 1:
            raise NotImplementedError(
                "OpSpec->Triton: gather supports exactly one indirect axis on one "
                "input (found more than one)"
            )
        value_arg, k_star = a, indirect_dims[0]
    if value_arg is None:
        raise NotImplementedError("OpSpec->Triton: gather has no indirect input")
    index_arg = next(a for a in inputs if a is not value_arg)

    # The gathered output-row axis is the index buffer's single iteration symbol.
    row_syms: set = set()
    for c in index_arg.device_coordinates:
        row_syms |= c.free_symbols
    if len(row_syms) != 1:
        raise NotImplementedError(
            "OpSpec->Triton: gather index buffer must have exactly one iteration "
            f"symbol (got {sorted(map(str, row_syms))})"
        )
    row_sym = next(iter(row_syms))
    row_dims = [
        k for k, c in enumerate(out.device_coordinates) if row_sym in c.free_symbols
    ]
    if len(row_dims) != 1:
        raise NotImplementedError(
            "OpSpec->Triton: gather output must carry the row symbol "
            f"{row_sym} on exactly one device dim (got dims {row_dims})"
        )
    return index_arg, value_arg, out, k_star, row_dims[0]


def _is_restickify_spec(spec: OpSpec) -> bool:
    """True if ``spec`` is a cross-stick restickify (a transpose copy).

    ``SpyreKernel.store`` labels a pointwise copy whose within-stick (last)
    device axis changes iteration symbol -- a transpose that moves which logical
    dim is sticked -- as ``RESTICKIFY_OP``; a same-stick copy stays ``identity``.
    """
    return getattr(spec, "op", None) == RESTICKIFY_OP


def _restickify_stick_symbol(arg: TensorArg) -> sympy.Symbol:
    """The single iteration symbol on ``arg``'s within-stick (last) device axis."""
    syms = arg.device_coordinates[-1].free_symbols
    if len(syms) != 1:
        raise NotImplementedError(
            "OpSpec->Triton: restickify within-stick axis must carry exactly one "
            f"iteration symbol (got {sorted(map(str, syms))})"
        )
    return next(iter(syms))


def _restickify_axis_role(coord: sympy.Expr) -> tuple[sympy.Symbol, str]:
    """Classify a restickify device axis coordinate as ``(symbol, role)``.

    ``role`` is one of ``full`` (a bare symbol ``s``), ``lo`` (the inner-stick
    axis ``Mod(s, stick)``), or ``hi`` (the outer-stick axis ``floor(s/stick)``).
    Raises for a constant (broadcast) or multi-symbol axis -- outside the
    bijection this cut supports.
    """
    syms = coord.free_symbols
    if len(syms) != 1:
        raise NotImplementedError(
            "OpSpec->Triton: restickify device axis must carry exactly one symbol "
            f"(got coord {coord}); broadcast / multi-symbol restickify unsupported"
        )
    s = next(iter(syms))
    if coord == s:
        return s, "full"
    if coord.has(sympy.Mod):
        return s, "lo"
    return s, "hi"


def _restickify_operands(
    spec: OpSpec,
) -> tuple[TensorArg, TensorArg, sympy.Symbol, sympy.Symbol]:
    """Identify and validate ``(in_arg, out, s_in, s_out)`` of a restickify.

    ``s_in`` / ``s_out`` are the input's and output's within-stick symbols; they
    differ (that is what makes the copy a cross-stick restickify).  Raises for
    arity != 1+1, an unclassifiable device axis, a non-cross-stick copy, or a
    work-divided within-stick symbol (the per-core stick split would not line
    up).
    """
    inputs = [a for a in spec.args if a.is_input]
    outputs = [a for a in spec.args if not a.is_input]
    if len(inputs) != 1 or len(outputs) != 1:
        raise NotImplementedError(
            "OpSpec->Triton: restickify must have exactly one input and one output "
            f"(got {len(inputs)} inputs, {len(outputs)} outputs)"
        )
    in_arg, out = inputs[0], outputs[0]
    for c in list(in_arg.device_coordinates) + list(out.device_coordinates):
        _restickify_axis_role(c)  # raises on broadcast / multi-symbol axes
    s_in = _restickify_stick_symbol(in_arg)
    s_out = _restickify_stick_symbol(out)
    if s_in == s_out:
        raise NotImplementedError(
            f"OpSpec->Triton: restickify within-stick symbol unchanged ({s_in}); "
            "expected a cross-stick restickify"
        )
    for s in (s_in, s_out):
        if int(spec.iteration_space.get(s, (0, 1))[1]) != 1:
            raise NotImplementedError(
                "OpSpec->Triton: restickify with a work-divided within-stick "
                f"symbol ({s}) not supported yet. Retry with fewer SENCORES."
            )
    return in_arg, out, s_in, s_out


def _restickify_atoms(
    arg: TensorArg, block: list[int], stick: int, split_syms: set
) -> list[tuple[sympy.Symbol, str, int]]:
    """Decompose ``arg``'s per-core tile into stick atoms, in device-axis order.

    Each device axis becomes one or two ``(symbol, part, size)`` atoms: a *split*
    symbol (the in/out within-stick symbols) carried as a bare ``full`` axis is
    broken into its ``(hi, lo)`` stick pair (sizes ``size // stick`` and
    ``stick``); every other axis stays a single atom.  This is the common
    granularity in which the input and output tiles are a pure permutation of
    each other.
    """
    atoms: list[tuple[sympy.Symbol, str, int]] = []
    for coord, size in zip(arg.device_coordinates, block):
        s, role = _restickify_axis_role(coord)
        if role == "full" and s in split_syms:
            atoms.append((s, "hi", int(size) // stick))
            atoms.append((s, "lo", stick))
        else:
            atoms.append((s, role, int(size)))
    return atoms


def _restickify_plan(
    in_arg: TensorArg,
    out: TensorArg,
    in_block: list[int],
    out_block: list[int],
) -> tuple[list[int], list[int], list[int]]:
    """Reshape/permute/reshape turning the input tile into the output tile.

    Returns ``(reshape1, permute, reshape2)``: split the input's full within-
    stick-*output* axis into its stick pair (``reshape1``), permute the atoms
    into the output's device-axis order (``permute``), then merge the output's
    full within-stick-*input* axis back from its stick pair (``reshape2`` == the
    output block).  Raises ``NotImplementedError`` for shapes outside the
    supported bijection (mismatched stick, non-cross-stick, or atom sets that do
    not line up per core).
    """
    stick = int(in_block[-1])
    if int(out_block[-1]) != stick:
        raise NotImplementedError(
            "OpSpec->Triton: restickify with differing in/out stick sizes "
            f"({stick} vs {int(out_block[-1])}) not supported"
        )
    s_in = _restickify_stick_symbol(in_arg)
    s_out = _restickify_stick_symbol(out)
    split = {s_in, s_out}
    in_atoms = _restickify_atoms(in_arg, in_block, stick, split)
    out_atoms = _restickify_atoms(out, out_block, stick, split)

    in_index: dict[tuple[sympy.Symbol, str], tuple[int, int]] = {
        (s, part): (i, size) for i, (s, part, size) in enumerate(in_atoms)
    }
    if {(s, p) for s, p, _ in out_atoms} != set(in_index):
        raise NotImplementedError(
            "OpSpec->Triton: restickify input/output atoms do not match "
            f"({sorted((str(s), p) for s, p, _ in in_atoms)} vs "
            f"{sorted((str(s), p) for s, p, _ in out_atoms)})"
        )
    permute: list[int] = []
    for s, part, size in out_atoms:
        i, in_size = in_index[(s, part)]
        if in_size != size:
            raise NotImplementedError(
                "OpSpec->Triton: restickify atom size mismatch for "
                f"{(str(s), part)} ({in_size} vs {size}); likely a work-divided "
                "stick axis. Retry with fewer SENCORES."
            )
        permute.append(i)
    reshape1 = [size for _, _, size in in_atoms]
    reshape2 = [int(b) for b in out_block]
    return reshape1, permute, reshape2


def _check_reshape_is_order_preserving(
    in_arg: TensorArg,
    out: TensorArg,
    axis: int,
    in_block: list[int],
    out_block: list[int],
) -> None:
    """Raise unless reshaping the reduced tile to the output block is a no-op
    on element order (i.e. only unit axes are added/removed, no permute).

    A bare reshape is correct only when the row-major enumeration of the
    surviving input coordinates equals that of the output coordinates.  We
    approximate that by requiring the non-unit-*block* coordinates (in axis
    order) to match on both sides -- a block-size-1 axis holds a single element
    and so does not affect ordering.  A genuine permute would need a transpose
    and is not supported yet.
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
            f"({surviving} -> {produced_out}); permute not supported yet"
        )
