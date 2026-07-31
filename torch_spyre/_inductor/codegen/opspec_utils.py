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

"""Backend-agnostic *pointwise-core* spec-reading helpers shared by the OpSpec
backends.

Both OpSpec backends -- the Triton source generator
(``SpyreTritonKernel``, on the fork) and the planned KTIR emitter
(``generate_ktir``) -- consume the same *finished* ``OpSpec``/``LoopSpec`` list
and must agree on the "decision / arithmetic" it implies: grouping, per-core
tile/block shape, loop offsets, call arguments, and reshape / broadcast
alignment.  Those computations are **pure** (sympy / int over the ``op_specs``)
-- no ``tl.*``, no MLIR builder, no live Inductor kernel state -- so they live
here as plain functions rather than as base-class methods (the two backends
deliberately share *functions*, not a base class; see
``OPSPEC_BACKEND_FUNCTIONS.md`` -- there is no ``SpyreOpSpecKernel``).

This module holds the **pointwise core** -- the helpers every OpSpec backend
needs regardless of op family.  The op-specific arithmetic lives in sibling
modules that import from here as needed:

- ``opspec_utils_reduction.py`` -- reduction-axis selection, outer-stick reduce,
  reshape order-preservation.
- ``opspec_utils_matmul.py`` -- matmul / bmm operand permutation.
- ``opspec_utils_gather.py`` -- gather (``aten.index``) operand identification.
- ``opspec_utils_restickify.py`` -- cross-stick restickify reshape/permute plan.

Keeping the pointwise core separate lets it be extracted for a first upstream
PR without dragging the op-specific modules along.

**This module (and every sibling) must stay Triton-free** (guard:
``grep -n "triton" opspec_utils*.py`` is empty) so the KTIR path can import it
without pulling Triton in.  Emission primitives (``texpr``/``tl.*`` on the
Triton side, ``ktdp.*``/``linalg.*`` on the KTIR side) stay in the respective
backends.
"""

from __future__ import annotations

import dataclasses

import sympy
from torch._inductor.virtualized import V
from torch.utils._sympy.functions import ModularIndexing

from torch_spyre._inductor.op_spec import OpSpec, TensorArg
from torch_spyre._inductor.pass_utils import coeff_through_floor


@dataclasses.dataclass
class _LoopCtx:
    """Loop-emission context for a ``LoopSpec`` body group.

    ``var`` is the loop-variable name; ``count`` the trip count; ``tiled`` the
    set of *real* iteration-space symbols advanced by this loop; ``subs`` maps
    each such symbol ``s`` to ``s + var * per_tile_range`` for offsetting
    full-size operands' coordinates.

    The body ops' ``tiled_symbols[0]`` hold *minted* per-(op, level) symbols
    (``_tile_adv_<op>_lvl<n>``), not the real ``c{i}`` iteration symbols, so
    the real symbol and its per-tile stride are recovered from each full-size
    operand's ``device_tile_advance_expr`` -- see ``coarse_loop_subs``.
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


def _tile_axis_offset(
    coords: list[sympy.Expr], strides: list[int], coeff: int
) -> tuple[sympy.Symbol, int]:
    """Recover the (real symbol, per-tile offset) a device-element advance encodes.

    ``coeff`` is one loop level's per-iteration advance in device *elements*
    (row-major over ``device_size``, as ``device_tile_advance_expr`` stores it).
    Decompose it against the row-major ``strides`` into mixed-radix per-axis
    offsets: exactly one device axis must carry it (a single logical dim is
    tiled), and that axis's coordinate must be a bare iteration symbol ``s`` so
    the offset can be applied as ``s -> s + loop * offset``.  A leftover, a
    multi-axis split, or a non-bare (sticked / folded) axis is a tiling pattern
    this cut does not support yet.
    """
    remaining = int(coeff)
    hits: list[tuple[int, int]] = []
    for axis, stride in enumerate(strides):
        off, remaining = divmod(remaining, int(stride))
        if off:
            hits.append((axis, off))
    if remaining != 0 or len(hits) != 1:
        raise NotImplementedError(
            "OpSpec->Triton: coarse-tile advance spans more than one device "
            f"axis (coeff={coeff}, strides={strides}); not supported yet"
        )
    axis, offset = hits[0]
    coord = coords[axis]
    if not isinstance(coord, sympy.Symbol):
        raise NotImplementedError(
            "OpSpec->Triton: coarse-tile advance lands on a non-bare device "
            f"axis (coord={coord}); tiling a sticked/folded dim is not "
            "supported yet"
        )
    return coord, offset


def coarse_loop_subs(group: list[OpSpec], loop_var: str) -> tuple[set, dict]:
    """Build the ``(tiled, subs)`` for a coarse-tiled ``LoopSpec`` body group.

    Each body op's ``tiled_symbols[0]`` names the *minted* per-(op, level)
    symbols the innermost loop advances; the real iteration symbol and its
    per-tile stride live in each full-size operand's
    ``device_tile_advance_expr`` (``coeff_through_floor`` extracts one level's
    device-element advance, ``_tile_axis_offset`` maps it back to a real
    symbol).  Register-threaded intermediates carry no advance expr and are
    left untouched.  Returns the set of real tiled symbols and a substitution
    mapping each to ``s + loop_var * per_tile_range``.
    """
    loop_sym = sympy.Symbol(loop_var)
    tiled: set = set()
    subs: dict = {}
    for spec in group:
        if not spec.tiled_symbols:
            continue
        for arg in spec.args:
            if arg.device_tile_advance_expr is None:
                continue
            strides = _row_major_strides(list(arg.device_size))
            for minted in spec.tiled_symbols[0]:
                coeff = int(coeff_through_floor(arg.device_tile_advance_expr, minted))
                if coeff == 0:
                    continue
                real_sym, per_tile = _tile_axis_offset(
                    list(arg.device_coordinates), strides, coeff
                )
                tiled.add(real_sym)
                subs[real_sym] = real_sym + loop_sym * per_tile
    return tiled, subs


def _is_broadcast_axis(coord: sympy.Expr) -> bool:
    """True if a device axis is a broadcast (fixed at element 0).

    A broadcast operand (e.g. a scalar ``spyre_constant_tensor`` reciprocal, or
    a per-row reduction result replicated over the outer-stick dim) is read at
    element ``0`` along the broadcast axis and replicated by the consuming op.
    Its coordinate is the pure constant ``0``.

    Such an axis has physical extent ``1`` even though ``device_size`` may report
    the full stick width: the inner-stick ``device_size`` is force-set to
    ``elems_per_stick`` (see ``ir.py`` layout rules) regardless of broadcasting,
    so a descriptor built straight from ``device_size`` would address 64 elements
    of a 1-element buffer and read past its end.
    """
    e = sympy.sympify(coord)
    return not e.free_symbols and e == 0


def _physical_device_extents(arg: TensorArg) -> list[int]:
    """``device_size`` with broadcast axes clamped to their true extent (1).

    Use this -- not ``arg.device_size`` -- to build a tensor descriptor's
    ``shape`` / ``strides`` and per-core ``block_shape`` so a broadcast operand
    addresses only its real elements.  Non-broadcast axes are unchanged, so
    ordinary operands are unaffected (fast path stays byte-identical).

    Clamping applies only to a *genuine scalar / all-broadcast* operand (e.g. a
    ``spyre_constant_tensor`` reciprocal), which is physically a single element
    -- so a descriptor built straight from ``device_size`` (whose inner-stick
    dim is force-set to the stick width) would address 64 elements of a
    1-element buffer and read past its end.

    A buffer that has a *real data axis* (a coordinate referencing an iteration
    symbol) is allocated at its full ``device_size`` by
    ``ktir_empty_with_layout``, so its broadcast axes ARE physically backed;
    clamping them would instead under-address a real stick.  A per-row reduction
    result is the canonical case: its within-stick coordinate is the broadcast
    constant ``0`` (one value per row, replicated over the stick), yet the buffer
    is a full 64-wide stick.  Its descriptor must keep extent 64 -- matching the
    retired subclass path -- so an outer-stick-only reduce (``[128, 64]``)
    reshapes cleanly to ``[1, 128, 64]`` rather than to an impossible
    ``[1, 128, 1]``.
    """
    coords = arg.device_coordinates
    if any(sympy.sympify(c).free_symbols for c in coords):
        # A real data axis is present -> the buffer is physically full-size, so
        # every axis is backed; leave all extents unclamped.
        return [int(s) for s in arg.device_size]
    return [
        1 if _is_broadcast_axis(c) else int(s) for s, c in zip(arg.device_size, coords)
    ]


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


# Device-dim coordinate kinds, used to align pointwise operands whose device
# axes are ordered differently from the op's output tile (broadcast alignment).
_DIM_CONST = "const"  # no iteration-space symbol (an inserted / broadcast dim)
_DIM_MULTI = "multi"  # more than one symbol (not a simple device axis)
_DIM_BARE = "bare"  # coord == sym            (an un-sticked dim)
_DIM_INNER_STICK = "inner_stick"  # coord == sym % stick  (within-stick lanes)
_DIM_OUTER_STICK = "outer_stick"  # coord == sym // stick (outer-stick chunks)


def _dim_info(coord: sympy.Expr) -> tuple[str, sympy.Symbol | None]:
    """Classify a device-dim coordinate as ``(kind, sym)``.

    Two device axes are the "same" logical dim iff they share a ``(kind, sym)``
    -- e.g. a weight's ``sym // stick`` outer-stick axis matches an output's
    ``sym // stick`` axis even when they sit at different positions.  A ``const``
    dim (extent 1, no symbol) carries no data and is dropped / broadcast.
    """
    syms = coord.free_symbols
    if not syms:
        return (_DIM_CONST, None)
    if len(syms) > 1:
        return (_DIM_MULTI, None)
    sym = next(iter(syms))
    if isinstance(coord, sympy.Symbol):
        return (_DIM_BARE, sym)
    if isinstance(coord, (sympy.Mod, ModularIndexing)):
        return (_DIM_INNER_STICK, sym)
    return (_DIM_OUTER_STICK, sym)


def _align_reshape_plan(
    in_coords: list[sympy.Expr],
    in_block: list[int],
    out_coords: list[sympy.Expr],
    out_block: list[int],
) -> tuple[list[int], list[int] | None] | None:
    """Plan to reshape + broadcast a pointwise operand tile into the op's output
    device-axis order.

    A pointwise op's operands may carry their device axes in a different order
    (and rank) from the output tile: a per-row reduction result broadcasts over
    the outer-stick dim, a channel weight broadcasts over the row dim, etc.
    ``tl.*`` elementwise only auto-broadcasts *leading* unit dims, so a
    misaligned operand (outer-stick where the output has rows) must be reshaped
    to the output order first.

    Each output device axis is matched to an input axis by ``(kind, sym)`` (see
    ``_dim_info``); the inner-stick axis maps last -> last.  Unmatched output
    axes get extent 1 (to be broadcast).  Returns ``(reshape_to, broadcast_to)``
    -- ``tl.reshape`` the operand to ``reshape_to`` (skip if it already equals
    ``in_block``) then ``tl.broadcast_to`` ``broadcast_to`` (``None`` to skip).
    Returns ``None`` when the operand already matches the output (fast path:
    no reshape / broadcast emitted, keeping simple kernels byte-identical).

    Raises ``NotImplementedError`` for a cross-stick transpose (an input axis
    with extent > 1 that no output axis matches, or matched axes that would need
    a permute) -- that needs a restickify, not a reshape.
    """
    in_block = [int(b) for b in in_block]
    out_block = [int(b) for b in out_block]
    if list(in_coords) == list(out_coords) and in_block == out_block:
        return None

    in_info = [_dim_info(c) for c in in_coords]
    in_rank = len(in_coords)
    out_rank = len(out_coords)

    reshape_to = [1] * out_rank
    used: set[int] = set()
    # Inner-stick lanes: the last input axis always maps to the last output axis.
    reshape_to[out_rank - 1] = in_block[in_rank - 1]
    used.add(in_rank - 1)
    matched_seq: list[int] = []  # input axes matched to output axes, in out order
    for o in range(out_rank - 1):
        okind, osym = _dim_info(out_coords[o])
        if okind in (_DIM_CONST, _DIM_MULTI):
            continue
        for a in range(in_rank - 1):
            if a in used:
                continue
            if in_info[a] == (okind, osym):
                used.add(a)
                reshape_to[o] = in_block[a]
                matched_seq.append(a)
                break

    # A pure reshape preserves the row-major element order: the matched input
    # axes must appear in increasing order.  If not, the operand would need a
    # transpose (restickify) to align -- not supported on this path.
    if any(matched_seq[i] >= matched_seq[i + 1] for i in range(len(matched_seq) - 1)):
        raise NotImplementedError(
            "OpSpec->Triton: pointwise operand needs a transpose (restickify) "
            "to align device axes; not supported yet"
        )
    prod_in = 1
    for b in in_block:
        prod_in *= b
    prod_reshape = 1
    for b in reshape_to:
        prod_reshape *= b
    if prod_in != prod_reshape:
        # An input axis with extent > 1 was dropped -> real data would be lost;
        # aligning it needs a cross-stick transpose (restickify).
        raise NotImplementedError(
            "OpSpec->Triton: pointwise operand needs a cross-stick transpose "
            "(restickify) to align device axes; not supported yet"
        )
    broadcast_to = out_block if reshape_to != out_block else None
    return (reshape_to, broadcast_to)


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
    one tile, so that dim is first divided by ``count``.  An operand that does
    *not* advance per iteration -- a ``per_tile_fixed`` scratch tile or a
    register-threaded intermediate whose ``device_size`` is already per-tile --
    carries no ``device_tile_advance_expr`` and holds one tile already, so it is
    left alone.  ``device_tile_advance_expr is not None`` is the precise signal
    (post-WSR ``tiled_symbols`` name minted symbols, so ``per_tile_fixed`` alone
    no longer distinguishes a full-size operand from a per-tile pool).
    """
    device_size = _physical_device_extents(arg)
    coords = arg.device_coordinates
    last = len(device_size) - 1
    tile_this_arg = loop_ctx is not None and arg.device_tile_advance_expr is not None

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
    if any("hbm_pool" in a.allocation for a in tensor_args.values()):
        call_args.append("_pool")
    seen: set[str] = set()
    for i in used:
        name = actuals[i]
        if name not in seen:
            seen.add(name)
            call_args.append(name)
    return call_args
