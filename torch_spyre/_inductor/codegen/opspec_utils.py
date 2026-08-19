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

"""OpSpec-reading helpers.

The KTIR emitter consumes the *finished* ``OpSpec`` list and must agree on the
"decision / arithmetic" it implies: buffer identity, row-major strides,
iteration-space grouping, and reshape / broadcast alignment of pointwise
operands.  Those computations are **pure** (sympy / int over the ``op_specs``)
-- no backend emission primitives, no MLIR builder, no live Inductor kernel
state -- so they live here as plain functions rather than as base-class methods.

They are therefore **emitter-agnostic**: they read the ``OpSpec`` contract
without emitting any target IR, so one set can serve every OpSpec consumer.  The
KTIR emitter is simply the first such consumer (hence the KTIR references
below); future emitters are intended to share them, which is why the module is
named for the ``OpSpec`` it reads rather than for KTIR.

``__all__`` is the contract: those names are what a consumer may import.  Anything
underscore-prefixed is a step inside one of them -- reachable from a test, but not
something to build on.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import sympy
from torch._inductor.virtualized import V
from torch.utils._sympy.functions import FloorDiv, ModularIndexing

from torch_spyre._inductor.constants import RESTICKIFY_OP
from torch_spyre._inductor.op_spec import IndirectAccess, OpSpec, TensorArg
from torch_spyre._inductor.pass_utils import coeff_through_floor

__all__ = [
    "PARALLEL",
    "REDUCTION",
    "align_reshape_plan",
    "buf_id",
    "core_divisions",
    "per_core_extent",
    "placeholder_axes",
    "reduction_indexing",
    "row_major_strides",
]


def _size_hint(expr) -> int:
    """Concrete size hint for an iteration-space range expression."""
    if isinstance(expr, (int, sympy.Integer)):
        return int(expr)
    return int(V.graph.sizevars.size_hint(expr))


def row_major_strides(device_size: Sequence[int]) -> list[int]:
    """Row-major (C-contiguous) strides for a device-size list."""
    n = len(device_size)
    strides = [1] * n
    for i in range(n - 2, -1, -1):
        strides[i] = strides[i + 1] * int(device_size[i + 1])
    return strides


def buf_id(arg: TensorArg) -> str:
    """Stable identity of the buffer an op arg refers to, for register threading.

    Keys on the op-spec ``name`` (the buffer name): unique per buffer and
    identical whether the buffer appears as an input or an output, so a
    fused-away intermediate threads its register value without aliasing.
    ``arg_index`` cannot serve as the identity -- distinct fused-away
    intermediates all carry the unassigned sentinel ``-1``.

    ``name`` must therefore be populated on every projected op arg (see
    ``create_tensor_arg``).  A ``None`` name means an unnamed arg reached
    projection, which would silently alias on ``-1``; raise loudly instead of
    falling back.
    """
    if arg.name is None:
        # ValueError (not NotImplementedError): under TORCH_SPYRE_KTIR=1
        # create_tensor_arg always populates the name, so a None here is a
        # broken internal invariant, not an unsupported-capability case.
        raise ValueError(
            "buf_id: TensorArg.name is None -- every projected op arg must "
            "carry a buffer name for register-threading identity (arg_index is "
            "-1 for fused intermediates and cannot disambiguate them)"
        )
    return arg.name


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
_DIM_BARE = "bare"  # coord == sym (a non-stick dim)
_DIM_WITHIN_STICK = "within_stick"  # coord == sym % stick  (within-stick lanes)
_DIM_OUTER_STICK = "outer_stick"  # coord == sym // stick (outer-stick chunks)


def _dim_info(coord: sympy.Expr) -> tuple[str, sympy.Symbol | None]:
    """Classify a device-dim coordinate as ``(kind, sym)``.

    Two device axes are the "same" logical dim iff they share a ``(kind, sym)``
    -- e.g. a weight's ``sym // stick`` outer-stick axis matches an output's
    ``sym // stick`` axis even when they sit at different positions.  A ``const``
    dim (extent 1, no symbol) carries no data and is dropped / broadcast.

    A coordinate carrying more than one iteration symbol (a single physical axis
    that folds two logical dims, e.g. ``a*8 + b``) is not a simple device axis.
    No legal reshape produces one today -- a plain reshape is a pure view whose
    strides stay stick-aligned (single symbol per axis), and a within-stick fold
    is rejected earlier at layout selection (the within-stick dim must be a full
    64-element stick).  Raise loudly rather than carrying a dead classification:
    if this fires, a new frontend construct reached alignment and both this
    helper and ``align_reshape_plan`` need real multi-symbol support.
    """
    syms = coord.free_symbols
    if not syms:
        return (_DIM_CONST, None)
    if len(syms) > 1:
        raise NotImplementedError(
            f"OpSpec alignment: device coordinate {coord!r} folds multiple "
            f"iteration symbols {syms} into one physical axis; no supported "
            "reshape produces this, so multi-symbol alignment is not implemented"
        )
    sym = next(iter(syms))
    if isinstance(coord, sympy.Symbol):
        return (_DIM_BARE, sym)
    if isinstance(coord, (sympy.Mod, ModularIndexing)):
        return (_DIM_WITHIN_STICK, sym)
    # Both spellings of the outer-stick index: torch's FloorDiv, and the plain
    # sympy ``floor(c/64)`` the projection actually produces -- the pointwise path
    # never noticed the difference because identical operand and output
    # coordinates take the fast path out of ``align_reshape_plan`` before any
    # classification happens.
    if isinstance(coord, (FloorDiv, sympy.floor)):
        return (_DIM_OUTER_STICK, sym)
    # A single-symbol coordinate that is none of the known device-axis forms
    # (e.g. ``2*d0 + 1``) is not a plain outer-stick chunk index.  Raise loudly
    # rather than silently classifying it as outer-stick -- same policy as the
    # multi-symbol case above; a later increment that legitimately produces such
    # a coordinate must extend this classifier explicitly.
    raise NotImplementedError(
        f"OpSpec alignment: single-symbol device coordinate {coord!r} is not a "
        "bare symbol, within-stick (Mod/ModularIndexing), or outer-stick "
        "(FloorDiv) form; classification is not implemented"
    )


def align_reshape_plan(
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
    Elementwise broadcast typically only auto-aligns *leading* unit dims, so a
    misaligned operand (outer-stick where the output has rows) must be reshaped
    to the output order first.

    Each output device axis is matched to an input axis by ``(kind, sym)`` (see
    ``_dim_info``); the within-stick axis maps last -> last.  Unmatched output
    axes get extent 1 (to be broadcast).  Returns ``(reshape_to, broadcast_to)``
    -- reshape the operand to ``reshape_to`` (skip if it already equals
    ``in_block``) then broadcast to ``broadcast_to`` (``None`` to skip).
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
    # Within-stick lanes: the last input axis always maps to the last output axis.
    reshape_to[out_rank - 1] = in_block[in_rank - 1]
    used.add(in_rank - 1)
    matched_seq: list[int] = []  # input axes matched to output axes, in out order
    for o in range(out_rank - 1):
        okind, osym = _dim_info(out_coords[o])
        if okind == _DIM_CONST:
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
            "OpSpec alignment: pointwise operand needs a transpose (restickify) "
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
            "OpSpec alignment: pointwise operand needs a cross-stick transpose "
            "(restickify) to align device axes; not supported yet"
        )
    broadcast_to = out_block if reshape_to != out_block else None
    return (reshape_to, broadcast_to)


def operand_indexing(
    in_coords: Sequence[sympy.Expr],
    in_extent: Sequence[int],
    out_coords: Sequence[sympy.Expr],
    out_extent: Sequence[int],
) -> tuple[int | None, ...]:
    """One pointwise operand's map row against the output's axes.

    The parallel counterpart of ``reduction_indexing``, and the same shape of
    answer: one entry per *operand* axis, being the axis of the OUTPUT that axis
    walks -- so the row is a projection of the output's iteration nest, which is
    what an all-parallel ``linalg.generic``'s ``indexing_maps`` states.  The
    iteration dims are the output's own axes, in order, because a pointwise op
    writes every element of its output exactly once.

    ``None`` is an axis the operand does not walk at all: its coordinate carries
    no iteration symbol and its extent is one element, so every iteration reads
    that one element and the map's result position is the CONSTANT 0.  That is the
    statistic read (``(d0,d1,d2) -> (d1, 0)``), the splat (``(d0,d1) -> (d0, 0)``)
    and the row broadcast (``(d0,d1,d2) -> (d0, 0, d2)``) -- three forms, one rule.

    Axes are matched by ``_dim_info`` classification, not by expression equality,
    so an outer-stick ``sym // 64`` matches an outer-stick ``sym // 64`` whatever
    it is spelled as, exactly as ``align_reshape_plan`` matches them.

    Refuses, rather than guessing, three shapes that would otherwise read the
    wrong elements silently:

    * an operand axis matching no output axis -- there is no dim for it to walk;
    * a matched axis whose extent differs from the output's -- that is a stretch
      or a shrink, and a map cannot say either;
    * matched axes out of increasing order -- a permutation, which needs a
      transpose (restickify) rather than a map, the same refusal
      ``reduction_indexing`` makes.

    ``in_extent``/``out_extent`` are the TILE extents, not the buffer's: a
    ``linalg`` operand's shape is the tile that was loaded, and a broadcast
    operand is loaded at one element on the axis it does not walk.
    """
    out_info = [_dim_info(coord) for coord in out_coords]
    row: list[int | None] = []
    matched: list[int] = []
    used: set[int] = set()
    for axis, coord in enumerate(in_coords):
        info = _dim_info(coord)
        size = int(in_extent[axis])
        if info == (_DIM_CONST, None):
            if size != 1:
                raise NotImplementedError(
                    f"OpSpec alignment: operand device axis {axis} is a constant "
                    f"coordinate over {size} elements, so it walks no iteration "
                    "dim and yet carries more than the one element a constant map "
                    "position reads"
                )
            row.append(None)
            continue
        candidates = [o for o in range(len(out_coords)) if o not in used]
        found = next((o for o in candidates if out_info[o] == info), None)
        if found is None:
            raise NotImplementedError(
                f"OpSpec alignment: operand device axis {axis} ({coord!r}) matches "
                "no output device axis, so there is no iteration dim for it to "
                "walk; aligning it needs a transpose (restickify)"
            )
        if int(out_extent[found]) != size:
            raise NotImplementedError(
                f"OpSpec alignment: operand device axis {axis} runs {size} "
                f"elements against the output axis {found}'s "
                f"{int(out_extent[found])}; a broadcast map reads one element or "
                "the whole axis, not a stretch of it"
            )
        used.add(found)
        matched.append(found)
        row.append(found)
    if any(matched[i] >= matched[i + 1] for i in range(len(matched) - 1)):
        raise NotImplementedError(
            "OpSpec alignment: the operand's device axes are permuted against the "
            "output's, which needs a transpose (restickify) rather than a map"
        )
    return tuple(row)


def placeholder_axes(
    coords: Sequence[sympy.Expr], extent: Sequence[int]
) -> tuple[int, ...]:
    """Output axes standing in for something the op does not write.

    A projection keeps an axis the op does not produce in the output's
    ``device_size`` as a unit extent at a constant coordinate, so the output is
    the same rank as the input even though it carries less.  A consumer wants
    those axes gone: the accepted KTIR form stores a rank-2 tile into a rank-2
    view, and keeping them would demand a reshape between the compute op and the
    store.

    **Unary on purpose** -- it asks nothing about the inputs.  A constant
    coordinate alone does not identify a placeholder: an on-stick reduction's
    output carries *two* constant coordinates, one a placeholder and one the
    broadcast lane the store really walks, and only the extent separates them.
    Folding this into a relational helper is what made the on-stick shape reach a
    coordinate-matching refusal before anyone could ask the unary question.
    """
    return tuple(
        axis
        for axis, coord in enumerate(coords)
        if _dim_info(coord)[0] == _DIM_CONST and int(extent[axis]) == 1
    )


# The two ``linalg`` iterator names, as ``iterator_types`` spells them.  Produced
# here because ``reduction_indexing`` is what decides which dim is which; a
# consumer that builds a ``linalg.generic`` passes them through unchanged.
PARALLEL = "parallel"
REDUCTION = "reduction"


def reduction_indexing(
    in_coords: Sequence[sympy.Expr],
    in_extent: Sequence[int],
    out_coords: Sequence[sympy.Expr],
    out_extent: Sequence[int],
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    """``(iters, in_map, out_map)`` for one unary reduction over a squeezed output.

    The iteration nest a reduction wants, stated as the three things a
    ``linalg.generic`` would have to say and a ``linalg.reduce`` says implicitly:
    one iterator per iteration dim, and per operand one dim index per result
    position (a *projection*, so ``(0, 1, 2)`` is ``(d0..d3) -> (d0,d1,d2)``).
    Plain tuples rather than dialect types, so this module keeps knowing nothing
    about MLIR.

    An iteration dim is a distinct ``(kind, sym)`` classification of a device
    coordinate (see ``_dim_info``), numbered in the order the *input* axes
    introduce them.  Two things follow that a set-complement over kept axes
    cannot express:

    * an output axis whose coordinate is a bare constant of extent > 1 is not a
      placeholder but a **broadcast lane** -- a real iteration dim the store walks
      that the input does not have -- so it gets a fresh dim rather than a
      refusal.  That is the whole on-stick reduction, and the axis a reduction
      reduces on the way in can therefore be kept on the way out.
    * an axis can be both, which is why the answer is per-operand maps and not a
      flat list of reduced axes.

    ``out_coords``/``out_extent`` must already have their placeholder axes dropped
    (``placeholder_axes``): an extent-1 constant would otherwise be
    indistinguishable from a degenerate broadcast lane.

    Refuses three shapes, each of which would otherwise read the wrong elements
    silently: a kept axis that matches no input axis or matches one out of
    increasing order (that is a transpose, which needs a restickify); a kept axis
    whose extent changed; and a nest in which nothing reduces at all.  The last
    is not implied by the caller having labelled the spec a reduction -- that says
    what was *asked for*, this reads what the coordinates *are* -- and without it
    an all-parallel nest reaches ``dimensions = []``, which does not build.
    """
    dims: list[tuple[str, sympy.Symbol | None]] = []
    extents: list[int] = []
    in_map: list[int] = []
    for axis, coord in enumerate(in_coords):
        info = _dim_info(coord)
        size = int(in_extent[axis])
        if info in dims:
            # A repeated classification is one dim walked twice, so the two axes
            # must agree about how far it runs.
            dim = dims.index(info)
            if extents[dim] != size:
                raise NotImplementedError(
                    f"OpSpec reduction: input device axes {in_map.index(dim)} and "
                    f"{axis} are both {coord!r} but run {extents[dim]} and {size} "
                    "elements; one iteration dim cannot have two ranges"
                )
        else:
            dims.append(info)
            extents.append(size)
            dim = len(dims) - 1
        in_map.append(dim)

    # Only the dims the input introduced are matchable; anything past this is a
    # broadcast lane allocated below, and a second one must not match the first.
    from_input = len(dims)
    out_map: list[int] = []
    matched: list[int] = []  # the input dims kept, in output order
    for axis, coord in enumerate(out_coords):
        info = _dim_info(coord)
        size = int(out_extent[axis])
        if info in dims[:from_input]:
            dim = dims.index(info)
            if extents[dim] != size:
                raise NotImplementedError(
                    f"OpSpec reduction: kept output axis {axis} has extent {size} "
                    f"but its input axis has extent {extents[dim]}; a reduction "
                    "does not resize an axis it keeps"
                )
            matched.append(dim)
        elif info == (_DIM_CONST, None):
            # The broadcast lane: a real iteration dim with no input axis behind
            # it.  ``placeholder_axes`` has already taken the extent-1 constants,
            # so what is left here carries elements.
            dims.append(info)
            extents.append(size)
            dim = len(dims) - 1
        else:
            raise NotImplementedError(
                f"OpSpec reduction: output device axis {axis} ({coord!r}) matches "
                "no input axis; the surviving axes are permuted, which needs a "
                "transpose rather than a reduction"
            )
        out_map.append(dim)

    # A pure reduction preserves the row-major order of the axes it keeps.
    if any(matched[i] >= matched[i + 1] for i in range(len(matched) - 1)):
        raise NotImplementedError(
            "OpSpec reduction: the surviving axes are permuted, which needs a "
            "transpose rather than a reduction"
        )
    reduced = set(range(len(dims))) - set(out_map)
    if not reduced:
        raise NotImplementedError(
            "OpSpec reduction: the output carries every input device axis, so "
            "there is no axis to reduce"
        )
    iters = tuple(REDUCTION if dim in reduced else PARALLEL for dim in range(len(dims)))
    return iters, tuple(in_map), tuple(out_map)


# ---------------------------------------------------------------------------
# Work division: the core grid, as the iteration space states it
# ---------------------------------------------------------------------------


def core_divisions(
    iteration_space: dict[sympy.Symbol, tuple[sympy.Expr, int]],
) -> tuple[list[tuple[sympy.Symbol, int, int]], int]:
    """``(divisions, total_cores)`` for one iteration space.

    ``OpSpec.iteration_space`` maps each symbol to ``(range, work_division)``,
    where the division is the number of cores that symbol's range is split
    across -- decided upstream by ``work_division.py`` from ``config.sencores``.
    The core grid is one flat index in ``[0, total_cores)``, read as a mixed-radix
    number over the divided symbols, so each division is returned as
    ``(sym, div, inner)`` **innermost-first**: that symbol's portion of the grid
    is ``(core_id // inner) % div``.

    ``total_cores`` is the product of the divisions, so an undivided space gives
    ``([], 1)`` -- a single-core kernel, stated by the contract rather than
    assumed.
    """
    split = [
        (sym, int(div)) for sym, (_range, div) in iteration_space.items() if div > 1
    ]
    total_cores = 1
    for _sym, div in split:
        total_cores *= div
    divisions: list[tuple[sympy.Symbol, int, int]] = []
    inner = 1
    for sym, div in reversed(split):  # innermost first
        divisions.append((sym, div, inner))
        inner *= div
    return divisions, total_cores


def per_core_extent(
    arg: TensorArg, divisors: dict[sympy.Symbol, int]
) -> tuple[list[int], list[sympy.Symbol | None]]:
    """``(extent, symbol)`` per device axis: one core's share, and what divides it.

    Each device axis carries at most one iteration symbol -- a bare ``c_i`` or an
    outer-stick ``c_i // stick`` -- so at most one divisor applies to it, and the
    axis' per-core extent is its full extent divided by that divisor.  The
    trailing axis is the within-stick one and is never divided: a stick is the
    unit of transfer, so splitting it across cores would split a stick.

    The returned symbol list says which division each axis follows (``None`` for
    an axis no division touches), which is what turns a division into a step
    along that axis.
    """
    extent = [int(s) for s in arg.device_size]
    coords = list(arg.device_coordinates)
    if divisors and len(coords) != len(extent):
        raise NotImplementedError(
            f"OpSpec work division: {arg.name!r} carries {len(coords)} device "
            f"coordinate(s) for {len(extent)} device axes, so which axis a "
            "divided symbol walks cannot be told"
        )
    per_core: list[int] = []
    symbols: list[sympy.Symbol | None] = []
    last = len(extent) - 1
    for axis, size in enumerate(extent):
        symbol = None
        if divisors and axis != last:
            kind, symbol = _dim_info(coords[axis])
            if kind == _DIM_CONST:
                symbol = None
        divisor = divisors.get(symbol, 1) if symbol is not None else 1
        if divisor > 1 and size % divisor:
            raise NotImplementedError(
                f"OpSpec work division: device axis {axis} of {arg.name!r} has "
                f"extent {size}, which {divisor} cores do not divide evenly; a "
                "ragged per-core tile is not supported"
            )
        per_core.append(size // divisor)
        symbols.append(symbol if divisor > 1 else None)
    return per_core, symbols


def _device_block_shape(
    arg: TensorArg,
    divisor_of: dict[sympy.Symbol, int],
    loop_ctx: _LoopCtx | None,
) -> list[int]:
    """Per-core ``block_shape`` for the access tile.

    Divides each non-stick device dim by the core divisor (``divisor_of``,
    this group's iteration-space work divisions) of the single OpSpec symbol
    on that dim's coordinate.  Each non-stick device dim carries exactly one
    iteration symbol -- a bare ``c_i`` or an outer-stick ``c_i // stick`` --
    so exactly one divisor applies per dim; a constant (broadcast) axis
    carries none and is left at full size.  The last device dim is the
    within-stick dim: always the full ``device_size[-1]`` (64 fp16 / 32 fp32 /
    128 int8), never divided across cores.

    In a counted loop, a full-size operand's ``device_size`` on the tiled dim
    spans the whole tensor (``count`` tiles), but each iteration loads only
    one tile, so that dim is first divided by ``count``.  An operand that does
    *not* advance per iteration -- a ``per_tile_fixed`` scratch tile or a
    register-threaded intermediate whose ``device_size`` is already per-tile --
    carries no ``device_tile_advance_expr`` and holds one tile already, so it is
    left alone.  ``device_tile_advance_expr is not None`` is the precise signal
    (post-WSR ``tiled_symbols`` name minted symbols, so ``per_tile_fixed`` alone
    no longer distinguishes a full-size operand from a per-tile pool).

    TODO(consolidate): the per-core divisor math here overlaps
    ``per_core_extent`` above; a later cleanup could delegate the divisor step
    to it, leaving this helper to add only the counted-loop ``loop_ctx`` layer.
    Kept standalone in the forward-port to preserve the generator's exact
    (test-verified) behavior.
    """
    device_size = [int(s) for s in arg.device_size]
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
        syms = coord.free_symbols
        if len(syms) > 1:
            raise NotImplementedError(
                f"OpSpec: device dim {k} coordinate {coord!r} folds multiple "
                f"iteration symbols {syms}; work-division of a multi-symbol "
                "axis is not supported"
            )
        divisor = divisor_of.get(next(iter(syms)), 1) if syms else 1
        block.append(max(1, size // max(1, divisor)))
    return block


# ---------------------------------------------------------------------------
# Counted-loop (``LoopSpec``) helpers
# ---------------------------------------------------------------------------
#
# A coarse-tiled scheduler node arrives as a single ``[LoopSpec(count, body)]``.
# These helpers hold the pure arithmetic a backend needs to wrap that body in a
# counted loop: the ``_LoopCtx`` emission context and the recovery of each real
# tiled iteration symbol and its per-tile stride from the body ops'
# ``device_tile_advance_expr``.  The pointwise path passes ``loop_ctx=None`` to
# ``_device_block_shape`` and never calls anything here.


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
            "OpSpec: coarse-tile advance spans more than one device "
            f"axis (coeff={coeff}, strides={strides}); not supported yet"
        )
    axis, offset = hits[0]
    coord = coords[axis]
    if not isinstance(coord, sympy.Symbol):
        raise NotImplementedError(
            "OpSpec: coarse-tile advance lands on a non-bare device "
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
            strides = row_major_strides(list(arg.device_size))
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


# ---------------------------------------------------------------------------
# Reduction helpers
# ---------------------------------------------------------------------------
#
# The pure arithmetic a *reduction* OpSpec implies: which iteration symbols are
# reduced, which input device axes carry them, the outer-stick subset Spyre
# actually reduces, and whether the reduced tile reshapes to the output block
# without a permute.


def _reduction_axes(in_arg: TensorArg, out_arg: TensorArg) -> tuple[set, list[int]]:
    """Reduced symbols and the input device axes that carry them.

    A reduction collapses one iteration-space symbol (e.g. ``torch.sum``'s
    reduced dim): it appears in the input's ``device_coordinates`` but not in
    the output's (the user-confirmed rule -- see ``sum`` SDSC artifacts).  The
    reduced symbols are therefore ``input_free_syms - output_free_syms``; the
    axes are the input device dimensions whose coordinate references one.

    A non-stick reduction (``dim=0`` on ``(128, 256)``) puts the reduced
    symbol on exactly one input axis -> a single-axis reduce.  A stick-dim
    reduction (``dim=1``) spreads it across the outer-stick and within-stick
    axes (two axes) -> not yet supported (needs a ``sum_stick`` primitive).

    TODO(consolidate): overlaps main's public ``reduced_axes`` (which returns
    reduced / placeholder *axis indices* via ``_dim_info`` coordinate matching);
    this returns the reduced-*symbol* set that ``_outer_stick_reduce_axes``
    needs, so it is kept distinct for the forward-port.
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


def _outer_stick_reduce_axes(
    in_arg: TensorArg, out_arg: TensorArg
) -> tuple[set, list[int], list[int]]:
    """Reduced symbols, all reduced input axes, and the *outer-stick* subset.

    A stick-dim reduction spreads the reduced symbol across two input device
    axes: the outer-stick axis (``coord = sym // stick``, a ``FloorDiv`` or a
    bare symbol) and the within-stick axis (``coord = sym % stick``, a
    ``Mod`` / ``ModularIndexing``).  Spyre reduces only the outer-stick axis;
    the backend implicitly reduces the within-stick (NE) dimension in hardware
    (see ``2602``).

    Returns ``(reduced_syms, all_axes, outer_stick_axes)`` where
    ``outer_stick_axes`` excludes any ``Mod`` / ``ModularIndexing`` axis.  A
    pure within-stick reduction yields an empty ``outer_stick_axes``.
    """
    reduced, axes = _reduction_axes(in_arg, out_arg)
    outer = [
        k
        for k in axes
        if not isinstance(in_arg.device_coordinates[k], (sympy.Mod, ModularIndexing))
    ]
    return reduced, axes, outer


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

    The innermost (within-stick) axis is *excluded* from the comparison: under
    the temporary outer-stick-only reduction the reduced input tile still
    carries the real within-stick coordinate (``Mod(c1, 64)``) while the
    reduction output broadcasts a single value across that axis (coordinate
    ``0``, ``stride -1``).  Those coordinates differ by construction; the axis
    is the same position and size on both sides, so the reshape stays order
    preserving.
    """
    n_in = len(in_arg.device_coordinates)
    surviving = [
        str(coord)
        for k, (coord, size) in enumerate(zip(in_arg.device_coordinates, in_block))
        if k != axis and size != 1 and k != n_in - 1
    ]
    n_out = len(out.device_coordinates)
    produced_out = [
        str(coord)
        for k, (coord, size) in enumerate(zip(out.device_coordinates, out_block))
        if size != 1 and k != n_out - 1
    ]
    if surviving != produced_out:
        raise NotImplementedError(
            "OpSpec: reduction output layout requires a permute "
            f"({surviving} -> {produced_out}); permute not supported yet"
        )


# ---------------------------------------------------------------------------
# Matmul / bmm helpers
# ---------------------------------------------------------------------------
#
# The device-axis permutation that places a sticked matrix dim's stick pair
# innermost so a backend can collapse it back into a matrix for its dot / matmul
# primitive.


def _matmul_operand_permutation(
    device_coords: list, batch_sym: sympy.Symbol | None = None
) -> list[int]:
    """Permutation placing the sticked matrix dim's stick pair innermost.

    A Spyre matmul operand's non-leading matrix dim (K for A, N for B) is
    stored as a stick split: an outer-stick dim (``FloorDiv(sym, stick)``) and
    the within-stick dim (``Mod(sym, stick)``), the latter always the last
    device dim.  ``_emit_matmul`` collapses the two innermost dims back into
    that matrix dim, so the outer-stick and within-stick dims must be adjacent
    and innermost, with the remaining dims (the leading matrix dim(s) --
    batch/M for A, batch/K for B) kept ahead of them.  When a batch dim is
    present (bmm) it must lead so the block reshapes to a batched matrix
    ``[B, M, K]`` / ``[B, K, N]`` for a batched matmul.

    Anchoring on the stick pair -- the two dims that share the within-stick
    dim's iteration symbol -- keeps this correct even when the row dim M is
    size 1 and its coordinate degenerates to a constant ``0`` (the decode-phase
    / GEMV case).  For a non-degenerate operand it yields the natural order, so
    a canonical row-major operand is left unchanged.
    """
    rank = len(device_coords)
    if rank < 3:
        return list(range(rank))  # already a (batched) matrix; nothing to move
    within_stick = rank - 1  # within-stick dim is always the last device dim
    within_stick_syms = device_coords[within_stick].free_symbols
    outer_stick = None
    if within_stick_syms:
        outer_stick = next(
            (
                k
                for k in range(rank - 1)
                if device_coords[k].free_symbols & within_stick_syms
            ),
            None,
        )
    if outer_stick is None:
        return list(range(rank))  # not stick-split; leave as-is
    leading = [k for k in range(rank) if k not in (outer_stick, within_stick)]
    # A batched-matmul operand must lead with its batch dim.
    if batch_sym is not None:
        b = next((k for k in leading if device_coords[k] == batch_sym), None)
        if b is not None:
            leading = [b] + [k for k in leading if k != b]
    return leading + [outer_stick, within_stick]


# ---------------------------------------------------------------------------
# Gather (``aten.index``) helpers
# ---------------------------------------------------------------------------
#
# Recognizing an indirect device coordinate and identifying the index / value /
# output operands and the gathered / row axes.


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
            "OpSpec: gather must have exactly two inputs (index + value) "
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
                "OpSpec: gather supports exactly one indirect axis on one "
                "input (found more than one)"
            )
        value_arg, k_star = a, indirect_dims[0]
    if value_arg is None:
        raise NotImplementedError("OpSpec: gather has no indirect input")
    index_arg = next(a for a in inputs if a is not value_arg)

    # The gathered output-row axis is the index buffer's single iteration symbol.
    row_syms: set = set()
    for c in index_arg.device_coordinates:
        row_syms |= c.free_symbols
    if len(row_syms) != 1:
        raise NotImplementedError(
            "OpSpec: gather index buffer must have exactly one iteration "
            f"symbol (got {sorted(map(str, row_syms))})"
        )
    row_sym = next(iter(row_syms))
    row_dims = [
        k for k, c in enumerate(out.device_coordinates) if row_sym in c.free_symbols
    ]
    if len(row_dims) != 1:
        raise NotImplementedError(
            "OpSpec: gather output must carry the row symbol "
            f"{row_sym} on exactly one device dim (got dims {row_dims})"
        )
    return index_arg, value_arg, out, k_star, row_dims[0]


# ---------------------------------------------------------------------------
# Restickify (cross-stick transpose) helpers
# ---------------------------------------------------------------------------
#
# Recognizing the op, classifying its device axes, identifying the in/out
# within-stick symbols, and computing the reshape/permute/reshape plan that
# turns the input tile into the output tile.


def _is_restickify_spec(spec: OpSpec) -> bool:
    """True if ``spec`` is a cross-stick restickify (a transpose copy).

    ``SpyreKernel.store`` labels a pointwise copy whose within-stick (last)
    device axis changes iteration symbol -- a transpose that moves which logical
    dim is sticked -- as ``RESTICKIFY_OP``; a same-stick copy stays ``identity``.
    """
    return getattr(spec, "op", None) == RESTICKIFY_OP


def _is_axis_permute_copy(spec: OpSpec) -> bool:
    """True if ``spec`` is a single-input copy whose device axes are a pure
    *permutation* of the input's (same within-stick axis, same atom set, a
    different order) -- e.g. the outer-axis transpose
    ``enforce_indirect_access_layout`` inserts to put a gather's indexed dim
    outermost in the value tensor's device layout.

    ``SpyreKernel.store`` labels such a copy ``IDENTITY_OP`` (its within-stick
    axis symbol is unchanged, so ``_is_restickify_spec`` does not match it), yet
    it physically moves data across device axes and must be emitted as a
    reshape/permute (``_restickify_plan``), not a plain strided pointwise copy.
    A true no-op copy (identical device-axis order), a broadcast, and a
    reduction all return False and stay on the pointwise path.
    """
    if getattr(spec, "is_reduction", False):
        return False
    inputs = [a for a in spec.args if a.is_input]
    outputs = [a for a in spec.args if not a.is_input]
    if len(inputs) != 1 or len(outputs) != 1:
        return False
    in_coords = list(inputs[0].device_coordinates)
    out_coords = list(outputs[0].device_coordinates)
    if len(in_coords) != len(out_coords):
        return False
    try:
        in_roles = [_restickify_axis_role(c) for c in in_coords]
        out_roles = [_restickify_axis_role(c) for c in out_coords]
    except NotImplementedError:
        # A broadcast / multi-symbol axis is not a bijective permute copy.
        return False
    # A changed within-stick (last) axis is a genuine cross-stick move that is
    # already ``RESTICKIFY_OP`` (handled by ``_is_restickify_spec``); here we
    # only claim the same-stick outer-axis permutation.
    if in_roles[-1] != out_roles[-1]:
        return False
    return in_roles != out_roles and sorted(map(str, in_roles)) == sorted(
        map(str, out_roles)
    )


def _restickify_stick_symbol(arg: TensorArg) -> sympy.Symbol:
    """The single iteration symbol on ``arg``'s within-stick (last) device axis."""
    syms = arg.device_coordinates[-1].free_symbols
    if len(syms) != 1:
        raise NotImplementedError(
            "OpSpec: restickify within-stick axis must carry exactly one "
            f"iteration symbol (got {sorted(map(str, syms))})"
        )
    return next(iter(syms))


def _restickify_axis_role(coord: sympy.Expr) -> tuple[sympy.Symbol, str]:
    """Classify a restickify device axis coordinate as ``(symbol, role)``.

    ``role`` is one of ``full`` (a bare symbol ``s``), ``lo`` (the within-stick
    axis ``Mod(s, stick)``), or ``hi`` (the outer-stick axis ``floor(s/stick)``).
    Raises for a constant (broadcast) or multi-symbol axis -- outside the
    bijection this cut supports.
    """
    syms = coord.free_symbols
    if len(syms) != 1:
        raise NotImplementedError(
            "OpSpec: restickify device axis must carry exactly one symbol "
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
            "OpSpec: restickify must have exactly one input and one output "
            f"(got {len(inputs)} inputs, {len(outputs)} outputs)"
        )
    in_arg, out = inputs[0], outputs[0]
    for c in list(in_arg.device_coordinates) + list(out.device_coordinates):
        _restickify_axis_role(c)  # raises on broadcast / multi-symbol axes
    s_in = _restickify_stick_symbol(in_arg)
    s_out = _restickify_stick_symbol(out)
    if s_in == s_out:
        raise NotImplementedError(
            f"OpSpec: restickify within-stick symbol unchanged ({s_in}); "
            "expected a cross-stick restickify"
        )
    for s in (s_in, s_out):
        if int(spec.iteration_space.get(s, (0, 1))[1]) != 1:
            raise NotImplementedError(
                "OpSpec: restickify with a work-divided within-stick "
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
            "OpSpec: restickify with differing in/out stick sizes "
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
            "OpSpec: restickify input/output atoms do not match "
            f"({sorted((str(s), p) for s, p, _ in in_atoms)} vs "
            f"{sorted((str(s), p) for s, p, _ in out_atoms)})"
        )
    permute: list[int] = []
    for s, part, size in out_atoms:
        i, in_size = in_index[(s, part)]
        if in_size != size:
            raise NotImplementedError(
                "OpSpec: restickify atom size mismatch for "
                f"{(str(s), part)} ({in_size} vs {size}); likely a work-divided "
                "stick axis. Retry with fewer SENCORES."
            )
        permute.append(i)
    reshape1 = [size for _, _, size in in_atoms]
    reshape2 = [int(b) for b in out_block]
    return reshape1, permute, reshape2
