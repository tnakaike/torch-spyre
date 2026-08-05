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

"""Counted-loop (``LoopSpec``) spec-reading helpers shared by the OpSpec
backends.

A coarse-tiled scheduler node arrives as a single ``[LoopSpec(count, body)]``.
This module holds the pure "decision / arithmetic" a backend needs to wrap that
body in a counted loop: the ``_LoopCtx`` emission context and the recovery of
each real tiled iteration symbol and its per-tile stride from the body ops'
``device_tile_advance_expr``.  Like every sibling here it is free of any
backend emission toolchain (sympy / int over the ``op_specs`` only) so any
OpSpec backend can import it.

Split out of ``opspec_utils.py`` (the pointwise core) so the pointwise-only
path never drags the loop machinery along: the pointwise path passes
``loop_ctx=None`` to ``_device_block_shape`` and never calls anything here.
"""

from __future__ import annotations

import dataclasses

import sympy

from torch_spyre._inductor.codegen.opspec_utils import _row_major_strides
from torch_spyre._inductor.op_spec import OpSpec
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
