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

"""Reduction-specific spec-reading helpers shared by the OpSpec backends.

Op-specific counterpart to ``opspec_utils.py`` (the pointwise core): the pure
"decision / arithmetic" a *reduction* OpSpec implies -- which iteration symbols
are reduced, which input device axes carry them, the outer-stick subset Spyre
actually reduces, and whether the reduced tile reshapes to the output block
without a permute.

Like ``opspec_utils.py`` this module must stay free of any backend emission
toolchain (no MLIR builder, no live Inductor kernel state) so any OpSpec
backend can import it.
"""

from __future__ import annotations

import sympy
from torch.utils._sympy.functions import ModularIndexing

from torch_spyre._inductor.op_spec import TensorArg


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
