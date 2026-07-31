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

"""Matmul-specific spec-reading helpers shared by the OpSpec backends.

Op-specific counterpart to ``opspec_utils.py`` (the pointwise core): the pure
"decision / arithmetic" a *matmul* / *bmm* OpSpec implies -- specifically the
device-axis permutation that places a sticked matrix dim's stick pair innermost
so the two backends can collapse it back into a matrix for ``tl.dot`` (Triton)
or the KTIR matmul primitive.

Like ``opspec_utils.py`` this module must stay **Triton-free** (no ``tl.*``,
no MLIR builder, no live Inductor kernel state) so both OpSpec backends -- the
Triton source generator and the planned KTIR emitter -- can import it.
"""

from __future__ import annotations

import sympy


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
