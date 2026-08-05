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

"""Restickify-specific spec-reading helpers shared by the OpSpec backends.

Op-specific counterpart to ``opspec_utils.py`` (the pointwise core): the pure
"decision / arithmetic" a *restickify* (cross-stick transpose copy) OpSpec
implies -- recognizing the op, classifying its device axes, identifying the
in/out within-stick symbols, and computing the reshape/permute/reshape plan that
turns the input tile into the output tile.

Like ``opspec_utils.py`` this module must stay free of any backend emission
toolchain (no MLIR builder, no live Inductor kernel state) so any OpSpec
backend can import it.
"""

from __future__ import annotations

import sympy

from torch_spyre._inductor.constants import RESTICKIFY_OP
from torch_spyre._inductor.op_spec import OpSpec, TensorArg


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
