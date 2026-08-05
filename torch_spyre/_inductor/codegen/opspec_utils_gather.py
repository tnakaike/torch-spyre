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

"""Gather-specific spec-reading helpers shared by the OpSpec backends.

Op-specific counterpart to ``opspec_utils.py`` (the pointwise core): the pure
"decision / arithmetic" a *gather* (``aten.index``) OpSpec implies --
recognizing an indirect device coordinate and identifying the index / value /
output operands and the gathered / row axes.

Like ``opspec_utils.py`` this module must stay free of any backend emission
toolchain (no MLIR builder, no live Inductor kernel state) so any OpSpec
backend can import it.
"""

from __future__ import annotations

import sympy

from torch_spyre._inductor.op_spec import IndirectAccess, OpSpec, TensorArg


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
