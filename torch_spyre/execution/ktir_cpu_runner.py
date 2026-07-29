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

"""Device-free execution of emitted KTIR via ``../ktir-cpu``.

``SpyreTritonAsyncCompile.triton()`` compiles a Triton kernel to KTIR but has no
Spyre device to launch it on. Instead of returning ``None`` (which makes the
generated wrapper's ``kernel.run(...)`` fail with ``'NoneType' has no attribute
'run'``), it returns a :class:`KtirCpuRunner`. The runner loads the KTIR text
into ``ktir_cpu.KTIRInterpreter`` and, on each ``.run(...)`` call, maps the
wrapper's positional arguments onto the KTIR function's declared parameters,
executes on the NumPy interpreter, and writes results back into the caller's
buffers.

``ktir_cpu`` is imported lazily (only when a kernel actually runs) so importing
this module never requires the interpreter to be installed.

This module also holds the host-side **physical-layout (stickification)**
helpers, since they exist only to feed this device-free path:

- ``ktir_empty_with_layout`` -- allocate a zeroed physical-layout tensor (the
  device-free stand-in for ``spyre_empty_with_layout``); allocating an empty
  buffer in physical shape is the trivial (data-free) case of stickification.
- ``ktir_stickify`` -- logical PyTorch tensor -> physical-layout tensor (reorder +
  pad), for real kernel inputs.
- ``ktir_destickify`` -- physical-layout tensor -> logical PyTorch tensor, for the
  values returned from the compiled function.

Layout contract: every tensor argument reaching ``.run()`` is already in
**physical** (sticked) layout -- the wrapper allocates buffers via
``ktir_empty_with_layout`` and stickifies real inputs, and destickifies the
returned outputs. So :class:`KtirCpuRunner` only bridges representation (physical
``torch.Tensor`` <-> physical ``np.ndarray``, since ktir-cpu speaks NumPy); it does
no layout conversion.

Physical layout is STANDARD element arrangement (no intra-stick permutation).
The reorder is governed by the identity
``logical_offset = sum_i device_coord[i] * stride_map[i]`` with ``stride_map[i]``
of ``-1`` (synthetic/pad) and ``0`` (broadcast) contributing 0. Only the split
"stick" dimension introduces padding; positions past the logical extent are
left zero.

Shared execution helper: this lives under ``torch_spyre/execution/`` (next to the
SDSC runner) rather than the Triton-only package, because the planned
OpSpec->KTIR emitter (``generate_ktir`` / ``async_compile.ktir``) reuses the same
device-free interpreter path and layout helpers.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np
import torch

from torch_spyre._inductor.logging_utils import get_inductor_logger

logger = get_inductor_logger("ktir_cpu_runner")

__all__ = [
    "KtirCpuRunner",
    "ktir_empty_with_layout",
    "ktir_stickify",
    "ktir_destickify",
]


# ---------------------------------------------------------------------------
# Physical-layout (stickification) helpers
# ---------------------------------------------------------------------------


def _row_major_strides(size: Sequence[int]) -> list[int]:
    strides = [1] * len(size)
    for k in range(len(size) - 2, -1, -1):
        strides[k] = strides[k + 1] * int(size[k + 1])
    return strides


def _gather_index_and_mask(
    logical_size: Sequence[int],
    device_size: Sequence[int],
    stride_map: Sequence[int],
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Map each physical position to a flat logical offset, plus a padding mask.

    Returns ``(idx, mask)`` both shaped ``device_size``: ``idx[p]`` is the flat
    row-major offset into the logical tensor that physical position ``p`` holds;
    ``mask[p]`` is False for padding positions (past the logical extent of the
    split stick dimension), or ``None`` when there is no padding.
    """
    device_size = [int(d) for d in device_size]
    stride_map = [int(s) for s in stride_map]
    rank = len(device_size)
    eps = device_size[-1]

    idx = np.zeros(device_size, dtype=np.int64)
    for i, sm in enumerate(stride_map):
        if sm > 0:  # -1 (synthetic/pad) and 0 (broadcast) contribute nothing
            shape = [device_size[i] if d == i else 1 for d in range(rank)]
            idx = idx + np.arange(device_size[i], dtype=np.int64).reshape(shape) * sm

    # Padding arises only from the split stick dim: the within-stick dim (last)
    # and an outer dim carry the same logical dim, and the logical extent may not
    # be a multiple of eps. Identify them via the stride_map and mask the tail.
    mask: Optional[np.ndarray] = None
    base = stride_map[-1]
    if base > 0:
        host_strides = _row_major_strides(logical_size)
        sticked = next((k for k, hs in enumerate(host_strides) if hs == base), None)
        # Padding exists only when the sticked logical dim does not fill whole
        # sticks (its extent is not a multiple of the stick width ``eps``). When
        # it divides evenly every stick is full, so there is no tail to mask --
        # regardless of any stride coincidence with an unrelated dim.
        if sticked is not None and int(logical_size[sticked]) % eps != 0:
            extent = int(logical_size[sticked])
            num_sticks = -(-extent // eps)  # ceil(extent / eps)
            # The outer-stick dim carries stride ``eps * base`` AND has exactly
            # ``num_sticks`` entries. Matching both disambiguates it from an
            # unrelated dim that merely shares the stride -- e.g. a leading head
            # dim whose logical stride equals ``eps`` when ``head_dim == eps``,
            # which would otherwise be mistaken for the outer-stick dim and
            # wrongly zero out all but its first slice.
            outer = next(
                (
                    i
                    for i, sm in enumerate(stride_map)
                    if sm == eps * base and int(device_size[i]) == num_sticks
                ),
                None,
            )
            if outer is not None:
                outer_ax = np.arange(device_size[outer], dtype=np.int64).reshape(
                    [device_size[outer] if d == outer else 1 for d in range(rank)]
                )
                within_ax = np.arange(eps, dtype=np.int64).reshape(
                    [eps if d == rank - 1 else 1 for d in range(rank)]
                )
                mask = np.broadcast_to(
                    (outer_ax * eps + within_ax) < extent, device_size
                )
    return idx, mask


def ktir_empty_with_layout(
    size: Sequence[int],
    stride: Sequence[int],
    dtype: torch.dtype,
    device_layout: Any,
) -> torch.Tensor:
    """Allocate a zeroed physical-layout tensor (drop-in for
    ``spyre_empty_with_layout``).

    ``size`` / ``stride`` are the logical shape / stride (unused for allocation);
    the physical shape is ``device_layout.device_size``. Returns a CPU
    ``torch.Tensor`` (not a Spyre-device tensor) so the device-free path can
    operate on it; the runner converts it to NumPy for ktir-cpu.
    """
    device_size = [int(d) for d in device_layout.device_size]
    if not device_size:
        raise ValueError("ktir_empty_with_layout: device_layout has empty device_size")
    return torch.zeros(device_size, dtype=dtype)


def ktir_stickify(logical: torch.Tensor, device_layout: Any) -> torch.Tensor:
    """Convert a logical PyTorch tensor to its physical (sticked) layout tensor."""
    device_size = [int(d) for d in device_layout.device_size]
    stride_map = [int(s) for s in device_layout.stride_map]
    idx, mask = _gather_index_and_mask(list(logical.shape), device_size, stride_map)

    src = logical.detach().cpu().contiguous().numpy().reshape(-1)
    idx_flat = idx.reshape(-1)
    if mask is not None:
        m = mask.reshape(-1)
        safe = np.where(m, idx_flat, 0)
        phys = np.where(m, src[safe], 0)
    else:
        phys = src[idx_flat]
    phys = np.ascontiguousarray(phys.reshape(device_size))
    logger.debug(
        "ktir_stickify: logical %s -> physical %s",
        tuple(logical.shape),
        tuple(device_size),
    )
    return torch.from_numpy(phys).to(logical.dtype)


def ktir_destickify(
    physical: torch.Tensor,
    logical_size: Sequence[int],
    device_layout: Any,
) -> torch.Tensor:
    """Convert a physical (sticked) layout tensor back to a logical PyTorch tensor."""
    device_size = [int(d) for d in device_layout.device_size]
    stride_map = [int(s) for s in device_layout.stride_map]
    logical_size = [int(d) for d in logical_size]
    idx, mask = _gather_index_and_mask(logical_size, device_size, stride_map)

    phys = physical.detach().cpu().contiguous().numpy().reshape(-1)
    out = np.zeros(math.prod(logical_size) if logical_size else 1, dtype=phys.dtype)
    idx_flat = idx.reshape(-1)
    if mask is not None:
        m = mask.reshape(-1)
        out[idx_flat[m]] = phys[m]
    else:
        out[idx_flat] = phys
    out = np.ascontiguousarray(out.reshape(logical_size))
    logger.debug(
        "ktir_destickify: physical %s -> logical %s",
        tuple(physical.shape),
        tuple(logical_size),
    )
    return torch.from_numpy(out).to(physical.dtype)


# ---------------------------------------------------------------------------
# Device-free KTIR runner
# ---------------------------------------------------------------------------


class KtirCpuRunner:
    """Runs one emitted KTIR function on ``ktir_cpu`` (device-free).

    Args:
        kernel_name: The KTIR function name (matches the ``@name`` in the KTIR
            module and the wrapper's kernel object name).
        ktir_text: The textual KTIR (KTDP-dialect MLIR) module.
    """

    def __init__(self, kernel_name: str, ktir_text: str) -> None:
        self.kernel_name = kernel_name
        self.ktir_text = ktir_text
        self._interp: Any = None
        self._arg_names: list[str] | None = None

    def _ensure_loaded(self) -> Any:
        if self._interp is None:
            from ktir_cpu import KTIRInterpreter
            from ktir_cpu.mlir_frontend.parser import MLIRFrontendParser

            # Use the mlir_ktdp-backed frontend parser (the same dialect the
            # backend emits with), not ktir-cpu's regex parser: the regex parser
            # mis-parses per-argument ``loc(...)`` attributes and reports only
            # the first function parameter.
            interp = KTIRInterpreter(parser=MLIRFrontendParser())
            interp.load(self.ktir_text)
            self._interp = interp
            self._arg_names = list(interp.arg_names(self.kernel_name))
            logger.debug(
                "KtirCpuRunner[%s]: loaded KTIR; args=%s",
                self.kernel_name,
                self._arg_names,
            )
        return self._interp

    def run(self, *args: Any, grid: Any = None, stream: Any = None, **kwargs: Any):
        """Execute the KTIR function.

        Positional ``args`` are the runtime kernel arguments in KTIR parameter
        order (pointer tensors followed by scalar sizes, as the wrapper emits
        them). ``grid`` / ``stream`` and any other kwargs are accepted and
        ignored (the grid is baked into the KTIR at compile time).
        """
        interp = self._ensure_loaded()
        names = self._arg_names or []
        # The wrapper passes only runtime args; a KTIR function may still declare
        # trailing constexpr params (e.g. BLOCK_SIZE) that ktir-cpu resolves from
        # the module. Bind the provided args to the leading params; too many args
        # is an error. (Our emitted KTIR has no constexpr params, so counts match.)
        if len(args) > len(names):
            raise RuntimeError(
                f"KtirCpuRunner[{self.kernel_name}]: KTIR declares {len(names)} "
                f"args {names} but .run() received {len(args)}"
            )

        call_kwargs: dict[str, Any] = {}
        # name -> original argument object, for writing outputs back in place.
        buffers: dict[str, Any] = {}
        for name, value in zip(names, args):
            if isinstance(value, torch.Tensor):
                # Already physical layout (stickified by the wrapper); just move
                # the representation to NumPy for ktir-cpu.
                arr = np.ascontiguousarray(value.detach().cpu().numpy())
                call_kwargs[name] = arr
                buffers[name] = value
            elif isinstance(value, np.ndarray):
                call_kwargs[name] = value
                buffers[name] = value
            else:
                # Scalar runtime argument (e.g. a dimension size).
                call_kwargs[name] = value

        outputs = interp.execute_function(self.kernel_name, **call_kwargs)

        # ktir_cpu reads every array argument back out of HBM; copy the ones we
        # were handed back into the caller's buffers so output/mutated tensors
        # reflect the computation.
        for name, orig in buffers.items():
            result = outputs.get(name)
            if result is None:
                continue
            if isinstance(orig, np.ndarray):
                orig[...] = result.reshape(orig.shape)
            else:  # physical torch.Tensor buffer -> write the physical result back
                orig.copy_(
                    torch.from_numpy(np.ascontiguousarray(result)).reshape(orig.shape)
                )
        return None

    # Some call sites invoke the kernel object directly; delegate to run().
    def __call__(self, *args: Any, **kwargs: Any):
        return self.run(*args, **kwargs)
