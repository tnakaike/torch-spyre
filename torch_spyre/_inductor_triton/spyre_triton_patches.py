# Copyright 2025 The Torch-Spyre Authors.
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

import sys
from contextlib import contextmanager
from types import ModuleType
from typing import Any

import torch
from torch._inductor import config
from torch._inductor.utils import get_current_backend
from torch._inductor.virtualized import V
from torch.fx.experimental.symbolic_shapes import has_free_unbacked_symbols

from torch_spyre._inductor.lowering import spyre_lowerings

# Spyre mm lowerings use BATCH_MATMUL_OP with a tuple-valued inner_fn, which
# SpyreKernel (SDSC) understands but TritonKernel cannot codegen.  The Triton
# path relies on PyTorch's standard aten.mm lowering so that use_native_matmul
# converts it to tl.dot.  These ops are popped from spyre_lowerings before
# enable_spyre_lowerings() installs them, and restored on context exit.
_TRITON_SKIP_MM_OPS = [
    torch.ops.aten.mm.default,
    torch.ops.aten.bmm.default,
]


def _patched_use_native_matmul(mat1, mat2) -> bool:
    """use_native_matmul with spyre added to supported device types.

    Identical to the upstream function except the device type check includes
    "spyre" alongside "cuda" and "xpu".
    """
    if not config.triton.native_matmul:
        return False
    if (
        config.triton.enable_persistent_tma_matmul
        and torch.utils._triton.has_triton_tma_device()
    ):
        raise AssertionError("native matmul doesn't support tma codegen yet")
    if config.triton.use_block_ptr:
        raise AssertionError("native matmul doesn't support block_ptr codegen yet")
    device_type = mat1.get_device().type
    if not (
        device_type in ("cuda", "xpu", "spyre")
        and get_current_backend(device_type) == "triton"
    ):
        return False
    triton_supported_dtype = [
        torch.int8,
        torch.uint8,
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    if mat1.dtype not in triton_supported_dtype:
        return False
    if mat2.dtype not in triton_supported_dtype:
        return False
    m, k, n = mat1.get_size()[-2], mat1.get_size()[-1], mat2.get_size()[-1]
    if any(map(has_free_unbacked_symbols, [m, k, n])):
        return False
    # Upstream rejects any degenerate dim (m/k/n <= 1) because GPU tl.dot does
    # not tile a size-1 dimension well.  On Spyre we still want the decode-phase
    # (seq_len == 1, so m == 1) GEMV-shaped linears to take the native-matmul
    # (tl.dot) path rather than fall back to extern_kernels.bmm, which cannot run
    # on-device.  Keep the k/n <= 1 guards (a genuinely degenerate contraction or
    # output), but allow m == 1 for spyre.
    if V.graph.sizevars.statically_known_leq(
        k, 1
    ) or V.graph.sizevars.statically_known_leq(n, 1):
        return False
    if device_type != "spyre" and V.graph.sizevars.statically_known_leq(m, 1):
        return False
    return True


def _load_module(name: str) -> ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        parts = name.rsplit(".", 1)
        parent = __import__(parts[0], fromlist=[parts[1]])
        mod = getattr(parent, parts[1])
    return mod


class SpyreTritonPatches:
    """Patches PyTorch Inductor functions to add Spyre device support."""

    def patch(self) -> list[tuple[ModuleType, Any]]:
        """Monkey-patch use_native_matmul in all modules that hold a local binding.

        mm.py and bmm.py each do `from mm_common import use_native_matmul`, so
        patching only mm_common would leave their local references stale.

        Returns:
            List of (module, original_function) pairs for restoration.
        """
        targets = [
            "torch._inductor.kernel.mm_common",
            "torch._inductor.kernel.mm",
            "torch._inductor.kernel.bmm",
        ]
        saved = []
        for name in targets:
            mod = _load_module(name)
            orig = getattr(mod, "use_native_matmul", None)
            if orig is not None:
                setattr(mod, "use_native_matmul", _patched_use_native_matmul)
                saved.append((mod, orig))
        return saved


@contextmanager
def spyre_triton_patches():
    """Context manager to apply Spyre-specific monkey patches for Triton compilation."""
    # Remove Spyre's BATCH_MATMUL_OP mm lowerings before enable_spyre_lowerings()
    # installs them.  spyre_triton_patches() is entered before enable_spyre_lowerings()
    # in patches.py, so the pop takes effect when the lowering pass runs.
    saved_mm = {op: spyre_lowerings.pop(op, None) for op in _TRITON_SKIP_MM_OPS}

    # Enable native matmul for the Spyre Triton path.  The default value is
    # False (TORCHINDUCTOR_NATIVE_MATMUL=0), but Spyre's Triton kernels always
    # handle mm via tl.dot — this must be True so SIMDKernel sets
    # is_native_matmul=True and SpyreTritonOverrides.dot() is invoked.
    saved_native_matmul = config.triton.native_matmul
    config.triton.native_matmul = True

    saved = SpyreTritonPatches().patch()
    try:
        yield
    finally:
        for mod, orig in saved:
            mod.use_native_matmul = orig
        for op, fn in saved_mm.items():
            if fn is not None:
                spyre_lowerings[op] = fn
        config.triton.native_matmul = saved_native_matmul
