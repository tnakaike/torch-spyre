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
from typing import Any, Optional, Sequence

import torch
import torch._inductor.lowering as original_lowerings
from torch._inductor import config
from torch._inductor.utils import get_current_backend
from torch._inductor.virtualized import V
from torch.fx.experimental.symbolic_shapes import has_free_unbacked_symbols

from torch_spyre._inductor.decompositions import spyre_decompositions
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.lowering import spyre_lowerings

# ---------------------------------------------------------------------------
# Triton-specific decompositions
#
# Some Spyre decompositions (in torch_spyre._inductor.decompositions) lower to
# fused Spyre HW ops that SDSC codegen handles but SpyreTritonOverrides does
# not.  For the Triton path we swap in decompositions built from standard
# reduction+pointwise aten ops, which TritonKernel can lower.  These are
# installed into ``spyre_decompositions`` only while spyre_triton_patches() is
# active (i.e. only when TORCH_SPYRE_TRITON=1) and restored on exit, so the
# SDSC decomposition table is left untouched.
# ---------------------------------------------------------------------------


def _spyre_triton_decomp_layer_norm(
    input: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """layer_norm decomposition for the Triton path.

    The SDSC decomposition (``spyre_layer_norm``) emits the fused HW ops
    ``spyre.exx2`` / ``spyre.layernormscale`` / ``spyre.layernormnorm``, which
    have SDSC codegen handlers but no ``SpyreTritonOverrides`` ops-handler
    method (so the Triton path fails with ``AttributeError: ... layernormscale``).
    Decompose instead into explicit reduction+pointwise aten ops that
    TritonKernel lowers cleanly.

    The mean is written as ``sum(...) / N`` rather than ``torch.mean`` on
    purpose: ``aten.mean.dim`` reaches codegen as an ``ops.reduction(..., 'mean')``
    that ``get_triton_reduction_function`` maps to ``tl.mean`` -- an op the Spyre
    Triton backend does not provide, so the bundle fails TTIR generation.
    ``sum`` maps to the supported ``tl.sum`` and the ``/ N`` is a plain pointwise
    op.  (PyTorch's native ``aten.var_mean`` decomposition is likewise avoided --
    its multi-output reduction hits ``TritonCSEVariable is not subscriptable`` in
    the Spyre reduction codegen.)
    """
    if len(normalized_shape) != 1:
        raise Unsupported(
            f"_spyre_triton_decomp_layer_norm: only supports normalized_shape of length 1, "
            f"got {normalized_shape}"
        )
    # F.layer_norm treats weight=None as identity and bias=None as zero.
    if weight is None:
        weight = input.new_ones(normalized_shape)
    if bias is None:
        bias = input.new_zeros(normalized_shape)
    n = normalized_shape[0]
    mean = torch.sum(input, dim=-1, keepdim=True) / n
    centered = input - mean
    var = torch.sum(centered * centered, dim=-1, keepdim=True) / n
    rstd = torch.rsqrt(var + eps)
    return centered * rstd * weight + bias


def _spyre_triton_decomp_rms_norm(
    input: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: Optional[torch.Tensor] = None,
    eps: Optional[float] = 1e-5,
) -> torch.Tensor:
    """rms_norm decomposition for the Triton path.

    Mirrors ``spyre_rms_norm`` but computes the mean-square as ``sum(...) / N``
    instead of ``torch.mean``: the latter reaches codegen as ``tl.mean``, which
    the Spyre Triton backend does not provide (bundle fails TTIR generation).
    Same fix as :func:`_spyre_triton_decomp_layer_norm`.
    """
    if len(normalized_shape) != 1:
        raise Unsupported(
            f"_spyre_triton_decomp_rms_norm: only supports normalized_shape of length 1, "
            f"got {normalized_shape}"
        )
    if eps is None:
        eps = torch.finfo(input.dtype).eps
    n = normalized_shape[0]
    mean_sq = torch.sum(input * input, dim=-1, keepdim=True) / n
    rstd = torch.rsqrt(mean_sq + eps)
    output = input * rstd
    if weight is not None:
        output = output * weight
    return output


def _spyre_triton_decomp_var_mean(self, dim=None, *, correction=None, keepdim=False):
    """var_mean.correction for the Triton path: two single-output reductions.

    ``aten.var_mean`` is a multi-output reduction; the Spyre Triton reduction
    codegen returns a single ``CSEVariable``, so the ``getitem`` on it fails with
    ``'TritonCSEVariable' object is not subscriptable`` (batch_norm and PyTorch's
    native layer_norm decomposition both emit it).  Split into explicit
    ``mean = sum(x)/N`` and ``var = sum((x-mean)^2)/(N-correction)``, each a normal
    single-output reduction.  Returns ``(var, mean)`` to match aten's order.
    """
    if correction is None:
        correction = 1
    dims = list(range(self.dim())) if dim is None else list(dim)
    n = 1
    for d in dims:
        n *= self.size(d)
    mean = torch.sum(self, dim=dims, keepdim=True) / n
    centered = self - mean
    var = torch.sum(centered * centered, dim=dims, keepdim=True) / (n - correction)
    if not keepdim:
        var = torch.squeeze(var, tuple(dims))
        mean = torch.squeeze(mean, tuple(dims))
    return var, mean


def _spyre_triton_decomp_silu(input: torch.Tensor) -> torch.Tensor:
    """silu for the Triton path: x * sigmoid(x), written via exp.

    The SDSC decomposition (``decompositions.py``) lowers ``aten.silu`` to the
    fused HW op ``spyre.silu``, whose ``lower_silu`` emits ``ops.silu`` -- a
    method ``SpyreTritonOverrides`` does not implement (``AttributeError: silu``).
    Decompose to ``x / (1 + exp(-x))`` instead: ``exp`` is mapped to ``tl.exp``
    (with fp32 upcast) by SpyreTritonOverrides, whereas ``sigmoid`` has no Spyre
    override.  (Same fix shape as the norm decompositions above.)
    """
    return input / (1.0 + torch.exp(-input))


# Op -> Triton-path decomposition.  Swapped into spyre_decompositions for the
# duration of the Triton compile; the SDSC entries are restored on exit.
_SPYRE_TRITON_DECOMPOSITIONS = {
    torch.ops.aten.layer_norm.default: _spyre_triton_decomp_layer_norm,
    torch.ops.aten.rms_norm.default: _spyre_triton_decomp_rms_norm,
    torch.ops.aten.var_mean.correction: _spyre_triton_decomp_var_mean,
    torch.ops.aten.silu.default: _spyre_triton_decomp_silu,
}


def _install_spyre_triton_decompositions() -> dict:
    """Swap Triton decompositions into ``spyre_decompositions``; return saved state."""
    saved: dict = {}
    for op, fn in _SPYRE_TRITON_DECOMPOSITIONS.items():
        saved[op] = spyre_decompositions.get(op)
        spyre_decompositions[op] = fn
    return saved


def _restore_decompositions(saved: dict) -> None:
    """Restore the SDSC decompositions swapped out by _install_spyre_triton_decompositions."""
    for op, fn in saved.items():
        if fn is not None:
            spyre_decompositions[op] = fn
        else:
            spyre_decompositions.pop(op, None)


# ---------------------------------------------------------------------------
# Triton-specific lowerings
#
# Spyre mm lowerings use BATCH_MATMUL_OP with a tuple-valued inner_fn, which
# SpyreKernel (SDSC) understands but TritonKernel cannot codegen.  The Triton
# path relies on PyTorch's standard aten.mm lowering so that use_native_matmul
# converts it to tl.dot.  These ops are popped from spyre_lowerings before
# enable_spyre_lowerings() installs them, and restored on context exit.
# ---------------------------------------------------------------------------


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
    # (seq_len == 1, so m == 1) GEMV-shaped linears AND degenerate contractions
    # (k == 1, e.g. an outer-product bmm) to take the native-matmul (tl.dot) path
    # rather than fall back to extern_kernels.bmm, which cannot run on-device.
    # Keep the n <= 1 guard (a genuinely degenerate output), but allow m == 1 and
    # k == 1 for spyre.
    if V.graph.sizevars.statically_known_leq(n, 1):
        return False
    if device_type != "spyre" and (
        V.graph.sizevars.statically_known_leq(m, 1)
        or V.graph.sizevars.statically_known_leq(k, 1)
    ):
        return False
    return True


def _spyre_triton_lowering_mm(native_fn: Any) -> Any:
    """Wrap a native matmul lowering with the degenerate-contraction shortcut.

    A matmul whose contraction dim ``K == 1`` is an outer product, so it reduces
    to a broadcasted pointwise multiply (``[.., M, 1] * [.., 1, N] -> [.., M, N]``)
    with no reduction at all.  This mirrors the SDSC ``lower_mm`` / ``lower_bmm``
    ``reduction_numel == 1`` branch (``result = lowering.mul(x, y)``) so both
    paths lower a K==1 matmul identically.  TritonKernel codegens the resulting
    ``mul`` as a plain pointwise kernel; native ``tl.dot`` cannot represent the
    K==1 case because the device layout sticks the size-1 K into a 64-element
    within-stick dim, leaving the two operands disagreeing on the contraction
    extent.  For ``K > 1`` we defer to the native (tl.dot) lowering unchanged.
    """

    def _lowering(x, y, *args, **kwargs):
        # K is the last dim of the first operand for both mm and bmm.
        if V.graph.sizevars.statically_known_equals(x.get_size()[-1], 1):
            return original_lowerings.mul(x, y)
        return native_fn(x, y, *args, **kwargs)

    return _lowering


# Op -> factory that builds the Triton-path lowering (wrapping the captured
# native tl.dot lowering), analogous to _SPYRE_TRITON_DECOMPOSITIONS.  Swapped
# into spyre_lowerings for the duration of the Triton compile; the SDSC entries
# are restored on exit.
_SPYRE_TRITON_LOWERINGS = {
    torch.ops.aten.mm.default: _spyre_triton_lowering_mm,
    torch.ops.aten.bmm.default: _spyre_triton_lowering_mm,
}


def _install_spyre_triton_lowerings() -> dict:
    """Swap Triton mm/bmm lowerings into ``spyre_lowerings``; return saved state.

    Removes Spyre's BATCH_MATMUL_OP mm/bmm lowerings (SpyreKernel's tuple-valued
    inner_fn cannot be codegen'd by TritonKernel) and installs the factory-built
    wrapper (see _spyre_triton_lowering_mm) that shortcuts the K==1 degenerate
    contraction to a pointwise mul while deferring to the native tl.dot lowering
    for K > 1.  The native lowering (tuned_mm / tuned_bmm) currently sits in
    original_lowerings.lowerings; capture it now, before enable_spyre_lowerings()
    installs our wrapper over it.
    """
    saved: dict = {}
    for op, make_lowering in _SPYRE_TRITON_LOWERINGS.items():
        saved[op] = spyre_lowerings.pop(op, None)
        native_fn = original_lowerings.lowerings.get(op)
        if native_fn is not None:
            spyre_lowerings[op] = make_lowering(native_fn)
    return saved


def _restore_lowerings(saved: dict) -> None:
    """Restore the Spyre lowerings swapped out by _install_spyre_triton_lowerings."""
    for op, fn in saved.items():
        if fn is not None:
            spyre_lowerings[op] = fn
        else:
            # We may have installed a K==1 shortcut wrapper above; drop it.
            spyre_lowerings.pop(op, None)


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
    # Install the Triton-specific lowerings and decompositions (see the
    # registries above).  spyre_triton_patches() is entered before
    # enable_spyre_lowerings() / enable_spyre_decompositions() in patches.py, so
    # these swaps are in place when those passes copy the tables into the active
    # compile.  Both are restored on context exit, leaving the SDSC tables intact.
    saved_mm = _install_spyre_triton_lowerings()
    saved_decompositions = _install_spyre_triton_decompositions()

    # Enable native matmul for the Spyre Triton path.  The default value is
    # False (TORCHINDUCTOR_NATIVE_MATMUL=0), but Spyre's Triton kernels always
    # handle mm via tl.dot — this must be True so SIMDKernel sets
    # is_native_matmul=True and SpyreTritonOverrides.dot() is invoked.
    saved_native_matmul = config.triton.native_matmul
    config.triton.native_matmul = True

    # Skip the SDSC-oriented split_multi_ops pre-scheduling pass in the Triton
    # path.  It splits multi-op pointwise loop bodies into separate buffers so
    # SpyreKernel can codegen them one op at a time; TritonKernel handles multi-op
    # pointwise bodies natively, so the split is unnecessary here.  It is also
    # actively harmful for a degenerate (k == 1) matmul, which Inductor collapses
    # into a Pointwise buffer carrying fp16 precision-cast `ops.to_dtype(...,
    # use_compute_types=...)` ops: the pass re-lowers those into a
    # prims.convert_element_type FX node and leaks the ops-handler-only
    # `use_compute_types` kwarg into the tensor-level lowering, which rejects it.
    # `CustomPreSchedulingPasses` captures split_multi_ops by identity in its
    # `passes` list at construction (before this context is entered), so patch
    # __call__ (resolved dynamically at codegen time) to filter it out.
    from torch_spyre._inductor.passes import CustomPreSchedulingPasses
    from torch_spyre._inductor.split_multi_ops import split_multi_ops

    saved_pre_call = CustomPreSchedulingPasses.__call__

    def _pre_call_skip_split_multi_ops(self: CustomPreSchedulingPasses, graph: Any):
        saved_passes = self.passes
        self.passes = [p for p in saved_passes if p is not split_multi_ops]
        try:
            return saved_pre_call(self, graph)
        finally:
            self.passes = saved_passes

    CustomPreSchedulingPasses.__call__ = _pre_call_skip_split_multi_ops  # type: ignore[method-assign]

    saved = SpyreTritonPatches().patch()
    try:
        yield
    finally:
        for mod, orig in saved:
            mod.use_native_matmul = orig
        _restore_lowerings(saved_mm)
        _restore_decompositions(saved_decompositions)
        config.triton.native_matmul = saved_native_matmul
        CustomPreSchedulingPasses.__call__ = saved_pre_call  # type: ignore[method-assign]
