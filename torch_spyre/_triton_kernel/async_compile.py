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

import os
import tempfile
from typing import Any, Optional

from torch._inductor.codecache import PyCodeCache
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.runtime.triton_compat import (
    ASTSource,
    GPUTarget,
    triton,
)

from torch_spyre._inductor import config
from torch_spyre._inductor.logging_utils import get_inductor_logger

logger = get_inductor_logger("async_compile")


def _asm_text(compiled: Any, ext: str) -> Optional[str]:
    """Return ``compiled.asm[ext]`` as text, or None if the stage is absent."""
    asm = getattr(compiled, "asm", None)
    if asm is None:
        return None
    try:
        blob = asm[ext]
    except (KeyError, TypeError):
        return None
    if isinstance(blob, (bytes, bytearray)):
        return blob.decode()
    return blob


def _extract_ktir(compiled: Any) -> Optional[str]:
    """Return the textual KTIR from a compiled kernel, or None if unavailable.

    The Spyre backend sets ``binary_ext = "ktir"``, so the KTDP-dialect module
    is stored (as printed MLIR text) under ``compiled.asm["ktir"]``.
    """
    return _asm_text(compiled, "ktir")


def _dump_ttir_ktir(kernel_name: str, compiled: Any) -> None:
    """Write the emitted TTIR and KTIR to disk, mirroring the SDSC path.

    Artifacts land under ``<cache_dir>/inductor-spyre/<kernel_name>_XXXX/`` (the
    same ``inductor-spyre`` root the SDSC bundle path uses), so the Triton path's
    intermediates can be inspected alongside SDSC's.
    """
    ttir = _asm_text(compiled, "ttir")
    ktir = _asm_text(compiled, "ktir")
    if ttir is None and ktir is None:
        return
    try:
        spyre_dir = os.path.join(cache_dir(), "inductor-spyre")
        os.makedirs(spyre_dir, exist_ok=True)
        out_dir = tempfile.mkdtemp(dir=spyre_dir, prefix=f"{kernel_name}_")
        for ext, text in (("ttir", ttir), ("ktir", ktir)):
            if text is None:
                continue
            path = os.path.join(out_dir, f"{kernel_name}.{ext}")
            with open(path, "w") as f:
                f.write(text)
        logger.debug("SpyreTriton: wrote TTIR/KTIR for %s to %s", kernel_name, out_dir)
    except OSError as exc:  # best-effort: never fail compilation over a dump
        logger.warning(
            "SpyreTriton: could not dump TTIR/KTIR for %s: %s", kernel_name, exc
        )


def _export_dir(compiled: Any) -> Optional[str]:
    """Unpack the compiled artifact and return dbo-opt's export directory.

    ``SpyreDriver.load_binary`` extracts the ``.spyrecode`` ZIP
    content-addressed under the Triton cache and returns the directory that
    *holds* ``spyreCodeDir/`` -- which is exactly what
    ``SpyreSDSCKernelRunner`` takes as its ``code_dir``, since it appends
    ``/spyreCodeDir`` itself.

    ``load_binary`` is called directly rather than through
    ``CompiledKernel._init_handles``, which would also build a launcher this
    path never uses.  It deliberately does not call ``prepare_kernel``: that
    needs an initialized Spyre runtime, so it is left to the runner's lazy
    ``jobplan`` on the launch path, after a device tensor has started the
    runtime.
    """
    from triton.runtime import driver

    # (module, function, n_regs, n_spills, n_max_threads).  On Spyre the program
    # *is* the directory, so module and function are the same path; the backend
    # ignores every argument but the artifact bytes.
    handles = driver.active.utils.load_binary(
        compiled.name, compiled.kernel, compiled.metadata.shared, 0
    )
    if not handles or handles[0] is None:
        return None
    return str(handles[0])


def _address_symbol_kinds(signature: Any) -> list:
    """One address symbol per pointer argument, keyed by its launch position.

    The KTIR entry function takes its base addresses as ``index`` arguments
    (``MaterializeBaseAddresses`` bakes them into constants only when the
    ``spyrecode`` stage is given ``base_addresses``, which this path does not
    do), so the runner has to bind one address per pointer argument.

    ``ConvertFunctions`` retypes only ``!tt.ptr`` arguments to ``index`` and
    leaves every other type alone, so the entry function's ``index`` arguments
    are the pointer arguments in signature order.  ``SymbolKind.arg_index`` is
    a position in the tuple handed to ``run()``, so an interleaved signature
    such as ``(ptr, i32, ptr)`` has to keep positions 0 and 2 rather than
    renumbering them to 0 and 1.
    """
    from torch_spyre._inductor.codegen.compute_ops import SymbolKind

    return [
        SymbolKind.kernel(pos)
        for pos, ty in enumerate(signature.values())
        if str(ty).startswith("*")
    ]


class SpyreTritonAsyncCompile:
    """Async compilation interface for Spyre Triton kernels."""

    def triton(self, kernel_name: str, source_code: str, device_str: str):
        cat = getattr(PyCodeCache.load(source_code), kernel_name)
        cfg = cat.configs[0]
        compile_meta = cat.triton_meta
        compile_meta["device_type"] = cat.device_props.type
        compile_meta["cc"] = cat.device_props.cc
        compile_meta["constants"].update(cfg.kwargs)
        compile_args = (
            ASTSource(
                cat.fn,
                compile_meta["signature"],
                compile_meta["constants"],
                compile_meta["configs"][0],
            ),
        )
        target = GPUTarget(
            compile_meta["device_type"],
            compile_meta["cc"],
            cat.device_props.warp_size_or_default,
        )
        # spyre_grid is injected by the OpSpec->Triton generator and carries the
        # per-axis program count for SpyreOptions.grid.  The DistributeWork MLIR
        # pass requires grid.size() == kernel pid rank.
        spyre_grid = compile_meta.get("spyre_grid", (32,))
        compile_kwargs = {
            "target": target,
            "options": {"grid": spyre_grid},
        }
        compiled = triton.compile(*compile_args, **compile_kwargs)

        # Persist TTIR/KTIR under <cache_dir>/inductor-spyre/, like the SDSC path.
        _dump_ttir_ktir(kernel_name, compiled)

        # Device-free path: run the emitted KTIR on ktir-cpu instead of a Spyre
        # device. Gated so the default (device) path is unchanged.
        if config.ktir_cpu:
            ktir_text = _extract_ktir(compiled)
            if ktir_text is None:
                logger.warning(
                    "TORCH_SPYRE_KTIR_CPU set but no KTIR found on the compiled "
                    "kernel %s; not returning a ktir-cpu runner.",
                    kernel_name,
                )
                return None
            from torch_spyre.execution.ktir_cpu_runner import KtirCpuRunner

            return KtirCpuRunner(kernel_name, ktir_text)

        # Device path.  The backend's ``spyrecode`` stage has already run the
        # dataflow scheduler over the emitted KTIR and produced a loadable
        # program, so there is no dbo-opt call to make here -- unlike the
        # OpSpec->KTIR emitter path, which invokes dbo-opt itself on a .ktir
        # file.  All that is left is to bind the launch arguments to the
        # program, which is what SpyreSDSCKernelRunner already does for the
        # SDSC bundle path; nothing in it is SDSC-specific.
        code_dir = _export_dir(compiled)
        if code_dir is None:
            logger.warning(
                "no Spyre program was unpacked for kernel %s; not returning a "
                "device runner.",
                kernel_name,
            )
            return None

        from torch_spyre.execution.kernel_runner import SpyreSDSCKernelRunner

        logger.debug("SpyreTriton: device runner for %s over %s", kernel_name, code_dir)
        return SpyreSDSCKernelRunner(
            kernel_name,
            code_dir,
            symbol_kinds=_address_symbol_kinds(compile_meta["signature"]),
        )

    def wait(self, scope: dict[str, Any]) -> None:
        pass
