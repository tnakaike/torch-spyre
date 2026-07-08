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

Layout contract: every tensor argument reaching ``.run()`` is already in
**physical** (sticked) layout -- the wrapper allocates buffers via
``ktir_layout.ktir_empty_with_layout`` and stickifies real inputs, and destickifies
the returned outputs. So this runner only bridges representation (physical
``torch.Tensor`` <-> physical ``np.ndarray``, since ktir-cpu speaks NumPy); it does
no layout conversion.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from torch_spyre._inductor.logging_utils import get_inductor_logger

logger = get_inductor_logger("ktir_cpu_runner")

__all__ = ["KtirCpuRunner"]


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
