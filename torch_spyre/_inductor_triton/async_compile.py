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

from typing import Any

from torch._inductor.codecache import PyCodeCache
from torch._inductor.runtime.triton_compat import (
    ASTSource,
    GPUTarget,
    cc_warp_size,
    triton,
)


class SpyreTritonAsyncCompile:
    """Async compilation interface for Spyre Triton kernels."""

    def triton(self, kernel_name: str, source_code: str, device_str: str) -> None:
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
            cc_warp_size(compile_meta["cc"]),
        )
        # spyre_grid is injected by SpyreTritonKernel.codegen_body() and
        # carries the per-axis program count for SpyreOptions.grid.  The
        # DistributeWork MLIR pass requires grid.size() == kernel pid rank.
        spyre_grid = compile_meta.get("spyre_grid", (32,))
        compile_kwargs = {
            "target": target,
            "options": {"grid": spyre_grid},
        }
        _ = triton.compile(*compile_args, **compile_kwargs)
        return None

    def wait(self, scope: dict[str, Any]) -> None:
        pass
