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

import tempfile
from collections.abc import Sequence
from typing import Any
import os
import subprocess
from typing import Union

from torch._inductor.codecache import PyCodeCache
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.runtime.triton_compat import (
    ASTSource,
    GPUTarget,
    cc_warp_size,
    triton,
)

from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.op_spec import (
    LoopSpec,
    OpSpec,
    UnimplementedOp,
    find_unimplemented,
)
from torch_spyre._inductor.codegen.bundle import generate_bundle
from .kernel_runner import SpyreSDSCKernelRunner, SpyreUnimplementedRunner

logger = get_inductor_logger("sdsc_compile")


def get_output_dir(kernel_name: str):
    spyre_dir = os.path.join(cache_dir(), "inductor-spyre")
    os.makedirs(spyre_dir, exist_ok=True)
    kernel_output_dir = tempfile.mkdtemp(dir=spyre_dir, prefix=f"{kernel_name}_")
    return kernel_output_dir


class SpyreAsyncCompile:
    def __init__(self) -> None:
        pass

    def sdsc(
        self, kernel_name: str, specs: Sequence[OpSpec | LoopSpec | UnimplementedOp]
    ):
        unimp = find_unimplemented(list(specs))
        if unimp is not None:
            logger.warning(
                f"WARNING: Compiling unimplemented {unimp.op} to runtime exception"
            )
            return SpyreUnimplementedRunner(kernel_name, unimp.op)

        # Generate SDSC Bundle from OpSpecs
        output_dir = get_output_dir(kernel_name)
        generate_bundle(kernel_name, output_dir, specs)

        # Invoke backend compiler of SDSC Bundle
        subprocess.run(["dxp_standalone", "--bundle", "-d", output_dir], check=True)

        return SpyreSDSCKernelRunner(kernel_name, output_dir)

    def wait(self, scope: dict[str, Any]) -> None:
        pass

    def triton(self, kernel_name: str, source_code: str, device_str: str):
        print("source_code={}".format(source_code))
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
        options = {
            "spyre_options": compile_meta["spyre_options"],
        }
        compile_kwargs = {
            "target": target,
            "options": options,
        }

        specs: list[Union[OpSpec | UnimplementedOp]] = compile_meta["spyre_options"][
            "op_specs"
        ]
        _ = triton.compile(*compile_args, **compile_kwargs)
        return self.sdsc(kernel_name, specs)
