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

"""OpSpec -> Triton *source generator* backend (see DESIGN-OpSpecToTriton.md).

Projects a finalized ``OpSpec``/``LoopSpec`` list to Triton source. This is the
only OpSpec backend that imports Triton (``triton.compile`` ingests its output),
so it is **lazy-imported** inside its env-var branch in
``torch_spyre/_inductor/__init__.py`` — importing ``torch_spyre`` never pulls in
Triton, and the KTIR path (a ``generate_ktir`` function in
``_inductor/codegen/``) stays Triton-free. See
``.claude/skills/spyre-triton/OPSPEC_BACKEND_FUNCTIONS.md``.
"""

from .async_compile import SpyreTritonAsyncCompile
from .spyre_triton_kernel import SpyreTritonKernel
from .spyre_triton_scheduling import SpyreTritonScheduling
from .spyre_triton_wrapper import SpyreTritonPythonWrapperCodegen

__all__ = [
    "SpyreTritonKernel",
    "SpyreTritonScheduling",
    "SpyreTritonAsyncCompile",
    "SpyreTritonPythonWrapperCodegen",
]
