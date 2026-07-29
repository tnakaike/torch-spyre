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

"""OpSpec-based Inductor backends for Spyre.

Each backend projects a finalized ``OpSpec``/``LoopSpec`` list to a target IR.
This package intentionally does **not** eagerly import its subpackages: each
backend is lazy-imported inside its env-var branch in
``torch_spyre/_inductor/__init__.py`` so that, e.g., the KTIR path never pulls
in Triton. See ``.claude/skills/spyre-triton/OPSPEC_BACKEND_FUNCTIONS.md``.
"""
