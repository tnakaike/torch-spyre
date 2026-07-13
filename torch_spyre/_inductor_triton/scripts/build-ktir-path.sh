#!/bin/bash
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

set -euo pipefail

# ----------------------------------------
# build-ktir-path.sh
# CI build script for the Triton / KTIR / ktir-cpu compilation path:
#   1. LLVM for triton (commit pinned in triton/cmake/llvm-hash-spyre.txt)
#   2. triton       (git@github.com:torch-spyre/triton.git    branch: main)
#   3. ktir-cpu     (git@github.com:torch-spyre/ktir-cpu.git  branch: main)
#
# torch-spyre itself is NOT built here: it is checked out and built
# separately (the `build-torch-spyre` composite action), and this script
# installs triton + ktir-cpu into that same venv (see VENV_DIR below).
#
# Assumptions / prerequisites:
#   - PROJECT_ROOT must be set in the environment (parent of torch-spyre,
#     triton, and ktir-cpu; e.g. /home/senuser).
#   - A Python venv exists at ${VENV_DIR} (defaults to ${PROJECT_ROOT}/.venv)
#     and uv is on PATH. In CI this is the pre-baked torch-spyre venv, so the
#     action passes VENV_DIR=${PROJECT_ROOT}/torch-spyre/.venv.
#   - SENLIB_INSTALL_DIR and DEEPTOOLS_INSTALL_DIR must be set in the
#     environment.
#   - SPYRE_COMMS_INSTALL_DIR must be set in the environment.
#   - SSH access to github.com is required to clone the private repositories.
#   - cmake, ninja, git, and a C++20-capable compiler (c++) must be on PATH.
# ----------------------------------------

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  echo "ERROR: PROJECT_ROOT must be set" >&2
  exit 1
fi

# Python environment to install triton + ktir-cpu into. Defaults to the
# dev-env layout (${PROJECT_ROOT}/.venv); CI overrides this to point at the
# venv that build-torch-spyre populated (${PROJECT_ROOT}/torch-spyre/.venv)
# so all three packages live in one environment.
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"

# Activate Python environment
source "${VENV_DIR}/bin/activate"
export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"

# Build output directory (mirrors dev-env.sh)
export PROJECT_BUILD="${PROJECT_ROOT}/build"
mkdir -p "${PROJECT_BUILD}"

uv pip install nanobind==2.9.2 ninja pybind11==3.0.1 build cmake~=3.26 regex wheel

# ----------------------------------------
# Helper: clone repo if absent, otherwise fetch + hard-reset to origin/<branch>
# Usage: clone_or_update <dir> <url> <branch>
# ----------------------------------------
clone_or_update() {
  local dir="$1"
  local url="$2"
  local branch="$3"

  if [[ ! -d "${dir}/.git" ]]; then
    echo "==> Cloning ${url} (branch: ${branch}) into ${dir}"
    git clone --branch "${branch}" "${url}" "${dir}"
  else
    echo "==> Updating $(basename "${dir}") to origin/${branch}"
    git -C "${dir}" fetch --prune origin
    git -C "${dir}" checkout "${branch}"
    git -C "${dir}" reset --hard "origin/${branch}"
  fi
}

# ----------------------------------------
# 1. Checkout repositories
#
# torch-spyre is intentionally omitted: it is checked out and built by the
# `build-torch-spyre` action before this script runs.
# ----------------------------------------
clone_or_update "${PROJECT_ROOT}/triton" \
  "git@github.com:torch-spyre/triton.git" "main"

clone_or_update "${PROJECT_ROOT}/ktir-cpu" \
  "git@github.com:torch-spyre/ktir-cpu.git" "main"

# ----------------------------------------
# 2. Build LLVM for Triton
# ----------------------------------------
echo "========================================"
echo "========== LLVM for Triton"
echo "========================================"

export TRITON_LLVM_BUILD_DIR="${PROJECT_BUILD}/llvm-triton"
export TRITON_LLVM_SRC_DIR="${PROJECT_ROOT}/llvm-project-triton"

# Read the pinned LLVM commit from triton's own hash file
LLVM_COMMIT="$(cat "${PROJECT_ROOT}/triton/cmake/llvm-hash-spyre.txt")"
echo "==> LLVM commit: ${LLVM_COMMIT}"

# Clone llvm-project-triton if absent; otherwise fetch
if [[ ! -d "${TRITON_LLVM_SRC_DIR}/.git" ]]; then
  echo "==> Cloning llvm-project for triton"
  git clone "https://github.com/llvm/llvm-project.git" "${TRITON_LLVM_SRC_DIR}"
fi
git -C "${TRITON_LLVM_SRC_DIR}" fetch --prune origin
git -C "${TRITON_LLVM_SRC_DIR}" checkout "${LLVM_COMMIT}"

mkdir -p "${TRITON_LLVM_BUILD_DIR}"
cd "${TRITON_LLVM_BUILD_DIR}"

cmake -G Ninja "${TRITON_LLVM_SRC_DIR}/llvm" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_EH=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_ENABLE_ZSTD=OFF \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE="$(which python3)" \
  -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld;clang" \
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU"

ninja

# ----------------------------------------
# 3. Build Triton
# ----------------------------------------
echo "========================================"
echo "========== Triton"
echo "========================================"

cd "${PROJECT_ROOT}/triton"
LLVM_INCLUDE_DIRS="${TRITON_LLVM_BUILD_DIR}/include" \
LLVM_LIBRARY_DIR="${TRITON_LLVM_BUILD_DIR}/lib" \
LLVM_SYSPATH="${TRITON_LLVM_BUILD_DIR}" \
LLVM_BUILD_DIR="${TRITON_LLVM_BUILD_DIR}" \
TRITON_BUILD_TUTORIALS=OFF \
TRITON_BUILD_PROTON=OFF \
uv pip install -e ".[spyre-test]" --no-build-isolation

# ----------------------------------------
# 4. Build ktir-cpu
# ----------------------------------------
echo "========================================"
echo "========== ktir-cpu"
echo "========================================"

FRONTEND_DIR="${PROJECT_ROOT}/triton/third_party/spyre/ktir-mlir-frontend"
MLIR_DIR="${PROJECT_ROOT}/build/llvm-triton/lib/cmake/mlir"

CMAKE_ARGS="-DMLIR_DIR=${MLIR_DIR}" uv pip install "${FRONTEND_DIR}"

cd "${PROJECT_ROOT}/ktir-cpu"
uv pip install -e ".[dev]"

echo "========================================"
echo "========== Build complete"
echo "========================================"
