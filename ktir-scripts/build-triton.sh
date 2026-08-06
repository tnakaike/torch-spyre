#!/bin/bash

# Exit on errors
set -e

# mlir_ktdp is NOT built here. It is built ONCE by the canonical builder
# build-mlir-ktdp.sh (from the ktir-mlir-frontend fork checkout) and reused by
# Triton and ktir-cpu. Triton's setup.py (spyre backend) still emits a helper
# install-ktdp-mlir-bindings.sh that would build mlir_ktdp from the VENDORED
# submodule (third_party/spyre/ktir-mlir-frontend) — that copy lacks the extra
# dialect Python bindings, so running it would clobber the fork build. We delete
# it at the end of this script. Toolchain order: ktir-mlir-frontend -> triton ->
# ktir-cpu.
#
# LLVM: Triton links against the SAME cached LLVM+MLIR install that
# build-mlir-ktdp.sh resolves via scripts/setup_mlir.py (downloaded/cached under
# ~/.cache/ktir-mlir/<artifact>/), NOT a separate from-source build. This
# guarantees libtriton.so and mlir_ktdp share one MLIR (they hold static MLIR
# globals in-process), and it means the colleague does not have to build LLVM
# from source (build-llvm.sh) just to build Triton.

echo "========================================"
echo "========== Triton"
echo "========================================"

# Resolve the cached LLVM+MLIR install the frontend was built against. Same hash
# file and setup_mlir.py invocation as build-mlir-ktdp.sh, so both resolve the
# identical artifact (cache hit — no re-download).
FRONTEND_DIR="${DTI_PROJECT_ROOT}/ktir-mlir-frontend"
LLVM_HASH=$(cat "$FRONTEND_DIR/cmake/llvm-hash.txt")

echo "==> resolving MLIR_DIR via setup_mlir.py (hash=$LLVM_HASH)"
MLIR_DIR=$(uv run --no-project python "$FRONTEND_DIR/scripts/setup_mlir.py" \
    --hash "$LLVM_HASH" \
    --repo "torch-spyre/ktir-mlir-frontend")
echo "==> MLIR_DIR=$MLIR_DIR"

# MLIR_DIR is <llvm-root>/lib/cmake/mlir; Triton wants the <llvm-root> install
# prefix (it has include/, lib/, and bin/llvm-config).
export TRITON_LLVM_BUILD_DIR="${MLIR_DIR%/lib/cmake/mlir}"
echo "==> TRITON_LLVM_BUILD_DIR=$TRITON_LLVM_BUILD_DIR"

if [[ ! -d "$TRITON_LLVM_BUILD_DIR/include" || ! -d "$TRITON_LLVM_BUILD_DIR/lib" ]]; then
    echo "TRITON_LLVM_BUILD_DIR does not look like an LLVM install: $TRITON_LLVM_BUILD_DIR"
    exit 1
fi

cd "$DTI_PROJECT_ROOT/triton"

echo "Building non-isolated Triton"
LLVM_INCLUDE_DIRS=$TRITON_LLVM_BUILD_DIR/include \
LLVM_LIBRARY_DIR=$TRITON_LLVM_BUILD_DIR/lib \
LLVM_SYSPATH=$TRITON_LLVM_BUILD_DIR \
LLVM_BUILD_DIR=$TRITON_LLVM_BUILD_DIR \
TRITON_BUILD_TUTORIALS=OFF \
TRITON_BUILD_PROTON=OFF \
uv pip install -e ".[spyre-test]" --no-build-isolation

# setup.py (spyre backend) generates a helper that would build/install mlir_ktdp
# from the vendored submodule (lacking the extra dialect Python bindings). Remove
# it so it can never clobber the canonical fork-built mlir_ktdp.
rm -f "$DTI_PROJECT_ROOT/triton/install-ktdp-mlir-bindings.sh"
