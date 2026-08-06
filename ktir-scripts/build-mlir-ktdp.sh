#!/bin/bash

# Canonical builder for mlir_ktdp (the KTIR MLIR Python bindings).
#
# This is the SINGLE source of truth for mlir_ktdp across the toolchain: Triton
# (build-triton.sh) and ktir-cpu (build.sh) consume the mlir_ktdp installed here
# and must NOT install a second copy. mlir_ktdp bundles its own MLIR and shares
# static MLIR globals with libtriton.so in-process, so both must link the SAME
# LLVM — guaranteed because this repo's cmake/llvm-hash.txt, Triton's
# cmake/llvm-hash-spyre.txt, and the submodule's llvm-hash.txt all pin the same
# artifact (resolved/cached by scripts/setup_mlir.py).
#
# Toolchain build order: ktir-mlir-frontend (this) -> triton -> ktir-cpu.

set -euo pipefail

# The ktir-mlir-frontend fork checkout (a sibling repo under DTI_PROJECT_ROOT),
# NOT this ktir-scripts directory. Resolve it explicitly so the script works
# from any cwd and regardless of where these scripts are collected.
FRONTEND_DIR="${DTI_PROJECT_ROOT}/ktir-mlir-frontend"
cd "$FRONTEND_DIR"

LLVM_HASH=$(cat "$FRONTEND_DIR/cmake/llvm-hash.txt")

echo "==> resolving MLIR_DIR via setup_mlir.py (hash=$LLVM_HASH)"
MLIR_DIR=$(uv run --no-project python "$FRONTEND_DIR/scripts/setup_mlir.py" \
    --hash "$LLVM_HASH" \
    --repo "torch-spyre/ktir-mlir-frontend")
echo "==> MLIR_DIR=$MLIR_DIR"

# --force-reinstall so this build authoritatively overwrites any mlir_ktdp that
# a previous submodule-based flow may have installed.
echo "==> building mlir_ktdp from $FRONTEND_DIR"
CMAKE_ARGS="-DMLIR_DIR=$MLIR_DIR" uv pip install --force-reinstall "$FRONTEND_DIR"

echo "==> verifying mlir_ktdp dialect bindings"
uv run --no-project python - <<'PY'
import mlir_ktdp
from mlir_ktdp.dialects import ktdp  # noqa: F401
missing = []
for _d in ("func", "arith", "math", "linalg", "scf", "tensor", "memref"):
    try:
        __import__(f"mlir_ktdp.dialects.{_d}")
    except ModuleNotFoundError:
        missing.append(_d)
print("mlir_ktdp:", mlir_ktdp.__file__)
if missing:
    print("WARNING: missing dialect bindings:", ", ".join(missing))
else:
    print("all requested dialect bindings present")
PY
