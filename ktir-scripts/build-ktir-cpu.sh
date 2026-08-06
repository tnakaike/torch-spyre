#!/usr/bin/env bash

# ktir-cpu must be the fork/branch that carries the MLIR Python-binding support
# consumed by this flow -- it is NOT yet upstreamed:
#   https://github.com/tnakaike/ktir-cpu/tree/nakaike/dev-3
# Check out that branch at ${DTI_PROJECT_ROOT}/ktir-cpu before running this.

cd ${DTI_PROJECT_ROOT}/ktir-cpu

echo "==> installing ktir-cpu (editable, with dev extras)"
uv pip install -e ".[dev]"
