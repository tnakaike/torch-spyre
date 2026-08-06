# Building and running the KTIR path (snapshot)

This branch (`dev/triton-opspec-snapshot`) lets you generate **KTIR** from
PyTorch programs through the Spyre Inductor backend, on two paths:

- **OpSpec → KTIR** (`run-ktir.sh`) — the direct KTIR emitter (`generate_ktir`),
  gated by `TORCH_SPYRE_KTIR=1`.
- **OpSpec → Triton → KTIR** (`run-triton.sh`) — the Triton source generator
  compiled through Triton's Spyre backend, gated by `TORCH_SPYRE_TRITON=1`.

Both run the emitted KTIR on **ktir-cpu**.

## Prerequisites

- The `dt-inductor` workspace, with these repos checked out as **siblings**
  under `$DTI_PROJECT_ROOT` (default `~/dt-inductor`):

  ```text
  $DTI_PROJECT_ROOT/
      ktir-mlir-frontend/     # KTIR MLIR dialect + Python bindings (mlir_ktdp)
      triton/                 # Triton with the Spyre backend
      ktir-cpu/               # KTIR reference executor (fork nakaike/dev-3)
      torch-spyre/            # this repo (the snapshot branch)
  ```

- **`ktir-cpu` must be the fork/branch with MLIR Python-binding support**, which
  is not yet upstreamed. Check out
  <https://github.com/tnakaike/ktir-cpu/tree/nakaike/dev-3> at
  `$DTI_PROJECT_ROOT/ktir-cpu`.

- The shared Python venv and environment in `$DTI_PROJECT_ROOT/.venv`.

- Network access to GitHub Releases on the first build: `mlir_ktdp` and Triton
  link a prebuilt LLVM+MLIR that `setup_mlir.py` downloads and caches under
  `~/.cache/ktir-mlir/`. Subsequent builds are cache hits (no network).

## Build

Build order is fixed — **ktir-mlir-frontend → triton → ktir-cpu** — because
Triton and ktir-cpu both consume the single `mlir_ktdp` (and its bundled MLIR)
installed by the frontend step. The build scripts live in `ktir-scripts/`.

Run them from this repo's root, with the venv activated:

```bash
# 1. mlir_ktdp — the KTIR MLIR Python bindings.
#    Downloads/caches the prebuilt LLVM+MLIR (via setup_mlir.py) and installs
#    mlir_ktdp with the extra dialect bindings (func/arith/math/linalg/scf/
#    tensor/memref) the emitter needs. This is the ONE authoritative copy.
./ktir-scripts/build-mlir-ktdp.sh

# 2. Triton (Spyre backend).
#    Links against the SAME cached LLVM+MLIR resolved in step 1 (cache hit, no
#    rebuild), so libtriton.so and mlir_ktdp share one MLIR in-process. It also
#    deletes triton's generated install-ktdp-mlir-bindings.sh so the vendored
#    submodule copy can never clobber the step-1 build.
./ktir-scripts/build-triton.sh

# 3. ktir-cpu — the KTIR reference executor.
./ktir-scripts/build-ktir-cpu.sh
```

Then install this repo (torch-spyre) into the venv if it is not already.

Notes:

- Do **not** run triton's `install-ktdp-mlir-bindings.sh`: it would build
  `mlir_ktdp` from triton's vendored submodule, which lacks the extra dialect
  Python bindings and would break the emitter. `build-triton.sh` removes it.

## Run

Both runners take two arguments:

```bash
./ktir-scripts/run-ktir.sh   <driver.py> <output-name>
./ktir-scripts/run-triton.sh <driver.py> <output-name>
```

- `<driver.py>` — a PyTorch driver (e.g. `add.py`).
- `<output-name>` — a label for the results subdirectory.

Run them **from this repo's root** (they write artifacts relative to the current
directory), with the venv activated:

```bash
# OpSpec -> KTIR emitter path:
./ktir-scripts/run-ktir.sh add.py add

# OpSpec -> Triton -> KTIR path:
./ktir-scripts/run-triton.sh add.py add
```

Each runner sets the backend gate (`TORCH_SPYRE_KTIR=1` for `run-ktir.sh`,
`TORCH_SPYRE_TRITON=1` for `run-triton.sh`), runs on `ktir-cpu`
(`TORCH_SPYRE_KTIR_CPU=1`) with `SENCORES=1`, then collects artifacts.

### Where results land

- `run-ktir.sh`   → `ktir-results/<output-name>/`
- `run-triton.sh` → `triton-results/<output-name>/`

Each results directory contains:

| Artifact | Description |
|---|---|
| `<output-name>.log` | full stdout/stderr (includes the `Max delta Compiled Spyre vs. CPU` correctness check) |
| `output_code.py`, `ir_*.txt`, fx graphs | Inductor `torch_compile_debug` artifacts |
| `*.ttir`, `*.ktir` (triton-dump) | Triton/KTIR intermediates dumped during compile |
| `<kernel>.ktir` | the **emitted KTIR** collected from an inductor output dir |

### Which drivers work

The OpSpec → KTIR emitter (`run-ktir.sh`) is **pointwise-only**. The Triton path
(`run-triton.sh`) covers a wider set (pointwise, reductions, matmul/bmm, gather,
norms, ffn).
