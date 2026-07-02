# Plan: Run the Triton path on CPU by isolating `torch_spyre._C`

**Status:** plan only — **blocked on environment setup** (a CPU box with Spyre
SDK **headers** but no runtime libs, and a `_C` build variant produced there).
Do not start the code changes until the environment step below is done.

**Goal (scoped):** make the Triton **codegen + verify** path importable and
runnable on a CPU-only machine — generate Triton/KTIR source and verify
coverage/non-overlap, with **no device execution**. Execution on a Triton CPU
backend is explicitly out of scope for this plan.

## Why this is needed

`torch_spyre/_C.so` dynamically links the entire Spyre runtime stack
(`libsendnn*`, `libflex`, `libspyre_comms`, `libdeeprt`, `libsenlib-dd2`,
deeptools `libdem/dsm/dm/dvs`, `libL3DlOpsScheduler`, all under
`/opt/ibm/spyre/...`). On a box without that stack, `import torch_spyre._C`
fails at the **dynamic-linker stage**. Because
[op_spec.py](../../../torch_spyre/_inductor/op_spec.py) does
`from torch_spyre._C import DataFormats` at module top, the *entire* Triton
codegen path fails to import before any codegen runs. The blocker is link-time,
not runtime init.

## Key finding: the Triton path only needs the *layout* subset of `_C`

Enumerating every `_C` symbol imported by the reused `_inductor/` modules that
the Triton path pulls in
([op_spec.py](../../../torch_spyre/_inductor/op_spec.py),
[spyre_kernel.py](../../../torch_spyre/_inductor/spyre_kernel.py),
[pass_utils.py](../../../torch_spyre/_inductor/pass_utils.py),
[propagate_layouts.py](../../../torch_spyre/_inductor/propagate_layouts.py),
[ir.py](../../../torch_spyre/_inductor/ir.py),
[lowering.py](../../../torch_spyre/_inductor/lowering.py),
[decompositions.py](../../../torch_spyre/_inductor/decompositions.py),
[padding.py](../../../torch_spyre/_inductor/padding.py),
[wrapper.py](../../../torch_spyre/_inductor/wrapper.py)):

| Category | Symbols | Needs HW runtime? |
|---|---|---|
| **Layout / compile-time** | `DataFormats`, `ElementArrangement`, `SpyreTensorLayout`, `get_device_dtype`, `get_elem_in_stick`, `encode_constant` | **No** |
| Allocation | `empty_with_layout`, `spyre_empty_with_layout`, `copy_tensor`, `as_strided_with_layout`, `to_with_layout`, `reinterpret_tensor*` | Yes (allocator) |
| Execution | `prepare_kernel`, `launch_jobplan`, `launch_kernel`, `JobPlan`, streams | Yes |
| Distributed | `createSpyreCCLBackend` | Yes (spyre_comms) |

The layout subset is implemented in
[module.cpp](../../../torch_spyre/csrc/module.cpp) +
[spyre_views.cpp](../../../torch_spyre/csrc/spyre_views.cpp) +
[types_mapping.h](../../../torch_spyre/csrc/types_mapping.h) and depends only on
**ATen + the flex/sendnn datatype enum header** — not the execution libs. So the
layout math is separable from the runtime at the source level.

Because the goal is **codegen + verify only**, the allocation/execution symbols
do **not** need to work — they only need to *exist* so imports resolve. They can
be stubbed to raise.

## Chosen approach: Option B — a runtime-free `_C` build variant

Reuses the **exact C++ layout math** (bit-identical to on-device), isolating the
runtime purely at the link + `#ifdef` boundary.

Rejected alternatives:

- **Option A (pure-Python `_C` shim):** would force re-implementing
  `SpyreTensorLayout`'s `device_size` / `stride_map` / sticking math in Python —
  exactly the logic that must stay identical to the C++ used on-device.
  Divergence risk not worth it when the SDK headers are available.
- **Option C (split `_C` into `_C_layout` + `_C_runtime` extensions):** cleanest
  long-term separation but a much larger refactor. Option B reaches the same
  outcome with a build flag + `#ifdef`s and can be promoted to C later if the
  split proves durable.

## Environment step (must happen first — implementation is blocked on it)

1. Provision a CPU-only box (or container) with:
   - PyTorch matching the repo's pinned version (provides `libc10`/`libtorch*`).
   - Spyre SDK **headers** on the include path (`sendatatype.hpp` and friends) —
     **headers only**; the runtime `.so`s are intentionally absent.
   - A working C++ toolchain for `CppExtension`.
2. Resolve the one open link question there (cannot inspect `/opt` from here):
   **is `elems_per_stick` header-inline, or a symbol in a datatype lib?**
   - If header-inline → the runtime-free build links **no** Spyre libs.
   - If it lives in a lib → link only that one lightweight **datatype** lib
     (e.g. `libsendnn_tensor` / `libflex`), never the execution/runtime libs.
   Determine by building the layout-only variant and reading the linker errors.
3. Produce the runtime-free `_C.so` on that box (see build flag below) and
   confirm `python3 -c "import torch_spyre._C; torch_spyre._C.DataFormats"`
   succeeds with no `/opt/ibm/spyre` runtime libs present (`ldd` shows none).

## Implementation plan (after environment is ready)

### 1. `setup.py` — `TORCH_SPYRE_NO_RUNTIME` build flag

- Read `TORCH_SPYRE_NO_RUNTIME` (default `0`).
- When `1`:
  - Drop `sendnn`, `sendnn_interface`, `flex`, `spyre_comms`, and the
    deeptools/senlib libs from `LIBRARIES`; keep only ATen/torch (+ at most the
    one datatype lib from step 2 above).
  - Add compile macro `-DTORCH_SPYRE_NO_RUNTIME`.
  - Skip the `SPYRE_COMMS_INSTALL_DIR` / `RUNTIME_INSTALL_DIR` hard requirements
    that currently `raise` when unset, so the build works without a runtime
    install (headers still needed for the datatype enum).

### 2. csrc partition, guarded on `TORCH_SPYRE_NO_RUNTIME`

- **Always compiled (runtime-free):** the layout bindings in `module.cpp`,
  `spyre_views.cpp`, `spyre_tensor_impl.cpp`, `types_mapping.h` — everything
  producing `DataFormats`, `ElementArrangement`, `SpyreTensorLayout`,
  `get_device_dtype`, `get_elem_in_stick`, `encode_constant`.
- **`#ifdef`'d out under the macro, replaced by stubs that still bind but raise
  `RuntimeError("Spyre runtime not available in CPU/NO_RUNTIME build")`:**
  `start_runtime`, `free_runtime`, `prepare_kernel`, `launch_jobplan`,
  `launch_kernel`, `JobPlan`, streams (`spyre_stream.cpp`), allocator
  (`spyre_allocator.cpp`, `spyre_mem.cpp`), `spyre_guard.cpp`,
  `spyre_device_enum.cpp`, CCL (`createSpyreCCLBackend`), and the
  allocator-backed `*_with_layout` / `reinterpret_tensor*` functions.
  Binding-but-raising keeps every top-level `from torch_spyre._C import ...`
  resolvable so the codegen path imports cleanly.
- Verify no *layout* function transitively calls a runtime symbol. Known edge:
  `get_device_dtype` → `elems_per_stick` (see step 2 of the environment
  section).

### 3. `torch_spyre/__init__.py` — import-time guard

- `_lazy_init()` already defers `start_runtime()`; make sure that in the
  no-runtime build `start_runtime`/`set_device` are no-ops (or the guard skips
  them) so `import torch_spyre` succeeds. Eager device ops stay unavailable;
  the compile path works.
- Consider a small env/marker (e.g. reading `TORCH_SPYRE_NO_RUNTIME` at import,
  or probing an attribute on `_C`) so Python code can branch where it must.

### 4. Verification (matches the skill's codegen+verify flow)

On the CPU box, run `bash run.sh my-examples/add.py` (start with `add.py` — the
simplest pointwise case). Expected/harmless end state:

- `AttributeError: 'NoneType' object has no attribute 'run'` at *execution* —
  the documented "backend not complete / no device" signal, not a codegen bug.
- Verify `torch_compile_debug/.../output_code.py` and the `triton-dump/*.ktir`
  for coverage/non-overlap exactly as **SKILL.md → "Verifying kernel coverage
  and non-overlap"** describes.

Then repeat for `sum.py` and `matmul.py`.

## Constraint note

The `csrc` / `setup.py` changes fall outside the usual "only modify
`_inductor_triton/`" rule, which scopes the *Python inductor* work. A
build-variant of the C++ extension is a separate, legitimate concern — confirm
with the team that a `TORCH_SPYRE_NO_RUNTIME` build flag is acceptable before
landing. Keep the default path (`0`) byte-for-byte unchanged.

## Open questions

1. `elems_per_stick` link location (blocks the exact `LIBRARIES` list) — resolve
   on the CPU box.
2. Does any layout path allocate a real tensor (needing the allocator) during
   `propagate_layouts`? If so, that call must be stubbed or routed to a plain
   CPU tensor in the no-runtime build. Audit during implementation.
3. Promote to Option C (two extensions) later? Decide once the `#ifdef`
   partition is proven and stable.
