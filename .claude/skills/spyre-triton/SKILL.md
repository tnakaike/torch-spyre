---
name: spyre-triton
description: "Implementation guide for SpyreTritonKernel: generating layout-aware Triton kernels from OpSpec/LoopSpec metadata. Covers the three-level dimension mapping, reusable components, dev workflow, and constraints."
---

# SpyreTritonKernel Implementation Guide

The design specification is in `2602-OpSpecToTriton.md` (same directory).
This file covers **how to implement and extend that design**: which code to
reuse, how to run and debug, and constraints that keep the implementation
maintainable.

Indirect access (gather) is specified in `2603-IndirectAccess.md` (same
directory). The load-side gather is **implemented** in `spyre_triton_kernel.py`
(`load()` → `_emit_gather_*`); scatter and fusion remain design-only.

`2604-MultiDIndexGather.md` (same directory) extends it to carry the index in
its native multi-D device layout (no rank-1 flatten). Its **frontend is
implemented**; the Triton-verifier relaxation and KTIR multi-D lowering are
TODO, so the current expected end state for a multi-D index is a Triton-verifier
rejection (`x offsets must be 1D`).

LX scratchpad usage is specified in `2605-LXPlanning.md` (same directory).
It is **design only**: fuse so intermediates stay in-kernel as SSA values
(no global memory), and fall back to an LX-tagged tensor descriptor
(`#ktdp.spyre_memory_space<LX>`) only for tensors that must cross a kernel
boundary.

`2606-KernelBundleLXModel.md` (same directory) is the **verified** model of how
the SDSC path forms kernels/bundles and where LX tensors live: one SDSC kernel
unless a FallbackKernel splits it; bundle splits come from the 6-tensor limit
(not LX bytes); LX tensors are shared *across* sub-bundles but **never cross a
kernel/launch boundary** (only HBM does). Read it before reasoning about LX or
fusion boundaries in either path.

`2607-InterCoreReduction.md` (same directory) is the **plan for M6**: honoring a
reduction-dim core split via the HW reduce-sum ring (`tl.inter_tile` +
`tl.wk_slice_coord`). It is **plan only and not yet implementable** — the
enabling backend builtins exist under `../triton/third_party/spyre/` but are
**not upstreamed yet**. Scope: `sum`-only, single split reduction axis,
standalone `SpyreTritonKernel`. Read it before working on inter-core reduction.

`PLAN-KTIR-CPU-Integration.md` (same directory) is the **plan** for executing
the emitted KTIR on CPU via `../ktir-cpu` (NumPy interpreter, no Spyre device)
so we can do a real numerics check instead of stopping at `NoneType…run`. It is
**plan only and blocked on environment setup**. Key decision: build ktir-cpu's
`mlir_ktdp` from the `ktir-mlir-frontend` **inside triton**
(`../triton/third_party/spyre/ktir-mlir-frontend`) so emitter and parser share
one dialect definition; all four repos are moving targets, so no pinning — fix
drift on demand. Seam is `SpyreTritonAsyncCompile.triton()` (which currently
discards `triton.compile()`'s result). Read it before wiring KTIR-CPU execution.

`PLAN-FlexRuntimeIsolation.md` (same directory) is a **BACKLOG** plan
(investigated, file-map verified) for building `torch_spyre._C` with **no libflex
link** via a `TORCH_SPYRE_NO_RUNTIME` flag — keeping only the layout-decision
logic (`SpyreTensorLayout` / `DataFormats` / `elems_per_stick`, all from the
header-only `util/sendefs.h`) and dropping/mocking the runtime. Deferred: the
preferred near-term approach is a **runtime toggle** that routes execution to
ktir-cpu while flex stays linked, so the SDSC+flex path remains available for
comparison in the same build (part of the KTIR-CPU work). This static isolation
is the fallback for flex-less environments. Read it before touching
`torch_spyre/csrc` or `setup.py` for a no-runtime build.

`NEXT-STEP-extern-bmm.md` (same directory) is the **immediate next step**
(investigated, fix pending): decode-phase `F.linear` cases (seq_len == 1, so
`m == 1`) fall back to `extern_kernels.bmm` instead of a Spyre Triton kernel,
because the `m <= 1` guard in `_patched_use_native_matmul`
(`spyre_triton_patches.py`) disables native matmul. Read it before touching the
matmul / `use_native_matmul` path.

## Quick-start

```bash
source ~/dev-env.sh       # required before any python3 invocation
bash run.sh my_script.py  # runs with all debug env vars pre-set
```

`run.sh` sets:

| Env var | Effect |
|---|---|
| `TORCH_SPYRE_TRITON=1` | Activate the Triton path (SpyreTritonScheduling + SpyreTritonKernel) |
| `TORCH_COMPILE_DEBUG=1` | Dump FX graphs, LoopLevel IR, kernels to `torch_compile_debug/` |
| `TRITON_KERNEL_DUMP=1` + `TRITON_DUMP_DIR=./triton-dump` | Dump Triton IR under `triton-dump/` |
| `TORCH_LOGS=output_code` | Print generated kernel source to stdout |
| `SPYRE_INDUCTOR_LOG_LEVEL=DEBUG` | Verbose Python-side Inductor/SpyreKernel logging |
| `TORCH_SPYRE_DEBUG=1` | Enable C++ debug logging |

Uncomment `TORCH_SPYRE_TRITON_FORCE=1` in `run.sh` to force the Triton path even
when `TORCH_SPYRE_TRITON` is not set (useful for single-kernel isolation).

## Key source files

| File | Role |
|---|---|
| `torch_spyre/_inductor_triton/spyre_triton_kernel.py` | **Only file you should modify** — `SpyreTritonKernel` subclass |
| `torch_spyre/_inductor_triton/spyre_triton_scheduler.py` | Wires `SpyreTritonKernel` into `TritonScheduling.create_kernel_choices()` |
| `torch_spyre/_inductor_triton/spyre_triton_wrapper.py` | Wrapper codegen (injects `SpyreTritonAsyncCompile`) |
| `torch_spyre/_inductor_triton/spyre_triton_patches.py` | `spyre_triton_patches()` — patches `use_native_matmul` for spyre |
| `torch_spyre/_inductor_triton/async_compile.py` | Triton compilation entry point for Spyre |
| `torch_spyre/_inductor/spyre_kernel.py` | `SpyreKernel` — primary source of reusable components |
| `torch_spyre/_inductor/op_spec.py` | `OpSpec`, `LoopSpec`, `TensorArg` data classes |
| `torch_spyre/_inductor/views.py` | `compute_coordinates()`, `align_tensors()` |
| `torch_spyre/_inductor/pass_utils.py` | `iteration_space()`, `apply_splits_from_index_coeff()` |
| `../triton/third_party/spyre/` | Spyre-specific Triton backend (KTDP lowering). **Ours to modify** — e.g. `lib/Dialect/KTDP/Transforms/LowerDescriptorMemory.cpp` |
| `../torch-spyre-triton/torch_spyre/_inductor/spyre_triton_kernel.py` | Prototype — good reference point for patterns |
| `/home/nakaike/dt-inductor/pytorch/torch/_inductor/` | PyTorch Inductor source — use for reading `TritonKernel`, `IterationRangesEntry`, `triton_heuristics.py`, etc. |

## Constraint: only modify `torch_spyre/_inductor_triton/`

Python-side implementation work must stay inside
`torch_spyre/_inductor_triton/`. Do **not** modify files in
`torch_spyre/_inductor/` (the existing non-Triton path must remain unchanged).

The Spyre Triton backend under `../triton/third_party/spyre/` **is** our
development area — there is **no high barrier** to modifying it. The KTDP
lowering passes there (notably
`lib/Dialect/KTDP/Transforms/LowerDescriptorMemory.cpp`) are fair game when a
feature needs backend support — for example, teaching `make_tensor_descriptor`
lowering to honor an LX memory space (see `2605-LXPlanning.md`). Keep changes
backward-compatible (default to existing HBM behavior).

Import freely from `torch_spyre/_inductor/` — its public API is stable:
`SpyreKernel`, `SpyreOpFuncs`, `TensorAccess`, `PointwiseOp`, `ReductionOp`,
`UnimplementedOp`, `compute_coordinates`, `align_tensors`, `iteration_space`,
`apply_splits_from_index_coeff`, `simplify_op_spec`.

**Logger:** use `get_inductor_logger` from `torch_spyre._inductor.logging_utils`,
not the stdlib `logging` module. `logging_utils` lives in `_inductor/`, not in
`_inductor_triton/`, so the import is:

```python
from torch_spyre._inductor.logging_utils import get_inductor_logger
logger = get_inductor_logger("spyre_triton_kernel")
```

`SPYRE_INDUCTOR_LOG_LEVEL=DEBUG` (set by `run.sh`) activates `logger.debug()`
output. Stdlib `logging.getLogger(__name__)` will not respond to that variable.

## Reusable components from `spyre_kernel.py`

Before adding code to `spyre_triton_kernel.py`, check whether the equivalent
already exists in `SpyreKernel`:

| You need | Reuse from `spyre_kernel.py` |
|---|---|
| Build `TensorArg` from a loaded buffer | `create_tensor_arg()` — calls `compute_coordinates()`, fills `device_size`, `device_coordinates`, `allocation` |
| Build `OpSpec` from args + op name | `create_op_spec()` — calls `apply_splits_from_index_coeff()`, handles `tiled_symbols` from `loop_info` |
| Wrap op_specs in a `LoopSpec` | `wrap_op_specs_in_loop()` |
| Simplify/align TensorArgs | `simplify_op_spec()` — call once per OpSpec before emitting |
| Look up device dtype / allocation | `tensor.layout.device_layout.device_dtype`, `tensor.layout.allocation` |
| Resolve mutation aliases | `V.graph.scheduler.mutation_real_name.get(name, name)` |
| Get the OpSpec index for a dep | `current_node.read_writes.reads` / `.writes` — iterate to find `dep.name == name` |

The prototype (`../torch-spyre-triton/…/spyre_triton_kernel.py`) inlines copies
of these helpers (`_create_tensor_arg`, `_create_op_spec`). In `torch-spyre`,
prefer calling the originals on a `SpyreKernel` instance or copying the minimal
delta rather than re-implementing.

## Three-level dimension mapping

The core algorithm maps:

```
Level 1 (Triton)   — xoffset / roffset (scalar program base)
Level 2 (Original) — per-original-dim scalar offsets from IterationRangesEntry
Level 3 (Device)   — device_coordinates.subs({c0: d0, c1: d1, ...})
```

### Building `triton_opspec_map`

`triton_opspec_map: dict[str, list[sympy.Symbol]]` maps each Triton prefix
(`"x"`, `"y"`, `"r0_"`, …) to the list of OpSpec symbols it covers.

**Default: index-coefficient method** (sufficient for current op coverage):

```python
it_space = iteration_space(self.current_node)   # {c0: range, c1: range, ...}
# For each Triton symbol t, find opspec symbol c_i such that
#   opspec_index.coeff(c_i) == triton_index.coeff(t)
# Uses self.range_tree_nodes[t].length for range, index.coeff(t) for stride.
```

See the prototype's `_create_triton_opspec_map()` for a working implementation
using `self.numels` and `V.graph.sizevars.size_hint()`.

**Alternative: construction-time hook** — override
`IterationRangesRoot.construct_entries(lengths)` to record
`entry.symbol() → (prefix, original_dim_idx)` at construction time. Use this
when tensors have equal strides, surviving size-1 dims, or broadcasts.

### XBLOCK override

Upstream heuristics (`triton_heuristics.py`) choose block sizes for GPUs —
they must be bypassed for Spyre. Set `XBLOCK` from `iteration_space`:

```
XBLOCK = product over all (range / core_divisor) in iteration_space
```

Store the computed value in `V.graph._spyre_triton_block_size` early (during
the first `store()` call, inside `__enter__` is too early). The heuristics
hook reads it back at autotuning time.

See `_get_triton_block_size()` in the prototype for the full computation using
`apply_splits_from_index_coeff()`.

### Per-core device shape (`block_shape`)

```python
# For each device dimension k:
#   find the first OpSpec symbol referenced by device_coordinates[k]
#   divide device_size[k] by that symbol's core_divisor
per_core_device_shape[k] = device_size[k] // core_divisors[first_symbol_of_dim_k]
```

**Within-stick dimension exception:** Distinguish two things that the word
"stick" loosely refers to (see
`docs/source/user_guide/tensors_and_layouts.md` and the `outer stick dim`
handling in `_inductor/views.py`):

- The **within-stick dimension** is the last device dimension
  (`device_rank - 1`). It holds the `var % stick_size` part and its
  `device_size` is always the max elements per stick — 128 bytes / dtype
  element size (64 at fp16/bf16, 32 at fp32, 128 at int8). This dimension is
  *never split*: `block_shape[device_rank - 1]` must always equal
  `device_size[-1]`, regardless of what the core divisor for the symbol in
  that coordinate would compute to. Failing to enforce this produces
  `block_shape` values like `[2, 1024, 2]` when the correct shape is
  `[2, 1024, 64]` — a silent correctness bug.
- The **outer stick dimension** is the tile-index part (`var // stick_size`)
  of whichever PyTorch dim is being sticked. It is a *separate* device
  dimension at an **arbitrary outer position — not `device_rank - 1`**. For
  example, PyTorch `[128, 256, 512]` → `device_size=[256, 8, 128, 64]`: the
  sticked dim (PyTorch dim 2) splits into device dim 1 (`8`, outer) and device
  dim 3 (`64`, within-stick). Work division *is* stick-aligned along this
  outer dimension, so individual cores own a whole number of sticks here and
  it may legitimately be divided by its core divisor.

Do not assume the outer stick dimension is at any fixed position; only the
within-stick dimension is pinned to `device_rank - 1`.

When `LoopSpec` is active, further divide by `loop_count` for tiled symbols
(non-stick dims only):

```python
per_loop_block_shape[k] = per_core_device_shape[k] // loop_count   # if dim k is tiled, k != last_dim
```

### Emitting tensor descriptors

```python
desc = tl.make_tensor_descriptor(
    base_ptr,
    shape=device_size,                         # full device tensor shape
    strides=row_major_strides(device_size),    # [prod(device_size[k+1:]), ..., 1]
    block_shape=per_loop_block_shape,
)
val = tl.load_tensor_descriptor(desc, device_offsets)   # device_offsets from Level 3
```

Hoist descriptors outside `LoopSpec` loops when `shape`, `strides`, and
`block_shape` are loop-invariant (they always are — only offsets change).

## Passing OpSpec metadata to the Triton runtime

`SpyreTritonKernel.codegen_body()` injects serialized `op_specs` and
`triton_opspec_map` into `self.triton_meta["spyre_options"]` before calling
`super().codegen_body()`. The Spyre Triton backend then picks this up at
compile time.

Use wrapper classes with `__repr__` returning `sympify('…')` calls for sympy
expressions — this is the same pattern used in the prototype
(`SympyExpr`, `OpSpecDict`, `TritonOpSpecMapDict`).

At `codegen_kernel()`, prepend the necessary imports:

```python
code.splice("from torch_spyre._inductor.op_spec import TensorArg, OpSpec, LoopSpec")
code.splice("from torch_spyre._inductor.spyre_kernel import UnimplementedOp")
code.splice("from torch_spyre._C import DataFormats")
code.splice("from sympy import sympify")
```

## `SpyreTritonKernel.__enter__` pattern

Use `super(TritonKernel, self).__enter__()` (skip `TritonKernel.__enter__`)
and install a custom ops handler:

```python
def __enter__(self):
    super(TritonKernel, self).__enter__()
    self.exit_stack.enter_context(
        V.set_ops_handler(SpyreTritonCSEProxy(self, SpyreTritonOverrides()))
    )
    self.exit_stack.enter_context(V.set_kernel_handler(self))
    return self
```

`SpyreTritonCSEProxy` intercepts pointwise ops to record
`cse_var → PointwiseOp` in `self.cse_var_to_pointwise`, which `store()` reads
back to recover the Spyre op name without re-implementing the op dispatch.

## `load()` / `store()` / `store_reduction()` structure

Each override should:

1. Call `super().load/store/store_reduction()` to produce the standard Triton
   code path (preserving upstream IR generation).
2. Build a `TensorAccess` using the `MemoryDep` index
   (`current_node.read_writes.reads/writes`), not the Triton index — the
   MemoryDep index uses OpSpec symbols (`c0`, `c1`, …), which `compute_coordinates()`
   needs.
3. Call `_create_tensor_arg()` / `_create_op_spec()` (or equivalent) to build
   and accumulate `self.op_specs`.
4. On first `store()`, call `_create_triton_opspec_map()` and store the block
   size into `V.graph._spyre_triton_block_size`.

Do not call `self.args.output(name)` unconditionally — check
`"pool" not in layout.allocation` first, matching `SpyreKernel.store()`.

## Matmul lowering: why `lower_mm` is skipped in the Triton path

`torch_spyre/_inductor/lowering.py` registers `lower_mm` for `aten.mm`.  It
creates `Reduction(reduction_type=BATCH_MATMUL_OP, input_node=[x, y],
inner_fn=tuple_fn)` — a Spyre-specific IR node with a tuple-valued inner
function that `SpyreKernel` (SDSC) understands.  `TritonKernel` cannot codegen
a tuple-valued inner_fn and will crash.

The Triton path needs PyTorch's **standard `aten.mm` lowering**, which creates
`Reduction(reduction_type='dot')`.  With `use_native_matmul=True` (patched in
`spyre_triton_patches.py`), `TritonKernel` converts that to `tl.dot`.

### How the switch is implemented

`spyre_triton_patches()` is entered **before** `enable_spyre_lowerings()` in
`_inductor/patches.py`.  At its entry, it pops `aten.mm.default` and
`aten.bmm.default` from `spyre_lowerings` so that `enable_spyre_lowerings()`
installs them without those overrides.  They are restored on exit.

See `_TRITON_SKIP_MM_OPS` in `spyre_triton_patches.py`.

### Could `lower_mm` be removed entirely (option 2)?

Yes, in principle.  `SpyreKernel` could be taught to handle
`reduction_type='dot'` (the standard native-matmul marker) and emit
`op='matmul'` in the OpSpec, eliminating `lower_mm` and `lower_bmm` from
`spyre_lowerings` altogether.  This would also require patching
`use_native_matmul` for the non-Triton path (currently only patched when
`TORCH_SPYRE_TRITON=1`).  This is a worthwhile future cleanup, but it requires
changes inside `_inductor/` (SpyreKernel + patches.py).  The current split
(option 3) is minimal and keeps `_inductor/` untouched.

### `SpyreTritonKernel.reduction()` and `reduction_type`

`reduction()` in `spyre_triton_kernel.py` currently asserts
`reduction_type == "sum"` and handles only that case (for `tl.sum` ops like
`torch.sum`).  When matmul is active, `reduction_type` is `"dot"`.  The
matmul case must be handled separately — see the matmul implementation notes
below.

## LoopSpec handling

`LoopSpec` enters `op_specs` via `wrap_op_specs_in_loop()` in `SpyreKernel`.
`SpyreTritonKernel` must emit the corresponding `for` loop in the Triton source.
The loop variable advances tiled device offsets:

```python
sym_step = (device_total_per_core) / loop_count   # elements per iteration
# offset for tiled device dim: base_offset + loop_var * sym_step
```

Read `OpSpec.tiled_symbols` (or `LoopSpec.tiled_symbols`) to identify which
iteration-space symbols are tiled by the enclosing loop.

## Test cases

Runnable examples live in `my-examples/`. They all use `test_harness.py`
which compiles the function with `torch.compile`, runs it on both CPU and
Spyre, and reports the max delta.

| File | What it tests | Notes |
|---|---|---|
| `add.py` | Elementwise add (1024×4096, fp16) | **Simplest case — start here when debugging** |
| `sum.py` | Reduction `torch.sum(dim=1)` (1024×4096, bfloat16) | `skip_eager=True` |
| `matmul.py` | Matrix multiply (64×128 @ 128×256, fp16) | `skip_eager=True` |

Run any of them with:

```bash
bash run.sh my-examples/add.py
bash run.sh my-examples/sum.py
bash run.sh my-examples/matmul.py
```

When you hit a regression or an unexpected failure, **fall back to `add.py`
first** — it exercises only elementwise pointwise codegen with no reduction or
layout complexity, making it the fastest way to isolate whether a problem is
in the core load/store/opspec path or op-specific logic.

## Debugging workflow

1. Run `bash run.sh my_script.py > tmp.log 2>&1` — redirect both stdout and
   stderr to `tmp.log`, then check it with `Read tmp.log` or `grep`. Do not
   wait for stdout; `run.sh` takes a long time and buffering makes real-time
   output unreliable.
2. Check `torch_compile_debug/run_*/torchinductor/*/` for:
   - `fx_graph_runnable.py` — post-Dynamo FX graph
   - `output_code.py` — final Inductor-generated wrapper + kernel
3. Check `triton-dump/*/` for the Triton IR (`.ttir` / `.ttgir`) emitted by
   the Spyre Triton backend.
4. Set `SPYRE_INDUCTOR_LOG_LEVEL=DEBUG` to get per-load/store logs showing
   `triton_is`, `opspec_is`, `triton_index`, `opspec_index`.

For a minimal reproducer, use `SENCORES=2` to shrink the work-division space
and reduce dump verbosity.

## Verifying kernel coverage and non-overlap

After generating a kernel (execution failure is expected and harmless), verify
that every output element is written by exactly one program.

**Step 1 — locate the IR.**

- Spyre KTIR: `triton-dump/<hash>/triton_unk_fused_*.ktir` (set by `TRITON_DUMP_DIR`)
- Standard TTIR: `find /tmp/torchinductor_<user>/triton/ -name "*.ttir"`
  (written by Triton's own cache even when `triton-dump` is not used)

**Step 2 — read the grid and per-program offset.**

From the KTIR `attributes {grid = [...]}` line and the offset arithmetic:

```
# 1D kernel (pointwise, reduction)
grid = [G]                              # G programs on x-axis
xoffset = program_id(x) * XBLOCK       # program i → elements [i*XBLOCK, i*XBLOCK+XBLOCK)

# 2D kernel (native matmul)
grid = [Gx, Gy]
y_offset = program_id(y) * YBLOCK      # y-program i → M-rows [i*YBLOCK, ...)
x_offset = program_id(x) * XBLOCK      # x-program j → N-cols [j*XBLOCK, ...)
```

**Step 3 — check coverage and non-overlap.**

- **Full coverage:** `G * XBLOCK == total_numel` for 1D; `Gy*YBLOCK == M` and
  `Gx*XBLOCK == N` for 2D matmul.
- **No overlap:** each output index maps to exactly one program — true when
  the per-program range `[i*BLOCK, i*BLOCK+BLOCK)` is a contiguous partition
  with no two programs sharing the same `i`.

**Worked examples (verified kernels):**

| Kernel | Shape | Grid | Per-program tile | Coverage | Overlap-free |
|---|---|---|---|---|---|
| `sum.py` | [128,256]→[128] | `[32]` | 4 rows (XBLOCK=4) | 32×4=128 ✓ | row `r` → prog `r//4` ✓ |
| `matmul.py` | [256,1024]@[1024,512] | `[1,32]` | 8M-rows×512N-cols | 32×8=256M, 1×512=512N ✓ | (m,n) → prog `(0, m//8)` ✓ |

**Note on sum.py arithmetic correctness (separate from partitioning):**
The `tt.reduce(axis=0)` in the sum TTIR reduces across K-sticks (device dim 0)
but not within each stick (64 elements, device dim 2). Only position
`[0, row, 0]` of the output stick holds a meaningful partial sum; positions
`[0, row, 1..63]` are stick-element-wise partial sums, not a single scalar.
The full intra-stick reduction is not yet emitted by `SpyreTritonKernel`.

## Expected errors (backend not yet complete)

`AttributeError: 'NoneType' object has no attribute 'run'` is the **expected
runtime error** when the Spyre Triton backend has not yet fully implemented a
given op.  It means the kernel compiled to `None` (async compilation returned
nothing).  This is NOT a Python codegen bug — the `output_code.py` kernel
source may be correct.  Check `output_code.py` to verify the generated Triton
source, then proceed to implement the missing backend lowering.

## Common pitfalls

- **Using Triton index instead of MemoryDep index for `compute_coordinates()`:**
  `TritonKernel.load/store` receives a Triton index over `x0`, `x1`, etc.
  `compute_coordinates()` expects an index over OpSpec symbols `c0`, `c1`, etc.
  Always fetch the index from `current_node.read_writes.reads/writes`.

- **Calling `args.output()` for scratchpad/pool buffers:** Follow
  `SpyreKernel.store()` — only register as output when
  `"pool" not in layout.allocation`.

- **`triton_opspec_map` built before `current_node` is set:** Build it lazily
  on first `store()` call (not in `__init__` or `__enter__`).

- **Missing `simplify_op_spec()` call:** Call it on every `OpSpec` before
  serializing in `codegen_body()` — it aligns singleton dimensions and
  reorders `device_coordinates` so device descriptors are consistent.

- **Dividing the within-stick dimension in `block_shape`:** The innermost
  device dimension (`device_rank - 1`) is always the within-stick dimension
  (64 elements at fp16/bf16, 32 at fp32, 128 at int8).  Work division is
  stick-aligned so it never splits this dimension.  Always set
  `block_shape[-1] = device_size[-1]` unconditionally.  Forgetting this
  produces silent wrong block sizes like `[2, 1024, 2]` when the correct value
  is `[2, 1024, 64]`.  Note this is *only* the within-stick dim — the **outer
  stick dimension** (the tile-index part of the sticked PyTorch dim) sits at an
  arbitrary outer position, is not `device_rank - 1`, and *is* divided across
  cores.

- **Re-implementing helpers already in `spyre_kernel.py`:** Prefer imports and
  delegation over inline copies; divergence causes silent correctness bugs.
