# SDSC Kernels, Bundles, and LX — Verified Model

Status: **verified by inspection** (June 2026). This records how the SDSC path
forms kernels and bundles, where LX tensors live, and how that contrasts with
the Triton path. It backs the assumptions in
[`2605-LXPlanning.md`](2605-LXPlanning.md). Findings marked *(verified)* were
confirmed by reading generated `output_code.py`; others cite
[`scratchpad_planning.md`](../../../docs/source/compiler/scratchpad_planning.md).

**Update (June 2026):** §1 and §6 were revised and §7 added. The earlier claim
that the Triton path "emits one kernel per group and never sub-splits" is
**wrong** — a `FusedSchedulerNode` does not meet `TritonKernel` requirements. A
budget-group maps to a `SpyreTritonKernelBundle` of `SpyreTritonKernel`s
executed in **one launch** with shared LX (the Triton analogue of the SuperDSC
bundle). Design in [`2605-LXPlanning.md`](2605-LXPlanning.md); enabler/gaps in
§6; a worked code example (per-bundled-kernel grids) in §7.

---

## 1. Three nested levels

```
ATen ops  →  SchedulerNodes  →  FusedSchedulerNode (scheduler group)  →  codegen
              (one per op)        (spyre_fuse_nodes: tensor-budget)

codegen splits the group by path:
  SDSC:   1 kernel per group, split into 1+ SuperDSC bundles
  Triton: 1+ SpyreTritonKernels per group, in 1 SpyreTritonKernelBundle (1 launch)
```

- **Scheduler group** = Inductor's fusion unit (`FusedSchedulerNode`).
  `SpyreHeuristics.can_fuse` in `choices.py` returns **`False`** (Inductor's own
  fusion is disabled); the grouping is actually done by the post-fusion pass
  `spyre_fuse_nodes` in `fusion.py`, driven by the **6-tensor-per-bundle limit**
  — *not* iteration-space compatibility. **Path-agnostic** — both paths install
  `CustomPostFusionPasses` (→ `spyre_fuse_nodes`) and `SpyreHeuristics`
  unconditionally (`patches.py:93,141`; only the matmul patch is
  `TORCH_SPYRE_TRITON`-gated), so the two paths receive byte-identical
  post-fusion groups *(verified: the `lx_cross_kernel` example produced the same
  `op0_op1_op2` group in both)*.
- **Sub-bundle (SuperDSC bundle)** = the SDSC backend's partition of one
  group's OpSpec sequence, bounded by the **6-tensor-per-bundle limit**. The
  Triton path needs an **analogous sub-level** (it was previously claimed it did
  not — see §6): a budget-group's OpSpecs span **heterogeneous iteration spaces /
  core mappings**, which a single upstream `TritonKernel` cannot codegen, so the
  group is sub-split into same-iteration-space `SpyreTritonKernel`s held in one
  `SpyreTritonKernelBundle` (one launch, shared LX).

## 2. The SDSC path emits a single kernel unless an op falls back

For a graph fully lowerable to SDSC ops, **everything bundles into one kernel**.
Kernel boundaries land **only at FallbackKernels** (ops with no Spyre lowering,
run via eager aten) — matching `scratchpad_planning.md`'s "bundle boundaries
should only land at FallbackKernels."

*(verified)* by reading `output_code.py` for three graphs:

| Graph | SDSC kernels | Note |
|---|---|---|
| `sum(x,0); sum(y*y)` | **1** (`sdsc_fused_mul_sum_0`) | 3 ops fused; intermediates LX/internal |
| `(a @ b) @ c` | **1** (`sdsc_fused_mm_0`) | two `batchmatmul` ops in one kernel; `a@b` in `{'pool': 0}` |
| `x*2; cumsum(.,1); .+1` | **3** | `aten.cumsum` is a fallback → splits into sdsc / aten / sdsc |

In the fallback case the wrapper threads intermediates between kernels as
**HBM buffers** (`buf1`, `buf3`, … passed to `.run()`), e.g.:

```python
sdsc_fused_mul_0.run(arg0_1, buf0, buf1)        # SDSC
buf2 = torch.ops.aten.cumsum.default(buf1, 1)   # FallbackKernel (eager)
sdsc_fused_add_1.run(buf3, buf4, buf5)          # SDSC
```

## 3. Matmul asymmetry: SDSC fuses it, Triton does not

A consequence of `lower_mm`:

- **SDSC:** `lower_mm` lowers `aten.mm` to a Spyre `batchmatmul` reduction IR
  node, which the scheduler **fuses** like any reduction. `(a@b)@c` → **one
  kernel** *(verified)*.
- **Triton:** `lower_mm` is skipped (`use_native_matmul` → `tl.dot`), and
  Inductor templates each matmul as **its own scheduler group**. `(a@b)@c` →
  **two kernels**, with `a@b` crossing a kernel boundary.

So a matmul chain crosses a boundary in **Triton but not SDSC** — it is the
genuine cross-kernel (Principle-2) scenario for the Triton path, and the one
SDSC sidesteps by bundling. There is **no single graph that crosses in both
paths via the same mechanism**: a FallbackKernel splits both, but a fallback
output is an HBM tensor, not an LX one.

## 4. The corrected LX model

```
single SDSC kernel  (unless a FallbackKernel forces a split)
   └─ split into ≥2 SuperDSC bundles  ← 6-tensor limit, NOT LX byte capacity
        └─ LX tensors PERSIST and are SHARED across those bundle boundaries
           (that is the whole point of LX planning)
```

Four points, with the common misconceptions corrected:

1. **Single kernel unless fallback.** ✓ (§2)
2. **In a kernel, torch-spyre uses LX as much as possible.** ✓ — pin as much as
   fits in ~1.6 MB to cut HBM traffic (scratchpad optimization, Job 2).
3. **Bundle split is driven by the 6-tensor limit, NOT LX bytes.** The
   6-tensor limit caps *how many tensors* a SuperDSC bundle references. LX byte
   capacity is handled **by tiling** (coarse/pre-Inductor tiling shrinks the
   working set to fit ~1.6 MB), not by splitting bundles. Softmax over
   `(512,1024)` splits into bundles because it touches >6 tensors, even though
   its data fits in LX.
4. **Bundles SHARE LX tensors — that is why the planner exists.** Per
   `scratchpad_planning.md`: "the planner assumes LX state persists across
   SuperDSC bundle boundaries… allocation decisions can span multiple bundles."
   Softmax stages 2–4 keep intermediates on LX *across* the bundles it is
   decomposed into; that cross-bundle reuse is the 27% speedup. (Common wrong
   intuition: "bundles split because LX is full, so they can't share LX" — both
   halves are false.)

## 5. Where LX lives, and what crosses what

- **LX tensor** = `allocation={'lx': ...}`, `arg_index=-1` — a baked per-core
  address, **never a kernel argument** *(verified in `output_code.py`)*. It
  therefore **cannot appear in a host-level `.run()` signature and cannot cross
  a kernel (launch) boundary.** LX persistence is entirely **within one
  kernel/launch**, across sub-bundles.
- **HBM tensor** = the only thing that crosses kernel boundaries (graph
  inputs/outputs, and intermediates between fallback-split kernels).
- **Caveat:** within-launch LX persistence is an *assumption* with a known gap
  — VF multi-tenancy may wipe LX at a bundle boundary. It holds because the
  bundles are one launch, but is not hardware-guaranteed.

## 6. Implications for the Triton path (revised June 2026)

The earlier version of this section assumed Triton "emits one kernel per group
and never sub-splits." That is **wrong**: a budget-group's OpSpecs span
heterogeneous iteration spaces / core mappings, and a single upstream
`TritonKernel` requires **one `(numel, rnumel)`, one tiling, one reduction
barrier**. So a `FusedSchedulerNode` does **not** map to one `TritonKernel`.

**One launch is mandatory.** torch-spyre cannot manage LX tensors across
kernel-launch boundaries (§5; cross-launch persistence is unverified and a
*harder* boundary than SDSC sub-bundles). So the Triton path must keep a whole
budget-group in **one launch** to reuse LX — it cannot split a group into
separate Triton launches.

**Two new constructs** (design; not yet implemented — see `2605-LXPlanning.md`):

- **`SpyreTritonKernel`** = a maximal run of **same-iteration-space** ops fused
  into one Triton kernel (one grid). This is a capability `SpyreKernel` lacks:
  SDSC has no "fused op" — each SDSC is exactly one op. A `SpyreTritonKernel`
  therefore corresponds to *several* same-iteration-space SDSCs collapsed into
  SSA dataflow (Principle 1 — intermediates are temps, no LX tensor).
- **`SpyreTritonKernelBundle`** = the sequence of `SpyreTritonKernel`s for one
  budget-group, executed in **one launch** with LX shared across them. This is
  the **direct Triton analogue of the SuperDSC bundle** (§4): LX persists and is
  shared across sub-kernels but never crosses the launch.

**The grid is the core mapping.** In the Spyre Triton backend
`prod(SpyreOptions.grid)` is the physical core count, consumed by `DistributeWork`
(`triton/third_party/spyre/lib/Dialect/KTDP/Transforms/DistributeWork.cpp`),
which lowers `tt.get_program_id` → `ktdp.get_compute_tile_id` and stamps a `grid`
attribute per function. So giving each `SpyreTritonKernel` its own grid **is**
giving it its own work division — the Triton analogue of SDSC's per-op core
remap (each SDSC carries its own `num_cores` / `work_slices`).

**Enabler / remaining gaps** (verified against the backend):

1. **Per-function grid.** `DistributeWork` today takes a single module-wide grid
   and asserts `grid.size() == numDims` for *every* function (invariant (c)), so
   all functions in a module must share one grid shape — but a bundle's
   sub-kernels have different grids. The pass already walks per-function and
   stamps a per-function `grid` attribute, so the change is localized: source
   the grid from a frontend-set `spyre.grid` attribute per function instead of
   the shared pass option. (The `DistributeWork.cpp` header calls this out and
   cites `PLAN_kernel_examples.md` G4, which does **not** exist in the tree — the
   pass body is the spec.)
2. **Multiple bundled-kernel functions + sequencing entry.** The module must hold several
   bundled-kernel functions plus a top-level entry that calls them in order (the
   launcher invokes one `get_entry_func_name()`), so sequencing happens
   device-side inside one launch.
3. **Inter-bundled-kernel barrier.** `DistributeWork` synthesizes no sync between
   bundled kernels. A barrier is needed only between bundled kernels with a **true (cross-core)**
   dependency; same-iteration-space-fused and core-local crossings need none.
4. **Cross-core movement via `tl.inter_tile`.** A bundled kernel boundary that
   **redistributes** data across cores (matmul-chain K split, cross-core
   reduction) is expressed with an explicit `tl.inter_tile` op over the data
   ring — **not** by tagging a descriptor with a foreign core. `reduce_to_one` /
   `all_reduce` exist today (M6); `reduce_scatter` / `broadcast` (needed for
   general redistribution) are the future enabler. `memory_space` stays
   core-agnostic (`{LX, HBM}`): each core only addresses its own LX, so
   same-work-division crossings use the plain `<LX>` baked-address descriptor and
   need no ring op.

**Host-runtime impact is small:** still one `triton.compile` → one
`CompiledKernel` → one `.run()` → one launch → one stream. A bundle adds
*bundled kernels inside one launch*, not launches; the work is in codegen (per-bundled-kernel
grids/configs) and the backend (items 1–4).

**Target representation** for the cross-kernel / Regime-2 LX tensor stays the
SDSC `output_code.py` form: `TensorArg(arg_index=-1, allocation={'lx': 0},
device_coordinates=[...])` — baked address, dropped from the kernel signature.

## 7. Worked example — `SpyreTritonKernelBundle` and per-bundled-kernel grids

**Status: illustrative / proposed.** Steps 2–4 below extend the existing
single-grid plumbing (`spyre_grid` in `triton_meta` → `SpyreOptions.grid` →
`DistributeWork(grid)`); only the single-grid version exists today. The example
shows the intended shape and answers the central question: **how does each
sub-kernel get its own grid?**

**Key idea:** a sub-kernel's grid is per-bundled-kernel *compile-time metadata*, not a
host launch argument. The host still issues one `.run()` (one launch, reserving
the max core budget). Each bundled kernel carries its own grid that `DistributeWork`
reads to resolve that bundled kernel's `tl.program_id`.

### 7.1 Generated Triton bundle source (`output_code.py`)

The whole bundle is registered with one
`async_compile.triton('bundle_0', '''…''', device_str='spyre')` call, exactly
like `results/add/output_code.py`; loads/stores use the descriptor method form
(`desc.load` / `desc.store`) the generated code emits. The per-bundled-kernel grids ride
in the **entry's `triton_meta`** (the `@triton_heuristics.fixed_config`
decorator on `bundle_0`) — the same channel the single-kernel path uses for
`spyre_grid`. `async_compile.triton()` reads them back from `triton_meta` and
forwards them as compile options (§7.2). The decorator below is trimmed; the
`inductor_meta` autotune fields are elided.

All memory is reached through `tl.make_tensor_descriptor`. The LX intermediate
uses the `memory_space="lx"` attribute (2605 §5 Option A / M1) over a **baked LX
offset** — it is *not* a kernel argument and is dropped from the wrapper
allocation. HBM tensors use the default (HBM) memory space.

```python
bundle_0 = async_compile.triton('bundle_0', '''
import triton
import triton.language as tl
from torch._inductor.runtime import triton_heuristics
from torch._inductor.runtime.hints import DeviceProperties

# Each bundled kernel is a SEPARATE function (noinline) so it survives to KTIR as its
# own func.func — DistributeWork can then stamp a per-function grid. If inlined,
# all program_id calls collapse into one function = one grid.

# Baked per-core LX byte offset (compile-time constant, NOT a kernel arg). Both
# bundled kernels build a descriptor over the SAME offset; the buffer never appears in a
# kernel signature or the wrapper allocation (2605 §5).
LX_TMP: tl.constexpr = 0       # = layout.allocation["lx"] for the intermediate

# bundled kernel #0 : pointwise chain   grid = (32,)   → 32 cores on axis x
@triton.jit(noinline=True)
def _stk0_pointwise(in_ptr, M, K, XBLOCK: tl.constexpr):
    pid = tl.program_id(0)                  # resolved against THIS bundled kernel's grid (32,)
    in_desc = tl.make_tensor_descriptor(    # HBM input (default memory_space)
        in_ptr, shape=[M, K], strides=[K, 1], block_shape=[XBLOCK, K])
    lx_desc = tl.make_tensor_descriptor(    # LX intermediate: baked base + tag
        LX_TMP, shape=[M, K], strides=[K, 1], block_shape=[XBLOCK, K],
        memory_space="lx")
    x = in_desc.load([pid * XBLOCK, 0])
    lx_desc.store([pid * XBLOCK, 0], tl.exp(x))         # stays on LX

# bundled kernel #1 : matmul            grid = (1, 32) → 32 cores on axis y
@triton.jit(noinline=True)
def _stk1_matmul(w_ptr, out_ptr, M, N, K,
                 XBLOCK: tl.constexpr, YBLOCK: tl.constexpr):
    pid_m = tl.program_id(0)                # axis 0 → grid[0] = 1
    pid_n = tl.program_id(1)                # axis 1 → grid[1] = 32
    lx_desc = tl.make_tensor_descriptor(    # SAME baked offset + tag → no HBM
        LX_TMP, shape=[M, K], strides=[K, 1], block_shape=[YBLOCK, K],
        memory_space="lx")
    w_desc = tl.make_tensor_descriptor(
        w_ptr, shape=[K, N], strides=[N, 1], block_shape=[K, XBLOCK])
    out_desc = tl.make_tensor_descriptor(
        out_ptr, shape=[M, N], strides=[N, 1], block_shape=[YBLOCK, XBLOCK])
    a = lx_desc.load([pid_m * YBLOCK, 0])
    b = w_desc.load([0, pid_n * XBLOCK])
    acc = tl.dot(a, b)
    out_desc.store([pid_m * YBLOCK, pid_n * XBLOCK], acc)

# bundle entry : the ONE thing the host launches (no lx_tmp arg — LX is baked).
# Its triton_meta carries the per-bundled-kernel grid table — the same channel the
# single-kernel path uses for 'spyre_grid'.
@triton_heuristics.fixed_config(
    config={'XBLOCK': 16384, 'YBLOCK': 8},
    filename=__file__,
    triton_meta={
        'signature': {'in_ptr': '*fp16', 'w_ptr': '*fp16', 'out_ptr': '*fp16',
                      'M': 'i32', 'N': 'i32', 'K': 'i32',
                      'XBLOCK': 'constexpr', 'YBLOCK': 'constexpr'},
        'device': DeviceProperties(type='spyre', index=0, cc=''),
        # per-bundled-kernel grids (read back by async_compile, §7.2):
        'spyre_grids': {'_stk0_pointwise': (32,), '_stk1_matmul': (1, 32)},
        'spyre_entry': 'bundle_0',
    },
    # inductor_meta={...},  # autotune hints etc. — elided
)
@triton.jit
def bundle_0(in_ptr, w_ptr, out_ptr, M, N, K,
             XBLOCK: tl.constexpr, YBLOCK: tl.constexpr):
    _stk0_pointwise(in_ptr, M, K, XBLOCK)
    spyre.barrier()                     # bundled kernel #1 reads LX written by bundled kernel #0
    _stk1_matmul(w_ptr, out_ptr, M, N, K, XBLOCK, YBLOCK)
''', device_str='spyre')
```

`_stk0` partitions the 32 cores 1-D `(32,)`; `_stk1` partitions the *same* 32
cores as `1×32`. That per-bundled-kernel remap is exactly SDSC's per-op work division.

**Cross-core caveat for this pair.** `_stk0` writes LX with rows split across 32
cores; `_stk1` (grid `(1, 32)`, `YBLOCK = M`) reads *all* M rows per n-tile —
rows written by *other* cores. So this specific crossing is a **redistribution**,
expressed with an explicit `tl.inter_tile` op over the data ring
(`reduce_scatter` / `broadcast` mode — the future enabler, §6 item 4), **not** by
tagging the descriptor with a foreign core. A *same-work-division* crossing (e.g.
another pointwise over the same row split) reads only what its own core wrote and
works with the plain `memory_space="lx"` shown.

### 7.2 Reading the grids back from `triton_meta` at compile time

`async_compile.triton()` reads the table out of the entry's `triton_meta`
(`cat.triton_meta`, populated by the `fixed_config` in §7.1) and forwards it as
compile options — the existing `spyre_grid = compile_meta.get("spyre_grid",
(32,))` read, generalized to a table:

```python
# async_compile.py  (Spyre)  — bundle-aware
compile_meta = cat.triton_meta                      # from bundle_0's fixed_config
spyre_grids = compile_meta.get("spyre_grids", {"": (32,)})
spyre_entry = compile_meta.get("spyre_entry", kernel_name)
compile_kwargs = {
    "target": target,
    # DistributeWork stamps each bundled-kernel func's spyre.grid from this table and
    # lowers its tl.program_id → ktdp.get_compute_tile_id over THAT grid.
    "options": {"spyre_grids": spyre_grids, "spyre_entry": spyre_entry},
}
triton.compile(*compile_args, **compile_kwargs)
```

```python
# wrapper — STILL one launch, one .run(); host reserves max(prod(grid)) cores.
# Note: no lx_tmp argument — the LX intermediate is a baked offset, not an arg.
bundle_0.run(in_ptr, w_ptr, out_ptr, M, N, K, stream=stream0)
```

Per-bundled kernel grids ride in `triton_meta` → `options`, **not** in `.run(grid=...)`.
The host launch only needs the core budget (the max over bundled kernels).

### 7.3 Codegen side — the bundle writes the table into `triton_meta`

Each bundled kernel is a `SpyreTritonKernel` that already computes its own grid via
`_compute_spyre_grid()`. The bundle writes the table into the entry's
`triton_meta`, mirroring `SpyreTritonKernel.codegen_body()` (which sets
`self.triton_meta["spyre_grid"]`):

```python
class SpyreTritonKernelBundle:
    def __init__(self, entry_name: str, triton_meta: dict):
        self.entry_name = entry_name
        self.triton_meta = triton_meta               # the ENTRY's triton_meta
        self.bundled_kernels: list[SpyreTritonKernel] = []   # one per same-IS run

    def add_bundled_kernel(self, kernel: SpyreTritonKernel, fn_name: str):
        kernel.bundled_kernel_name = fn_name
        self.bundled_kernels.append(kernel)

    def codegen_body(self):
        # mirror SpyreTritonKernel.codegen_body, which sets
        # self.triton_meta["spyre_grid"]; here we set the per-bundled-kernel table.
        self.triton_meta["spyre_grids"] = {
            r.bundled_kernel_name: r._compute_spyre_grid()   # existing method
            for r in self.bundled_kernels
        }
        self.triton_meta["spyre_entry"] = self.entry_name
```

### 7.4 Backend changes that consume the table

```python
# triton/third_party/spyre/backend/compiler.py
@dataclass
class SpyreOptions:
    # was:  grid: Tuple[int, ...] = (32,)
    spyre_grids: Dict[str, Tuple[int, ...]] = \
        field(default_factory=lambda: {"": (32,)})   # name → grid
    spyre_entry: str = ""
```

```text
# DistributeWork  (today: one shared `grid` for every function)
# Change: look the grid up per function from spyre_grids.
for fn in module.functions:                 # pass already walks per-function
    g = options.spyre_grids.get(fn.name, [32])
    assert len(g) == fn.pid_dims             # existing invariant (c)
    fn.setAttr("grid", g)                    # stamp THIS bundled kernel's grid
    lower tt.get_program_id → ktdp.get_compute_tile_id  over g
```

`DistributeWork` already walks per function and stamps a per-function `grid`
attribute (`lib/Dialect/KTDP/Transforms/DistributeWork.cpp`); the only change is
sourcing `g` from the table instead of the single shared option (§6 item 1).

### 7.5 The non-trivial assumptions

1. **`noinline` bundled kernels survive as separate `func.func`s**, so per-function grids
   apply. Inlining would collapse them into one function = one grid.
2. **`spyre.barrier()` between bundled kernels** for cross-core dependencies (§6 item 3).
   Same-iteration-space-fused and core-local crossings need none.
3. **`make_tensor_descriptor` gains a `memory_space` kwarg** (2605 §6 M1) that
   emits an attribute on `tt.make_tensor_descriptor`, and
   `LowerDescriptorMemory.cpp` reads it (defaulting to HBM) instead of
   hardcoding HBM. `memory_space` is core-agnostic (`{LX, HBM}`); cross-core
   redistribution is handled by `tl.inter_tile` (§6 item 4), not a descriptor
   tag.

### 7.6 Tiling loops are per-sub-kernel, never per-bundle

A tiling loop (`LoopSpec`, emitted from `wrap_op_specs_in_loop()`) advances the
tiled device offsets of every op in its body with **one loop counter**
(`base_offset + loop_var * sym_step`). For that single counter to be correct for
all the ops, they must share the tiled iteration-space symbols — i.e. the
**same iteration space**. Ops with different iteration spaces would need
different `sym_step`s, which one counter cannot express. So a tiling loop is
bounded to a single iteration space, by construction.

| | SDSC | Triton |
|---|---|---|
| What the loop wraps | several OpSpecs (= several **atomic** SDSCs, 1 op each) | the **fused** same-IS ops of one `SpyreTritonKernel` (SSA temps) |
| Where it sits | `LoopSpec` inside a SuperDSC bundle | tiling loop inside one `SpyreTritonKernel` |

These are the same structure at different granularity: SDSC's "multiple SDSCs in
one loop" ≡ Triton's "one `SpyreTritonKernel` (fused ops) with a tiling loop."

In the bundle hierarchy the loop lives one level *below* the bundle:

```
SpyreTritonKernelBundle   (1 launch, LX shared)
  └─ SpyreTritonKernel    (1 grid = 1 IS = 1 core mapping)
       └─ tiling loop (LoopSpec over that IS's tiled symbols)   ← per-sub-kernel
            └─ fused same-IS ops  (SSA temps)
```

Consequences:

- ✅ A tiling loop **inside** each `SpyreTritonKernel` (over its own IS).
- ❌ A single tiling loop **spanning** a bundle's sub-kernels — impossible: sub-kernels
  have different iteration spaces *by definition* (if two ops shared an IS they
  would already be fused into the **same** `SpyreTritonKernel`).

The bundle's cross-sub-kernel sequencing is therefore the entry function +
`spyre.barrier()` (§7.1), **not** a shared loop. Coarse tiling still helps the
LX budget (2605 §4), but only *within* a sub-kernel — it shrinks that bundled kernel's
per-iteration working set; it does not merge bundled kernels.
