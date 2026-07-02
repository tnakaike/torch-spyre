# Indirect Access (Gather/Scatter) in SpyreTritonKernel — Design

Status: **design only — not yet implemented.** This document specifies how
`SpyreTritonKernel` should compile indirect (gather) and, eventually, scatter
accesses onto the Spyre-enhanced Triton `tensor_descriptor.gather` / `.scatter`
primitives. It is the Triton-path counterpart to the SDSC-path design in
[`docs/source/compiler/indirect_access.md`](../../../docs/source/compiler/indirect_access.md),
and extends the kernel-generation design in
[`2602-OpSpecToTriton.md`](2602-OpSpecToTriton.md), which explicitly lists
gather/scatter as out of scope.

---

## 1. Scope and non-goals

**In scope**

- Load-side indirect access (gather): `aten.index`, e.g. `x[i]` and `x[i].exp()`.
- Mapping a torch-spyre tiled device layout — where the gathered axis is in the
  *middle* of the device dimensions — onto Triton's gather, which only addresses
  the *outermost* dimension indirectly.
- Codegen that uses `desc.gather(x_offsets, y_offset)` /
  `desc.scatter(value, x_offsets, y_offset)` — **never** pointer arithmetic
  (constraint from the task; also the explicit guidance in the Spyre backend's
  [`triton-ktir-lowering-thought-exercise.md`](../../../../triton/third_party/spyre/docs/triton-ktir-lowering-thought-exercise.md):
  pointer arithmetic for indirect patterns requires de-linearization the
  lowering cannot represent).

**Non-goals (for now)**

- Scatter / store-side indirect access (`aten.scatter`). The primitive
  (`desc.scatter`) exists and is symmetric (see §4), but, as in the SDSC path,
  only gather is targeted by the first implementation.
- Fused gather + downstream pointwise in a single op spec. The SDSC path keeps
  these as two op specs (identity copy + unary) and disables fusion pending
  backend support; the Triton path inherits the same staging.
- Multi-dimensional index tensors. Only 1-D index tensors (one indirect symbol
  per data dep) are supported, matching the SDSC limitation and the Triton
  verifier (`x_offsets` must be rank-1).

---

## 2. Background

### 2.1 How Inductor and the SDSC path model a gather

Inductor lowers `aten.index` to a `Pointwise` node whose `inner_fn` calls
`ops.indirect_indexing()`:

```
load(index_tensor, ...)        → tmp0          (int32 value)
indirect_indexing(tmp0, ...)   → i_sym         (used as a row address)
load(value_tensor, i_sym * N + c2)             (data load at runtime row)
```

In the resulting `MemoryDep` for the value tensor, the index expression
contains a symbol (`tmp0`) that is **not** in `dep.ranges` — it has no static
loop bound. `MemoryDep.is_indirect()` returns `True`.

The SDSC path replaces that symbol in the device-coordinate expression with the
opaque sympy atom `IndexLoad('arg1_1')` (defined in
[`op_spec.py`](../../../torch_spyre/_inductor/op_spec.py)), via
`compute_coordinates(..., indirect_load_subs=...)`. The value tensor's device
coordinates then look like (from `indirect_access.md`, `x:[128,256]`, `i:[3,192]`):

```
device_size        = [1, 4, 128, 64]
device_coordinates = [0, floor(c2/64), IndexLoad('arg1_1'), Mod(c2, 64)]
                       dim0   dim1(NS)     dim2 (indirect)     dim3 (NE)
```

The indirect axis (`IndexLoad`) sits at **device dim 2** — the middle of a
rank-4 tensor — flanked by the stick-outer (`NS`) and stick-inner (`NE`)
dimensions. The exact rank and dim index vary per layout (the real artifact in
§7 is rank-3 with `IndexLoad` at dim 1); what is invariant is that `IndexLoad`
lands in the *middle*, never at dim 0. This middle placement is the crux of the
problem (§5).

### 2.2 What the Triton path already has

`SpyreTritonKernel` (`spyre_triton_kernel.py`) generates one descriptor per
buffer with `tl.make_tensor_descriptor` and emits `desc.load([off…])` /
`desc.store([off…], val)`. It computes the same `device_coordinates` via
`compute_coordinates()` (in `load`/`store` → `_emit_tensor_descriptor` →
`_emit_scalar_offsets`), and already performs a **dimension permutation** for
matmul (`_matmul_dim_permutation`, `_emit_matmul_tensor_descriptor`) — moving a
chosen device dim to position 0 and permuting `device_size`, strides,
`block_shape`, and coords together. That permutation machinery is the natural
foundation for gather (§6.3).

Note: the value tensor's `MemoryDep` index, not the Triton index, must feed
`compute_coordinates()` — it carries OpSpec symbols (`c0, c1, …`) and, for a
gather, the indirect symbol. `SpyreTritonKernel` already fetches the dep from
`current_node.read_writes.reads/writes`.

---

## 3. The Triton gather/scatter primitive (Spyre extension)

Defined in [`triton/python/triton/language/core.py`](../../../../triton/python/triton/language/core.py)
(`tensor_descriptor_base.gather/scatter`) and lowered by
[`semantic.py`](../../../../triton/python/triton/language/semantic.py)
(`descriptor_gather` / `descriptor_scatter`). The Spyre backend lowers
`tt.descriptor_gather/scatter` → `ktdp.construct_indirect_access_tile` +
`ktdp.load/store`. Behavior is documented and pinned by tests in
[`triton/third_party/spyre/docs/patterns/memory.md`](../../../../triton/third_party/spyre/docs/patterns/memory.md).

API:

```python
result = desc.gather(x_offsets, y_offset)            # → tl.descriptor_gather
desc.scatter(value, x_offsets, y_offset)             # → tl.descriptor_scatter
```

Constraints (the ones that shape this design):

| # | Constraint | Source |
|---|---|---|
| C1 | **Indirect axis is always dim 0.** The index buffer is wired onto descriptor dim 0; there is no way to point it at another dim. | `memory.md` *gather-nd-permuted-strides*, *gather-nd-subscripts* |
| C2 | **`block_shape[0] == 1`** ("descriptor block must have exactly 1 row"). The N-D relaxation widened the rank rule but kept the leading-1 rule. | `semantic.py:1147`; `memory.md` *gather-nd-block-dim0* |
| C3 | **Rank-N descriptors allowed** (`block_shape >= 2D`) — Spyre extension; upstream Triton requires exactly 2D. | `semantic.py:1141-1146` |
| C4 | **Exactly two "movable" axes.** dim 0 = indirect (`ind(idx[c_x + d0])`); dim 1 = direct, offset by the single scalar `y_offset` (`c_y + d1`). Every dim `i ≥ 2` is a bare `d_i`: **no offset, full block extent always read.** | `memory.md` *gather-nd-subscripts*, *gather-3d* |
| C5 | **`x_offsets` is a 1-D tensor**, `shape[0] >= 8`. 2-D index tensors are rejected (flatten + reshape workaround). | `semantic.py:1150,1153`; `memory.md` *gather-2d-indices* |
| C6 | **`x_offsets` must come from a `tl.descriptor_load` of an `!tt.ptr<i32>` index buffer**, not a tensor-typed kernel argument — the lowering traces the index buffer's provenance. A non-zero load offset is captured into the indirect subscript map. | `memory.md` *gather* (`test_gather_with_x_offsets_arg_fails_to_legalize`, `…captures_x_offset`) |
| C7 | **Permuted strides are the only way to gather a physically-inner axis.** Declare the descriptor *shape* with the gathered axis at dim 0 and express the true physical layout via *strides*; the lowering reads the view shape from the descriptor's declared shape. | `memory.md` *gather-nd-permuted-strides* |
| C8 | TMA min-cols rule (`block_shape[1] >= 32/bitwidth*8`) is **dropped** for Spyre — dim 1 may be 1. | `semantic.py:1155-1162`; `memory.md` *gather-nd-trailing-one* |

The role split (C4) is the headline limitation: **at most one direct axis can
carry an offset.** This is what forces the layout reconciliation below.

> **Portability note (C7 + C3 are Spyre-only as used here).** The stride
> *mechanism* C7 relies on is generic: `make_tensor_descriptor`
> ([`semantic.py`](../../../../triton/python/triton/language/semantic.py#L1851))
> accepts any strides on every target, provided the innermost stride is `1`,
> the last block dim is ≥ 16 bytes, and leading strides are 16-byte multiples.
> The Spyre permuted-strides example honors this — it keeps the innermost
> contiguous (`[INNER_DIM, NUM_BLOCKS*INNER_DIM, 1]`) and only reorders the
> *leading* axes. **But gathering a *middle* axis this way requires a rank ≥ 3
> descriptor (C3), and rank-N gather is gated behind `target_info.is_spyre()`**
> (`semantic.py:1141-1146`): NVIDIA and other targets assert
> `len(desc.block_shape) == 2`. On NVIDIA the gather is hardware-bound to 2D —
> it lowers to `cp.async.bulk.tensor.2d.tile::gather4`
> ([`LoadStoreOpToLLVM.cpp:1611`](../../../../triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp#L1611));
> there is no rank-N TMA gather. And a 2D fallback cannot help, because the
> innermost-stride-1 rule forbids permuting the gathered axis out of dim 0 into
> the contiguous column position. So the C7-for-a-middle-axis technique this
> design depends on is effectively **Spyre-only**; it does not port to the
> NVIDIA TMA backend.

---

## 4. Gather and scatter are symmetric

`descriptor_scatter` mirrors `descriptor_gather` exactly (same rank rules,
leading-1 rule, single `y_offset`, rank-1 `x_offsets`) — see `memory.md`
*scatter-nd* (`test_scatter_3d_lowered`). A future scatter implementation reuses
the entire mapping in §6, substituting `desc.scatter(value, x_offsets,
y_offset)` for the gather on the store side. This document specifies gather;
scatter is a mechanical mirror once load-side works.

---

## 5. The core mismatch

Two independent gaps between the torch-spyre device layout and the Triton
primitive:

1. **Middle vs. outermost indirect axis.** torch-spyre places `IndexLoad` in
   the middle of the device dims (dim 1 of 3 in the §7 artifact; dim 2 of 4 in
   the `indirect_access.md` illustration). Triton requires the indirect axis at
   dim 0 (C1, C2).

2. **One offset axis vs. many.** A direct descriptor access in
   `SpyreTritonKernel` emits one scalar offset per device dim (the
   `[dim0, dim1, dim2, …]` list from `_emit_scalar_offsets`). The gather
   primitive offers only **one** direct offset (`y_offset`, on dim 1); all
   remaining dims read their full block extent with no offset (C4).

Gap (1) is solved by permutation + permuted strides (C7). Gap (2) is a genuine
expressiveness restriction that constrains which layouts/work-divisions are
representable, and drives the constraints in §6.4 and §6.7.

---

## 6. Design

### 6.1 Detecting an indirect load

In `SpyreTritonKernel.load(name, index)`, after the existing
`FixedTiledLayout`/allocation guards, detect the indirect case by inspecting the
value tensor's `MemoryDep`:

- `dep.is_indirect()` is `True`, **or**
- the dep index contains a symbol not present in `iteration_space(current_node)`
  (the indirect `tmpN`).

When detected, route to a new `_emit_gather_descriptor` + `_emit_descriptor_gather`
path instead of `_emit_tensor_descriptor` + `_emit_descriptor_load`. Non-indirect
loads are unaffected.

### 6.2 Staging `x_offsets` from the index buffer (C6)

The index tensor (`arg1_1` in the running example) is itself a `FixedTiledLayout`
buffer. Triton requires `x_offsets` to be a rank-1 tensor produced by a
`tl.descriptor_load` of an `i32` pointer (C6) — passing the loaded value tensor
directly, or a tensor-typed kernel arg, fails to legalize.

Plan:

1. Build a descriptor for the index buffer (`idx_desc`) with
   `tl.make_tensor_descriptor` over its int32 device layout.
2. Load the index slice for the current program:
   `x_offsets = idx_desc.load([…])`.
3. If the loaded index tile is not already rank-1, `tl.reshape` it to 1-D
   (C5). The 2-D-index workaround in `memory.md` (*gather-2d-indices*) —
   `reshape` to rank-1 before the gather, `reshape` the result back after — is
   the model. Per the current SDSC limitation, only 1-D index tensors are
   supported, so the reshape is for stick/grouping flattening, not for genuine
   multi-D index semantics.

The CSE variable holding the loaded index tile is recorded against the indirect
symbol so the value-tensor gather can find it. Inductor's standard
`indirect_indexing` machinery already runs the index `load()` during `inner_fn`
execution; the work here is to capture *which* CSE var is the rank-1 index
tile and route it to the gather as `x_offsets` rather than letting it flow
into pointer arithmetic.

> **Index-buffer offset capture (C6):** if the index `descriptor_load` uses a
> non-zero offset (e.g. a per-core slice of the index buffer), the Spyre
> lowering folds that offset into the indirect subscript map
> (`ind(idx[offset_m + d0])`). This is exactly how per-program work division of
> the index buffer is expressed — see §6.7.

### 6.3 Permuting the indirect axis to dim 0 (C1, C7)

Reuse the matmul permutation pattern. Compute `device_coords` for the value
tensor via `compute_coordinates()` (with the indirect-load substitution applied,
so the indirect coordinate is identifiable). Then:

1. Find the device dim `k*` whose coordinate is the indirect axis — the
   coordinate that contains `IndexLoad(...)` (or the raw indirect `tmpN` symbol
   before substitution).
2. Build a permutation that moves `k*` to position 0:
   `perm = [k*] + [i for i in range(rank) if i != k*]`
   (the same shape as `_matmul_dim_permutation`'s output).
3. Permute `device_size`, `strides` (row-major over the *original* device_size),
   `block_shape`, and `device_coords` by `perm`, exactly as
   `_emit_matmul_tensor_descriptor` does.

Per C7, the descriptor's declared **shape** now has the gathered axis at dim 0,
while the **strides** still describe the true physical layout — so the permuted
stride list is the mechanism, not a physical re-layout. The lowering derives the
memory view from the declared shape; nothing else changes on the backend side.

### 6.4 Offset assignment (C4)

After permutation the device dims are `[indirect, rest…]`. Assign offsets:

- **dim 0 (indirect):** addressed by `x_offsets`. **No scalar offset** in the
  descriptor offset list — the row address comes entirely from the index tile.
- **dim 1 (direct, `y_offset`):** the single dim that may carry a runtime scalar
  offset. Choose the dimension that *needs* a per-program/per-core base offset
  (see §6.7). If no direct dim needs an offset, pass `y_offset = 0`
  (`memory.md` shows `y_offset=0` variants are legal).
- **dims ≥ 2:** no offset — the gather reads the full block extent on each
  (C4). The kernel must therefore declare `block_shape` on these dims to cover
  exactly the extent it wants (§6.5). To slice such a dim, `reshape` the gather
  result afterward (`memory.md` *gather-nd-subscripts*).

**Representability rule:** at most one direct (non-indirect, non-trivial) device
dim may require an offset. If a layout/work-division would require offsets on two
or more direct dims, it is **not expressible** as a single gather. Detect this at
codegen time and raise a clear `UnimplementedOp`/`NotImplementedError` rather
than emitting a silently wrong kernel.

### 6.5 `block_shape` and the leading-1 rule (C2, C8)

- `block_shape[0]` (indirect axis) must be **1** (C2). The existing
  `_device_block_shape` would divide by core/tile divisors; for the gather path
  it must unconditionally set the dim-0 block to 1.
- The **stick dimension** rule from `_device_block_shape` still applies to
  whichever permuted dim is the stick-inner (`NE`): its block must equal the full
  device size of that dim and is never split across cores. With dim-1 = 1 now
  legal (C8), stick layouts with a singleton group dim are fine.
- The result tile shape is `[x_offsets.shape[0], *block_shape[1:]]`. Downstream
  consumers (the reshape into `tl.dot`, pointwise compute, or `store`) must agree
  with this shape; an extra `tl.reshape` may be needed to recover the logical
  per-output shape.

### 6.6 Emitting the gather

New helpers parallel to the existing descriptor load/store:

```python
# in load(), indirect branch:
idx_desc, idx_off = self._emit_index_descriptor(idx_name, idx_var, idx_dep, idx_layout)
x_offsets = self.cse.generate(self.loads, f"{idx_desc}.load([{idx_off}])", ...)
x_offsets = self.cse.generate(self.loads, f"tl.reshape({x_offsets}, [{K}])", ...)  # → rank-1

desc, y_offset = self._emit_gather_descriptor(name, var, dep, layout)   # permuted, block_shape[0]=1
val = self.cse.generate(self.loads, f"{desc}.gather({x_offsets}, {y_offset})", ...)
# optional: val = tl.reshape(val, <logical per-output shape>)
```

The descriptor itself (`_emit_gather_descriptor`) is hoisted/loop-invariant like
the existing `_emit_tensor_descriptor` (shape, strides, block_shape are
constant; only `x_offsets` and `y_offset` vary).

### 6.7 Work division: partition the index, not the indirect axis

The indirect axis cannot be split across cores by a descriptor offset (its block
is 1 and its addressing is the index tile, C1/C2). Work division of a gather is
therefore expressed by **partitioning `x_offsets`** — each program loads its own
slice of the index buffer via a non-zero offset on the *index* descriptor
(`idx_desc.load([per_core_offset])`), which the Spyre lowering folds into the
indirect subscript (C6, `memory.md` `…captures_x_offset`).

Implications for `_compute_core_division` / `_get_triton_block_size`:

- The number of gathered rows per program = `x_offsets.shape[0]` (≥ 8, C5).
- The `XBLOCK` and `spyre_grid` computation must treat the indirect (output-row)
  dimension as the partitioned axis, with the per-program index slice driving the
  grid, rather than splitting the value tensor's device dims.
- Direct dims that are split across cores must use the single `y_offset` (dim 1).
  Combined with §6.4's representability rule, this means: **a gather can be
  partitioned along the output-row (index) axis and along at most one direct
  device axis.**

---

## 7. Worked example: `x[i]` from `my-examples/gather_exp_1d.py`

Source: [`my-examples/gather_exp_1d.py`](../../../my-examples/gather_exp_1d.py)
— `x: f16[64,128]`, `i: i32[64]`, `result = x[i]` (and `x[i].exp()`). The
artifacts below are the **SDSC-path** output captured in
[`results/gather_exp_1d_sdsc/`](../../../results/gather_exp_1d_sdsc/)
(`output_code.py`, `ir_post_fusion.txt`). The Triton path must reproduce the
*same* device access; these are the concrete coordinates it has to map onto a
`desc.gather`.

The fused `op='identity'` op spec (from `output_code.py`):

```
iteration_space = {c0: (64, 2), c1: (128, 2)}     # c0 = output rows (P=64), c1 = N=128; work_division = 2 each
index  arg1_1 (i32): device_size=[1, 2, 32],  coords=[0, floor(c0/32), Mod(c0,32)],          name='arg1_1'
value  arg0_1 (f16): device_size=[2, 64, 64],  coords=[floor(c1/64), IndexLoad('arg1_1'), Mod(c1,64)]
output buf0   (f16): device_size=[2, 64, 64],  coords=[floor(c1/64), c0, Mod(c1,64)]
```

Note the value tensor is **rank-3** with the indirect axis at **dim 1** (not
rank-4/dim-2 as in the `indirect_access.md` illustration — real layouts vary):

```
arg0_1 device_size=[2, 64, 64]
       dim0 = floor(c1/64)        # NS  — N-stick-outer (N=128 → 2 sticks)
       dim1 = IndexLoad('arg1_1') # indirect row (the M=64 axis of x)
       dim2 = Mod(c1, 64)         # NE  — stick-inner (64)
```

The `op0_loop_body` in `ir_post_fusion.txt` confirms the Inductor shape:

```
index1 = 128*indirect0 + p1     # value load: row = indirect0, col = p1
load(arg1_1, p0); set_indirect0(load); load(arg0_1, index1); store(buf0, 128*p0 + p1, …)
```

Triton-path plan for the value load:

1. **Index stage (C6).** Build `idx_desc` over `arg1_1` (`device_size=[1,2,32]`,
   i32); `x_offsets = idx_desc.load([…])`; `reshape` to rank-1 length-64. (64 ≥ 8,
   so C5 is satisfied without padding.) Per-core slicing of the index goes on the
   index descriptor's load offset, folded into the subscript (§6.7).
2. **Identify indirect dim.** In the value coords, dim 1 holds `IndexLoad` →
   `k* = 1`.
3. **Permute** `perm = [1, 0, 2]` (same shape as `_matmul_dim_permutation`):
   - declared shape `[64, 2, 64]`; strides permuted from the physical
     row-major strides of `[2,64,64]` = `[4096, 64, 1]` → `[64, 4096, 1]` (C7),
   - `block_shape = [1, 2, 64]` → dim 0 = 1 (C2); dim 1 (`NS`) may take the
     `y_offset`; dim 2 (`NE`) full extent, no offset (C4).
4. **Offsets.** dim 0 ← `x_offsets`; dim 1 (`y_offset`) ← `NS` per-core base
   (0 when N is not split); dim 2 no offset (full `NE`).
5. **Emit** `val = src_desc.gather(x_offsets, y_offset)` → result
   `[64, 2, 64]`; `reshape`/store to match the output `buf0` layout
   (`coords=[floor(c1/64), c0, Mod(c1,64)]`, a direct write).

The two-offset budget (C4) holds: only the indirect axis (`x_offsets`) and one
direct axis (`NS` via `y_offset`) are non-trivial; `NE` is read in full. Because
the indirect axis maps to the output-row dimension `c0` (work_division 2), the
per-core split is realized by partitioning `x_offsets` (§6.7), **not** by
offsetting the indirect descriptor dim.

---

## 8. Integration points in `spyre_triton_kernel.py`

| Method | Change |
|---|---|
| `load()` | Detect indirect dep (§6.1); route to gather helpers; keep the non-indirect path untouched. |
| `_emit_gather_descriptor` (new) | Like `_emit_matmul_tensor_descriptor`: compute coords, find indirect dim, permute to dim 0, force `block_shape[0]=1`, return `(desc, y_offset, …)`. |
| `_emit_index_descriptor` (new) | Build the int32 index-buffer descriptor and emit the rank-1 `x_offsets` load (C6). |
| `_emit_descriptor_gather` (new) | Emit `desc.gather(x_offsets, y_offset)`, plus any result `reshape`. |
| `_device_block_shape` | Honor `block_shape[0]=1` for the gathered axis (or branch in the gather helper). |
| `_compute_core_division` / `_get_triton_block_size` / `_compute_spyre_grid` | Partition by the index/output-row axis, not the indirect device dim (§6.7). |
| `_dump_opspec_json` | Emit `IndexLoad('…')` in the value coords and name the index arg first, mirroring the SDSC op-spec ordering (index args first, value args, output). |

All changes stay within `torch_spyre/_inductor_triton/` (the standing constraint
from `SKILL.md`). Import `IndexLoad` and the indirect-subs helpers
(`indirect_load_subs_from_kernel`) from `_inductor/` rather than reimplementing.

A subtlety vs. the SDSC path: the actual `SpyreTritonKernel.__enter__` does
**not** install a custom CSE/ops proxy (unlike the `SpyreKernel`
`indirect_indexing` interception described in `indirect_access.md`). It relies on
upstream `TritonKernel`'s `indirect_indexing` handling, which mints the `tmpN`
symbols. The design must therefore recover the index→`x_offsets` association from
`TritonKernel`'s indirect bookkeeping (`indirect_vars` equivalent) plus the
`MemoryDep`, not from a Spyre ops handler. Confirming the exact upstream hook is
an implementation task (§10).

---

## 9. Constraints and limitations (summary)

- **One direct offset axis only** (C4). Layouts needing offsets on ≥ 2 direct
  device dims are unrepresentable as a single gather — fail loudly.
- **Indirect axis must become dim 0** via permutation + permuted strides (C1/C7);
  block on that axis is always 1 (C2).
- **`x_offsets` rank-1, ≥ 8 elements, from an `i32` descriptor load** (C5/C6).
- **1-D index tensors only**, matching the SDSC limitation.
- **Gather only**; scatter is a documented mirror (§4) but deferred.
- **No fusion** of gather with downstream pointwise into one op spec (matches
  SDSC, which keeps identity-copy + unary as two op specs).
- **Trailing dims read in full**; slicing requires a post-gather `reshape`.

---

## 10. Open questions

1. **Index→`x_offsets` plumbing.** Exactly which upstream `TritonKernel`
   structure holds the indirect symbol → loaded-index-CSE-var mapping, given
   `SpyreTritonKernel` does not install a Spyre ops proxy? (Candidates:
   `indirect_vars`, the `tmpN` CSE entries, `current_node` inner_fn replay.)
2. **`y_offset` selection** when more than one direct dim is non-trivial but the
   work division only actually offsets one — is the "fail loudly" rule too
   strict for real models, or do common gathers (embedding lookup, KV-cache page
   gather) always fit the one-offset budget?
3. **Reshape correctness** between the `[len(x_offsets), *block_shape[1:]]`
   gather result and the logical output tile expected by store / downstream
   compute — does it interact with the stick (`NE`) dimension cleanly?
4. **Minimum 8 rows (C5)** vs. small gathers (e.g. `x_offsets.shape[0] < 8`):
   pad, or reject?
5. **Scatter timing** — enable alongside gather, or strictly after gather is
   validated end-to-end?
</content>
</invoke>
