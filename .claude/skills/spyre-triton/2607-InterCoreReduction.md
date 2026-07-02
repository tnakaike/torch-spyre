# 2607 — Inter-Core Reduction (M6)

Status: **design / plan only — NOT yet implementable.** The enabling backend
builtins (`tl.inter_tile`, `tl.wk_slice_coord`) exist in
`../triton/third_party/spyre/` but are **not upstreamed yet**. A further builtin,
`tl.sum_stick` (explicit intra-stick reduction), **must still be added** to the
backend. Do not start the implementation until these land. This doc records the
plan so it is ready when they do.

Scope (decided): `sum` only (`combiner="add"`, `mode="reduce_to_one"`), a single
split reduction axis, **standalone** `SpyreTritonKernel` (no bundle path).
Excludes native matmul and the tiling-loop path. max/mul/mean and bundle
integration are follow-ons.

## Context — the gap

`SpyreTritonKernel` can divide an op's **spatial** dims across cores but not the
**reduction** dim. When SDSC's work division shards the reduction extent across
cores (few outputs vs. many cores, large reduction), each core holds only a
*partial* reduction that must be combined across cores — and there is currently
no way to express that combine. The case is silently mishandled today:

- `_compute_core_division()` preserves the reduction symbol's divisor in the M5
  path (`spyre_triton_kernel.py:1724-1725`), but
- `_compute_spyre_grid()` (`:418-439`) counts only `x/y/z` prefixes → the shard
  never becomes programs, and
- `_emit_split_logical_offsets()` (`:1247-1328`) splits only spatial symbols
  (`:1278`) → the reduction offset is never sharded, and there is no cross-core
  combine.

The tiling-loop fallback explicitly hard-sets reduction divisors to 1
(`:1755-1756`) — out of M6 scope.

## Enablers

### New builtin required: `tl.sum_stick` (backend addition)

The outer-stick and intra-stick reductions are represented **explicitly** as two
distinct ops, because they map to different hardware primitives and differ in
return-shape semantics:

- `tl.sum(x, axis)` reduces an **outer-stick** (or spatial) dim and **drops** it
  (`result_rank = rank − 1`), element-wise across sticks.
- `tl.sum_stick(x)` reduces the **within-stick** (innermost) dim but **cannot
  drop it** — a stick is physically mandatory on every tensor — so it collapses
  the 64 elements (fp16) *within* the stick and returns **a one-scalar stick of
  the same rank** (rank-preserving). This one-scalar-stick form is a supported
  backend representation.

`tl.sum_stick` is a **new builtin we must add**, parallel to `tl.inter_tile`: a
`tt` op (a reduce-variant flagged intra-stick) plus a KTDP lowering that emits
the within-stick HW reduce and yields the one-scalar-stick representation. It
lives under `../triton/third_party/spyre/` (ours to add). Same shape of work as
the `inter_tile`/`wk_slice_coord` builtins below.

### Present in the backend (no backend change needed for these)

- `tl.inter_tile(x, axis, combiner, mode, work_slices=...)` → `tt.inter_tile_reduce`
  → `LowerInterTile` → `ktdp.inter_tile_produce` + delivery. `x` needs a **unit
  leading dim**; result rank = partial rank − 1. `combiner∈{add,max,mul}`,
  `mode∈{all_reduce,reduce_to_one}` (reduce_scatter/broadcast not implemented).
  For `reduce_to_one`, only pick₀ (reduced-axis slice index == 0) stores.
- `tl.wk_slice_coord(work_slices, axis)` → `work_slices[program_id(0)][axis]` as
  runtime i32; folds to a `program_id(0)`-indexed select chain (no new IR op).
- `work_slices` = constexpr **list** indexed by tile_id, each entry
  `{axis_name: slice_index}`. `W[axis]=max+1`. Tiles sharing all non-reduced axis
  values form a group. Flat 1D grid with `prod(grid) == len(work_slices)`,
  indexed by `program_id(0)`.
- `LowerInterTile` runs **before** `DistributeWork` (`backend/compiler.py`).
- Canonical reference:
  `../triton/third_party/spyre/test/fixtures/inter_tile_reduce/`
  (`matmul_splitk_kernel` = reduce_to_one + pick₀; `meta.py`
  `_WORK_SLICES = [{"out": t//N, "in": t%N} for t in range(T)]`).

## Serialization channel (important correction)

This codebase serializes **only** `triton_meta["spyre_grid"]`
(`codegen_body:354-355` → `async_compile.py:52-55`). The `spyre_options` /
`SympyExpr` / `OpSpecDict` machinery referenced in SKILL.md is the **prototype**,
not this tree. Because `work_slices` is referenced **inside the kernel body**
(unlike backend-only `spyre_grid`), bake it as a **module-level constexpr literal**
in the generated source (the `LX_TMP` baked-constant pattern of
[`2606-KernelBundleLXModel.md`](2606-KernelBundleLXModel.md) §7.1) — no
signature/`.run()` arg plumbing. A list of int-dicts `repr()`s to a valid Triton
literal.

## Status (current increment, June 2026)

Done so far, on branch `dev/triton-inter-tile` (all in `spyre_triton_kernel.py`):

- **STEP 1** (earlier): detection (`_reduction_core_division`,
  `_is_inter_core_reduction`/`_compute_inter_core_reduction`), flat 1D grid +
  `_build_work_slices`, `codegen_body` M6 branch.
- **§4 + §6 DONE** (this increment): coordinate recovery via
  `tl.wk_slice_coord` and module-level baking of `work_slices`.
  - §6: `codegen_body` stashes the list on `self._work_slices` and sets only
    `triton_meta["spyre_grid"]`; `codegen_kernel` → `_bake_work_slices` inserts
    `work_slices = tl.constexpr([...])` at module scope (before the decorator).
    It is no longer placed in `triton_meta` (`async_compile.py` only forwards
    `spyre_grid`). **Must be `tl.constexpr(...)`, not a bare literal** — a global
    read from a `@jit` body otherwise raises `NameError("Cannot access global
    variable ... instantiated as constexpr")` (this is the `x = tl.constexpr(v)`
    form, not the unsupported `x: tl.constexpr = v` annotation).
  - §4: `_emit_split_logical_offsets` under M6 emits `c0 =
    tl.wk_slice_coord(work_slices, "c0")` for spatial split dims and shards the
    reduction offset **onto the outer-stick dim only** — `c1 = r0_offset +
    tl.wk_slice_coord(work_slices, "c1") * extent` (extent = range // work_div,
    ADDED to the range-tree offset, not replacing it). The non-M6 (M5)
    `program_id` radix path is unchanged.
  - The M6 decision is cached in `set_current_node` (where `current_node` is
    live) and read back in `codegen_body`; `_is_inter_core_reduction()` is now a
    cached getter. Risk 1 stick-alignment is enforced in
    `_compute_inter_core_reduction` (bail to non-M6 if the reduction extent is
    not divisible by `work_div × stick_size`); **>1 split reduction axis raises
    `NotImplementedError`** (not supported).
  - Scope decision (this increment): **only the outer-stick dim is treated as the
    reduction dim** — the within-stick collapse (`tl.sum_stick`) and the
    cross-core combine (`tl.inter_tile` + pick₀ store) are §5, deferred.
  - Verified `results/sum_stick/output_code.py`: module-level `work_slices`
    (len 32), `spyre_grid:(32,)`, `c0 = tl.wk_slice_coord(...)`, `c1 = r0_offset
    + tl.wk_slice_coord(work_slices, "c1") * 2048` (shard 0 → outer-sticks
    [0,32), shard 1 → [32,64)), block_shape `[32,1,64]`. Frontend still can't
    compile `tl.wk_slice_coord` (not upstreamed) — output_code generation is the
    gate. Regressions add/sum/sum_sum reach the `NoneType … run` backend stop;
    pre-commit clean.
- **§5 cross-core combine DONE** (this increment): `tl.inter_tile` +
  pick₀ store guard, on top of the existing on-core `tl.sum`.
  - `reduction()` under M6: after the on-core `tl.sum` (`local_partial`),
    reshape `result_shape -> [1, *result_shape]` (unit leading dim required by
    the inter_tile verifier) and emit `tl.inter_tile(partial, axis=<group>,
    combiner="add", mode="reduce_to_one", work_slices=work_slices)`; the
    collapsed result carries `result_shape` again so `store_reduction` is
    unchanged shape-wise.
  - **`axis=` is the GROUP-defining spatial dim, not the reduced dim** (the key
    correction). `LowerInterTile.cpp` partitions tiles by the *value* of the
    `axis` key (buildGroupSets): equal value → same group, and the shards
    *varying within* a group are reduced. So `axis = c0` (output-row group);
    passing the reduced `c1` gives non-contiguous groups `{0,2,4,…}` →
    `error: group 0 is not contiguous`. New helper `_group_axis_name()` returns
    the single spatial split sym; `_reduced_reduction_axis_name()` (the reduced
    sym) is used only for the pick₀ guard.
  - `store_reduction()` under M6 passes `pick0_axis=<reduced sym>` to
    `_emit_descriptor_store`, which guards the store with
    `if tl.wk_slice_coord(work_slices, "<reduced>") == 0:` (both `if` and the
    indented store are `DeferredLine(name, …)` so they survive/drop together —
    Risk 4).
  - **Verified end-to-end through KTIR** on `sum_stick.py` (`torch.sum([16,4096],
    dim=1)`): `output_code.py` has `tl.inter_tile(..., axis="c0",
    mode="reduce_to_one")` + `if tl.wk_slice_coord(work_slices,"c1")==0:` store;
    `.ktir` has `grid=[32]`, `groups` = 16 (one per `c0`),
    `producer_tiles_per_group` covers `[2g,2g+1]` (contiguous, gsize 2),
    `consumer_tiles_per_group` a **single equality** (reduce_to_one pick₀),
    `linalg.add` combiner, `inter_tile_reduce` result `tensor<1x64xf32>` (unit
    dim collapsed), `scf.if { ktdp.store }`, no residual `tt.inter_tile_reduce`.
    Reaches the expected `NoneType … run` stop (execution not wired). Regressions
    add/sum/sum_sum emit zero inter_tile and reach the same stop; pre-commit clean.
- **§5 REMAINING (`tl.sum_stick`)**: explicit within-stick (intra-stick) reduce.
  The builtin is **not in the backend yet**. Deferred by decision this increment;
  the on-core `tl.sum` reduces the outer-stick axis and the backend implicitly
  reduces the within-stick (NE) dim (`_get_reduction_axis` contract), so the
  current kernel is treated as complete pending KTIR-numerics confirmation. Add
  `tl.sum_stick` only if that within-stick assumption fails to hold.

## Planned unification: `work_slices` subsumes the `program_id` radix

(Design decision, June 2026 — deferred, gated on the same upstreaming as §5.)

`tl.wk_slice_coord(work_slices, axis)` is defined as
`work_slices[program_id(0)][axis]` — a lookup table indexed by `program_id(0)`.
The M5 spatial radix `idx_s = (program_id(0) // inner_cores) % div_s` is exactly
the **closed form** of that lookup for the special case where the table is the
row-major enumeration of the spatial split dims. The two orderings already
agree: `_build_work_slices` (spatial outer-first, then reduced; innermost varies
fastest) and the `_emit_split_logical_offsets` M5 radix (`reversed(split)`,
`inner_cores *= div`) enumerate identically, so for a 1D kernel
`work_slices[pid]["c0"]` equals `(pid // inner) % div`. **So the radix is just
`work_slices` specialized to spatial-only, row-major; `work_slices` is the strict
generalization** (it additionally carries the reduced axis the radix dropped —
the whole M6 gap).

Target end state (once the builtins land): build `work_slices` for **every 1D
pointwise/reduction kernel** and drive coordinate recovery uniformly off
`tl.wk_slice_coord`, deleting the `program_id` radix branch in
`_emit_split_logical_offsets`. This is behavior-preserving for those kernels.

Two cases that do **not** collapse into it for free, and must stay exceptions
(or get a later generalization):

1. **Native matmul (2D grid).** `work_slices` is indexed by `program_id(0)` only
   (flat 1D, `prod(grid)==len(work_slices)`); matmul uses an essential 2D grid
   `program_id(0)`+`program_id(1)` (e.g. `(1,32)`). Unifying it means flattening
   the 2D grid into 1D `work_slices` **and** teaching `DistributeWork` to map it
   — a real backend change, not a rename.
2. **Tiling-loop path** (`program_id` + a `tile_idx` loop) — a separate
   mechanism, not covered by `work_slices`.

Blocker: doing this now would make *all* Spyre Triton kernels depend on the
not-yet-upstreamed `wk_slice_coord`/`work_slices` builtins, breaking the
add/sum/matmul kernels that currently compile to KTIR on the upstream backend.
Unify only after the builtins are upstream (same gate as §5).

**Grid side already landed (June 2026).** The first concrete step of this
unification — making the grid flat 1D for every descriptor/M5 kernel — is done
and needs no new builtin. `_compute_spyre_grid` now returns `(total,)` (product
of the per-axis program counts) for all non-native-matmul kernels, since
`_emit_split_logical_offsets` drives the body off `program_id(0)` alone. This
fixed a real bug on `my-examples/gather_idx1d_src2d.py`: Inductor tiled the
gather into `y`+`x` range trees, so the old `_compute_spyre_grid` reported
`spyre_grid=(2,2)` while the body's radix (`c0=(pid0//2)*32`, `c1=(pid0%2)*64`)
used only `program_id(0)` → DistributeWork errored `grid rank 2 does not match
kernel's pid dimensionality 1`, and `program_id(0)∈[0,2)` meant the `pid0//2`
tile never ran. With `(4,)` it passes DistributeWork and reaches the `NoneType …
run` stop. Native matmul keeps its genuine 2D grid (`(1,32)`); add/sum stay
`(32,)`. The remaining unification step (replace the `program_id` radix itself
with `wk_slice_coord`) is what stays gated on upstreaming.

## Implementation plan (all in `torch_spyre/_inductor_triton/spyre_triton_kernel.py`)

### 1. Detect the M6 case (single branch point)
- `_reduction_core_division()` — partition `self._triton_to_opspec` by
  `range_tree_nodes[ts].prefix.startswith("r")` (filter as `:1272`/`:1740`),
  return reduction syms with `_core_division.get(os,1) > 1`.
- `_is_inter_core_reduction()` = has-reduction-split **and** `not
  is_native_matmul` **and** `_tiling_loop_count is None` **and**
  `inside_reduction`. Compute once after `_compute_core_division()` (`:307`);
  cache `self._inter_core_reduction` / `self._reduction_div`.
- Bail to non-M6 if >1 split reduction sym, or if reduction extent not divisible
  by `reduction_shards × stick_size` (Risk 1).

### 2. Flat 1D grid + `work_slices` (shared radix helper)
- Extract the radix loop `_emit_split_logical_offsets:1295-1307` into shared
  `_decode_axis_index(...)` so grid build, offset decode (§3), and coord recovery
  (§4) cannot drift. Returns Python int (work_slices) or Triton string (offsets).
- `_build_work_slices() -> (grid_tuple, list[dict])`:
  - Axis order = spatial split syms (outer-first by `range_tree_nodes[ts].divisor`
    desc, as `:1278-1284`) **+ reduced sym innermost/fastest** (K-shards form a
    contiguous group; reduced-slice `==0` = pick₀).
  - Names = `str(opspec_sym)` (reduced axis name = reduced sym str).
  - `W[spatial]=core_division[sym]`, `W[reduced]=reduction_shards`.
  - `total = prod(spatial_cores)*reduction_shards`; build the list via shared
    radix. `grid=(total,)`.
- `_compute_spyre_grid()`: under M6 return `_build_work_slices()[0]`; else
  unchanged.

### 3. Shard reduction offset + descriptor block_shape
- `_emit_split_logical_offsets` under M6: add reduced sym to `split` (innermost),
  `extent=max(1, s_range//reduction_shards)`, and **add** `shard_base=coord*extent`
  to the reduction symbol's existing range-tree offset (`:1318-1328`), not replace.
- `_device_block_shape()` (`:1567-1602`): reduced sym's `_core_division` is already
  `reduction_shards`, so the **outer-stick (FloorDiv/NS)** dim auto-shrinks via
  `:1591-1594`; **within-stick (NE) dim stays pinned** (`:1587-1589`). The shard
  lands on the outer-stick dim — exactly what `_get_reduction_axis()` (`:540-598`,
  skips Mod `:587`) selects. No further change.

### 4. Coordinate recovery via `tl.wk_slice_coord`
- Prologue: per axis emit `<axis> = tl.wk_slice_coord(work_slices, "<name>")`; use
  `<axis>*extent` for offset bases, replacing the inline `(pid//inner)%div` string
  in `bases` (`:1306`) while keeping `extent`. Matches fixtures; pick₀ guard needs
  the reduced coord as a Triton value anyway.

### 5. Emit explicit intra-stick + outer-stick reductions + `tl.inter_tile` + pick₀ store
- `reduction()` (`:600-653`) under M6 emits the three reduction levels explicitly
  (only the outer-stick dim is sharded → only it touches `inter_tile`; the
  intra-stick reduce is unconditionally local):
  - **intra-stick (local, never sharded):** `local_intra = tl.sum_stick(<value>)`
    — collapses the within-stick (innermost) dim to a one-scalar stick,
    rank-preserving.
  - **outer-stick, on-core portion:** `local_partial = tl.sum(local_intra,
    axis="<outer_stick_axis>")` — accumulates the one-scalar sticks this core
    owns; replaces the old single `triton_fn(value, axis)` (`:650`).
  - **outer-stick, cross-core portion:** `partial = tl.reshape(local_partial,
    [1, *result_shape])`, then `result = tl.inter_tile(partial,
    axis="<outer_stick_reduced>", combiner="add", mode="reduce_to_one",
    work_slices=work_slices)`; return `result`.
  - Sum is commutative, so steps 1 and 2 may be reordered, but this order is the
    natural one (collapse the stick, then accumulate scalars across sticks/cores).
  - Emit `tl.sum_stick` on the within-stick axis and `tl.sum`/`tl.inter_tile` on
    the outer-stick axis — do not conflate the two axes (Risk 2).
- `store_reduction()` (`:655-713`) under M6: wrap reshape + `_emit_descriptor_store`
  (`:706-713`) in `if <reduced_axis_var> == 0:` (fixture `kernel.py:184`). Store
  uses `DeferredLine` (`:1242`) — emit the guard as a deferred/indented block so it
  survives splicing/DCE (Risk 4).

### 6. Thread `work_slices` into the source
- Bake module-level `work_slices = [ {...}, ... ]` in `codegen_kernel` (`:319-337`).
- Set `triton_meta["spyre_grid"]=(total,)` via `_compute_spyre_grid()` so
  `prod(grid)==len(work_slices)` (Risk 3). No `async_compile.py` change expected;
  verify `spyre_grid` flow.

## Verification

1. New driver `my-examples/sum_reduce_dim0.py` (model on `sum.py`,
   `skip_eager=True`): `torch.sum(x, dim=0)` of `[4096, 64]` → `[64]` — 64 outputs
   = 1 stick (≤ cores), reduction 4096 forced to split. `SENCORES=2`/`4` to shrink.
2. `bash run.sh my-examples/sum_reduce_dim0.py > tmp.log 2>&1`.
3. `output_code.py`: flat 1D `spyre_grid` = `spatial_cores × reduction_shards`;
   baked `work_slices` with `len==prod(grid)` + reduced-axis key;
   `tl.wk_slice_coord`; explicit `tl.sum_stick(...)` (intra-stick) → `tl.sum(...,
   axis=outer_stick)` (on-core) → `tl.reshape(...,[1,...])` → `tl.inter_tile(...,
   "add","reduce_to_one",...)`; `if <reduced_var>==0:` guard; reduction-dim
   block_shape shrunk (NOT inner-stick).
4. `.ktir` (`triton-dump/<hash>/*.ktir`): `grid=[N]` with `N==len(work_slices)`;
   the intra-stick reduce (`tl.sum_stick` lowering) collapses the within-stick dim
   to a one-scalar stick (result-shape check — innermost dim still full stick width,
   one effective element); `ktdp.inter_tile_produce` + `ktdp.inter_tile_reduce`
   present, `tt.inter_tile_reduce` absent; `consumer_tiles_per_group`
   single-equality (reduce_to_one); `linalg.add` in the reduce region (mirror
   `meta.py` `extra_checks`).
5. Numerics: test_harness max-delta vs CPU `torch.sum`. A hardware
   `AttributeError: 'NoneType'...run` is harmless if the run isn't wired; IR shape
   + CPU-ref delta are the gates.

## Risks / open questions

1. **Stick alignment (highest).** Shard must land on the outer-stick dim;
   reduction extent must be divisible by `reduction_shards × stick_size`, else the
   boundary cuts mid-stick → silent wrong block_shape. Assert + bail.
2. **Explicit intra-stick reduction** (SKILL.md sum.py note). The within-stick
   reduction must be emitted explicitly via `tl.sum_stick` *before* the
   unit-leading-dim reshape feeding `tl.inter_tile`, so `inter_tile` combines
   one-scalar sticks, not un-reduced 64-element sticks (position-wise add → silently
   wrong). The frontend owns emitting the right op on the right axis — `tl.sum_stick`
   on the within-stick (innermost) axis, `tl.sum`/`tl.inter_tile` on the outer-stick
   axis — and must not conflate the two. Pin with the `.ktir` result-shape check.
3. **`prod(grid)==len(work_slices)` & SENCORES.** Confirm `SpyreOptions.grid`
   tolerates `(total,)` and how SENCORES < total interacts with DistributeWork.
4. **pick₀ guard over `DeferredLine`.** Ensure the `if` survives deferral/DCE.
5. **Multi-axis ordering.** ≥2 spatial split dims + reduced dim must use identical
   innermost-fastest ordering in grid build, offsets, and `wk_slice_coord` —
   guaranteed by the shared §2 radix helper.

## Cross-references

- [`SKILL.md`](SKILL.md) — three-level dim mapping, within-stick rule, reduction notes.
- [`2606-KernelBundleLXModel.md`](2606-KernelBundleLXModel.md) — bundle/per-function-grid
  model (the later milestone that would carry a split-reduction sub-kernel) and the
  baked-constant (`LX_TMP`) serialization pattern reused in §6.
