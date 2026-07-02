# Multi-D (>1D) Index Gather — Design

Status: **frontend implemented; Triton-side deferred.** This extends the gather
design in [`2603-IndirectAccess.md`](2603-IndirectAccess.md). It removes the
rank-1 *flattening* of the index buffer and instead carries the index in its
native multi-dimensional device layout all the way into `desc.gather`
("option 2").

---

## 1. Why (the problem with flattening)

Milestone-1 gather (`2603` §7) flattened the index to a rank-1 `x_offsets` by
building a fresh `[total]` descriptor over the index buffer. That is lossless
for a **logically 1-D** index, but it discards device-layout structure in
general:

- The flat offset is the *device-linear* position
  `dot(device_coords, row_major_strides(device_size))`, which equals the logical
  position only when `stride_map == row_major_strides(device_size)` — i.e. **no
  interior padding**.
- For a genuine >1D index (e.g. a 2-D `[4, 100]` page grid), an inner dim's
  stick padding makes `stride_map ≠ row_major_strides`, so the device-linear
  flatten **interleaves padding gaps** and scrambles logical order. Example:
  `[4,100]` → `device_size=[4,4,32]`, `stride_map=[100,32,1]` but
  `row_major=[128,32,1]` → logical `[i,j]` lands at device-linear `i*128+j` vs
  logical `i*100+j` (a 28-element gap after each row).

So flattening is a 1-D-only convenience and a dead-end for >1D. The right
representation keeps the index multi-dimensional.

## 2. The Triton ↔ KTIR capability gap

- **KTIR** (`construct_indirect_access_tile`) is built on affine integer-sets /
  maps — inherently n-D capable. But the *current lowering pass*
  (`buildGatherSubscriptMaps`, `traceToSourceMemoryView` in
  `triton/third_party/spyre/lib/Dialect/KTDP/Transforms/LowerDescriptorMemory.cpp`)
  is wired for a **rank-1** index (single iteration variable `x_offsets[c_x + d_0]`,
  and an assert that the index access tile carries exactly one base index).
- **Triton** rejects a >1D `x_offsets` outright at the verifier.

A flattening reshape can't satisfy the backend either: the lowering traces
`x_offsets` provenance to a *direct* `descriptor_load` (no reshape between) — see
`2603` and `traceToSourceMemoryView`.

Two ways to close the gap (both backend/Triton-side, outside `_inductor/`):

1. **Trace through reshape** in the backend to recover the device shape — rejected: it
   reconstructs deliberately-discarded structure, fragile.
2. **Carry >1D end-to-end** (chosen): Triton accepts a multi-D `x_offsets`; KTIR
   builds a multi-D index iteration space. Each IR means what it says.

## 3. Frontend (DONE — in `spyre_triton_kernel.py`)

The torch-spyre frontend already emits the multi-D gather:

- `_emit_index_xoffsets(indirect_sym)` returns the **upstream multi-D index
  load** directly as `x_offsets` (its CSE var name is the indirect `SymT.TMP`
  symbol's name), plus `idx_block` (the index's per-core device block) and
  `num_rows = prod(idx_block)`. No fresh descriptor, no flatten — this also
  removes the milestone-1 "two index loads" wart (one `desc_0.load([dim0,dim1])`
  serves both the bounds-check and `x_offsets`).
- `_emit_gather_descriptor` is unchanged (indirect axis → dim 0, permuted
  strides C1/C7, `block_shape[0]=1` C2).
- `_emit_descriptor_gather` builds the result with shape
  `[*idx_shape, *block_shape[1:]]`, then `tl.reshape`s the leading index dims
  down to the output's single row dim (`prod(idx_shape) == num_rows`, row-major →
  row `flatten(i0,i1,…)`, matching output-row order).
- `_SpyreGatherCSEProxy` (stacked in `__enter__`) overrides `indirect_indexing`
  to mint the index symbol directly, **skipping** the upstream negative-index
  wrap + bounds-check codegen (`ops.add(index, index_expr(size))` + `where` +
  `device_assert`).  That upstream codegen broadcasts the loaded index against
  iteration-shaped operands, which is incompatible with our device-tile-shaped
  descriptor loads (e.g. loaded index shape `(1,32)` vs `(XBLOCK,)`); it
  happened to broadcast for a 2-D source but asserts for a 1-D-flattened (≥3-D)
  source.  The Spyre gather addresses with the raw index, so the wrap/assert are
  unused — this mirrors the SDSC `SpyreKernel.indirect_indexing`.

**Generated kernel for `gather_exp_1d.py`** (`x: f16[64,128]`, `i: i32[64]`):

```python
desc_0 = tl.make_tensor_descriptor(in_ptr0, shape=[2, 32], strides=[32, 1], block_shape=[2, 32])
tmp0   = desc_0.load([dim0, dim1])             # single 2-D index load (+ bounds-check)
desc_1 = tl.make_tensor_descriptor(in_ptr1, shape=[64, 2, 64], strides=[64, 4096, 1], block_shape=[1, 2, 64])
tmp6   = desc_1.gather(tmp0, y_off)            # x_offsets = tmp0 (tensor<2x32>) — multi-D
tmp7   = tl.reshape(tmp6, [64, 2, 64])         # collapse index dims to output rows
desc_2.store([dim_1_0, dim_1_1, dim_1_2], tmp7)
```

**Expected failure (now):** the unrelaxed Triton verifier rejects the 2-D
`x_offsets`:
```
CompilationError: x offsets must be 1D, but got ['constexpr[2]', 'constexpr[32]']
```
This is the intended boundary until the Triton side (§4) lands — analogous to
the documented `AttributeError: 'NoneType'…run` backend-incomplete state.

## 4. Triton side (TODO — relax verifier to accept >1D `x_offsets`)

All changes gated to the Spyre target so the NVIDIA TMA path (hardware-2D
gather) is untouched. Result shape becomes `[*x_offsets.shape, *block_shape[1:]]`
(reduces to the current `[x_offsets.shape[0], *block_shape[1:]]` at rank-1).

### `triton/python/triton/language/semantic.py` — `descriptor_gather` (~1134-1168)
- The rank-1 assert (`len(x_offsets.shape) == 1`, ~1150): gate behind
  `not target_info.is_spyre()`; on Spyre require `>= 1`.
- The `>= 8` rows check (~1153): on Spyre use `math.prod(x_offsets.shape) >= 8`.
- Result type (~1165): `tl.block_type(desc.dtype, [*x_offsets.shape, *desc.block_shape[1:]])`.
- Mirror in `descriptor_scatter` (~1170-1199) for consistency (shared verifier).

### `triton/lib/Dialect/Triton/IR/Ops.cpp` — verifier (1485-1584)
All under the existing `#ifdef TRITON_BUILD_TTIR_ONLY` (Spyre) gate:
- `verifyGatherScatterResultType` (1488): allow `indicesType.getRank() >= 1`.
- Generalize the row checks for multi-D indices: result's **leading
  `indices.rank` dims must equal `indices.shape`** (replaces line 1526's
  `result[0] == indices[0]`); the `>= 8` check (1506) uses the product of those
  leading dims.
- `verifyGatherScatterOp` (1559-1571): result rank ==
  `indicesType.getRank() + blockType.getRank() - 1`; trailing dims
  `result[indices.rank ..]` must equal `block[1 ..]` (shift the per-dim loop by
  `indices.rank - 1`).
- **Requires a Triton C++ rebuild.**

### Tests
- `triton/third_party/spyre/test/test_lower_desc_memory.py::test_gather_2d_indices_rejected`
  changes premise: the Triton verifier now *accepts* a 2-D index; it fails later
  in KTIR. Update to assert verifier-accept + KTIR-failure (or `xfail` pending
  §5). Grep `test_core` / `test_tensor_descriptor` for other rank-1-rejection
  tests and gate them on target.

After §4: `bash run.sh my-examples/gather_exp_1d.py` should produce a valid TTIR
carrying `tt.descriptor_gather %desc[%idx2d, %y]` (2-D indices), passing the
Triton verifier, and then **fail in `make_ktir`** — the expected boundary until §5.

## 5. KTIR side (TODO — the execution work)

`LowerDescriptorMemory.cpp` must carry a multi-D index iteration space:
- `traceToSourceMemoryView` / `resolveIndexView`: drop the rank-1 assert
  (`indices.size() == 1`); accept a multi-D index access tile.
- `buildGatherSubscriptMaps`: build `x_offsets[off_0 + d_0, off_1 + d_1, …]`
  (one index iteration variable per index dim) instead of a single `c_x + d_0`.
- `buildIndirectAccessTile`: thread the per-dim index offsets.

This is what makes the gather actually execute; it is the genuinely invasive
part the `2603` *gather-2d-indices* note flagged ("carrying a multi-D iteration
space through `buildGatherSubscriptMaps`").

## 6. Verification (end state per phase)

| Phase | `run.sh my-examples/gather_exp_1d.py` end state |
|---|---|
| Frontend only (now) | Multi-D gather emitted; Triton verifier rejects 2-D `x_offsets` (`x offsets must be 1D`). |
| + §4 Triton side | Valid TTIR (`tt.descriptor_gather` with 2-D indices); fails in `make_ktir`. |
| + §5 KTIR side | Lowers to `ktdp.construct_indirect_access_tile` with a multi-D index; runnable. |

Regression at every phase: `add.py` / `matmul.py` / `sum.py` unaffected (no
gather), and the NVIDIA Triton path unchanged (Spyre-gated).

Examples: `gather_idx1d_src2d.py` (2-D source) and `gather_idx1d_src3d.py`
(3-D source).  Both now codegen the multi-D gather without the
`indirect_indexing` broadcast `AssertionError` and stop at the `x offsets must
be 1D` boundary.

## 7. Output store: permute the output-row axis to dim 0 (DONE)

The gather result is row-first (`[num_rows, *block[1:]]` after the reshape), so
the output store must put the dense output-row device dim at dim 0 to match.
The output-row dim is the one whose coordinate is the **index's iteration
symbol** (e.g. `c0`, from the int32 index dep `arg1_1[c0]`), captured in
`self._gather_row_sym` during the gather load.

`store()` passes `row_sym=self._gather_row_sym` to
`_emit_symbol_first_tensor_descriptor`, which then uses
`_gather_output_permutation` (target the `row_sym` dim) instead of
`_symbol_first_permutation` (first bare symbol).  The latter was wrong for a
≥3-D source: e.g. output coords `[c1, floor(c2/64), c0, Mod(c2,64)]` have `c1`
as the first bare symbol, so the descriptor was left unpermuted and its
`block_shape` did not match the row-first result.

Now (3-D source) both line up:
```
desc_2 = tl.make_tensor_descriptor(out_ptr0, shape=[64,128,4,64],
                                   strides=[64,16384,4096,1], block_shape=[64,128,4,64])
tmp2   = tl.reshape(tmp1, [64,128,4,64])
desc_2.store([...], tmp2)              # shapes match
```
The permutation is applied in lockstep to shape/strides/block_shape/offsets, so
the device addresses are unchanged (`dot(perm(coords), perm(strides)) ==
dot(coords, strides)`) — it is a pure axis relabeling consistent with the OpSpec
device layout.  Row-axis work division (`num_rows` vs per-program rows) stays
consistent because both the index block and the output-row block divide that
axis by the same core divisor.

## 8. Future work: fused gather + compute (when fusion is enabled)

**Not needed today:** `torch_spyre/_inductor/choices.py` disables all fusion
(`can_fuse` / `can_fuse_vertical` / `can_fuse_horizontal` → `False`), so a gather
is never in the same kernel as a downstream compute op.  `x[i] + y` becomes
kernel A (`buf0 = x[i]`, identity gather + store) and kernel B
(`add(buf0, y)` over two plain tensors) — no in-kernel operand-shape mismatch.

**When fusion is enabled** (to avoid the `buf0` HBM round-trip), a fused
`x[i] + y` would load the gather result (row-first tile `[num_rows, *block[1:]]`)
and a co-operand `y` (its own device-block tile) in the same kernel; an
elementwise op then requires matching tile shapes.  Make all tiles row-first:

1. **Detect the gather up front** (in `set_current_node`, not on the gather
   load): if the node has an indirect read dep, mark it a gather kernel and set
   `self._gather_row_sym` from the int32 index dep.  Up-front because a
   co-operand `load()` can run *before* the gather load in the inner_fn replay.
2. **Propagate the row-first permutation to every descriptor** in a gather
   kernel: each non-gather load and the store uses
   `_gather_output_permutation(coords, self._gather_row_sym)` (output-row symbol
   → dim 0), the same permutation §7 added for the store.  Then every tile is
   row-first, the gather result's post-reshape shape matches the co-operands,
   and elementwise ops align.  The gather *value* descriptor stays
   indirect-axis-first (`block_shape[0]=1`); its reshape already lands the
   result row-first.

Caveats:
- **Unary ops** (`x[i].exp()`) are already fine even fused — single operand,
  shape preserved; no propagation needed.
- **Broadcast operands** (e.g. `x[i] + bias`, `bias: [N]`) have no row axis, so
  the propagation must be broadcast-aware: permute only the axes the operand
  actually has and leave size-1/broadcast dims alone, rather than assuming a
  row dim.
- Canonical shape is the **post-reshape** row-first `[num_rows, *block[1:]]`.

Defer implementation until fusion is actually turned on (it can't be exercised
before then).
