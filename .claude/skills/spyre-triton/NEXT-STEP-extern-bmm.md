# Immediate next step: `extern_kernels.bmm` for decode-phase linear (m == 1)

**Status:** guard fixed; matmul `tl.dot` codegen fixed and verified for 2-D,
M==1 (GEMV), real batched bmm, and batch>1 broadcast (linear-style). Linear
end-to-end still blocked by two *separate* backlog items: the weight-transpose
restickify (`kernel_0`) and the `tl.arange` power-of-2 limit for non-pow2 M in
the 3-D matmul grid. See "Progress 2026-07-02" below.
**Owner:** continuing together (2026-07-02).

## Problem

Running the Llama-3.1-8B linear cases through the Triton path
(`run_test.sh` → `tests/models/test_model_ops.py --model Meta-Llama-3.1-8B-Instruct -k linear`)
emits `extern_kernels.bmm` for 7 of the `F.linear` cases instead of a Spyre
Triton kernel. `extern_kernels.bmm` cannot run on Spyre.

Reproducers:

- `my-examples/linear.py` — q_proj shape `[1,12,4096] @ [4096,4096]` (m=12).
  This one is fine: it takes the native-matmul path.
- The failing cases are the **decode-phase** linears with seq_len == 1:
  down_proj `(1,1,14336)@(1,14336,4096)` and lm_head `(1,1,4096)@(1,4096,128256)`.

## Root cause

`_patched_use_native_matmul` in
`torch_spyre/_inductor_triton/spyre_triton_patches.py` (lines 74-79) inherits
the upstream degenerate-dim guard verbatim:

```python
m, k, n = mat1.get_size()[-2], mat1.get_size()[-1], mat2.get_size()[-1]
if (
    V.graph.sizevars.statically_known_leq(m, 1)   # m == 1 (seq_len=1) trips this
    or V.graph.sizevars.statically_known_leq(k, 1)
    or V.graph.sizevars.statically_known_leq(n, 1)
):
    return False
```

Chain: `m == 1` → returns `False` → native matmul disabled → PyTorch's standard
`aten.bmm` lowering falls back to its ATen `ExternKernelChoice`
(`extern_kernels.bmm`). Spyre cannot rescue it via its own lowering either,
because `aten.bmm.default` is popped from `spyre_lowerings` (it's in
`_TRITON_SKIP_MM_OPS`, lines 33-36), so there is no BATCH_MATMUL_OP lowering to
catch it.

## Evidence (from a full linear-cases run, `test.log`)

| Path taken | count | LHS (`mat1`) shape | `m` |
|---|---|---|---|
| `extern_kernels.bmm` | 7 | `(1,1,14336)` ×1, `(1,1,4096)` ×6 | 1 |
| native `tl.dot` (Triton) | 4 | prefill shapes | >1 |

Exact correlation: every `extern_kernels.bmm` case has `mat1 = (1, 1, K)`
(m == 1); every `tl.dot` case has m > 1.

## Fix direction (to do)

1. Relax the guard in `_patched_use_native_matmul` for Spyre so `m == 1` (the
   decode/GEMV case) still takes native-matmul → `tl.dot`. Keep the `k <= 1` /
   `n <= 1` degenerate checks; only `m <= 1` corresponds to the real seq_len=1
   workload.
2. Verify `SpyreTritonKernel` / the Spyre Triton backend can codegen and lower
   an `m == 1` `tl.dot` end-to-end (block_shape, reduction axis, KTIR).
3. Re-run `run_test.sh` (linear cases) and confirm the 7 extern cases now route
   into a Triton bundle; then check kernel coverage/non-overlap per SKILL.md.

## How to reproduce

```bash
bash run_test.sh > test.log 2>&1        # renamed run_linear_test.sh + debug env vars
grep -c "extern_kernels.bmm(" test.log  # expect 7 before the fix, 0 after
```

## Progress 2026-07-02

**Done (verified):**

1. **Guard relaxed** — `_patched_use_native_matmul` now allows `m == 1` on
   spyre (keeps the `k/n <= 1` guards). `extern_kernels.bmm` is gone; the
   decode-phase linears take the native `tl.dot` path.
2. **m == 1 matmul codegen fixed** in `spyre_triton_kernel.py`:
   - **`_matmul_operand_permutation`** (replaces the bare-symbol
     `_symbol_first_permutation` / `_batch_symbol_first_permutation` for matmul
     operands): anchors on the *stick pair* (outer-stick + within-stick dims
     share the within-stick's iteration symbol; within-stick is always last),
     placing them adjacent + innermost with batch/row dims leading (batch first
     for bmm). Correct even when M == 1 collapses the row coordinate to a
     constant `0` — the old bare-symbol search then found nothing and left the
     K stick dims non-adjacent (produced `dot([1,64,64], [4096,128])`).
   - **`SpyreTritonOverrides.dot()` rank reconciliation via `_fold_leading_dims`**:
     a linear-derived bmm has a batched activation but a broadcast (un-batched)
     weight, so the collapsed operands differ in rank (`A [batch, M, K]` vs
     `B [K, N]`). Because the weight is shared across batch *and* M, those
     leading dims are all matmul rows, so they are **folded into a single row
     dim** on the higher-rank operand: `[batch, M, K] -> [batch*M, K]` (the
     size-1 `[1, M, K] -> [M, K]` is the degenerate case). A real bmm (both
     operands batched, equal rank) never folds. The store side already reshapes
     the `[rows, N]` dot result back to the output block shape.
   - Verified (all reach the expected backend-incomplete state `AttributeError:
     'NoneType' object has no attribute 'run'`, i.e. Python/Triton codegen is
     correct; no regressions):

     | Example | dot | notes |
     |---|---|---|
     | `matmul.py` | `[8,1024]@[1024,512]` | 2-D, unchanged |
     | `bmm.py` | `[4,4,256]@[4,256,512]` | real batch=4, no fold |
     | `mm_m1.py` | `[1,4096]@[4096,128]` | 2-D M=1 GEMV |
     | `bmm_bcast_b2.py` | `[32,4096]@[4096,128]` | batch=2 broadcast (3-D@2-D), M=16, folded |

     `mm_m1.py` and `bmm_bcast_b2.py` codegen correctly end-to-end (no transpose
     kernel). The `F.linear` examples (`linear.py` m=1, `linear_m12.py` m=12,
     `linear_batch2.py` batch=2) all have well-formed `kernel_1` dots but are
     still blocked by `kernel_0` (below).

**Remaining blocker (separate, M-independent): `kernel_0` weight transpose.**
`F.linear` = `x @ w.T`; the `permute` of `w` becomes a standalone restickify
copy (`kernel_0` in the bundle) that materializes the transposed weight
(`arg1_1 → buf1`), which the matmul `kernel_1` consumes. `kernel_0` loads a
`[64,128,64]` tile but stores a `[2,4096,64]` tile — same element count, but a
genuine cross-dim transpose, so the store shape mismatches the loaded block and
Triton rejects it. This fails identically for m=1 and m=12 (weight is
`[4096,4096]` either way), so it is **not** an m==1 issue.

Two directions to fix (decide together):
- (A) Fix the pointwise transpose/restickify codegen so the load tile is
  transposed to the store tile (e.g. `tl.trans`, or make store block == load
  block via matched descriptor strides).
- (B) Eliminate `kernel_0` by loading the weight transposed directly in the
  matmul kernel via descriptor strides (fuse the transpose into `kernel_1`'s
  B-operand load; `desc_1` there already uses non-row-major strides
  `[64,262144,1]`), so no separate restickify buffer is needed.

Reproducers: `my-examples/linear.py` (m=1), `my-examples/mm_m1.py` (2-D m=1,
codegens clean), `my-examples/linear_m12.py` (m=12, same `kernel_0` failure),
`my-examples/linear_batch2.py` (batch=2), `my-examples/bmm_bcast_b2.py`
(batch=2 broadcast, codegens clean — no transpose kernel).

**Second backlog item — `tl.arange` power-of-2 for non-pow2 M in the 3-D matmul
grid.** The `torch.matmul(3-D, 2-D)` path uses `BatchMatmulGrid3D` (z=batch,
y=M, x=N) and emits `tl.arange(0, YBLOCK)` with `YBLOCK == M`. Triton requires
`arange` ranges to be powers of 2, so a non-pow2 M (e.g. 12) fails with
`arange's range must be a power of 2`. Use a pow2 M (e.g. 16) to exercise
batch>1 for now; padding M to a power of 2 in the grid path is a separate
long-term fix. (The `F.linear` path uses a 1-D grid bundle and does not hit
this — it's blocked by `kernel_0` first.)
