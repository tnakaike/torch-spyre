# LX Planning in the Triton Path — Design

Status: **design only; not implemented.** This describes how the on-core LX
scratchpad should be used by the Triton path (`SpyreTritonKernel`), and how it
differs from the SDSC (non-Triton) path's scratchpad planning.

Background on LX itself — the 2 MB per-core scratchpad, the greedy solver, the
softmax stages — is in
[`docs/source/compiler/scratchpad_planning.md`](../../../docs/source/compiler/scratchpad_planning.md).
This document only covers the **Triton-path** representation question.

---

## 1. The core idea in one sentence

Prefer **fusion** so intermediates never leave the kernel (they are SSA values,
not tensors); fall back to an **LX-tagged tensor descriptor** only for the rare
tensor that genuinely must cross a kernel boundary.

This inverts the SDSC model. There, the scratchpad planner *owns* LX: it assigns
explicit per-core addresses and bakes them into every OpSpec
(`component_="lx"`). In the Triton path, the first choice is to make the
intermediate disappear entirely.

## 2. Why the regimes differ — the runtime model

Torch-spyre does **not** pass tensors across kernel-launch boundaries. The SDSC
path bundles many SDSC ops into a *single* host kernel and keeps intermediates
in LX **internally**, across SDSC sub-units that all live inside that one
launch. The reason the SDSC path splits a bundle into multiple SDSCs at all is
an SDSC-backend implementation detail (e.g. the per-bundle tensor limit), not a
property the Triton path has to reproduce.

The Triton-path analogue of "one bundle" is **one Triton kernel** (= one
`triton.compile` = one launch). So the design goal is to make the set of ops the
SDSC path would bundle land in a single Triton kernel — and then the
cross-kernel LX question mostly evaporates.

Two regimes follow:

| Regime | When | Representation |
|---|---|---|
| **Intra-kernel** | Producer and consumer fuse into one Triton kernel | SSA temp — *def then use*, no descriptor, no global memory (Principle 1) |
| **Cross-kernel** | Fusion is impossible; a tensor must outlive a kernel | LX-tagged tensor descriptor at a baked address (Principle 2) |

## 3. Principle 1 — Fusion first: in-kernel def/use, no global memory

When the producer and consumer of an intermediate are in the **same** Triton
kernel, Inductor fuses the intermediate into an SSA value (`tmpN` CSE var). It
is *defined* by one op and *used* by the next directly — never stored to or
loaded from memory:

```python
# exp(exp(x)) fused into one kernel — y is never materialized
tmp0 = desc_in.load([...])
tmp1 = tl.exp(tmp0.to(tl.float32)).to(tl.float16)   # def y
tmp2 = tl.exp(tmp1.to(tl.float32)).to(tl.float16)   # use y (no store/reload)
desc_out.store([...], tmp2)
```

This is **strictly better** than a planned LX buffer: no store, no reload, no
address, no allocation. It already works today for fused pointwise chains, and
it is the right outcome for softmax (`max → sub → exp → sum → div`) and any
pointwise/reduction chain Inductor can fuse.

The lever here is **fusion in the scheduler** (`SpyreTritonScheduling` /
Inductor), not anything in `spyre_triton_kernel.py`. "Implementing LX" for these
graphs means *maximizing fusion*, so the intermediate is represented as
dataflow, not as a tensor on LX.

### Where fusion breaks

Inductor does not fuse everything into one kernel. It splits at:

- **matmul** (`tl.dot` is its own kernel),
- **reduction barriers** where the full reduced result must exist before the
  next pass,
- **multi-consumer** intermediates read by several downstream kernels,
- **shape/tiling changes** between producer and consumer.

Attention (`mm → softmax → mm`) is the canonical graph that will *not* collapse
to a single Triton kernel. Those intermediates fall into Regime 2.

### Open question: who owns LX for materialized in-kernel intermediates?

Even inside one Triton kernel, a coarse intermediate that cannot stay an SSA
value gets bufferized by the **KTIR backend**, and *that* allocator decides LX
vs HBM — the same way a GPU compiler manages shared memory. If the KTIR
backend already bufferizes within-kernel intermediates to LX, torch-spyre's
scratchpad planner is **irrelevant** to the Triton path and we plumb nothing.
If it spills them to HBM, the fix is in the KTIR backend, not in
`spyre_triton_kernel.py`. **This must be verified** by inspecting the `.ktir`
dump of a fused multi-op kernel (see §8).

## 4. LX-aware fusion: estimating the per-core working set

Fusion (Principle 1) is only safe up to LX capacity. Fuse too many ops and the
combined working set exceeds ~1.6 MB usable per core, and the KTIR backend has
to spill to HBM — silently undoing the win. So fusion should be **bounded by an
LX estimate**, and torch-spyre already has the pieces to make one.

**What to estimate.** The peak bytes simultaneously live on *one core's*
scratchpad over the fused op sequence:

```text
LX_estimate = max over the fused op sequence of
              Σ (per-core tile bytes of each live buffer at that point)
```

Two facts make this tractable at fusion-decision time:

- **Footprint is per-core tiles, not whole tensors.** LX is core-local, so what
  counts is the *post-work-division* tile — `block_shape` (or
  `per_loop_block_shape` under a `LoopSpec`), not `device_size`.
  `SpyreTritonKernel` already computes these for every descriptor, so the
  per-tile byte sizes are in hand. Coarse tiling (a `LoopSpec`) shrinks the
  per-iteration footprint and so directly relaxes the fusion bound.
- **Peak-live-set is a liveness walk the scratchpad planner already does.**
  `LifetimeBoundBuffer` + the greedy/first-fit solvers in
  `_inductor/scratchpad/` compute the high-water mark of simultaneously-live
  buffers. The same walk, run over the *fusion group's* op list with per-core
  tile sizes, is the estimate.

### The "loads in LX, results streamed" assumption

A useful first-order model is: **a loaded tensor occupies LX; a compute result
does not.** This maps onto a real hardware behavior — staged tensors (loaded
inputs, output tiles being assembled) sit in LX, while a pipelined compute
result flows through the compute units and is consumed without its own LX
residency. For an elementwise/streaming chain (`load → exp → exp → store`) it is
accurate: the result reuses the dying input's slot (in-place) or never lands.

As a *safe* estimate it is optimistic, because some results **must** be
materialized and therefore do occupy LX:

| Result that must live in LX | Why it cannot be streamed away |
|---|---|
| **Reduction output** | Different shape; accumulated across tiles before any consumer runs |
| **Matmul accumulator** (`tl.dot`) | The `C` tile is read-modify-written across the K loop |
| **Multi-consumer value** | Must persist while several downstream ops read it |
| **Shape/broadcast change** | Producer and consumer tiles differ — no in-place reuse |

So the precise rule is: *a compute result adds to LX only when it cannot be
in-placed onto a dying input* — exactly the distinction the planner already
encodes (`OP_GOOD_FOR_LX_INPLACE`). The assumption gives a **lower bound**;
"every result needs its own tile" gives an **upper bound**; the in-place
classification picks the right one per op.

### Calibration is what makes the estimate actionable

The estimate is only useful if it **matches what the KTIR backend actually
does**, since the backend — not torch-spyre — owns LX allocation for
within-kernel intermediates (§3, "who owns LX"; open question in §8). If the
model assumes
results stream through but the backend bufferizes every intermediate to LX, an
over-fused kernel spills and the estimate was wrong in the unsafe direction.

So the "does this result occupy LX?" rule should be **calibrated to the
backend's real policy** (the `.ktir`-dump probe in §8), not chosen for
convenience. Once calibrated, the estimate's role is to **gate fusion**: keep
fusing while predicted peak-live < ~1.6 MB/core, stop before the backend would
have to spill.

## 5. Principle 2 — Cross-kernel tensors: Option A (memory_space attribute)

For the genuinely-unfusable case (§3, "where fusion breaks"), a tensor must
cross a kernel boundary. Keep it on LX instead of round-tripping HBM.

**Representation.** Emit a normal tensor descriptor, but:

- its **base** is the baked LX byte offset from `layout.allocation["lx"]` (a
  compile-time constant, *not* a kernel-argument pointer),
- it is **tagged** `memory_space = LX`, so the lowering produces
  `#ktdp.spyre_memory_space<LX>` instead of the hardcoded HBM,
- the buffer is **dropped from the kernel signature and the wrapper
  allocation** — it is core-local persistent state, not a runtime arg.

Both the producer kernel and the consumer kernel emit a descriptor over the
**same** LX offset. Because each core's scratchpad uses the same offset, this is
the executing-core's LX — tag it `<LX>`. `memory_space` is **core-agnostic**:
every core only ever addresses its *own* LX. Cross-core access (one core needing
data another core produced) is **not** a foreign-core descriptor tag — it is an
explicit `tl.inter_tile` op over the data ring (`reduce_to_one` / `all_reduce`
today; `reduce_scatter` / `broadcast` are the future enabler for
redistribution). See [`2606-KernelBundleLXModel.md`](2606-KernelBundleLXModel.md)
§6.

This mirrors the SDSC path's `component_="lx"` + baked per-core
`startAddressCoreCorelet_` exactly — just expressed through a Triton descriptor.

**Frontend work** (`spyre_triton_kernel.py`): skip `args.input/output` for LX
buffers (today they fall back to `super().load()`); emit the baked-address
descriptor; attach the memory-space tag. **Wrapper work**: drop the HBM
allocation for LX buffers.

**Caveat.** Cross-launch LX persistence is a known correctness gap: under VF
multi-tenancy the runtime may wipe LX at a bundle boundary (see
`scratchpad_planning.md`, "LX state survives kernel boundaries"). Two separate
Triton launches are a *harder* boundary than SDSC sub-bundles. This is the main
reason Principle 1 is strongly preferred and Principle 2 is a fallback, not the
default.

## 6. Does Option A require changing `make_tensor_descriptor`?

**Short answer: yes — but only to carry a memory-space *signal*; the geometry
handling is untouched, and the substantive change is in the lowering.**

Today `tl.make_tensor_descriptor(base, shape, strides, block_shape)` is an
upstream Triton builtin with **no** memory-space parameter, and the Spyre
lowering hardcodes HBM:

```cpp
// LowerDescriptorMemory.cpp  (buildBaseMemoryView)
auto memSpaceAttr = mlir::ktdp::SpyreMemorySpaceAttr::get(
    ctx, mlir::ktdp::SpyreMemorySpaceKind::HBM, /*core=*/-1);
```

`LowerDescriptorMemory.cpp` **must** change regardless of approach: stop
hardcoding HBM and instead read a memory-space signal (defaulting to HBM, so
every existing kernel is unaffected). The KTDP vocabulary already exists
(`SpyreMemorySpaceKind::{unspecified, LX, HBM}`); nothing populates it from the
frontend yet.

How the signal reaches the lowering is the sub-decision:

| Mechanism | Changes `make_tensor_descriptor`? | Trade-off |
|---|---|---|
| **M1 — explicit attribute** `make_tensor_descriptor(..., memory_space="lx")` | Yes: builtin gains an optional kwarg that emits an attribute on `tt.make_tensor_descriptor`; lowering reads it | Cleanest; LX-ness is visible in the kernel source and self-describing like shape/strides. Cost: patching/forking the upstream builtin + its op verifier |
| **M2 — Spyre-side tag** Keep the builtin; attach the attribute via a Spyre-specific path/wrapper op | Not the upstream signature, but still new descriptor-creation code | Avoids touching upstream, but invents a Spyre construct and a correlation step |
| **M3 — provenance inference** Lowering infers LX when the descriptor base is a compile-time constant (not an arg pointer) | No | No builtin change, but implicit and fragile: conflates "constant base" with "LX" |

**Recommendation: target M1.** The descriptor is already self-describing for
geometry; adding `memory_space` (default HBM) keeps it self-describing for
placement and maps 1:1 onto the existing KTDP attribute. `memory_space` stays
**core-agnostic** (`{LX, HBM}`) — cross-core movement is an explicit
`tl.inter_tile` op over the data ring, not a descriptor tag. M3 is a tempting
no-frontend-change shortcut but bakes in an assumption that will block later
work.

In all three, the *amount* of new logic in `make_tensor_descriptor` is small —
it forwards one optional attribute. The real work is teaching the lowering to
honor it.

## 7. Comparison to the SDSC path

| | SDSC path | Triton path (this design) |
|---|---|---|
| Who owns LX | scratchpad planner (explicit addresses) | fusion first; KTIR backend for in-kernel materialized temps; planner only for Regime 2 |
| Intra-kernel intermediate | LX buffer with baked address | SSA temp — not materialized |
| Cross-kernel intermediate | LX across SDSC sub-bundles in one launch | LX-tagged descriptor across two launches (fallback) |
| LX tag mechanism | `component_="lx"` in OpSpec | `#ktdp.spyre_memory_space<LX>` from a descriptor attribute |
| Buffer as kernel arg | excluded when `"lx"`/`"pool"` in allocation | excluded (drop from signature + wrapper) |

## 8. Open questions / to verify

1. **Decisive:** Does the KTIR backend bufferize within-kernel intermediates to
   LX or HBM? Inspect `triton-dump/<hash>/*.ktir` for a fused multi-op kernel
   (`exp(exp(x))` or softmax). This decides whether Principle 1 needs *any*
   torch-spyre LX plumbing or is already handled downstream.
2. **Fusion coverage:** Where exactly does Inductor split the target graphs?
   Confirm which intermediates fall into Regime 2 in practice.
3. **Persistence:** Does LX survive between two Triton launches on this runtime,
   or only within one? If only within one, Regime 2 is unsafe until SpyreCode /
   non-terminal-kernel hints land, and the design should restrict LX to
   Principle 1.
4. **Plumbing:** Is `allocation["lx"]` consumed by the Triton path at all?
   Depends entirely on (1).

## 9. References

- [`docs/source/compiler/scratchpad_planning.md`](../../../docs/source/compiler/scratchpad_planning.md)
  — LX hardware, solvers, the bundle-persistence assumption and its gap.
- [`2602-OpSpecToTriton.md`](2602-OpSpecToTriton.md) — the Triton-path
  descriptor model this builds on.
- `triton/third_party/spyre/lib/Dialect/KTDP/Transforms/LowerDescriptorMemory.cpp`
  — where the HBM hardcode lives.
- `triton/third_party/spyre/ktir-mlir-frontend/include/Ktdp/KtdpAttrs.td` —
  `SpyreMemorySpaceKind` / `SpyreMemorySpaceAttr` (`<LX>`, `<HBM>`). The attr
  still carries an optional `core` parameter, but **this design does not use
  it** — cross-core data movement is a `tl.inter_tile` op, not a `core=N`
  descriptor tag.
- `torch_spyre/_inductor/scratchpad/allocator.py` — where
  `layout.allocation["lx"]` is assigned (SDSC path).
