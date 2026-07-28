# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpSpec -> Triton source generator (see DESIGN-OpSpecToTriton.md).

``SpyreOpSpecTritonKernel`` is **not** a ``TritonKernel`` subclass.  It is a
``SpyreKernel`` (the SDSC frontend) that, after the op_specs are finalized,
*generates* Triton ``tl.*`` source by walking the finished ``OpSpec`` list.

The whole three-level dimension mapping that makes ``SpyreTritonKernel``
(~2555 lines, a ``TritonKernel`` subclass) large is unnecessary here: Inductor
never mints ``x/y/r0_`` symbols for us, so there is nothing to reconcile back to
the OpSpec symbols ``c0/c1/...``.  We *choose* the Triton program axes from the
OpSpec symbols by construction:

    one program axis per work-divided OpSpec symbol   (c0 -> x, c1 -> y, ...)
    per-core block extent   = device_size[k] // core_divisor
    grid                    = prod(core_divisors)
    program offsets         = mixed-radix decomposition of program_id(0)
    tensor descriptor       = make_tensor_descriptor(shape=device_size,
                                                     strides=row_major(device_size),
                                                     block_shape=per_core_shape)

This first cut covers pointwise ops (``add.py``).  Reductions, ``tl.dot`` and
``LoopSpec`` raise ``NotImplementedError`` — they are the next bring-up steps.

Fuse vs. split
--------------
``SpyreKernel`` (the SDSC frontend) builds one op_spec per single op and, because
SDSC drives a whole kernel from one grid, forces every op in a scheduler node to
share one work division.  The Triton path is not bound by that: we partition a
node's op_specs into *groups* that share an identical iteration space (same
symbols, ranges, and work divisions) and emit **one Triton kernel per group**.
Ops with matching iteration space fuse (tiles thread through registers); ops with
a different iteration space or work division are split into separate kernels,
each with its own grid and ``.run()`` call.  A value produced by one group and
consumed by another crosses the group boundary through its HBM buffer (every op
output already carries an HBM allocation + descriptor), since register threading
only holds within a single kernel.
"""

import dataclasses
import os

import sympy
from torch._inductor.codegen.triton import TritonKernel, texpr
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import V
from torch.utils._sympy.functions import FloorDiv

from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, TensorArg
from torch_spyre._inductor.spyre_kernel import SpyreKernel

logger = get_inductor_logger("opspec_triton_kernel")

# Placeholder substituted with the real kernel name in the scheduler's
# define_kernel (codegen_kernel runs before the name is assigned).
KERNEL_NAME_PLACEHOLDER = "__KERNEL_NAME__"

# Pointwise binary op name -> infix tl expression operator.
_BINARY_OPS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
}


@dataclasses.dataclass
class _KernelPlan:
    """One emitted Triton kernel: its source plus the ``.run`` call arguments.

    ``call_args`` are the caller-side buffer names (in signature order, deduped,
    with a leading ``_pool`` if the group touches pool memory) that the scheduler
    passes to ``<name>.run(...)`` for this group's kernel.
    """

    source: str
    call_args: list[str]


def _normalize_floor_div(expr: sympy.Expr) -> sympy.Expr:
    """Replace ``sympy.floor(a / n)`` with ``FloorDiv(a, n)`` for integer floordiv.

    ``compute_coordinates`` builds coordinates with Python ``//``, which sympy
    stores as ``floor(a / n)``.  ``texpr`` renders that as a floating-point
    ``libdevice.floor(...)``, which is wrong for integer device offsets.  This
    rewrites it to the Inductor-native ``FloorDiv`` form so it prints as integer
    floor division.  (Lifted from ``spyre_triton_kernel.py`` on ``dev/triton``.)
    """

    def _rewrite(e: sympy.Expr) -> sympy.Expr:
        if isinstance(e, sympy.floor):
            inner = e.args[0]
            coeff, base = inner.as_coeff_Mul()
            if isinstance(coeff, sympy.Rational) and coeff.p > 0 and coeff.q > 1:
                return FloorDiv(_normalize_floor_div(base) * coeff.p, coeff.q)
            return FloorDiv(_normalize_floor_div(inner), 1)
        if e.args:
            new_args = [_rewrite(a) for a in e.args]
            if any(na is not oa for na, oa in zip(new_args, e.args)):
                return e.func(*new_args)
        return e

    return _rewrite(expr)


def _buf_id(arg: TensorArg) -> object:
    """Stable identity of the buffer an op arg refers to, for register threading.

    A fused-away intermediate carries ``arg_index == -1`` (the unassigned
    sentinel), so distinct intermediates collide on ``arg_index``.  The op-spec
    ``name`` is the buffer name, unique per buffer and identical whether the
    buffer appears as an input or an output, so it is the reliable key; fall back
    to ``arg_index`` only when a name is absent.
    """
    return arg.name if arg.name is not None else ("idx", arg.arg_index)


def _coord_str(coord: sympy.Expr) -> str:
    """Render a device coordinate expression as Triton scalar-index source."""
    return texpr(_normalize_floor_div(coord))


def _row_major_strides(device_size: list[int]) -> list[int]:
    """Row-major (C-contiguous) strides for a device-size list."""
    n = len(device_size)
    strides = [1] * n
    for i in range(n - 2, -1, -1):
        strides[i] = strides[i + 1] * int(device_size[i + 1])
    return strides


class SpyreOpSpecTritonKernel(SpyreKernel):
    """SpyreKernel that emits Triton source instead of the flex OpSpec literal.

    All op_spec building (``load``/``store``/``store_reduction``/
    ``create_op_spec``/``create_tensor_arg``) is inherited unchanged from
    ``SpyreKernel`` — the mature layout frontend comes for free.  Only the final
    projection to source is replaced.
    """

    def codegen_kernels(self) -> list[_KernelPlan]:
        """Finalize the op_specs and emit one Triton kernel per fusion group.

        ``super().codegen_kernel()`` runs ``simplify_op_spec`` on each op and
        assigns ``tensor_arg.arg_index`` + ``allocation["hbm"]`` (we need
        ``arg_index`` to map each ``TensorArg`` to its kernel parameter; the flex
        literal it returns is discarded).  The finalized op_specs are then
        partitioned by iteration space and each group is projected to its own
        Triton kernel — see the module docstring's "Fuse vs. split".
        """
        super().codegen_kernel()
        plans = [self._emit_group(group) for group in self._partition_specs()]
        for plan in plans:
            self._dump_source(plan.source)
        return plans

    # -- fusion grouping ----------------------------------------------------

    def _partition_specs(self) -> list[list[OpSpec]]:
        """Split the op_specs into maximal contiguous same-iteration-space runs.

        Ops that share an iteration space (symbols + ranges + work divisions)
        fuse into one kernel; a change of iteration space starts a new group.
        Grouping is *contiguous* so the kernels run in op order — a later op that
        reads an earlier op's (now HBM) output is always sequenced after it.
        """
        groups: list[list[OpSpec]] = []
        prev_key: object = None
        for spec in self.op_specs:
            self._validate_spec(spec)
            assert isinstance(spec, OpSpec)  # narrowed by _validate_spec
            key = self._iteration_space_key(spec)
            if not groups or key != prev_key:
                groups.append([spec])
            else:
                groups[-1].append(spec)
            prev_key = key
        if not groups:
            raise NotImplementedError("OpSpec->Triton: no ops to emit")
        return groups

    @staticmethod
    def _validate_spec(spec) -> None:
        """Assert ``spec`` is one of the ops this first cut supports."""
        if isinstance(spec, LoopSpec):
            raise NotImplementedError(
                "OpSpec->Triton: LoopSpec (counted loop) not supported yet"
            )
        if not isinstance(spec, OpSpec):
            raise NotImplementedError(
                f"OpSpec->Triton: unexpected spec {type(spec).__name__}"
            )
        if spec.is_reduction:
            raise NotImplementedError("OpSpec->Triton: reduction ops not supported yet")
        if spec.op not in _BINARY_OPS:
            raise NotImplementedError(
                f"OpSpec->Triton: op '{spec.op}' not supported yet"
            )

    @staticmethod
    def _iteration_space_key(spec: OpSpec) -> tuple:
        """Hashable canonical form of ``spec.iteration_space`` for grouping.

        Two ops fuse iff this key matches: same symbols, same ranges, and same
        work divisions.  Symbols/ranges are compared by their string form so the
        key is order-independent and hashable.
        """
        return tuple(
            sorted(
                (str(sym), str(rng), int(div))
                for sym, (rng, div) in spec.iteration_space.items()
            )
        )

    # -- generator core -----------------------------------------------------

    def _emit_group(self, group: list[OpSpec]) -> _KernelPlan:
        """Emit one Triton kernel (source + ``.run`` args) for a fusion group."""
        # Kernel parameter names / caller buffer names, in arg_index order
        # (in_ptr0, in_ptr1, ..., out_ptr0).  TensorArg.arg_index indexes into
        # these parallel lists (assigned by super().codegen_kernel()).
        argdefs, actuals, _sig, _types = self.args.python_argdefs()
        param_names = [a.name for a in argdefs]

        # Unique *materialized* tensor args of this group, first occurrence per
        # arg_index.  A negative arg_index marks a buffer that was never made a
        # kernel parameter (a fused-away intermediate): it is threaded through
        # registers by _emit_ops, so it gets no descriptor and no signature slot.
        tensor_args: dict[int, TensorArg] = {}
        for spec in group:
            for a in spec.args:
                if (
                    isinstance(a, TensorArg)
                    and a.arg_index >= 0
                    and a.arg_index not in tensor_args
                ):
                    tensor_args[a.arg_index] = a
        used = sorted(tensor_args)

        # Every op in a group shares one iteration space (grouping invariant).
        it_space = group[0].iteration_space
        grid = 1
        for _rng, div in it_space.values():
            grid *= int(div)

        # Build the body at column 0; buf.indent() re-indents it under `def`.
        body = IndentedBuffer()
        self._emit_logical_offsets(body, it_space)
        desc_of, dims_of = self._emit_descriptors(
            body, tensor_args, param_names, it_space
        )
        self._emit_ops(body, group, desc_of, dims_of)

        signature = ", ".join(param_names[i] for i in used)
        header = self._emit_header(grid, used)
        buf = IndentedBuffer()
        buf.splice(header)
        buf.writeline(f"def {KERNEL_NAME_PLACEHOLDER}({signature}):")
        with buf.indent():
            buf.splice(body.getvalue())

        return _KernelPlan(
            source=buf.getvalue(),
            call_args=self._group_call_args(tensor_args, used, actuals),
        )

    @staticmethod
    def _group_call_args(
        tensor_args: dict[int, TensorArg],
        used: list[int],
        actuals: list[str],
    ) -> list[str]:
        """Caller-side buffer names for this group's ``.run`` call.

        Mirrors ``SpyreKernel.call_kernel``: a leading ``_pool`` when the group
        touches pool memory, then the used arg buffers in arg_index order,
        deduplicated (an in-place op lists the same buffer as input and output).
        """
        call_args: list[str] = []
        if any("pool" in a.allocation for a in tensor_args.values()):
            call_args.append("_pool")
        seen: set[str] = set()
        for i in used:
            name = actuals[i]
            if name not in seen:
                seen.add(name)
                call_args.append(name)
        return call_args

    def _emit_logical_offsets(
        self,
        body: IndentedBuffer,
        it_space: dict[sympy.Symbol, tuple[sympy.Expr, int]],
    ) -> None:
        """Emit ``# Triton -> Logical layouts``: c0/c1/... program-id bases.

        The Spyre grid is 1D (``program_id(0)`` = flattened core index).  Each
        work-divided symbol owns one ``range // div`` slice; the flat program id
        is decomposed mixed-radix with the innermost split symbol varying
        fastest, matching the SDSC device-space split.
        """
        body.writeline("# Triton -> Logical layouts")

        # Split symbols (div > 1) in iteration order (outermost first).
        split = [(s, rng, div) for s, (rng, div) in it_space.items() if div > 1]
        total_cores = 1
        for _s, _rng, div in split:
            total_cores *= div

        bases: dict[sympy.Symbol, str] = {}
        inner_cores = 1
        for sym, rng, div in reversed(split):  # innermost first
            extent = max(1, int(self._size_hint(rng)) // div)
            idx = "tl.program_id(0)"
            if inner_cores > 1:
                idx = f"({idx} // {inner_cores})"
            if inner_cores * div != total_cores:
                idx = f"({idx} % {div})"
            bases[sym] = idx if extent == 1 else f"({idx}) * {extent}"
            inner_cores *= div

        for sym in it_space:
            body.writeline(f"{sym} = {bases.get(sym, '0')}")

    def _emit_descriptors(
        self,
        body: IndentedBuffer,
        tensor_args: dict[int, TensorArg],
        param_names: list[str],
        it_space: dict[sympy.Symbol, tuple[sympy.Expr, int]],
    ) -> tuple[dict[int, str], dict[int, list[str]]]:
        """Emit device-layout dim vars + one descriptor per tensor arg.

        Returns ``(desc_of, dims_of)`` keyed by ``arg_index``: the descriptor
        variable name and the list of device-dim offset variable names.
        """
        divisor_of = {sym: int(div) for sym, (_rng, div) in it_space.items()}
        desc_of: dict[int, str] = {}
        dims_of: dict[int, list[str]] = {}
        # Cache device-dim var names by coordinate signature so operands with an
        # identical device layout share dim0/dim1/... (as SDSC/dev-triton do).
        coords_seen: dict[tuple[str, ...], list[str]] = {}

        for arg_index in sorted(tensor_args):
            arg = tensor_args[arg_index]
            coords = list(arg.device_coordinates)
            key = tuple(str(c) for c in coords)
            if key not in coords_seen:
                group = len(coords_seen)
                dim_names = []
                body.writeline("# Logical layouts -> Device layouts")
                for k, coord in enumerate(coords):
                    name = f"dim{k}" if group == 0 else f"dim_{group}_{k}"
                    body.writeline(f"{name} = {_coord_str(coord)}")
                    dim_names.append(name)
                coords_seen[key] = dim_names
            dims_of[arg_index] = coords_seen[key]

        for arg_index in sorted(tensor_args):
            arg = tensor_args[arg_index]
            device_size = [int(s) for s in arg.device_size]
            strides = _row_major_strides(device_size)
            block_shape = self._device_block_shape(arg, divisor_of)
            desc = f"desc_{arg_index}"
            body.writeline(
                f"{desc} = tl.make_tensor_descriptor("
                f"{param_names[arg_index]}, "
                f"shape={device_size}, "
                f"strides={strides}, "
                f"block_shape={block_shape})"
            )
            desc_of[arg_index] = desc

        return desc_of, dims_of

    def _device_block_shape(
        self, arg: TensorArg, divisor_of: dict[sympy.Symbol, int]
    ) -> list[int]:
        """Per-core ``block_shape`` for ``tl.make_tensor_descriptor``.

        Divides each non-stick device dim by the product of core divisors
        (``divisor_of``, this group's iteration-space work divisions) of the
        OpSpec symbols appearing in that dim's coordinate.  The last device dim
        is the inner-stick dim: always the full ``device_size[-1]`` (64 fp16 /
        32 fp32 / 128 int8), never divided across cores.
        """
        device_size = [int(s) for s in arg.device_size]
        coords = arg.device_coordinates
        last = len(device_size) - 1

        block = []
        for k, coord in enumerate(coords):
            if k == last:
                block.append(device_size[k])
                continue
            divisor = 1
            for sym in coord.free_symbols:
                divisor *= divisor_of.get(sym, 1)
            block.append(max(1, device_size[k] // max(1, divisor)))
        return block

    def _emit_ops(
        self,
        body: IndentedBuffer,
        specs: list[OpSpec],
        desc_of: dict[int, str],
        dims_of: dict[int, list[str]],
    ) -> None:
        """Emit load / compute / store for each pointwise op.

        Values produced by an earlier op (its output buffer) are reused by a
        later op that reads the same buffer, so a fused chain threads tiles
        through registers rather than reloading.  A fused-away intermediate
        (``arg_index < 0``) is never materialized: its value stays in a register
        and is not stored to HBM.  Buffers are identified by name so distinct
        intermediates (all sharing the ``-1`` sentinel) do not alias.
        """
        tmp_counter = 0
        produced: dict[object, str] = {}  # buffer id -> tmp var holding its value

        def _load(arg: TensorArg) -> str:
            nonlocal tmp_counter
            key = _buf_id(arg)
            if key in produced:
                return produced[key]
            # A register-threaded intermediate must have been produced earlier in
            # this kernel; only materialized args (arg_index >= 0) load from HBM.
            offsets = ", ".join(dims_of[arg.arg_index])
            var = f"tmp{tmp_counter}"
            tmp_counter += 1
            body.writeline(f"{var} = {desc_of[arg.arg_index]}.load([{offsets}])")
            produced[key] = var
            return var

        for spec in specs:
            inputs = [a for a in spec.args if a.is_input]
            outputs = [a for a in spec.args if not a.is_input]
            assert len(outputs) == 1, "pointwise op must have exactly one output"
            assert len(inputs) == 2, f"binary op '{spec.op}' needs two inputs"

            lhs = _load(inputs[0])
            rhs = _load(inputs[1])
            out_var = f"tmp{tmp_counter}"
            tmp_counter += 1
            body.writeline(f"{out_var} = {lhs} {_BINARY_OPS[spec.op]} {rhs}")

            out = outputs[0]
            # Only write HBM for a materialized output; a fused-away intermediate
            # (arg_index < 0) is consumed from the register in a later op.
            if out.arg_index >= 0:
                offsets = ", ".join(dims_of[out.arg_index])
                body.writeline(
                    f"{desc_of[out.arg_index]}.store([{offsets}], {out_var})"
                )
            produced[_buf_id(out)] = out_var

    # -- preamble -----------------------------------------------------------

    def _emit_header(self, grid: int, used: list[int]) -> str:
        """Imports + ``@triton.jit`` decorator preamble.

        A ``@triton_heuristics.fixed_config`` decorator (with real
        ``triton_meta`` restricted to this group's ``used`` args) is attempted so
        ``output_code.py`` is structurally a Spyre Triton kernel; if any metadata
        helper is unavailable it falls back to a plain ``@triton.jit`` so source
        is always emitted (execution may still fail at ``.run`` — expected for
        this source-only cut).
        """
        buf = IndentedBuffer()
        buf.splice(TritonKernel.gen_common_triton_imports())
        buf.writeline("")
        decorator = self._fixed_config_decorator(grid, used)
        if decorator is not None:
            buf.splice(decorator)
        buf.writeline("@triton.jit")
        return buf.getvalue()

    def _fixed_config_decorator(self, grid: int, used: list[int]):
        """Build a ``@triton_heuristics.fixed_config(...)`` block, or None.

        ``triton_meta`` is built from only this group's ``used`` args (a subset
        of ``python_argdefs``), so the signature matches the emitted kernel.
        """
        try:
            from torch._inductor.codegen.triton import config_of, signature_to_meta
            from torch._inductor.runtime.hints import DeviceProperties

            argdefs, _call, signature, _types = self.args.python_argdefs()
            argdefs = [argdefs[i] for i in used]
            signature = [signature[i] for i in used]
            triton_meta = {
                "signature": signature_to_meta(
                    signature, size_dtype=None, argdefs=argdefs
                ),
                "device": DeviceProperties.create(
                    V.graph.get_current_device_or_throw()
                ),
                "constants": {},
                "configs": [config_of(signature)],
                "native_matmul": False,
                "spyre_grid": [grid],
            }
            inductor_meta = {
                "grid_type": "Grid1D",
                "kernel_name": KERNEL_NAME_PLACEHOLDER,
                "mutated_arg_names": [],
                "no_x_dim": False,
            }
            buf = IndentedBuffer()
            buf.writeline("@triton_heuristics.fixed_config(")
            with buf.indent():
                buf.writeline("config={},")
                buf.writeline("filename=__file__,")
                buf.writeline(f"triton_meta={triton_meta!r},")
                buf.writeline(f"inductor_meta={inductor_meta!r},")
            buf.writeline(")")
            return buf.getvalue()
        except Exception as exc:  # pragma: no cover - fidelity best-effort
            logger.debug("OpSpec->Triton: fixed_config unavailable (%s)", exc)
            return None

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _size_hint(expr) -> int:
        """Concrete size hint for an iteration-space range expression."""
        if isinstance(expr, (int, sympy.Integer)):
            return int(expr)
        return int(V.graph.sizevars.size_hint(expr))

    def _dump_source(self, src: str) -> None:
        """Write the emitted source to ./opspec-triton-dump/ for inspection."""
        try:
            out_dir = os.path.join(os.getcwd(), "opspec-triton-dump")
            os.makedirs(out_dir, exist_ok=True)
            idx = len(os.listdir(out_dir))
            path = os.path.join(out_dir, f"opspec_kernel_{idx}.py")
            with open(path, "w") as fh:
                fh.write(src)
            logger.debug("OpSpec->Triton: dumped kernel to %s", path)
        except OSError as exc:  # pragma: no cover
            logger.debug("OpSpec->Triton: dump failed (%s)", exc)
