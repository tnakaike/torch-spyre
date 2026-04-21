import sympy
from typing import Optional, Any, Sequence
from torch._inductor.codegen.triton import TritonKernel, FixedTritonConfig
from torch._inductor.virtualized import StoreMode, V
from torch._inductor.codegen.common import CSEVariable
from torch._inductor.utils import IndentedBuffer, sympy_subs
from .errors import Unsupported
from .ir import FixedTiledLayout
from .spyre_kernel import TensorAccess, UnimplementedOp
from .op_spec import OpSpec, TensorArg
from .views import compute_coordinates
from .pass_utils import iteration_space, apply_splits_from_index_coeff
from .constants import SPYRE_FP32_OPS
from .logging_utils import get_inductor_logger
from torch_spyre._C import DataFormats
import logging


class SympyExpr:
    """Wrapper for sympy expressions that serializes to sympify() calls"""

    def __init__(self, expr: sympy.Expr):
        self.expr = str(expr)

    def __repr__(self):
        return f"sympify('{self.expr}')"


class IterationSpaceDict:
    """Wrapper for iteration_space dict that serializes properly"""

    def __init__(self, it_space: dict[sympy.Symbol, tuple[sympy.Expr, int]]):
        self.items = [
            (SympyExpr(k), (SympyExpr(v[0]), v[1])) for k, v in it_space.items()
        ]

    def __repr__(self):
        items_str = ", ".join(f"{k!r}: ({v[0]!r}, {v[1]})" for k, v in self.items)
        return f"{{{items_str}}}"


class TensorArgDict:
    """Wrapper for TensorArg that serializes properly"""

    def __init__(self, arg: TensorArg):
        self.is_input = arg.is_input
        self.arg_index = arg.arg_index
        self.device_dtype = arg.device_dtype
        self.device_size = arg.device_size
        self.device_coordinates = [SympyExpr(e) for e in arg.device_coordinates]
        self.allocation = arg.allocation

    def __repr__(self):
        coords_str = ", ".join(repr(c) for c in self.device_coordinates)
        # Format device_dtype as DataFormats.ENUM_NAME instead of <DataFormats.ENUM_NAME: value>
        dtype_str = f"DataFormats.{self.device_dtype.name}"
        return (
            f"TensorArg("
            f"is_input={self.is_input}, "
            f"arg_index={self.arg_index}, "
            f"device_dtype={dtype_str}, "
            f"device_size={self.device_size!r}, "
            f"device_coordinates=[{coords_str}], "
            f"allocation={self.allocation!r})"
        )


class OpSpecDict:
    """Wrapper for OpSpec that serializes properly"""

    def __init__(self, op_spec: OpSpec):
        self.op = op_spec.op
        self.is_reduction = op_spec.is_reduction
        self.iteration_space = IterationSpaceDict(op_spec.iteration_space)
        self.args = [TensorArgDict(arg) for arg in op_spec.args]
        self.op_info = op_spec.op_info

    def __repr__(self):
        args_str = ", ".join(repr(arg) for arg in self.args)
        return (
            f"OpSpec("
            f"op={self.op!r}, "
            f"is_reduction={self.is_reduction}, "
            f"iteration_space={self.iteration_space!r}, "
            f"args=[{args_str}], "
            f"op_info={self.op_info!r})"
        )


class UnimplementedOpDict:
    """Wrapper for UnimplementedOp that serializes properly"""

    def __init__(self, op: UnimplementedOp):
        self.op = op.op

    def __repr__(self):
        return f"UnimplementedOp(op={self.op!r})"


class TritonOpSpecMapDict:
    """Wrapper for triton_opspec_map that serializes properly"""

    def __init__(self, mapping: dict[str, list[sympy.Symbol]]):
        self.items = [
            (prefix, [SympyExpr(sym) for sym in symbols])
            for prefix, symbols in mapping.items()
        ]

    def __repr__(self):
        items_str = ", ".join(
            f"{prefix!r}: [{', '.join(repr(sym) for sym in symbols)}]"
            for prefix, symbols in self.items
        )
        return f"{{{items_str}}}"


logger = get_inductor_logger("spyre_triton_kernel")


class SpyreTritonKernel(TritonKernel):
    def __init__(
        self,
        tiling: dict[str, sympy.Expr],
        min_elem_per_thread=0,
        optimize_mask=True,
        fixed_config: Optional[FixedTritonConfig] = None,
        hint_override: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            tiling,
            min_elem_per_thread,
            optimize_mask,
            fixed_config,
            hint_override,
            **kwargs,
        )
        self.op_specs: list[OpSpec | UnimplementedOp] = []
        self.spyre_kernel_args: list[tuple[str, TensorArg]] = []
        # Track loaded tensor args to use in store
        self.loaded_tensor_args: dict[str, TensorArg] = {}
        # Mapping from Triton prefixes (x, y, z, r0_, ...) to OpSpec iteration space symbols (c0, c1, ...)
        # Shows which OpSpec dimensions are flattened into each Triton dimension
        self.triton_opspec_map: dict[str, list[sympy.Symbol]] = {}

    def codegen_kernel(self, name=None) -> str:
        original_code = super().codegen_kernel(name)

        code = IndentedBuffer()
        code.splice("from torch_spyre._inductor.op_spec import TensorArg, OpSpec")
        code.splice("from torch_spyre._inductor.spyre_kernel import UnimplementedOp")
        code.splice("import torch")
        code.splice("from torch_spyre._C import DataFormats, SpyreTensorLayout")
        code.splice("from sympy import sympify")
        return code.getvalue() + original_code

    def get_triton_iteration_space(self, index: sympy.Expr) -> dict[str, int]:
        """
        Extract the Triton iteration space from an index expression.

        Args:
            index: Triton index expression containing symbols like x0, x1, r0_0, etc.

        Returns:
            Dictionary mapping symbol names to their ranges, ordered from outermost to innermost,
            e.g., {'x3': 16, 'x2': 32, 'x1': 64, 'x0': 128} where x3 is the outermost dimension
        """
        triton_symbols = index.free_symbols

        # Collect symbols with their coefficients in the index expression
        symbol_coeffs = []
        for sym in triton_symbols:
            if isinstance(sym, sympy.Symbol) and sym in self.range_tree_nodes:
                # Get coefficient of this symbol in the index expression
                coeff = index.coeff(sym)
                if coeff is not None:
                    symbol_coeffs.append((sym, coeff))

        # Sort by coefficient (descending) to get outermost to innermost order
        # Outermost dimension has largest coefficient (stride)
        symbol_coeffs.sort(key=lambda x: V.graph.sizevars.size_hint(x[1]), reverse=True)

        # Build ordered dictionary
        triton_is = {}
        for sym, _ in symbol_coeffs:
            sym_name = str(sym)
            range_entry = self.range_tree_nodes[sym]
            triton_is[sym_name] = V.graph.sizevars.size_hint(range_entry.length)

        return triton_is

    def get_triton_block_size(self) -> dict[str, int]:
        """
        Extract core division information from OpSpec iteration space via triton_opspec_map.

        Returns:
            Dictionary mapping Triton dimension prefixes to block size per core (elements per core),
            e.g., {'x': 64, 'r0_': 128} means XBLOCK should be 64 elements per core
        """
        if not self.triton_opspec_map:
            raise RuntimeError(
                "triton_opspec_map is not available - cannot compute block sizes"
            )

        if not hasattr(self, "current_node") or self.current_node is None:
            raise RuntimeError(
                "current_node is not available - cannot compute block sizes"
            )

        # Get OpSpec iteration space with core divisions
        it_space = iteration_space(self.current_node)
        ir_node = self.current_node.node

        # Get core division from IR node
        core_division: dict[sympy.Symbol, int] = {}
        if hasattr(ir_node, "op_it_space_splits"):
            write_index = next(iter(self.current_node.read_writes.writes)).index
            read_index = next(iter(self.current_node.read_writes.reads)).index
            core_division = apply_splits_from_index_coeff(
                ir_node.op_it_space_splits,  # type: ignore[attr-defined]
                write_index,
                read_index,
                it_space,
            )
            logger.debug(f"Core division for {ir_node}: {core_division}")
        else:
            raise RuntimeError(f"ir_node {ir_node} does not have op_it_space_splits")

        # Map Triton dimensions to block sizes per core based on OpSpec mapping
        spyre_triton_block_size: dict[str, int] = {}

        for triton_prefix, opspec_symbols in self.triton_opspec_map.items():
            if not opspec_symbols:
                # No OpSpec dimensions mapped to this Triton dimension - no splitting needed
                continue

            # Calculate total number of cores for this Triton dimension
            # by multiplying the core divisions of all mapped OpSpec dimensions
            total_cores = 1
            total_size = 1
            for sym in opspec_symbols:
                cores = core_division.get(sym, 1)
                total_cores *= cores
                # Get the size of this dimension from iteration space
                if sym in it_space:
                    size_expr = it_space[sym]
                    # Evaluate the size if it's a sympy expression
                    size_val = V.graph.sizevars.size_hint(size_expr)
                    total_size *= size_val

            # Calculate block size per core: total_size / n_cores
            # Include all dimensions, even when total_cores == 1 (no splitting)
            block_per_core = total_size // total_cores
            spyre_triton_block_size[triton_prefix] = max(1, block_per_core)

        logger.debug(f"spyre_triton_block_size: {spyre_triton_block_size}")
        return spyre_triton_block_size

    def _create_triton_opspec_map(self) -> dict[str, list[sympy.Symbol]]:
        """
        Create a mapping from Triton dimension prefixes (x, y, z, r0_, r1_, ...) to
        OpSpec iteration space symbols (c0, c1, ...).

        Algorithm:
        1. For matmul/dot operations: Use direct 1:1 mapping based on size matching
        2. For other operations: Use the flattening algorithm

        For matmul, Triton creates separate dimensions for each axis (y, x, r0_)
        which map directly to OpSpec dimensions (c0, c1, c2).
        """
        if self.current_node is None:
            raise RuntimeError(
                "Cannot map Triton to OpSpec dimensions: current_node is None"
            )

        # Get the OpSpec iteration space (c0, c1, c2, ...)
        it_space = iteration_space(self.current_node)
        # Sort by symbol name to get consistent ordering (c0, c1, c2, ...)
        opspec_symbols = sorted(it_space.keys(), key=lambda x: str(x))

        # Get the Triton dimension prefixes, sorted by type and name
        triton_prefixes = sorted(
            self.numels.keys(), key=lambda p: (1 if p.startswith("r") else 0, p)
        )

        # Initialize the mapping with empty lists
        mapping: dict[str, list[sympy.Symbol]] = {
            prefix: [] for prefix in triton_prefixes
        }

        # Get size hints for all dimensions
        opspec_size_hints = {
            sym: V.graph.sizevars.size_hint(it_space[sym]) for sym in opspec_symbols
        }

        triton_size_hints = {
            prefix: V.graph.sizevars.size_hint(self.numels[prefix])
            for prefix in triton_prefixes
        }

        # Check if this is a matmul/dot operation by looking at the reduction type
        is_matmul = False
        if (
            hasattr(self.current_node, "node")
            and self.current_node.node is not None
            and hasattr(self.current_node.node, "data")
        ):
            data = self.current_node.node.data
            if hasattr(data, "reduction_type"):
                is_matmul = data.reduction_type in ["dot", "matmul"]

        # For matmul/dot, use direct 1:1 mapping
        if is_matmul and len(triton_prefixes) == len(opspec_symbols):
            logger.debug("Using direct 1:1 mapping for matmul/dot operation")
            # Create a direct mapping: each Triton dimension maps to one OpSpec dimension
            # Sort both by size to match them correctly
            triton_by_size = sorted(triton_prefixes, key=lambda p: triton_size_hints[p])
            opspec_by_size = sorted(opspec_symbols, key=lambda s: opspec_size_hints[s])

            for triton_prefix, opspec_sym in zip(triton_by_size, opspec_by_size):
                mapping[triton_prefix] = [opspec_sym]
                logger.debug(
                    f"  {triton_prefix} ({triton_size_hints[triton_prefix]}) -> {opspec_sym} ({opspec_size_hints[opspec_sym]})"
                )
        else:
            # Use the original flattening algorithm for non-matmul operations
            # Reverse to traverse from innermost (last) dimension
            opspec_symbols_reversed = list(reversed(opspec_symbols))
            triton_prefixes_reversed = list(reversed(triton_prefixes))

            # Track which OpSpec dimensions have been matched
            opspec_idx = 0

            # Traverse Triton dimensions from innermost
            for triton_prefix in triton_prefixes_reversed:
                tnumel = triton_size_hints[triton_prefix]

                # Skip dummy dimensions (size 1)
                if tnumel == 1:
                    continue

                # Try to match with product of consecutive OpSpec dimensions
                matched_symbols = []
                product = 1

                # Accumulate OpSpec dimensions until product matches tnumel
                temp_idx = opspec_idx
                while temp_idx < len(opspec_symbols_reversed):
                    sym = opspec_symbols_reversed[temp_idx]
                    product *= opspec_size_hints[sym]
                    matched_symbols.append(sym)

                    if product == tnumel:
                        # Exact match found
                        # Store in original order (not reversed)
                        mapping[triton_prefix] = list(reversed(matched_symbols))
                        opspec_idx = temp_idx + 1
                        break
                    elif product > tnumel:
                        # Product exceeded without exact match
                        raise RuntimeError(
                            f"Cannot map Triton dimension '{triton_prefix}' (numel={tnumel}) "
                            f"to OpSpec dimensions. Product {product} exceeds target. "
                            f"Attempted symbols: {matched_symbols}"
                        )

                    temp_idx += 1
                else:
                    # Ran out of OpSpec dimensions without matching
                    raise RuntimeError(
                        f"Cannot map Triton dimension '{triton_prefix}' (numel={tnumel}) "
                        f"to OpSpec dimensions. Accumulated product: {product}, "
                        f"symbols: {matched_symbols}"
                    )

        logger.debug(f"Triton to OpSpec mapping: {mapping}")
        return mapping

    def codegen_body(self):
        if self.triton_meta is not None:
            # Convert op_specs to serializable format using wrapper classes
            # These classes have __repr__ methods that generate proper sympify() calls
            serializable_specs = []
            for op_spec in self.op_specs:
                if isinstance(op_spec, UnimplementedOp):
                    serializable_specs.append(UnimplementedOpDict(op_spec))
                else:
                    serializable_specs.append(OpSpecDict(op_spec))

            # Convert triton_opspec_map to serializable format
            serializable_mapping = TritonOpSpecMapDict(self.triton_opspec_map)

            self.triton_meta["spyre_options"] = {
                "op_specs": serializable_specs,
                "triton_opspec_map": serializable_mapping,
            }
        return super().codegen_body()

    def create_tensor_arg(
        self, is_input: bool, name: str, tensor: TensorAccess
    ) -> TensorArg:
        """Create a TensorArg following the same pattern as SpyreKernel"""
        if self.current_node is None:
            raise RuntimeError("current_node is None")

        device_coords = compute_coordinates(
            tensor.layout.device_layout.device_size,  # type: ignore[arg-type]
            tensor.layout.device_layout.stride_map,  # type: ignore[arg-type]
            var_ranges=iteration_space(self.current_node),
            index=tensor.index,
        )
        tensor_arg = TensorArg(
            is_input,
            -1,
            tensor.layout.device_layout.device_dtype,
            tensor.layout.device_layout.device_size,
            device_coords,
            tensor.layout.allocation,
        )
        if not tensor.layout.allocation:
            self.spyre_kernel_args.append((name, tensor_arg))
        return tensor_arg

    def create_op_spec(
        self,
        op: str,
        is_reduction: bool,
        args: Sequence[TensorArg],
        op_info: dict[str, Any],
    ) -> OpSpec:
        """Create an OpSpec following the same pattern as SpyreKernel"""
        for arg in args:
            if arg.device_dtype == DataFormats.IEEE_FP32 and op not in SPYRE_FP32_OPS:
                raise Unsupported(f"{op} on {arg.device_dtype}")
            elif arg.device_dtype not in [
                DataFormats.IEEE_FP32,
                DataFormats.SEN169_FP16,
            ]:
                raise Unsupported(f"operation on {arg.device_dtype}")

        if self.current_node is None:
            raise RuntimeError("current_node is None")

        it_space = iteration_space(self.current_node)

        ir_node = self.current_node.node  # ComputedBuffer
        core_division: dict[sympy.Symbol, int] = {}
        if hasattr(ir_node, "op_it_space_splits"):
            write_index = next(iter(self.current_node.read_writes.writes)).index
            read_index = next(iter(self.current_node.read_writes.reads)).index
            core_division = apply_splits_from_index_coeff(
                ir_node.op_it_space_splits,  # type: ignore[attr-defined]
                write_index,
                read_index,
                it_space,
            )

        it_space_extended = {
            k: (v, core_division.get(k, 1)) for k, v in it_space.items()
        }

        return OpSpec(
            op,
            is_reduction,
            it_space_extended,
            args,
            op_info,
        )

    def _create_opspec_index(self, name: str, is_load: bool) -> sympy.Expr:
        """
        Create an index expression from the memory dependency for a specific tensor.

        This retrieves the index from MemoryDep which already contains the proper
        linearized index in terms of iteration space variables (c0, c1, ...).

        MemoryDep.index already has the correct variable ordering, stride information,
        and properly handles both reduction and non-reduction dimensions.

        Args:
            name: The tensor name to find the MemoryDep for
            is_load: True if this is a load operation (read), False if store (write)

        Returns:
            The index expression from the matching MemoryDep
        """
        if self.current_node is None:
            raise RuntimeError("current_node is None")

        # Find the MemoryDep that matches the tensor name and access type
        deps = (
            self.current_node.read_writes.reads
            if is_load
            else self.current_node.read_writes.writes
        )

        for dep in deps:
            if dep.name == name:
                return dep.index

        raise RuntimeError(
            f"Could not find MemoryDep for {'load' if is_load else 'store'} of {name}"
        )

    def load(self, name: str, index: sympy.Expr):
        """Codegen a load from an InputBuffer and track the TensorAccess"""
        buf = V.graph.get_buffer(name)
        layout = buf.get_layout()
        if not isinstance(layout, FixedTiledLayout):
            raise Unsupported(f"{name} does not have FixedTiledLayout")
        index = sympy_subs(index, V.graph.sizevars.precomputed_replacements)
        if not layout.allocation:
            _ = self.args.input(name)

        # Create TensorArg for this load and store it
        assert self.current_node is not None
        opspec_index = self._create_opspec_index(name, is_load=True)
        tensor_access = TensorAccess(name, opspec_index, layout)
        tensor_arg = self.create_tensor_arg(True, name, tensor_access)
        self.loaded_tensor_args[name] = tensor_arg

        if logger.isEnabledFor(logging.DEBUG):
            # Get iteration spaces for debugging
            triton_is = self.get_triton_iteration_space(index)
            opspec_is = iteration_space(self.current_node) if self.current_node else {}
            logger.debug(
                f"load: name={name} triton_is={triton_is} triton_index={index} "
                f"opspec_is={opspec_is} opspec_index={opspec_index}"
            )

        return super().load(name, index)

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> None:
        """Store and create OpSpec following SpyreKernel pattern"""
        _ = self.args.output(name)
        buf = V.graph.get_buffer(name)
        layout = buf.get_layout()
        if not isinstance(layout, FixedTiledLayout):
            raise Unsupported(f"{name} does not have FixedTiledLayout")
        index = sympy_subs(index, V.graph.sizevars.precomputed_replacements)

        # Create index from iteration space variables
        # This is needed because compute_coordinates expects iteration space variables
        opspec_index = self._create_opspec_index(name, is_load=False)

        dst = TensorAccess(name, opspec_index, layout)
        real_dst_name = V.graph.scheduler.mutation_real_name.get(name, name)
        if real_dst_name != name:
            # Skip allocating an output buffer; this name is an alias to another buffer
            V.graph.removed_buffers.add(name)

        op_info: dict[str, Any] = {}
        if hasattr(self.current_node, "op_dim_splits"):
            op_info["op_dim_splits"] = self.current_node.op_dim_splits  # type: ignore[union-attr]
        if hasattr(self.current_node, "n_cores_used"):
            op_info["n_cores_used"] = self.current_node.n_cores_used  # type: ignore[union-attr]

        # Create output TensorArg
        output_tensor_arg = self.create_tensor_arg(False, real_dst_name, dst)

        if logger.isEnabledFor(logging.DEBUG):
            # Get iteration spaces for debugging
            triton_is = self.get_triton_iteration_space(index)
            opspec_is = iteration_space(self.current_node) if self.current_node else {}
            logger.debug(
                f"store: name={name} triton_is={triton_is} triton_index={index} "
                f"opspec_is={opspec_is} opspec_index={opspec_index}"
            )

        # Collect all tensor args in the order they appear in kernel arguments
        # Get the actual argument list from the kernel
        actuals = self.args.python_argdefs()[1]

        # Build args list in the order of kernel arguments
        args: list[TensorArg] = []
        for arg_name in actuals:
            if arg_name in self.loaded_tensor_args:
                # Input argument - use the TensorArg created during load
                tensor_arg = self.loaded_tensor_args[arg_name]
                tensor_arg.arg_index = len(args)
                args.append(tensor_arg)
            elif arg_name == real_dst_name:
                # Output argument
                output_tensor_arg.arg_index = len(args)
                args.append(output_tensor_arg)

        # Create the mapping from Triton dimensions to OpSpec iteration space
        # This should be done once per kernel, so check if it's empty
        if not self.triton_opspec_map:
            self.triton_opspec_map = self._create_triton_opspec_map()
            # Print the mapping for debugging
            logger.debug(f"Triton to OpSpec mapping: {self.triton_opspec_map}")
            logger.debug(f"Triton numels: {self.numels}")

            # current_node is always available here (already used earlier in this function)
            assert self.current_node is not None, (
                "current_node should be set in store()"
            )
            it_space = iteration_space(self.current_node)
            logger.debug(f"OpSpec iteration space: {it_space}")

            # Compute and store ALL metadata NOW while we have current_node
            # Store in V.graph so it's available during heuristics
            spyre_triton_block_size = self.get_triton_block_size()
            if spyre_triton_block_size:
                # Store spyre_triton_block_size directly in V.graph
                setattr(V.graph, "_spyre_triton_block_size", spyre_triton_block_size)
                logger.debug(
                    f"Stored spyre_triton_block_size in V.graph: {spyre_triton_block_size}"
                )

        # Create and store the OpSpec
        # For Triton kernels, we use a generic operation name
        op_spec = self.create_op_spec("add", False, args, op_info)
        self.op_specs.append(op_spec)

    def store_reduction(self, name: str, index: sympy.Expr, value: CSEVariable) -> None:
        """Store reduction result and create OpSpec following SpyreKernel pattern"""
        _ = self.args.output(name)
        buf = V.graph.get_buffer(name)
        layout = buf.get_layout()
        if not isinstance(layout, FixedTiledLayout):
            raise Unsupported(f"{name} does not have FixedTiledLayout")
        index = sympy_subs(index, V.graph.sizevars.precomputed_replacements)

        # Create index from iteration space variables
        opspec_index = self._create_opspec_index(name, is_load=False)

        dst = TensorAccess(name, opspec_index, layout)
        real_dst_name = V.graph.scheduler.mutation_real_name.get(name, name)
        if real_dst_name != name:
            # Skip allocating an output buffer; this name is an alias to another buffer
            V.graph.removed_buffers.add(name)

        op_info = {}
        if hasattr(self.current_node.node.data, "op_info"):  # type: ignore[union-attr]
            op_info.update(self.current_node.node.data.op_info)  # type: ignore[union-attr]

        # Determine the reduction operation type from the IR node
        data = self.current_node.node.data  # type: ignore[union-attr]
        if not hasattr(data, "reduction_type"):
            raise RuntimeError(
                f"Reduction node missing reduction_type attribute: {data}"
            )

        # Map Triton reduction types to SDSC operation names
        reduction_type = data.reduction_type
        if reduction_type in ["dot", "matmul"]:
            reduction_op = "matmul"
        elif reduction_type == "sum":
            reduction_op = "sum"
        else:
            raise Unsupported(f"Unsupported reduction type: {reduction_type}")

        if logger.isEnabledFor(logging.DEBUG):
            triton_is = self.get_triton_iteration_space(index)
            opspec_is = iteration_space(self.current_node) if self.current_node else {}
            logger.debug(
                f"store_reduction: name={name} triton_is={triton_is} triton_index={index} "
                f"opspec_is={opspec_is} opspec_index={opspec_index}"
            )

        # Collect all tensor args in the order they appear in kernel arguments
        actuals = self.args.python_argdefs()[1]

        # Build args list: inputs first, then output
        args: list[TensorArg] = []
        for arg_name in actuals:
            if arg_name in self.loaded_tensor_args:
                # Input argument
                tensor_arg = self.loaded_tensor_args[arg_name]
                tensor_arg.arg_index = len(args)
                args.append(tensor_arg)
            elif arg_name == real_dst_name:
                # Output argument
                output_tensor_arg = self.create_tensor_arg(False, real_dst_name, dst)
                output_tensor_arg.arg_index = len(args)
                args.append(output_tensor_arg)

        # Create the mapping from Triton dimensions to OpSpec iteration space
        if not self.triton_opspec_map:
            self.triton_opspec_map = self._create_triton_opspec_map()
            logger.debug(f"Triton to OpSpec mapping: {self.triton_opspec_map}")
            logger.debug(f"Triton numels: {self.numels}")

            assert self.current_node is not None, (
                "current_node should be set in store_reduction()"
            )
            it_space = iteration_space(self.current_node)
            logger.debug(f"OpSpec iteration space: {it_space}")

            # Compute and store block size metadata
            spyre_triton_block_size = self.get_triton_block_size()
            if spyre_triton_block_size:
                setattr(V.graph, "_spyre_triton_block_size", spyre_triton_block_size)
                logger.debug(
                    f"Stored spyre_triton_block_size in V.graph: {spyre_triton_block_size}"
                )

        # Create and store the OpSpec with is_reduction=True
        op_spec = self.create_op_spec(reduction_op, True, args, op_info)
        self.op_specs.append(op_spec)

        return super().store_reduction(name, index, value)
