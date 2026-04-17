import sympy
from typing import Optional, Any, Sequence
import torch
from torch._inductor.codegen.triton import TritonKernel, FixedTritonConfig
from torch._inductor.virtualized import StoreMode, V
from torch._inductor.codegen.common import CSEVariable
from torch._inductor.utils import IndentedBuffer, sympy_subs
from .errors import Unsupported
from .ir import FixedTiledLayout
from .spyre_kernel import TensorAccess, UnimplementedOp
from .op_spec import OpSpec, TensorArg
from .views import compute_coordinates
from .pass_utils import iteration_space, map_ir_splits_to_scheduler
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


logger = get_inductor_logger("spyre_triton_kernel")


def get_dtype_bytes(dtype: torch.dtype) -> int:
    """Get the size in bytes for a given torch dtype."""
    dtype_sizes = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float64: 8,
        torch.int32: 4,
        torch.int64: 8,
        torch.int16: 2,
        torch.int8: 1,
        torch.uint8: 1,
        torch.bool: 1,
        torch.complex64: 8,
        torch.complex128: 16,
    }
    return dtype_sizes.get(dtype, 4)  # Default to 4 bytes if unknown


def calculate_min_block_size(dtype: torch.dtype, min_bytes: int = 128) -> int:
    """
    Calculate minimum block size based on data type to ensure at least min_bytes
    are accessed per block.

    Args:
        dtype: The torch data type of the tensor
        min_bytes: Minimum number of bytes to access (default: 128)

    Returns:
        Minimum block size (power of 2)
    """
    dtype_bytes = get_dtype_bytes(dtype)
    min_elements = min_bytes // dtype_bytes

    # Round up to next power of 2
    import math

    if min_elements <= 1:
        return 1
    return 2 ** math.ceil(math.log2(min_elements))


def enforce_min_block_sizes(
    config_dict: dict[str, int],
    tensor_dtypes: dict[str, torch.dtype],
    min_bytes: int = 128,
) -> dict[str, int]:
    """
    Enforce minimum block sizes based on tensor data types.

    Args:
        config_dict: Dictionary with block size configuration (e.g., {"XBLOCK": 64, "YBLOCK": 32})
        tensor_dtypes: Dictionary mapping dimension prefixes to tensor dtypes
                      (e.g., {"x": torch.float32, "y": torch.bfloat16, "r0_": torch.float16})
        min_bytes: Minimum bytes per block (default: 128)

    Returns:
        Updated config_dict with enforced minimum block sizes
    """
    updated_config = config_dict.copy()

    # Map block names to dimension prefixes
    block_to_dim = {
        "XBLOCK": "x",
        "YBLOCK": "y",
        "ZBLOCK": "z",
        "R0_BLOCK": "r0_",
        "R1_BLOCK": "r1_",
        "R2_BLOCK": "r2_",
    }

    for block_name, dim_prefix in block_to_dim.items():
        if block_name in updated_config and dim_prefix in tensor_dtypes:
            dtype = tensor_dtypes[dim_prefix]
            min_block = calculate_min_block_size(dtype, min_bytes)

            # Enforce minimum
            if updated_config[block_name] < min_block:
                logger.debug(
                    f"Increasing {block_name} from {updated_config[block_name]} to {min_block} "
                    f"for dtype {dtype} (min_bytes={min_bytes})"
                )
                updated_config[block_name] = min_block

    return updated_config


def patch_triton_config_with_min_blocks(min_bytes: int = 128):
    """
    Monkey-patch PyTorch Inductor's triton config functions to enforce minimum block sizes.

    This should be called during Spyre initialization to override the heuristics.

    Args:
        min_bytes: Minimum bytes per block dimension (default: 128)

    Returns:
        Dictionary of original functions for restoration
    """
    import sys

    # Get the actual module from sys.modules to ensure we patch the right instance
    triton_heuristics_module = sys.modules.get(
        "torch._inductor.runtime.triton_heuristics"
    )
    if triton_heuristics_module is None:
        # Module not loaded yet, import it
        from torch._inductor.runtime import triton_heuristics as th_module

        triton_heuristics_module = th_module

    # Cast to Any to allow dynamic attribute access for monkey patching
    triton_heuristics: Any = triton_heuristics_module

    # Save original functions
    _original_triton_config = triton_heuristics.triton_config
    _original_triton_config_tiled_reduction = (
        triton_heuristics.triton_config_tiled_reduction
    )
    _original_match_target_block_product = triton_heuristics.match_target_block_product

    def patched_triton_config(size_hints, x, y=None, z=None, num_warps=None, **kwargs):
        """Patched version that enforces minimum block sizes."""
        config = _original_triton_config(size_hints, x, y, z, num_warps, **kwargs)

        # Try to infer dtypes from the current kernel context
        tensor_dtypes = {}
        try:
            if hasattr(V, "kernel") and hasattr(V.kernel, "args"):
                # Get dtypes from kernel arguments
                for name in V.kernel.args.input_buffers:
                    buf = V.graph.get_buffer(name)
                    dtype = buf.get_dtype()
                    # Map to dimension - this is a heuristic
                    if "x" not in tensor_dtypes:
                        tensor_dtypes["x"] = dtype
                    if "y" not in tensor_dtypes and y is not None:
                        tensor_dtypes["y"] = dtype
                    if "z" not in tensor_dtypes and z is not None:
                        tensor_dtypes["z"] = dtype
        except Exception as e:
            logger.debug(f"Could not infer dtypes for block size enforcement: {e}")
            # Use a conservative default
            tensor_dtypes = {k: torch.float32 for k in ["x", "y", "z"]}

        config.kwargs = enforce_min_block_sizes(config.kwargs, tensor_dtypes, min_bytes)
        return config

    def patched_triton_config_tiled_reduction(
        size_hints, x, r, num_stages=1, num_warps=None, **kwargs
    ):
        """Patched version for tiled reductions."""
        print("patched_triton_config_tiled_reduction called")
        config = _original_triton_config_tiled_reduction(
            size_hints, x, r, num_stages, num_warps, **kwargs
        )

        # Infer dtypes
        tensor_dtypes = {}
        try:
            if hasattr(V, "kernel") and hasattr(V.kernel, "args"):
                for name in V.kernel.args.input_buffers:
                    buf = V.graph.get_buffer(name)
                    dtype = buf.get_dtype()
                    if "x" not in tensor_dtypes:
                        tensor_dtypes["x"] = dtype
                    if "r0_" not in tensor_dtypes:
                        tensor_dtypes["r0_"] = dtype
        except Exception as e:
            logger.debug(f"Could not infer dtypes for reduction: {e}")
            tensor_dtypes = {"x": torch.float32, "r0_": torch.float32}

        config.kwargs = enforce_min_block_sizes(config.kwargs, tensor_dtypes, min_bytes)
        return config

    def patched_match_target_block_product(
        size_hints, tiling_scores, target_block_product, min_block_size=1
    ):
        """Patched version that respects dtype-based minimums."""
        print("patched_match_target_block_product called")
        # First get the original block sizes
        block_sizes = _original_match_target_block_product(
            size_hints, tiling_scores, target_block_product, min_block_size
        )

        # Infer dtypes and enforce minimums
        tensor_dtypes = {}
        try:
            if hasattr(V, "kernel") and hasattr(V.kernel, "args"):
                for name in V.kernel.args.input_buffers:
                    buf = V.graph.get_buffer(name)
                    dtype = buf.get_dtype()
                    for dim in block_sizes.keys():
                        if dim not in tensor_dtypes:
                            tensor_dtypes[dim] = dtype
        except Exception as e:
            logger.debug(f"Could not infer dtypes for block product matching: {e}")
            tensor_dtypes = {k: torch.float32 for k in block_sizes.keys()}

        # Enforce minimums for each dimension
        for dim, dtype in tensor_dtypes.items():
            if dim in block_sizes:
                min_block = calculate_min_block_size(dtype, min_bytes)
                if block_sizes[dim] < min_block:
                    logger.debug(
                        f"Increasing {dim} block from {block_sizes[dim]} to {min_block} "
                        f"for dtype {dtype}"
                    )
                    block_sizes[dim] = min_block

        return block_sizes

    def patched_make_matmul_triton_config(sizes, num_warps, num_stages):
        """Patched version for matmul configs."""
        # Call original
        config = _original_make_matmul_triton_config(sizes, num_warps, num_stages)

        # For matmul, assume float16 as default (most common for Spyre)
        # Map XBLOCK/YBLOCK to x/y dimensions
        tensor_dtypes = {"x": torch.float16, "y": torch.float16, "r0_": torch.float16}

        # Enforce minimums
        original_kwargs = dict(config.kwargs)
        config.kwargs = enforce_min_block_sizes(config.kwargs, tensor_dtypes, min_bytes)

        # Log if any changes were made
        if original_kwargs != config.kwargs:
            logger.debug(
                f"Enforced min block sizes for matmul: {original_kwargs} -> {config.kwargs}"
            )

        return config

    # Save original make_matmul_triton_config
    _original_make_matmul_triton_config = triton_heuristics.make_matmul_triton_config

    # Apply patches
    triton_heuristics.triton_config = patched_triton_config
    triton_heuristics.triton_config_tiled_reduction = (
        patched_triton_config_tiled_reduction
    )
    triton_heuristics.match_target_block_product = patched_match_target_block_product
    triton_heuristics.make_matmul_triton_config = patched_make_matmul_triton_config

    print(
        f"[SPYRE] Patched Triton heuristics to enforce min_bytes={min_bytes} per block"
    )
    logger.info(f"Patched Triton heuristics to enforce min_bytes={min_bytes} per block")

    # Return original functions for restoration
    return {
        "triton_config": _original_triton_config,
        "triton_config_tiled_reduction": _original_triton_config_tiled_reduction,
        "match_target_block_product": _original_match_target_block_product,
        "make_matmul_triton_config": _original_make_matmul_triton_config,
    }


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

    def codegen_kernel(self, name=None) -> str:
        original_code = super().codegen_kernel(name)
        code = IndentedBuffer()
        code.splice("from torch_spyre._inductor.op_spec import TensorArg, OpSpec")
        code.splice("from torch_spyre._inductor.spyre_kernel import UnimplementedOp")
        code.splice("import torch")
        code.splice("from torch_spyre._C import DataFormats, SpyreTensorLayout")
        code.splice("from sympy import sympify")
        return code.getvalue() + original_code

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

            self.triton_meta["spyre_options"] = {"op_specs": serializable_specs}
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
            core_division = map_ir_splits_to_scheduler(
                ir_node.op_it_space_sizes,  # type: ignore[attr-defined]
                ir_node.op_it_space_splits,  # type: ignore[attr-defined]
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

    def _create_index_from_iteration_space(self) -> sympy.Expr:
        """
        Create an index expression directly from iteration space variables.

        This creates a linearized index from the iteration space (c0, c1, ...)
        that can be used with compute_coordinates to get proper device coordinates.
        """
        if self.current_node is None:
            raise RuntimeError("current_node is None")

        # Get the iteration space which has the c0, c1, etc. variables
        it_space = iteration_space(self.current_node)

        # Create a linearized index from iteration space variables
        # For a 2D iteration space {c0: 64, c1: 128}, create: c0 * 128 + c1
        # This matches the row-major layout
        iter_vars = sorted(it_space.keys(), key=lambda x: str(x))

        if len(iter_vars) == 0:
            return sympy.Integer(0)
        elif len(iter_vars) == 1:
            return iter_vars[0]
        else:
            # Build linearized index: c0 * size1 * size2 * ... + c1 * size2 * ... + c2 * ... + cn
            index = sympy.Integer(0)
            for i, var in enumerate(iter_vars):
                stride = sympy.Integer(1)
                # Calculate stride as product of all subsequent dimension sizes
                for j in range(i + 1, len(iter_vars)):
                    stride *= it_space[iter_vars[j]]
                index += var * stride
            return index

    def load(self, name: str, index: sympy.Expr):
        """Codegen a load from an InputBuffer and track the TensorAccess"""
        buf = V.graph.get_buffer(name)
        layout = buf.get_layout()
        if not isinstance(layout, FixedTiledLayout):
            raise Unsupported(f"{name} does not have FixedTiledLayout")
        index = sympy_subs(index, V.graph.sizevars.precomputed_replacements)
        if not layout.allocation:
            _ = self.args.input(name)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"kernel_load: {name}, shape={[int(s) for s in layout.size]}, "
                f"device_size={list(layout.device_layout.device_size)}"
            )

        # Create TensorArg for this load and store it
        assert self.current_node is not None
        iter_index = self._create_index_from_iteration_space()
        tensor_access = TensorAccess(name, iter_index, layout)
        tensor_arg = self.create_tensor_arg(True, name, tensor_access)
        self.loaded_tensor_args[name] = tensor_arg

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"load: name={name} triton_index={index} iter_index={iter_index}"
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
        iter_index = self._create_index_from_iteration_space()

        dst = TensorAccess(name, iter_index, layout)
        real_dst_name = V.graph.scheduler.mutation_real_name.get(name, name)
        if real_dst_name != name:
            # Skip allocating an output buffer; this name is an alias to another buffer
            V.graph.removed_buffers.add(name)

        op_info: dict[str, Any] = {}
        if hasattr(self.current_node, "op_dim_splits"):
            op_info["op_dim_splits"] = self.current_node.op_dim_splits  # type: ignore[union-attr]
        if hasattr(self.current_node, "n_cores_used"):
            op_info["n_cores_used"] = self.current_node.n_cores_used  # type: ignore[union-attr]

        if logger.isEnabledFor(logging.DEBUG):
            value_type = type(value).__name__
            logger.debug(
                f"kernel_store: {name} (type: {value_type}), shape={[int(s) for s in layout.size]}, "
                f"device_size={list(layout.device_layout.device_size)}, op_info={op_info}"
            )

        # Create output TensorArg
        output_tensor_arg = self.create_tensor_arg(False, real_dst_name, dst)

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

        # Create and store the OpSpec
        # For Triton kernels, we use a generic operation name
        op_spec = self.create_op_spec("add", False, args, op_info)
        self.op_specs.append(op_spec)

        return super().store(name, index, value, mode)
