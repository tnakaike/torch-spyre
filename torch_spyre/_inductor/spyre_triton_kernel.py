import sympy
from typing import Optional, Any
import torch
from torch._inductor.codegen.triton import TritonKernel, FixedTritonConfig
from torch._inductor.virtualized import StoreMode, V
from torch._inductor.codegen.common import CSEVariable
from torch._inductor.utils import IndentedBuffer, sympy_subs
from .errors import Unsupported
from .ir import FixedTiledLayout
from .spyre_kernel import (
    UnimplementedOp,
    DimensionInfo,
    TensorAccess,
    analyze_tensor_access,
    create_op_spec,
)
from .pass_utils import (
    map_dims_to_vars,
    wildcard_symbol,
)
from .op_spec import OpSpec, TensorArg
from .logging_utils import get_inductor_logger
import logging

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
        self.di: list[DimensionInfo] = []
        self.tensor_args: dict[str, TensorArg] = {}

    def codegen_kernel(self, name=None) -> str:
        original_code = super().codegen_kernel(name)
        code = IndentedBuffer()
        code.splice("from torch_spyre._inductor.op_spec import TensorArg, OpSpec")
        code.splice("import torch")
        code.splice("from torch_spyre._C import DataFormats, SpyreTensorLayout")
        return code.getvalue() + original_code

    def codegen_body(self):
        self.triton_meta["spyre_options"] = {"op_specs": self.op_specs}
        return super().codegen_body()

    def derive_dim_info(self, access: TensorAccess) -> list[DimensionInfo]:
        """
        Return the iteration space implied by the tensor access
        """
        var_ranges = self.var_ranges()
        if var_ranges:
            dim_map = map_dims_to_vars(access.layout, access.index)
            return [
                DimensionInfo(dim_map[v], int(var_ranges.get(dim_map[v], 1)))
                for v in sorted(dim_map)
            ]
        else:
            return [DimensionInfo(wildcard_symbol(0), 1)]

    def create_tensor_arg(
        self, is_input: bool, name: str, tensor: TensorAccess, di: list[DimensionInfo]
    ) -> TensorArg:
        scales = analyze_tensor_access(di, tensor)
        tensor_arg = TensorArg(
            is_input,
            -1,
            tensor.layout.dtype,
            scales,
            tensor.layout.allocation,
            tensor.layout.device_layout,
        )
        self.tensor_args[name] = tensor_arg
        return tensor_arg

    def load(self, name: str, index: sympy.Expr):
        """Codegen a load from an InputBuffer"""
        var = self.args.input(name)
        buf = V.graph.get_buffer(name)
        layout = buf.get_layout()
        if not isinstance(layout, FixedTiledLayout):
            raise Unsupported(f"{name} does not have FixedTiledLayout")
        index = sympy_subs(index, V.graph.sizevars.precomputed_replacements)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"kernel_load: {name}, shape={[int(s) for s in layout.size]}, "
                f"device_size={list(layout.device_layout.device_size)}"
            )

        input = TensorAccess(name, index, layout).unsqueeze_if_sparse()
        _ = self.create_tensor_arg(True, var, input, self.derive_dim_info(input))

        return super().load(name, index)

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> None:
        var = self.args.output(name)
        buf = V.graph.get_buffer(name)
        layout = buf.get_layout()
        if not isinstance(layout, FixedTiledLayout):
            raise Unsupported(f"{name} does not have FixedTiledLayout")
        index = sympy_subs(index, V.graph.sizevars.precomputed_replacements)
        dst = TensorAccess(name, index, layout).unsqueeze_if_sparse()
        op_info = {}
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

        _ = self.create_tensor_arg(False, var, dst, di=self.derive_dim_info(dst))
        self.op_specs.append(
            create_op_spec(
                "add",
                False,
                dims=self.get_dimension_info(),
                args=self.create_args(),
                op_info=op_info,
            )
        )

        return super().store(name, index, value, mode)

    def get_dimension_info(self) -> list[DimensionInfo]:
        di: list[DimensionInfo] = []
        if len(self.di) == 0:
            var_ranges = self.var_ranges()
            symbols = reversed(sorted(var_ranges.keys(), key=lambda x: str(x)))
            for s in symbols:
                di.append(DimensionInfo(s, int(var_ranges[s])))
        return di

    def create_args(self) -> list[TensorArg]:
        args: list[TensorArg] = []
        actuals = self.args.python_argdefs()[1]
        print(f"create_args actuals={actuals} args={self.args}")
        for index, name in enumerate(actuals):
            if name.startswith("buf"):
                var = self.args.output(name)
            else:
                var = self.args.input(name)
            arg = self.tensor_args[var]
            arg.arg_index = index
            args.append(arg)
        return args

        # def reduction(
        #     self,
        #     dtype: torch.dtype,
        #     src_dtype: torch.dtype,
        #     reduction_type: str,
        #     value: Union[CSEVariable, tuple[CSEVariable, ...]],
        # ) -> Union[CSEVariable, tuple[CSEVariable, ...]]:
        #     """
        #     Override reduction to handle matmul and batchmatmul reduction types.
        #
        #     For matmul reductions, we translate them to the native matmul pattern:
        #     pointwise multiply (ops.dot) followed by reduction("dot").
        #     This generates proper tl.dot operations in Triton.
        #     """
        #     if reduction_type in [MATMUL_REDUCTION_OP, BATCH_MATMUL_OP]:
        #         # Generate tl.dot directly for matmul operations
        #         # Don't use parent's "dot" reduction as it's incompatible with our setup
        #         if isinstance(value, tuple) and len(value) == 2:
        #             from torch._inductor.codegen.triton import TritonCSEVariable
        #             from torch.utils._sympy.value_ranges import ValueRanges
        #
        #             # For tl.dot, we need [M, K] @ [K, N]
        #             # Both operands have shape [M, K]
        #             # Use tl.trans for transpose (Triton's transpose function)
        #
        #             left_shape = value[0].shape
        #             right_shape = value[1].shape
        #
        #             # Compute output shape
        #             # [M, K] @ [K, M] -> [M, M]
        #             if left_shape and right_shape and len(left_shape) >= 2:
        #                 output_shape = (left_shape[0], right_shape[0])
        #             else:
        #                 output_shape = None
        #
        #             # Use tl.trans with dims=(1, 0) to transpose the 2D tensor
        #             result = TritonCSEVariable(
        #                 name=f"tl.dot({value[0]}, tl.trans({value[1]}, 1, 0))",
        #                 bounds=ValueRanges.unknown(),
        #                 dtype=dtype,
        #                 shape=output_shape,
        #             )
        #             return result
        #         else:
        #             raise Unsupported(f"Unexpected value type for {reduction_type}: {type(value)}")
        #
        #     # For other reduction types, use the parent implementation
        #     return super().reduction(dtype, src_dtype, reduction_type, value)  # type: ignore[arg-type]
