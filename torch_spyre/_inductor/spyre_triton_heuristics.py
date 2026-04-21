"""
Triton heuristics patching for Spyre.

This module provides functionality to monkey-patch PyTorch Inductor's Triton
heuristics to set appropriate block sizes for Spyre hardware.
"""

from typing import Any
from torch._inductor.virtualized import V
from .logging_utils import get_inductor_logger


logger = get_inductor_logger("spyre_triton_heuristics")


def set_spyre_triton_block_size(
    config_dict: dict[str, int],
    spyre_triton_block_size: dict[str, int],
) -> dict[str, int]:
    """
    Set block sizes for Spyre.

    Args:
        config_dict: Original block size configuration
        spyre_triton_block_size: Block size for Spyre

    Returns:
        Updated config_dict with block sizes adjusted for Spyre
    """
    logger.debug("set_spyre_triton_block_size")
    logger.debug(f"  Original config: {config_dict}")

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
        if block_name not in updated_config:
            continue

        block_size = spyre_triton_block_size.get(dim_prefix)
        if block_size is None:
            raise RuntimeError(f"Block size for {dim_prefix} is not specified")

        if block_size < 1:
            raise RuntimeError(f"Illegal block size for {dim_prefix}: {block_size}")

        # Ensure block size is at least 1 and a power of 2
        import math

        # Round up to nearest power of 2
        p2_block_size = 2 ** math.ceil(math.log2(block_size))

        if p2_block_size != block_size:
            logger.info(
                f"Block size is rounded up from {block_size} to {p2_block_size}"
            )

        updated_config[block_name] = p2_block_size

    logger.debug(f"  Updated config: {updated_config}")
    return updated_config


def get_spyre_triton_block_size() -> dict[str, int]:
    """
    Get spyre_triton_block_size from V.graph.

    Returns:
        Dictionary mapping dimension prefixes to block sizes

    Raises:
        RuntimeError: If V.graph does not have _spyre_triton_block_size attribute
                     or if spyre_triton_block_size is not set
    """
    if not hasattr(V.graph, "_spyre_triton_block_size"):
        raise RuntimeError("V.graph does not have _spyre_triton_block_size attribute")

    spyre_triton_block_size = getattr(V.graph, "_spyre_triton_block_size")

    if not spyre_triton_block_size:
        raise RuntimeError("spyre_triton_block_size is not set")

    logger.debug(f"spyre_triton_block_size={spyre_triton_block_size}")
    return spyre_triton_block_size


def patch_triton_config_for_spyre():
    """
    Monkey-patch PyTorch Inductor's triton config functions to set the block size
    for Spyre.

    This should be called during Spyre initialization to override the heuristics.

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
    _original_triton_config_reduction = triton_heuristics.triton_config_reduction
    _original_triton_config_tiled_reduction = (
        triton_heuristics.triton_config_tiled_reduction
    )
    _original_match_target_block_product = triton_heuristics.match_target_block_product
    _original_make_matmul_triton_config = triton_heuristics.make_matmul_triton_config

    def patched_triton_config(size_hints, x, y=None, z=None, num_warps=None, **kwargs):
        """Patched version that sets the block size for Spyre."""
        logger.debug("patched_triton_config called")

        config = _original_triton_config(size_hints, x, y, z, num_warps, **kwargs)

        # Get spyre_triton_block_size from V.graph
        spyre_triton_block_size = get_spyre_triton_block_size()

        # Build size hint dictionary
        size_hint_dict = {"x": x}
        if y is not None:
            size_hint_dict["y"] = y
        if z is not None:
            size_hint_dict["z"] = z

        config.kwargs = set_spyre_triton_block_size(
            config.kwargs, spyre_triton_block_size
        )

        return config

    def patched_triton_config_reduction(
        size_hints,
        x,
        r,
        num_stages=1,
        num_warps=None,
        register_intensive=False,
        dynamic_scale_rblock=True,
        reduction_hint=None,
        min_num_warps=None,
    ):
        """Patched version for reductions to set the block size for Spyre."""
        logger.debug("patched_triton_config_reduction called")

        config = _original_triton_config_reduction(
            size_hints,
            x,
            r,
            num_stages,
            num_warps,
            register_intensive,
            dynamic_scale_rblock,
            reduction_hint,
            min_num_warps,
        )

        # Get spyre_triton_block_size from V.graph
        spyre_triton_block_size = get_spyre_triton_block_size()

        config.kwargs = set_spyre_triton_block_size(
            config.kwargs, spyre_triton_block_size
        )

        return config

    def patched_triton_config_tiled_reduction(
        size_hints, x, y, r, num_stages=1, register_intensive=False
    ):
        """Patched version for tiled reductions to set the block size for Spyre."""
        logger.debug("patched_triton_config_tiled_reduction called")

        config = _original_triton_config_tiled_reduction(
            size_hints, x, y, r, num_stages, register_intensive
        )

        # Get spyre_triton_block_size from V.graph
        spyre_triton_block_size = get_spyre_triton_block_size()

        config.kwargs = set_spyre_triton_block_size(
            config.kwargs, spyre_triton_block_size
        )

        return config

    def patched_match_target_block_product(
        size_hints, tiling_scores, target_block_product, min_block_size=1
    ):
        """Patched version that sets the block size for Spyre."""
        logger.debug("patched_match_target_block_product called")

        # First get the original block sizes
        block_sizes = _original_match_target_block_product(
            size_hints, tiling_scores, target_block_product, min_block_size
        )

        # Get spyre_triton_block_size from V.graph
        spyre_triton_block_size = get_spyre_triton_block_size()

        # Convert block_sizes format to config format for set_spyre_triton_block_size
        config_dict = {}
        dim_to_block = {
            "x": "XBLOCK",
            "y": "YBLOCK",
            "z": "ZBLOCK",
            "r0_": "R0_BLOCK",
            "r1_": "R1_BLOCK",
            "r2_": "R2_BLOCK",
        }
        for dim, block_name in dim_to_block.items():
            if dim in block_sizes:
                config_dict[block_name] = block_sizes[dim]

        # Apply core splitting
        updated_config = set_spyre_triton_block_size(
            config_dict, spyre_triton_block_size
        )

        # Convert back to block_sizes format
        for dim, block_name in dim_to_block.items():
            if block_name in updated_config:
                block_sizes[dim] = updated_config[block_name]

        return block_sizes

    def patched_make_matmul_triton_config(
        sizes: dict[str, int], num_warps: int, num_stages: int
    ):
        """Patched version for matmul to set the block size for Spyre."""
        logger.debug("patched_make_matmul_triton_config called")
        logger.debug(f"  Input sizes: {sizes}")

        # First get the original config
        config = _original_make_matmul_triton_config(sizes, num_warps, num_stages)

        # Get spyre_triton_block_size from V.graph
        spyre_triton_block_size = get_spyre_triton_block_size()

        # Apply Spyre block sizes
        config.kwargs = set_spyre_triton_block_size(
            config.kwargs, spyre_triton_block_size
        )

        logger.debug(f"  Updated config: {config.kwargs}")
        return config

    # Apply patches
    triton_heuristics.triton_config = patched_triton_config
    triton_heuristics.triton_config_reduction = patched_triton_config_reduction
    triton_heuristics.triton_config_tiled_reduction = (
        patched_triton_config_tiled_reduction
    )
    triton_heuristics.match_target_block_product = patched_match_target_block_product
    triton_heuristics.make_matmul_triton_config = patched_make_matmul_triton_config

    logger.info("Patched Triton heuristics for Spyre")

    # Return original functions for restoration
    return {
        "triton_config": _original_triton_config,
        "triton_config_reduction": _original_triton_config_reduction,
        "triton_config_tiled_reduction": _original_triton_config_tiled_reduction,
        "match_target_block_product": _original_match_target_block_product,
        "make_matmul_triton_config": _original_make_matmul_triton_config,
    }


# Made with Bob
