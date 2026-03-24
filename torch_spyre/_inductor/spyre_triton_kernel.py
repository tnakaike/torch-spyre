import sympy
from typing import Optional
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
