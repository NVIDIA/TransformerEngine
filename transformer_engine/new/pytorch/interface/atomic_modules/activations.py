from ....common import ops, DType
from ..base_modules.compute_pipeline_module_base import (
    ComputePipelineModuleBase,
)


class ReLU(ComputePipelineModuleBase):
    def __init__(
        self,
    ):
        super().__init__(
            ops.Relu(
                "relu",
                DType.Infer,
                DType.Infer,
            )
        )


class GELU(ComputePipelineModuleBase):
    def __init__(
        self,
    ):
        super().__init__(
            ops.Relu(
                "relu",
                DType.Infer,
                DType.Infer,
            )
        )
