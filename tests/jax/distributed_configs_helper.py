# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import jax
from itertools import product
from transformer_engine.jax.sharding import ShardingType
from transformer_engine.jax.softmax import SoftmaxType
from transformer_engine.jax.fused_attn import AttnBiasType, AttnMaskType\


class DistributedConfigsHelper(object):

    def __init__(self, num_gpus=len(jax.devices())):
        super().__init__()
        self.device_count = min(num_gpus, 8)
        if self.device_count < 2:
            self.layernorm_refs = []
            self.softmax_types = []
            self.softmax_refs = []
            self.self_attn_bias_types = []
            self.self_attn_mask_types = []
            self.self_attn_refs = []
            self.cross_attn_mask_types = []
            self.cross_attn_refs = []
            return

        mesh_configs = [
            ((self.device_count, 1),    ("dp", None), ShardingType.DP),
            ((self.device_count, 1),    ("tp", None), ShardingType.TP_COL),
            ((self.device_count, 1),    ("tp", None), ShardingType.TP_ROW)
        ]
        if self.device_count >= 4:
            mesh_configs += [
                ((self.device_count//2, 2), ("dp", "tp"), ShardingType.DP_TP_COL),
                ((self.device_count//2, 2), ("dp", "tp"), ShardingType.DP_TP_ROW),
            ]
        if self.device_count >= 6:
            mesh_configs += [
                ((2, self.device_count//2), ("dp", "tp"), ShardingType.DP_TP_COL),
                ((2, self.device_count//2), ("dp", "tp"), ShardingType.DP_TP_ROW),
            ]

        layernorm_collectives = {
            ShardingType.DP        : {'all-reduce': 2, 'other': 0},
            ShardingType.TP_COL    : {'all-reduce': 0, 'other': 0},
            ShardingType.DP_TP_COL : {'all-reduce': 2, 'other': 0}
        }
        self.layernorm_refs = [
            mesh_config + (layernorm_collectives[mesh_config[2]], ) \
                for mesh_config in mesh_configs \
                    if mesh_config[2] not in (ShardingType.TP_ROW, ShardingType.DP_TP_ROW)
        ]

        self.softmax_types = [
            SoftmaxType.SCALED,
            SoftmaxType.SCALED_MASKED,
            SoftmaxType.SCALED_UPPER_TRIANG_MASKED
        ]
        softmax_collectives = {
            ShardingType.DP        : {'all-reduce': 1, 'other': 0},
            ShardingType.TP_COL    : {'all-reduce': 1, 'other': 0},
            ShardingType.TP_ROW    : {'all-reduce': 1, 'other': 0},
            ShardingType.DP_TP_COL : {'all-reduce': 1, 'other': 0},
            ShardingType.DP_TP_ROW : {'all-reduce': 1, 'other': 0}
        }
        self.softmax_refs = [
            mesh_config + (softmax_collectives[mesh_config[2]], ) for mesh_config in mesh_configs
        ]

        self.self_attn_bias_types = [
            AttnBiasType.NO_BIAS,
            AttnBiasType.PRE_SCALE_BIAS,
            AttnBiasType.POST_SCALE_BIAS
        ]
        self.self_attn_mask_types = [
            AttnMaskType.CAUSAL_MASK,
            AttnMaskType.PADDING_MASK,
            AttnMaskType.NO_MASK
        ]
        self_attn_collectives = {
            ShardingType.DP : {
                AttnBiasType.NO_BIAS         : {'all-reduce': 1, 'other': 0},
                AttnBiasType.PRE_SCALE_BIAS  : {'all-reduce': 2, 'other': 0},
                AttnBiasType.POST_SCALE_BIAS : {'all-reduce': 2, 'other': 0},
            },
            ShardingType.TP_COL : {
                AttnBiasType.NO_BIAS         : {'all-reduce': 1, 'other': 0},
                AttnBiasType.PRE_SCALE_BIAS  : {'all-reduce': 1, 'other': 0},
                AttnBiasType.POST_SCALE_BIAS : {'all-reduce': 1, 'other': 0}
            },
            ShardingType.DP_TP_COL : {
                AttnBiasType.NO_BIAS         : {'all-reduce': 1, 'other': 0},
                AttnBiasType.PRE_SCALE_BIAS  : {'all-reduce': 2, 'other': 0},
                AttnBiasType.POST_SCALE_BIAS : {'all-reduce': 2, 'other': 0}
            },
        }
        self.self_attn_refs = [
            mesh_config + (bias_type, self_attn_collectives[mesh_config[2]][bias_type]) \
                for mesh_config, bias_type in product(mesh_configs, self.self_attn_bias_types) \
                    if mesh_config[2] not in (ShardingType.TP_ROW, ShardingType.DP_TP_ROW)
        ]

        self.cross_attn_mask_types = [
            AttnMaskType.PADDING_MASK,
            AttnMaskType.NO_MASK
        ]
        self.cross_attn_refs = [
            mesh_config + ({'all-reduce': 1, 'other': 0}, ) \
                for mesh_config in mesh_configs \
                    if mesh_config[2] not in (ShardingType.TP_ROW, ShardingType.DP_TP_ROW)
        ]
