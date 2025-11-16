from contextlib import contextmanager
import copy
from dataclasses import dataclass

@dataclass
class LinearLowbitContext:
    q_forward_input = "Cast2Fp4e2m1"
    q_forward_weight = "Cast2Fp4e2m1"
    q_backward_input = "Cast2Fp4e2m1"
    q_backward_weight = "Cast2Fp4e2m1"
    q_backward_outputgrad = "Cast2Fp4e2m1"

        # SVD & low-rank 配置
    activation_lowrank_niter = 2
    backward_lowrank_niter = 2
    q_scalar = 1.0
    enable_activation_svd = False
    activation_lowrank_svd = -1
    enable_backward_svd = False
    backward_lowrank_svd = -1
    activation_broadcast_dim = -1
    backward_broadcast_dim = -1
    activation_longtail_schedule = "none"
    backward_longtail_schedule = "none"
    enable_lowbit = True
    forward_svd_rank = -1
    enable_weight_svd = False
    gradacc_broadcast = False
    load_history=False

    def __repr__(self) -> str:
        """Pretty full-text representation of LinearLowbitContext."""
        def fn_name(f):
            return f.__name__ if callable(f) else repr(f)

        # schedules = ", ".join(self.schedule_list.keys())

        return (
            f"LinearLowbitContext(\n"
            f"  q_forward_input={fn_name(self.q_forward_input)},\n"
            f"  q_forward_weight={fn_name(self.q_forward_weight)},\n"
            f"  q_backward_input={fn_name(self.q_backward_input)},\n"
            f"  q_backward_weight={fn_name(self.q_backward_weight)},\n"
            f"  q_backward_outputgrad={fn_name(self.q_backward_outputgrad)},\n"
            f"  activation_lowrank_niter={self.activation_lowrank_niter},\n"
            f"  backward_lowrank_niter={self.backward_lowrank_niter},\n"
            f"  q_scalar={self.q_scalar},\n"
            f"  enable_activation_svd={self.enable_activation_svd},\n"
            f"  activation_lowrank_svd={self.activation_lowrank_svd},\n"
            f"  enable_backward_svd={self.enable_backward_svd},\n"
            f"  backward_lowrank_svd={self.backward_lowrank_svd},\n"
            f"  activation_broadcast_dim={self.activation_broadcast_dim},\n"
            f"  backward_broadcast_dim={self.backward_broadcast_dim},\n"
            f"  activation_longtail_schedule='{self.activation_longtail_schedule}',\n"
            f"  backward_longtail_schedule='{self.backward_longtail_schedule}',\n"
            f"  enable_lowbit={self.enable_lowbit},\n"
            f"  forward_svd_rank={self.forward_svd_rank},\n"
            f"  enable_weight_svd={self.enable_weight_svd}\n"
            f"  gradacc_broadcast={self.gradacc_broadcast}\n"
            f"  load_history={self.load_history}\n"
            f")"
        )
    # === 新增：clone 方法 ===
    def clone(self):
        new_obj = self.__class__()  # 创建新实例
        for k, v in self.__dict__.items():  # 拷贝实例属性
            setattr(new_obj, k, copy.deepcopy(v))
        # 如果类属性未实例化到 __dict__ 中，也复制它们
        for k, v in self.__class__.__dict__.items():
            if not k.startswith("__") and not callable(v) and k not in new_obj.__dict__:
                setattr(new_obj, k, copy.deepcopy(v))
        return new_obj

@contextmanager
def get_metis_context(**kwargs):
    """
    用于临时修改 LinearLowbitContext 全局配置的上下文管理器。
    进入时按 kwargs 修改，退出时自动恢复。
    
    示例：
        with get_metis_context(q_scalar=0.5, enable_lowbit=False):
            # 临时使用低比特关闭配置
            ...
    """
    old_state = {}
    # print("entering metis context with ", kwargs)
    try:
        # 保存旧值并设置新值
        for key, value in kwargs.items():
            if hasattr(LinearLowbitContext, key):
                old_state[key] = getattr(LinearLowbitContext, key)
                setattr(LinearLowbitContext, key, value)
            else:
                raise AttributeError(f"LinearLowbitContext has no attribute '{key}'")
        # debugpy.breakpoint()
        yield
    finally:
        # 恢复原始值
        # print("exiting metis context with ", old_state)
        for key, value in old_state.items():
            setattr(LinearLowbitContext, key, value)



@contextmanager
def load_svd_history():
    old_gradacc_status = LinearLowbitContext.load_history
    LinearLowbitContext.load_history = True
    # setattr(LinearLowbitContext, "gradacc_broadcast", True)
    try:
        yield
    finally:
        LinearLowbitContext.load_history = old_gradacc_status

    