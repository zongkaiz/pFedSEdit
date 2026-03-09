from dataclasses import dataclass
from typing import List, Literal

from util.hparams import HyperParams


@dataclass
class MEMITHyperParams(HyperParams):
    # Method
    model_name: str
    layers: List[int]
    layer_selection: Literal["all", "random"]
    fact_token: Literal[
        "last", "subject_first", "subject_last", "subject_first_after_last"
    ]
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
    mom2_update_weight: float

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str
    alpha: float = 35.0 # FedLEKE 编辑成功阈值 (需要根据模型调整)
    upper_bound: float = 150.0 # FedLEKE 跳过编辑的上限阈值 (需要根据模型调整)
    max_iterations: int = 1 # FedLEKE 最大迭代次数 (设为 1 则不进行迭代)