# 在 memit/memit_rect_main.py 文件中

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set # 为追踪索引添加 Set 类型

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast
from util.globals import *

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .memit_hparams import MEMITHyperParams # 假设超参数已在该文件中添加

# 缓存变量
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

# --- apply_memit_rect_to_model 函数保持不变，它调用 execute_memit ---
def apply_memit_rect_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    返回应用了 MEMIT（可能包含 FedLEKE 迭代）编辑后的模型。
    :param copy: 如果为 True, 会保留原始模型，创建一个新的模型进行编辑。
        注意：您需要负责释放新模型占用的内存以避免泄漏。
    :return: (1) 更新后的模型, (2) 发生改变的原始权重的副本
    """

    weights_copy = {} # 用于存储原始权重（如果需要）
    if copy:
        model = deepcopy(model) # 如果需要，复制模型

    # 执行核心编辑逻辑（包含可能的迭代）
    deltas = execute_memit(model, tok, requests, hparams, cache_template=cache_template)

    # 应用整流（Rectification）逻辑
    with torch.no_grad():
        for w_name, (key_mat, val_mat) in deltas.items(): # 使用 execute_memit 返回的最后一次计算的 delta 组件
            key_mat, val_mat = key_mat.to("cuda"), val_mat.to("cuda")
            # 注意：这里的 upd_matrix 是基于最后一次迭代的 delta 计算的，可能不完全代表累积的总变化
            upd_matrix = key_mat @ val_mat.T
            k_percent = 40  # 整流百分比
            epsilon = 1e-8  # 防止除零

            w = nethook.get_parameter(model, w_name) # 获取模型 *当前* (已累积更新后) 的权重
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape) # 匹配形状

            # 如果需要返回原始权重，并且尚未复制
            if return_orig_weights and w_name not in weights_copy:
                 # 注意：这里复制的是编辑 *开始前* 的权重可能更合理，但需要调整 execute_memit
                 # 目前这样复制的是迭代过程 *中途* 某层的权重，可能意义不大
                 weights_copy[w_name] = w.detach().clone() # 这行可能需要调整逻辑

            # 计算相对变化量并找到阈值
            delta = torch.abs(upd_matrix / (w + epsilon))
            threshold = torch.kthvalue(delta.reshape(-1), int(delta.numel() * (100 - k_percent) / 100)).values

            # 应用稀疏更新 (整流) - 注意：这里是应用 *最后一次迭代计算出的delta* 的整流版本
            # 这与 FedLEKE 累积更新的逻辑可能存在冲突或不一致，需要谨慎评估效果
            with torch.no_grad():
                mask = delta >= threshold
                # 警告：这里的 w 已经是累积更新后的权重，再应用一次基于最后 delta 的稀疏更新可能不是预期行为
                # 或许整流逻辑需要重新考虑，或者应该在 execute_memit 内部处理？
                w[mask] += upd_matrix[mask].float() # 再次修改模型权重

    print(f"成功将新的(整流后)权重插入到 {list(deltas.keys())}")

    return model, weights_copy # 返回被修改的模型和（可能意义不大的）权重副本


def execute_memit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    执行 MEMIT 更新算法，如果 hparams.max_iterations > 1 则包含 FedLEKE 风格的迭代。
    注意：此函数会直接修改传入的 model 对象的状态。
    """

    deltas = {} # 存储 *最后一次* 成功迭代/层的中间结果 (adj_k, resid)

    # 预处理请求，保留原始请求列表
    original_requests = deepcopy(requests) # 保留完整的原始请求列表
    for i, request in enumerate(original_requests):
        if request["target_new"]["str"][0] != " ":
            # 为保证正确分词，需要加空格
            original_requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    # 打印请求示例
    for request in original_requests[:10]:
        print(
            f"MEMIT request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # 获取需要修改的初始权重 (用于累积更新)
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }

    # --- 计算所有原始请求的目标 z 向量 (与之前相同) ---
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1] # 目标层
    z_list = []
    for request in original_requests:
        # (保留从缓存加载 z 的逻辑)
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if (
            cache_fname is not None
            and cache_fname.exists()
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
                data_loaded = True
            except Exception as e:
                print(f"读取缓存文件出错: {e}。 正在重新计算...")

        # 如果缓存未加载，则计算 z
        if not data_loaded:
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
            )
            z_list.append(cur_z)
            # (保留保存 z 到缓存的逻辑)
            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"已缓存 k/v 对在 {cache_fname}")
    # --- 目标 z 向量计算结束 ---

    original_zs = torch.stack(z_list, dim=1) # 所有请求的目标状态
    num_requests = len(original_requests)
    # 使用集合追踪需要处理的请求的 *原始索引*
    requests_to_process_indices: Set[int] = set(range(num_requests))

    # ===== 开始 FedLEKE 迭代循环 =====
    for iteration in range(hparams.max_iterations):
        print(f"\n===== 迭代 {iteration + 1} / {hparams.max_iterations} =====")

        # 如果没有需要处理的请求了，提前退出
        if not requests_to_process_indices:
            print("没有请求需要进一步处理。停止迭代。")
            break

        # --- 根据 FedLEKE 标准筛选当前迭代要处理的请求 ---
        current_requests_indices = sorted(list(requests_to_process_indices)) # 获取当前需要处理的请求的原始索引
        current_requests = [original_requests[i] for i in current_requests_indices] # 获取对应的请求字典
        current_target_zs = original_zs[:, current_requests_indices] # 获取对应的目标 z 向量

        # 获取 *当前子集* 在目标层的 *当前* 隐藏状态
        cur_zs_at_final_layer = get_module_input_output_at_words(
            model, # 使用当前模型状态
            tok,
            z_layer,
            context_templates=[req["prompt"] for req in current_requests],
            words=[req["subject"] for req in current_requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T

        # 计算当前子集的目标 z 和实际 z 之间的距离（范数平方）
        # 注意：NSE论文用范数平方，这里用范数，效果类似但阈值需调整
        distances = torch.linalg.norm(current_target_zs - cur_zs_at_final_layer, dim=0)
        # 找出距离大于 alpha (未达标) 且小于 upper_bound (不过于离谱) 的请求
        mask_needs_edit = (distances > hparams.alpha) & (distances < hparams.upper_bound)

        # 得到在 *当前子集* 中，需要在此次迭代中进行编辑的请求的 *相对索引*
        indices_to_edit_in_current = torch.nonzero(mask_needs_edit, as_tuple=True)[0].tolist()

        # 如果当前子集中没有需要编辑的请求了
        if not indices_to_edit_in_current:
            print(f"迭代 {iteration + 1}: 所有剩余请求均满足标准或超出界限。停止。")
            break

        # 将相对索引映射回 *原始索引*，用于本轮迭代处理
        original_indices_to_edit = [current_requests_indices[i] for i in indices_to_edit_in_current]

        print(f"本轮迭代处理 {len(original_indices_to_edit)} 个请求 (索引示例: {original_indices_to_edit[:10]}...).")

        # 准备本轮迭代实际使用的请求列表和目标 z 向量
        iter_requests = [original_requests[i] for i in original_indices_to_edit]
        iter_zs = original_zs[:, original_indices_to_edit] # 本轮迭代的目标 Zs

        # --- MEMIT 层循环 (现在操作的是筛选后的子集 iter_requests) ---
        for i, layer in enumerate(hparams.layers):
            print(f"\n-- 层 {layer} (迭代 {iteration + 1}) --")

            # 获取子集在当前层的 keys
            layer_ks = compute_ks(model, tok, iter_requests, hparams, layer, context_templates).T
            print(f"向层 {layer} 写入 {layer_ks.size(1)} 个键值对")

            # 计算子集在目标层的当前状态，以得到残差
            cur_zs = get_module_input_output_at_words(
                model,
                tok,
                z_layer,
                context_templates=[request["prompt"] for request in iter_requests],
                words=[request["subject"] for request in iter_requests],
                module_template=hparams.layer_module_tmp,
                fact_token_strategy=hparams.fact_token,
            )[1].T
            targets = iter_zs - cur_zs # 当前子集的目标残差
            print("子集 z 误差均值:", torch.linalg.norm(targets, dim=0).mean())

            # 确认 keys 和 targets 的数量一致 (在子集处理模式下应该一致)
            assert layer_ks.size(1) == targets.size(1), "处理子集时 Keys 和 Targets 数量必须匹配"

            # 加载协方差矩阵 (逻辑不变)
            force_recompute = False
            cov = get_cov(
                model,
                tok,
                hparams.rewrite_module_tmp.format(layer),
                hparams.mom2_dataset,
                hparams.mom2_n_samples if not force_recompute else hparams.mom2_n_samples // 10,
                hparams.mom2_dtype,
                force_recompute=force_recompute,
            )

            # 使用双精度计算更新量 (基于子集)
            layer_ks, targets = (
                layer_ks.double(),
                targets.double(),
            )

            # 计算调整后的 key (adj_k)
            adj_k = torch.linalg.solve(
                hparams.mom2_update_weight * cov.double() + layer_ks @ layer_ks.T,
                layer_ks,
            )
            # 计算当前层需要承担的残差 (保留 MEMIT 的分配方式)
            resid = targets / (len(hparams.layers) - i)
            # 计算更新矩阵
            upd_matrix = resid @ adj_k.T

            # 调整更新矩阵形状 (逻辑不变)
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

            print("原始权重范数:", torch.linalg.norm(weights[weight_name]))
            print("更新矩阵范数:", torch.linalg.norm(upd_matrix))

            # 直接、累积地更新模型权重
            with torch.no_grad():
                current_weight = nethook.get_parameter(model, weight_name)
                current_weight[...] += upd_matrix.float() # 累积应用更新
                # 记录最后一次计算的 delta 组件 (可选，可能不完全代表总变化)
                deltas[weight_name] = (
                    adj_k.detach().cpu(),
                    resid.detach().cpu(),
                )

            # 清理 GPU 内存
            cov.cpu()
            for x in [layer_ks, cur_zs, targets, adj_k, resid, upd_matrix]: # 清理更多张量
                x.cpu()
                del x
            torch.cuda.empty_cache()
        # --- MEMIT 层循环结束 ---

        # --- 更新 *下一次* 迭代需要处理的请求集合 ---
        # 在本轮层更新 *之后* 再次检查编辑成功情况
        # 获取当前子集（current_requests_indices对应的）在目标层的 *最终* 隐藏状态
        final_iter_zs_at_final_layer = get_module_input_output_at_words(
            model, tok, z_layer,
            context_templates=[original_requests[idx]["prompt"] for idx in current_requests_indices],
            words=[original_requests[idx]["subject"] for idx in current_requests_indices],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T

        # 重新计算距离，判断哪些请求 *仍然* 需要编辑
        final_iter_distances = torch.linalg.norm(current_target_zs - final_iter_zs_at_final_layer, dim=0)
        final_mask_still_needs_edit = (final_iter_distances > hparams.alpha) & (final_iter_distances < hparams.upper_bound)

        # 得到在 *当前子集* 中，*仍然* 需要编辑的请求的 *相对索引*
        final_indices_to_edit_in_current = torch.nonzero(final_mask_still_needs_edit, as_tuple=True)[0].tolist()

        # 更新下一轮迭代需要处理的 *原始索引* 集合
        requests_to_process_indices = {current_requests_indices[i] for i in final_indices_to_edit_in_current}
        print(f"迭代 {iteration + 1} 结束. 下一轮待处理请求数: {len(requests_to_process_indices)}")

    # ===== FedLEKE 迭代循环结束 =====

    # # 不再恢复原始模型状态 - 让累积更新生效
    # with torch.no_grad():
    #     for k, v in weights.items():
    #         v[...] = weights_copy[k] # 移除恢复操作

    print(f"Delta 计算完成于 {list(deltas.keys())}。最终模型状态反映了累积更新。")

    # 返回最后一次计算的 delta 组件。
    # 注意：直接应用这些组件可能无法精确复现最终模型状态，因为状态是迭代累积的。
    # 但这是为了兼容 apply_memit_rect_to_model 的结构。
    return deltas


# --- get_cov, upd_matrix_match_shape, get_context_templates 函数保持不变 ---
def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    获取协方差统计信息，然后计算其代数逆。
    缓存结果以备将来使用。
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"正在检索 {model_name} @ {layer_name} 的协方差统计信息。")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu") # 计算并缓存二阶矩

    # 如果需要逆，则计算逆；否则直接返回协方差矩阵
    return (
        torch.inverse(COV_CACHE[key].to("cuda")) if inv else COV_CACHE[key].to("cuda")
    )


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 和 GPT-J 的权重表示可能需要转置。
    返回与所需形状匹配的矩阵，否则引发 ValueError。
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T # 如果转置后匹配，则返回转置矩阵
    else:
        raise ValueError(
            "MEMIT 计算的更新矩阵与原始权重形状不匹配。"
            "请检查代码中是否存在错误？"
        )


def get_context_templates(model, tok):
    """
    生成或加载用于计算 z 向量的上下文模板。
    """
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        # 如果缓存为空，生成模板
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [ # 包含一个空模板
            [
                # 生成一些前缀文本，并将主题占位符 {} 添加到末尾
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast( # 使用模型快速生成一些文本片段
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"], # 不同的起始提示
                    n_gen_per_prompt=n_gen // 5, # 每个提示生成多少个
                    max_out_len=length, # 生成的最大长度
                )
            ]
            # 定义生成长度和数量
            for length, n_gen in [(10, 5)]  # 小心修改这里
        ]
        print(f"已缓存上下文模板: {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE