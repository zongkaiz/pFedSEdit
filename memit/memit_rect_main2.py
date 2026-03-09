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

# --- apply_memit_rect_to_model 现在主要负责调用 execute_memit ---
def apply_memit_rect_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    copy=False,
    return_orig_weights=False, # 这个参数现在意义不大，因为原始权重在 execute_memit 中不保存
    cache_template: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    返回应用了 MEMIT（包含 FedLEKE 迭代和层内整流）编辑后的模型。
    注意：此函数现在主要调用 execute_memit 来完成实际工作。
    :param copy: 如果为 True, 会保留原始模型，创建一个新的模型进行编辑。
    :return: (1) 更新后的模型, (2) 一个空字典（或根据需要调整 execute_memit 的返回值）
    """

    if copy:
        model = deepcopy(model) # 如果需要，在开始前复制模型

    # 执行核心编辑逻辑（包含迭代和内部整流）
    # execute_memit 现在直接修改 model，并且不需要返回 deltas 用于外部整流
    execute_memit(model, tok, requests, hparams, cache_template=cache_template)

    print(f"模型已通过迭代和层内整流进行更新。")

    # weights_copy 现在没有明确的源，因为原始权重没在 execute_memit 中保存返回
    # 如果确实需要返回编辑前的权重，需要在 execute_memit 开始时复制并返回
    weights_copy = {}
    if return_orig_weights:
        print("警告: return_orig_weights=True 在当前实现下可能不会返回有意义的原始权重。")

    return model, weights_copy # 返回被修改的模型


def execute_memit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    cache_template: Optional[str] = None,
) -> None: # 不再返回 deltas
    """
    执行 MEMIT 更新算法，包含 FedLEKE 风格的迭代 和 MEMIT-Rect 风格的层内整流。
    注意：此函数会直接修改传入的 model 对象的状态。
    """

    # 预处理请求，保留原始请求列表 (同前)
    original_requests = deepcopy(requests)
    for i, request in enumerate(original_requests):
        if request["target_new"]["str"][0] != " ":
            original_requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    for request in original_requests[:10]:
        print(
            f"MEMIT request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # --- 计算所有原始请求的目标 z 向量 (同前) ---
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []
    for request in original_requests:
        # (保留缓存逻辑)
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
        if not data_loaded:
            cur_z = compute_z( model, tok, request, hparams, z_layer, context_templates,)
            z_list.append(cur_z)
            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez( cache_fname, **{ "v_star": cur_z.detach().cpu().numpy(), },)
                print(f"已缓存 k/v 对在 {cache_fname}")
    # --- 目标 z 向量计算结束 ---

    original_zs = torch.stack(z_list, dim=1)
    num_requests = len(original_requests)
    requests_to_process_indices: Set[int] = set(range(num_requests))

    # ===== 开始 FedLEKE 迭代循环 =====
    for iteration in range(hparams.max_iterations):
        print(f"\n===== 迭代 {iteration + 1} / {hparams.max_iterations} =====")

        if not requests_to_process_indices:
            print("没有请求需要进一步处理。停止迭代。")
            break

        # --- 根据 FedLEKE 标准筛选当前迭代要处理的请求 (同前) ---
        current_requests_indices = sorted(list(requests_to_process_indices))
        current_requests = [original_requests[i] for i in current_requests_indices]
        current_target_zs = original_zs[:, current_requests_indices]

        cur_zs_at_final_layer = get_module_input_output_at_words(
            model, tok, z_layer,
            context_templates=[req["prompt"] for req in current_requests],
            words=[req["subject"] for req in current_requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T

        distances = torch.linalg.norm(current_target_zs - cur_zs_at_final_layer, dim=0)
        mask_needs_edit = (distances > hparams.alpha) & (distances < hparams.upper_bound)
        indices_to_edit_in_current = torch.nonzero(mask_needs_edit, as_tuple=True)[0].tolist()

        if not indices_to_edit_in_current:
            print(f"迭代 {iteration + 1}: 所有剩余请求均满足标准或超出界限。停止。")
            break

        original_indices_to_edit = [current_requests_indices[i] for i in indices_to_edit_in_current]
        print(f"本轮迭代处理 {len(original_indices_to_edit)} 个请求 (索引示例: {original_indices_to_edit[:10]}...).")

        iter_requests = [original_requests[i] for i in original_indices_to_edit]
        iter_zs = original_zs[:, original_indices_to_edit]

        # --- MEMIT 层循环 ---
        for i, layer in enumerate(hparams.layers):
            print(f"\n-- 层 {layer} (迭代 {iteration + 1}) --")

            # 获取子集 keys (同前)
            layer_ks = compute_ks(model, tok, iter_requests, hparams, layer, context_templates).T
            print(f"向层 {layer} 写入 {layer_ks.size(1)} 个键值对")

            # 计算子集残差 (同前)
            cur_zs = get_module_input_output_at_words(
                model, tok, z_layer,
                context_templates=[request["prompt"] for request in iter_requests],
                words=[request["subject"] for request in iter_requests],
                module_template=hparams.layer_module_tmp,
                fact_token_strategy=hparams.fact_token,
            )[1].T
            targets = iter_zs - cur_zs
            print("子集 z 误差均值:", torch.linalg.norm(targets, dim=0).mean())
            assert layer_ks.size(1) == targets.size(1), "处理子集时 Keys 和 Targets 数量必须匹配"

            # 加载协方差矩阵 (同前)
            force_recompute = False
            cov = get_cov(
                model, tok, hparams.rewrite_module_tmp.format(layer),
                hparams.mom2_dataset,
                hparams.mom2_n_samples if not force_recompute else hparams.mom2_n_samples // 10,
                hparams.mom2_dtype, force_recompute=force_recompute,
            )

            # 计算更新量 (同前)
            layer_ks, targets = ( layer_ks.double(), targets.double(), )
            adj_k = torch.linalg.solve(
                hparams.mom2_update_weight * cov.double() + layer_ks @ layer_ks.T, layer_ks,
            )
            resid = targets / (len(hparams.layers) - i)
            upd_matrix = resid @ adj_k.T

            # 调整形状 (同前)
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            upd_matrix = upd_matrix_match_shape(upd_matrix, nethook.get_parameter(model, weight_name).shape)

            print("原始权重范数:", torch.linalg.norm(nethook.get_parameter(model, weight_name)))
            print("计算出的更新矩阵范数:", torch.linalg.norm(upd_matrix))

            # ******** 开始：整合 Rectification 逻辑 ********
            with torch.no_grad():
                w = nethook.get_parameter(model, weight_name) # 获取当前权重
                k_percent = 40  # 整流百分比 (可以考虑也放入 hparams)
                epsilon = 1e-8  # 防止除零

                # 计算相对变化量和阈值
                delta_rel = torch.abs(upd_matrix.float() / (w + epsilon)) # 转 float 计算相对变化
                threshold = torch.kthvalue(delta_rel.reshape(-1), int(delta_rel.numel() * (100 - k_percent) / 100)).values
                mask = delta_rel >= threshold # 计算 mask

                # 只应用 mask 范围内的更新
                # 注意 upd_matrix 之前是 double, 现在转 float 应用
                w[mask] += upd_matrix[mask].float()
                print(f"应用整流更新: {mask.sum().item()} / {mask.numel()} 个元素被更新。")
            # ******** 结束：整合 Rectification 逻辑 ********

            # 清理 GPU 内存 (同前)
            cov.cpu()
            for x in [layer_ks, cur_zs, targets, adj_k, resid, upd_matrix, delta_rel, mask]: # 多清理几个张量
                if isinstance(x, torch.Tensor): x.cpu()
                del x
            torch.cuda.empty_cache()
        # --- MEMIT 层循环结束 ---

        # --- 更新下一次迭代需要处理的请求集合 (同前) ---
        final_iter_zs_at_final_layer = get_module_input_output_at_words(
            model, tok, z_layer,
            context_templates=[original_requests[idx]["prompt"] for idx in current_requests_indices],
            words=[original_requests[idx]["subject"] for idx in current_requests_indices],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T
        final_iter_distances = torch.linalg.norm(current_target_zs - final_iter_zs_at_final_layer, dim=0)
        final_mask_still_needs_edit = (final_iter_distances > hparams.alpha) & (final_iter_distances < hparams.upper_bound)
        final_indices_to_edit_in_current = torch.nonzero(final_mask_still_needs_edit, as_tuple=True)[0].tolist()
        requests_to_process_indices = {current_requests_indices[i] for i in final_indices_to_edit_in_current}
        print(f"迭代 {iteration + 1} 结束. 下一轮待处理请求数: {len(requests_to_process_indices)}")

    # ===== FedLEKE 迭代循环结束 =====

    print(f"execute_memit 完成。模型状态已累积更新并应用了层内整流。")

    # 函数不再需要返回 deltas
    return None # 或者可以返回一些状态信息

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