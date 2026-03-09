import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import shutil
from itertools import islice
from time import time
from typing import Tuple, Union
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import random
from baselines.ft import FTHyperParams, apply_ft_to_model
from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    MultiCounterFactDataset,
    MQUAKEDataset,
    get_tfidf_vectorizer,
    KnownsDataset,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from experiments.py.eval_utils_mquake import compute_rewrite_quality_mquake
from memit import MEMITHyperParams
from memit.compute_z import get_module_input_output_at_words, compute_z
from memit.memit_main import apply_memit_to_model, get_context_templates
from memit.memit_seq_main import apply_memit_seq_to_model
from memit.memit_rect_main1 import apply_memit_rect_to_model
from AlphaEdit import AlphaEditHyperParams
from AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model, get_cov
from rome import ROMEHyperParams, apply_rome_to_model
from util import nethook
from util.globals import *
from fedsleke import NSEHyperParams
from fedsleke.fedsleke_main import apply_nse_to_model
from glue_eval.glue_eval import GLUEEval
ALG_DICT = {
    "AlphaEdit": (AlphaEditHyperParams, apply_AlphaEdit_to_model),
    "MEMIT_seq": (MEMITHyperParams, apply_memit_seq_to_model),
    "MEMIT_prune": (MEMITHyperParams, apply_memit_to_model),
    "MEMIT_rect": (MEMITHyperParams, apply_memit_rect_to_model),
    "FedLEKE": (NSEHyperParams, apply_nse_to_model),
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
}

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
    "mquake": (MQUAKEDataset, compute_rewrite_quality_mquake),
}

def construct_federated_dataset(args, ds):
    """
    为联邦模式构建并返回一个新的数据集。

    该函数根据 `args` 中提供的联邦采样参数，对原始数据集 `ds` 进行重构。
    它会创建一个大小为 `args.total_dataset_size` 的新数据集，其中包含
    特定比例的主关系数据和相似关系数据。

    Args:
        args: 命令行参数对象，必须包含所有 federated_* 相关参数。
        ds: 初始数据集对象 (例如 MultiCounterFactDataset)。

    Returns:
        一个由数据记录组成的列表，代表新构建的数据集。
    """
    # 1. 验证联邦模式所需的参数
    if not all([args.primary_relation_id, args.total_dataset_size, args.federated_index_path]):
        raise ValueError(
            "在联邦模式下, --primary_relation_id, --total_dataset_size, "
            "和 --federated_index_path 参数必须被提供。"
        )

    # 2. 加载联邦索引元数据
    print(f"正在从 {args.federated_index_path} 加载联邦索引...")
    with open(args.federated_index_path, 'r', encoding='utf-8') as f:
        fed_index = json.load(f)
    
    relations_ordered = fed_index["relations_ordered"]
    similarity_matrix = np.array(fed_index["relation_similarity_matrix"])
    relation_to_cases = fed_index["relation_to_cases"]

    # 创建一个反向映射以便快速查找
    relation_to_idx = {rel: i for i, rel in enumerate(relations_ordered)}

    if args.primary_relation_id not in relation_to_idx:
        raise ValueError(f"主关系 '{args.primary_relation_id}' 在联邦索引中未找到。")

    # 3. 识别主关系和相似关系
    primary_idx = relation_to_idx[args.primary_relation_id]
    
    # 获取主关系的相似度得分并降序排序
    sim_scores = similarity_matrix[primary_idx]
    sorted_indices = np.argsort(sim_scores)[::-1]

    # 寻找前 N 个最相似的关系 (排除主关系自身)
    similar_relation_ids =[]
    for idx in sorted_indices:
        if idx == primary_idx:
            continue
        if len(similar_relation_ids) < args.num_similar_relations:
            similar_relation_ids.append(relations_ordered[idx])
        else:
            break
    
    print(f"主关系 (本地数据): {args.primary_relation_id}")
    print(f"找到 {len(similar_relation_ids)} 个最相似的关系 (非本地数据): {similar_relation_ids}")

    # 4. 将 case_id 分层到本地和非本地数据池
    # 首先，获取所有实际存在于已加载数据集 `ds` 中的 case_id
    case_id_to_record = {record['case_id']: record for record in ds}
    available_case_ids = set(case_id_to_record.keys())

    # 从索引中获取 case_id，并用可用 case_id 进行过滤
    local_case_ids = set(relation_to_cases.get(args.primary_relation_id,))
    local_case_ids.intersection_update(available_case_ids)

    non_local_case_ids = set()
    for rel_id in similar_relation_ids:
        non_local_case_ids.update(relation_to_cases.get(rel_id,))
    non_local_case_ids.intersection_update(available_case_ids)
    
    # 确保本地和非本地数据池之间没有重叠
    non_local_case_ids -= local_case_ids

    total_local_available = len(local_case_ids)
    total_non_local_available = len(non_local_case_ids)
    print(f"可用本地数据 ({args.primary_relation_id}) 数量: {total_local_available}")
    print(f"可用非本地数据 (来自相似关系) 数量: {total_non_local_available}")

    if total_local_available == 0:
        raise ValueError("主关系没有任何可用的数据，无法继续。")

    # 5. 计算批次构成，并处理数据稀缺情况
    num_batches = math.ceil(args.total_dataset_size / args.num_edits)
    
    requested_local_per_batch = math.floor(args.num_edits * args.local_ratio_in_batch)
    total_local_needed = requested_local_per_batch * num_batches

    if total_local_needed > total_local_available:
        # 检测到数据稀缺：计算每个批次可持续的最大本地样本数
        sustainable_local_per_batch = math.floor(total_local_available / num_batches)
        final_local_per_batch = sustainable_local_per_batch
        
        new_ratio = final_local_per_batch / args.num_edits
        print("\n" + "="*60)
        print("!! 警告：检测到主关系数据稀缺!!")
        print(f"您请求的本地数据比例 ({args.local_ratio_in_batch}) 共需要 {total_local_needed} 个样本, 但当前只有 {total_local_available} 个可用。")
        print(f"为保证每个批次的本地数据量相同，已自动调整。现在每个批次将包含 {final_local_per_batch} 个本地样本。")
        print(f"因此，实际的本地数据比例将约为 {new_ratio:.4f}。")
        print("="*60 + "\n")
    else:
        final_local_per_batch = requested_local_per_batch

    final_non_local_per_batch = args.num_edits - final_local_per_batch

    # 6. 组装最终的数据集
    final_dataset =[]
    
    # 打乱数据池以便随机采样
    local_pool = list(local_case_ids)
    random.shuffle(local_pool)
    non_local_pool = list(non_local_case_ids)
    random.shuffle(non_local_pool)
    
    if total_non_local_available > 0 and total_non_local_available < final_non_local_per_batch:
        print("提示：非本地数据不足以充满一个批次，将在不同批次中重复使用。")

    for _ in range(num_batches):
        batch_case_ids =[]
        
        # 采样本地数据
        for _ in range(final_local_per_batch):
            if not local_pool:
                # 此情况不应发生（因为有稀缺性检查），但作为安全措施
                break 
            batch_case_ids.append(local_pool.pop())

        # 采样非本地数据
        for _ in range(final_non_local_per_batch):
            if not non_local_pool:
                if total_non_local_available == 0: break # 完全没有非本地数据
                # 如果数据池耗尽，则重新打乱并复用
                non_local_pool = list(non_local_case_ids)
                random.shuffle(non_local_pool)
            batch_case_ids.append(non_local_pool.pop())
            
        # 在批次内部进行打乱，混合本地和非本地数据
        random.shuffle(batch_case_ids)
        
        # 将 case_id 转换回完整的数据记录
        for case_id in batch_case_ids:
            final_dataset.append(case_id_to_record[case_id])

    # 裁剪到用户指定的精确总大小
    return final_dataset[:args.total_dataset_size]
def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    dir_name: str,
    num_edits: int = 1,
    use_cache: bool = False,
    # 在这里添加新的参数
    federated_mode: bool = False,
    federated_index_path: str = None,
    primary_relation_id: str = None,
    num_similar_relations: int = 3,
    local_ratio_in_batch: float = 0.9,
    total_dataset_size: int = None,
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Determine run directory
    # Create new dir if not continuing from prev run OR prev run doesn't exist
    if (
        continue_from_run is None
        or not (run_dir := RESULTS_DIR / dir_name / continue_from_run).exists()
    ):
        continue_from_run = None
    if continue_from_run is None:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")
    if "MEMIT" in alg_name:
    # Get run hyperparameters
        params_path = (
            run_dir / "params.json"
            if continue_from_run is not None
            else HPARAMS_DIR / "MEMIT" / hparams_fname
        )
    else:
        params_path = (
            run_dir / "params.json"
            if continue_from_run is not None
            else HPARAMS_DIR / alg_name / hparams_fname
        )
    hparams = params_class.from_json(params_path)
    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    print(f"Executing {alg_name} with parameters {hparams}")

    # Instantiate vanilla model
    if type(model_name) is str:
        print("Instantiating model")
        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path

    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    if num_edits > 1:
        assert ds_name != "cf", f"{ds_name} does not support multiple edits"

    ds_class, ds_eval_method = DS_DICT[ds_name]
    # 注意：在联邦模式下，我们先加载完整数据集，所以这里的 dataset_size_limit 会被忽略
    ds_load_size = None if federated_mode else dataset_size_limit
    ds = ds_class(DATA_DIR, tok=tok, size=ds_load_size)

    # =================================================================
    # 开始：联邦模式集成代码块
    # =================================================================
    if federated_mode:
        print("联邦模式已启用。正在构建联邦数据集...")

        # --- 自动选择索引文件路径的逻辑 ---
        index_path_to_use = federated_index_path
        if index_path_to_use is None:
            # 如果用户没有指定路径，则根据 ds_name 自动生成
            index_path_to_use = f"/media/h3c/users/zongkai/AlphaEdit-main/data/federated_index_{ds_name}.json"
            print(f"未指定索引文件路径，将根据 ds_name='{ds_name}' 自动使用: {index_path_to_use}")
        else:
            # 如果用户指定了路径，则使用用户提供的路径
            print(f"已手动指定索引文件路径: {index_path_to_use}")
        # --- 逻辑结束 ---

        class ArgsNamespace:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        
        fed_args = ArgsNamespace(
            primary_relation_id=primary_relation_id,
            total_dataset_size=total_dataset_size,
            federated_index_path=index_path_to_use,  # <-- 使用我们新确定的路径
            num_similar_relations=num_similar_relations,
            local_ratio_in_batch=local_ratio_in_batch,
            num_edits=num_edits
        )
        ds = construct_federated_dataset(fed_args, ds)
        print(f"联邦数据集构建完成，共包含 {len(ds)} 条记录。")
    # =================================================================
    # 结束：联邦模式集成代码块
    # =================================================================

    # Get cache templates
    cache_template = None
    if use_cache:
        if any(alg in alg_name for alg in ["MEMIT","AlphaEdit", "MEMIT_seq", "MEMIT_prune", "MEMIT_rect"]):
            cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_MEMIT"
                / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
            )
        else:
            cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_{alg_name}"
                / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
            )
        print(f"Will load cache from {cache_template}")
    if alg_name == "MEMIT_rect":
        cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_{alg_name}"
                / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        for record in ds:
            # Retrieve k/v pair if already stored in cache
            cache_fname = (
                Path(
                    str(cache_template).format(
                        hparams.layers[-1], hparams.clamp_norm_factor, record["case_id"]
                    )
                )
                if cache_template is not None
                else None
            )
            data_loaded = False
            if (
                cache_fname is not None  # Require cache template
                and cache_fname.exists()  # Cache file must exist
            ):
                continue
            # Compute k/v pair if not loaded from cache
            if not data_loaded:
                context_templates = get_context_templates(model, tok)
                cur_z = compute_z(
                    model,
                    tok,
                    {"case_id": record["case_id"], **record["requested_rewrite"]},
                    hparams,
                    hparams.layers[-1],
                    context_templates,
                )
                if cache_fname is not None:
                    cache_fname.parent.mkdir(exist_ok=True, parents=True)
                    np.savez(
                        cache_fname,
                        **{
                            "v_star": cur_z.detach().cpu().numpy(),
                        },
                    )
                    print(f"Cached k/v pair at {cache_fname}")
    if any(alg in alg_name for alg in ["AlphaEdit", "MEMIT_seq", "MEMIT_prune", "FedLEKE"]):
        # Iterate through dataset
        W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
        if hparams.model_name == "gpt2-xl":
            cache_c = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
            if alg_name == "AlphaEdit":
                P = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
        elif hparams.model_name in ["EleutherAI_gpt-j-6B","Llama3-8B","phi-1.5"]:
            cache_c = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
            if alg_name == "AlphaEdit":
                P = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
        del W_out
    if alg_name == "AlphaEdit":
        for i, layer in enumerate(hparams.layers):
            P[i,:,:] = get_project(model,tok,layer,hparams)
        torch.save(P, "null_space_project.pt")
    # hs = get_module_input_output_at_words(
    #         model,
    #         tok,
    #         hparams.layers[-1],
    #         context_templates=[request["template"] for request in eval_ds],
    #         words=[request["subject"] for request in eval_ds],
    #         module_template=hparams.layer_module_tmp,
    #         fact_token_strategy=hparams.fact_token,
    #     )[1].T
    # torch.save(hs, "pre_edit_hs.pt")
    # del hs
    glue_save_location = str(run_dir) + '/' + 'glue_eval/'
    os.makedirs(glue_save_location, exist_ok=True)
    cnt = 0
    for record_chunks in chunks(ds, num_edits):
        case_result_template = str(run_dir / "{}_edits-case_{}.json")
        print(f"=================================================================={cnt+1}_edit==================================================================")
        # Is the chunk already done?
        already_finished = True
        for record in record_chunks:
            if not Path(
                case_result_template.format(num_edits, record["case_id"])
            ).exists():
                already_finished = False
                break
        if already_finished:
            continue
        
        # Compute weight changes + record weights that changed
        case_ids = [record["case_id"] for record in record_chunks]
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
        etc_args = dict(cache_template=cache_template) if any(alg in alg_name for alg in ["ROME", "MEMIT","AlphaEdit", "MEMIT_seq", "MEMIT_prune", "FedLEKE"]) else dict()
        seq_args = dict(cache_c=cache_c) if any(alg in alg_name for alg in ["AlphaEdit", "MEMIT_seq", "FedLEKE"]) else dict()
        nc_args = dict(P = P) if any(alg in alg_name for alg in ["AlphaEdit"]) else dict()
        #if cnt == 0 and args.downstream_eval_steps > 0:#do initial GLUE EVAL WITH ORIGINAL MODEL
        #    glue_results = {'edit_num': -1}
#
        #    out_file = glue_save_location + "base.json"
        #    
        #    glue_eval = GLUEEval(model, tok, number_of_tests = 100)
         #   glue_results = glue_eval.evaluate(glue_results, out_file, nli_flag = True, sst_flag = True, cola_flag=True, rte_flag=True, mmlu_flag = True, mrpc_flag = True)

         #   #store the individual overall result file
         #   output_filename = out_file.replace('.json', '_glue.json')
         #   with open(output_filename, "w") as f:
          #      json.dump(glue_results, f, indent=4)
        start = time()
        if any(alg in alg_name for alg in ["AlphaEdit", "MEMIT_seq", "FedLEKE"]):
            edited_model, cache_c = apply_algo(
                model,
                tok,
                [
                    {"case_id": record["case_id"], **rewrite_dict}
                    for record in record_chunks
                    for rewrite_dict in (
                        record["requested_rewrite"]
                        if isinstance(record["requested_rewrite"], list)
                        else [record["requested_rewrite"]]
                    )
                ],
                hparams,
                **args_conserve_memory,
                **etc_args,
                **seq_args,
                **nc_args,
            )
        elif alg_name == "MEMIT_prune":
            if cnt == 0:
                edited_model, weights_copy = apply_algo(
                    model,
                    tok,
                    [
                        {"case_id": record["case_id"], **rewrite_dict}
                        for record in record_chunks
                        for rewrite_dict in (
                            record["requested_rewrite"]
                            if isinstance(record["requested_rewrite"], list)
                            else [record["requested_rewrite"]]
                        )
                    ],
                    hparams,
                    return_orig_weights=True,
                    **args_conserve_memory,
                    **etc_args,
                )
                # Initialize the upd_matrix dictionary
                upd_matrix = {}
            else:
                edited_model, _ = apply_algo(
                    model,
                    tok,
                    [
                        {"case_id": record["case_id"], **rewrite_dict}
                        for record in record_chunks
                        for rewrite_dict in (
                            record["requested_rewrite"]
                            if isinstance(record["requested_rewrite"], list)
                            else [record["requested_rewrite"]]
                        )
                    ],
                    hparams,
                    return_orig_weights=False,
                    **args_conserve_memory,
                    **etc_args,
                )
            if cnt == (dataset_size_limit/num_edits) - 1:
            # Calculate the weight update matrix
                with torch.no_grad():
                    for k, v in weights_copy.items():
                        current_weight = nethook.get_parameter(model, k)
                        upd_matrix[k] = current_weight - v.to("cuda")
                        # Calculate max singular value of the original weight
                        _, S_orig, _ = torch.svd(v)
                        max_sigma = S_orig.max().item()

                        # Adjust the upd_matrix singular values
                        U_upd, S_upd, V_upd = torch.svd(upd_matrix[k])
                        adjusted_S = torch.where(
                            S_upd > max_sigma,
                            torch.log(S_upd) - torch.log(torch.tensor(max_sigma, device='cuda')) + max_sigma,
                            S_upd
                        )
                        upd_matrix[k] = torch.matmul(U_upd, torch.matmul(torch.diag(adjusted_S), V_upd.t()))

                # Apply the adjusted updates to the model
                with torch.no_grad():
                    for k in upd_matrix:
                        original_weight = nethook.get_parameter(model, k)
                        adjusted_weight = original_weight + upd_matrix[k]
                        original_weight.copy_(adjusted_weight)
        else:
            edited_model, _ = apply_algo(
                model,
                tok,
                [
                    {"case_id": record["case_id"], **rewrite_dict}
                    for record in record_chunks
                    for rewrite_dict in (
                        record["requested_rewrite"]
                        if isinstance(record["requested_rewrite"], list)
                        else [record["requested_rewrite"]]
                    )
                ],
                hparams,
                return_orig_weights=False,
                **args_conserve_memory,
                **etc_args,
            )
        exec_time = time() - start
        cnt+=1
        print("Execution took", exec_time)
        # Evaluate new model
    
        if args.downstream_eval_steps > 0 and cnt % args.downstream_eval_steps == 0:
            glue_results = {
                        'edit_num': cnt*num_edits,
                        'case_id': case_ids
                        }

            out_file = glue_save_location + "case_{}.json".format(record["case_id"])#stores the last case ID of the batch

            glue_eval = GLUEEval(model, tok, number_of_tests = 100)
            glue_results = glue_eval.evaluate(glue_results, out_file, nli_flag = True, sst_flag = True, cola_flag=True, rte_flag=True, mmlu_flag = True, mrpc_flag = True)
                    
            #store the individual overall result file
            output_filename = out_file.replace('.json', '_glue.json')
            with open(output_filename, "w") as f:
                json.dump(glue_results, f, indent=4)
    # hs = get_module_input_output_at_words(
    #         edited_model,
    #         tok,
    #         hparams.layers[-1],
    #         context_templates=[request["template"] for request in eval_ds],
    #         words=[request["subject"] for request in eval_ds],
    #         module_template=hparams.layer_module_tmp,
    #         fact_token_strategy=hparams.fact_token,
    #     )[1].T
    # torch.save(hs, "post_edit_hs_memit.pt")
    start = time()
    gen_test_vars = [snips, vec]
    for record in ds:
        out_file = Path(case_result_template.format(num_edits, record["case_id"]))
        if out_file.exists():
            print(f"Skipping {out_file}; already exists")
            continue
        metrics = {
            "case_id": record["case_id"],
            "grouped_case_ids": case_ids,
            "num_edits": num_edits,
            "requested_rewrite": record["requested_rewrite"],
            "time": exec_time,
            "post": ds_eval_method(
                edited_model,
                tok,
                record,
                *(
                    gen_test_vars
                    if record["case_id"] % generation_test_interval == 0
                    else [None, None]
                ),  # Only test generation every generation_test_interval cases
            ),
        }
        # Dump metrics in .json
        with open(out_file, "w") as f:
            json.dump(metrics, f, indent=1)

        # Restore original weights
        # with torch.no_grad():
        #     for k, v in weights_copy.items():
        #         nethook.get_parameter(model, k)[...] = v.to("cuda")

        print("Evaluation took", time() - start)
def get_project(model, tok, layer, hparams):
    force_recompute = False
    cov = get_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples
        if not force_recompute
        else hparams.mom2_n_samples // 10,
        hparams.mom2_dtype,
        force_recompute=force_recompute,
    ).cpu()
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    threshold = hparams.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    print(len(small_singular_indices))
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T
def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["AlphaEdit","MEMIT_rect", "MEMIT_seq","MEMIT_prune", "MEMIT", "ROME", "FT", "MEND","FedLEKE"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["mcf", "cf", "zsre", "mquake"],
        default="mcf",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--downstream_eval_steps",
        type=int,
        default=0,
        help="If we want to do sequential editing or not",
    )
    parser.add_argument(
        "--federated_mode",
        action="store_true",
        help="启用联邦采样模式，以构建具有特定数据分布的数据集。",
    )
    parser.add_argument(
        "--federated_index_path",
        type=str,
        default=None,
        help="用于联邦模式的、包含关系相似度信息的JSON文件路径。",
    )
    parser.add_argument(
        "--primary_relation_id",
        type=str,
        default=None,
        help="在联邦模式中，作为'本地'数据源的关系ID。",
    )
    parser.add_argument(
        "--num_similar_relations",
        type=int,
        default=3,
        help="从多少个最相似的关系中采样'非本地'数据。",
    )
    parser.add_argument(
        "--local_ratio_in_batch",
        type=float,
        default=0.9,
        help="每个批次中本地（主关系）数据的期望比例。",
    )
    parser.add_argument(
        "--total_dataset_size",
        type=int,
        default=None,
        help="在联邦模式下要构建的数据集的总大小。",
    )




    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.generation_test_interval,
        args.conserve_memory,
        dir_name=args.alg_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
        # 在这里传递新的参数
        federated_mode=args.federated_mode,
        federated_index_path=args.federated_index_path,
        primary_relation_id=args.primary_relation_id,
        num_similar_relations=args.num_similar_relations,
        local_ratio_in_batch=args.local_ratio_in_batch,
        total_dataset_size=args.total_dataset_size,
    )
