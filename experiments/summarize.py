import sys
import os

# 将项目的根目录添加到 Python 的模块搜索路径中
# __file__ 是当前脚本的路径
# os.path.dirname() 是获取路径的父目录
# 两层 dirname 就是父目录的父目录，也就是项目的根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# --- 原来的代码 ---
from util.globals import *
import collections
import json
from pprint import pprint
from typing import List, Optional
import sys
from pathlib import Path  # 确保导入Path

import numpy as np
from scipy.stats import hmean

from util.globals import *

def summarize(
    dir_name = None,
    runs: Optional[List] = None,
    first_n_cases=None,
    abs_path=False,
    get_uncompressed=False,
):  # runs = None -> all runs
    summaries = []
    uncompressed = []

    # 确定目标目录
    target_dir = RESULTS_DIR / dir_name if not abs_path else Path(dir_name)
    if not target_dir.exists():
        print(f"目录不存在: {target_dir}")
        return []

    for run_dir in target_dir.iterdir():
        # 跳过非目录文件
        if not run_dir.is_dir():
            continue

        # 按runs参数过滤
        if runs is not None and all(run not in str(run_dir) for run in runs):
            continue

        # 获取所有case_*.json文件并按文件名排序
        files = list(run_dir.glob("*case_*.json"))
        # 按文件名自然排序（确保case_1, case_2, ..., case_100的顺序）
        files.sort(key=lambda x: int(str(x).split("_")[-1].split(".")[0]))
        
        # 只保留前first_n_cases个文件（如果指定）
        if first_n_cases is not None and first_n_cases > 0:
            files = files[:first_n_cases]  # 取前N个文件
        
        # 如果没有符合条件的文件，跳过
        if not files:
            print(f"在{run_dir}中未找到case文件")
            continue

        cur_sum = collections.defaultdict(lambda: [])
        file_wise_results = {}
        for case_file in files:
            try:
                with open(case_file, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                print(f"解析错误，跳过文件: {case_file}")
                continue
            except Exception as e:
                print(f"处理文件{case_file}时出错: {e}")
                continue

            # 无需依赖case_id，直接处理所有筛选后的文件
            if "time" in data:
                cur_sum["time"].append(data["time"])

            # 以下是原有的指标计算逻辑（保持不变）
            for prefix in ["pre", "post"]:
                # Probability metrics for which new should be lower (better) than true
                for key in ["rewrite_prompts_probs", "paraphrase_prompts_probs"]:
                    if prefix not in data or key not in data[prefix]:
                        continue

                    sum_key_discrete = f"{prefix}_{key.split('_')[0]}_success"
                    sum_key_cont = f"{prefix}_{key.split('_')[0]}_diff"

                    cur_sum[sum_key_discrete].append(
                        np.mean(
                            [
                                x["target_true"] > x["target_new"]
                                for x in data[prefix][key]
                            ]
                        )
                    )
                    cur_sum[sum_key_cont].append(
                        np.mean(
                            [
                                np.exp(-x["target_new"]) - np.exp(-x["target_true"])
                                for x in data[prefix][key]
                            ]
                        )
                    )

                # Probability metrics for which true should be lower (better) than new
                sum_key_discrete = f"{prefix}_neighborhood_success"
                sum_key_cont = f"{prefix}_neighborhood_diff"
                key = "neighborhood_prompts_probs"
                if prefix in data and key in data[prefix]:
                    cur_sum[sum_key_discrete].append(
                        np.mean(
                            [
                                x["target_true"] < x["target_new"]
                                for x in data[prefix][key]
                            ]
                        )
                    )
                    cur_sum[sum_key_cont].append(
                        np.mean(
                            [
                                np.exp(-x["target_true"]) - np.exp(-x["target_new"])
                                for x in data[prefix][key]
                            ]
                        )
                    )

                # Accuracy-based evaluation metrics
                for key in ["rewrite", "paraphrase", "neighborhood"]:
                    sum_key = f"{prefix}_{key}_acc"
                    key = f"{key}_prompts_correct"

                    if prefix not in data or key not in data[prefix]:
                        continue

                    cur_sum[sum_key].append(np.mean(data[prefix][key]))

                # Generation metrics that can be directly averaged
                for key in ["ngram_entropy", "reference_score", "essence_score"]:
                    if prefix in data and key in data[prefix]:
                        cur_sum[f"{prefix}_{key}"].append(data[prefix][key])

        if len(cur_sum) == 0:
            print(f"{run_dir}中没有有效数据用于汇总")
            continue

        num_items = len(cur_sum[next(iter(cur_sum.keys()))])
        metadata = {
            "run_dir": str(run_dir),
            "num_cases": num_items,
        }

        uncompressed.append(dict(cur_sum, **metadata))

        cur_sum = {k: (np.mean(v), np.std(v)) for k, v in cur_sum.items()}
        for k, v in cur_sum.items():
            if all(exclude not in k for exclude in ["essence_score", "time"]):
                cur_sum[k] = tuple(np.around(z * 100, 2) for z in v)

        for prefix in ["pre", "post"]:
            for k_efficacy, k_generalization, k_specificity in [
                (
                    f"{prefix}_rewrite_success",
                    f"{prefix}_paraphrase_success",
                    f"{prefix}_neighborhood_success",
                ),
            ]:
                if all(k in cur_sum for k in [k_efficacy, k_generalization, k_specificity]):
                    hmean_list = [
                        cur_sum[k_efficacy][0],
                        cur_sum[k_generalization][0],
                        cur_sum[k_specificity][0],
                    ]
                    cur_sum[f"{prefix}_score"] = (hmean(hmean_list), np.nan)
                    break

        print(metadata)
        pprint(cur_sum)
        summaries.append(cur_sum)

    return uncompressed if get_uncompressed else summaries

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir_name", type=str, help="Name of directory to scan for runs."
    )
    parser.add_argument(
        "--runs",
        type=str,
        default=None,
        help="逗号分隔的run名称，仅评估这些run",
    )
    parser.add_argument(
        "--first_n_cases",
        type=int,
        default=None,
        help="只处理每个run下的前N个case文件",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="绝对路径模式（可选）",
    )
    args = parser.parse_args()

    summarize(
        args.dir_name,
        None if args.runs is None else args.runs.split(","),
        args.first_n_cases,
        args.path is not None  # 若提供--path则启用绝对路径
    )
