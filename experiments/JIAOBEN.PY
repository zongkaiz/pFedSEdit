import os
import time
from datetime import datetime

# 基础命令模板
base_cmd = (
    "python3 -m experiments.evaluate "
    "--alg_name=MEMIT_rect "
    "--model_name=meta-llama/Meta-Llama-3-8B-Instruct "
    "--hparams_fname=Llama3-8B.json "
    "--ds_name=mcf "
    "--num_edits=100 "
    "--federated_mode "
    "--primary_relation_id=\"P27\" "
    "--num_similar_relations=3 "
    "--local_ratio_in_batch=0.9 "
    "--downstream_eval_steps={eval_steps} "
    "--total_dataset_size={dataset_size}"
)

# 定义参数范围（100到2000，步长100，共20个任务）
dataset_sizes = list(range(100, 2001, 100))
total_tasks = len(dataset_sizes)

# 记录开始时间
start_time = datetime.now()
print(f"===== 开始执行所有任务，共 {total_tasks} 个 =====")
print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

for i, dataset_size in enumerate(dataset_sizes, 1):
    eval_steps = dataset_size // 100
    cmd = base_cmd.format(eval_steps=eval_steps, dataset_size=dataset_size)
    
    # 打印当前任务信息（进度+参数）
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{current_time}] 任务 {i}/{total_tasks} 开始:")
    print(f"  total_dataset_size = {dataset_size}")
    print(f"  downstream_eval_steps = {eval_steps}")
    print(f"  执行命令: {cmd[:50]}...")  # 只显示命令前50字符，避免过长
    
    # 执行命令并记录耗时
    task_start = time.time()
    exit_code = os.system(cmd)  # 执行命令，返回退出码（0表示成功）
    task_end = time.time()
    task_duration = round((task_end - task_start) / 60, 2)  # 转换为分钟
    
    # 打印任务结果
    status = "成功" if exit_code == 0 else "失败"
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 任务 {i}/{total_tasks} {status}，耗时 {task_duration} 分钟\n")

# 所有任务完成后汇总
end_time = datetime.now()
total_duration = round((end_time - start_time).total_seconds() / 3600, 2)  # 转换为小时
print(f"===== 所有 {total_tasks} 个任务执行完毕 =====")
print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"总耗时: {total_duration} 小时")