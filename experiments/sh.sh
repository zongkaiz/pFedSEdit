nohup python3 -m experiments.evaluate \
    --alg_name=FedLEKE \
    --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
    --hparams_fname=Llama3-8B.json \
    --ds_name=mcf \
    --dataset_size_limit=10 \
    --num_edits=2 \
    --downstream_eval_steps=30 > /media/h3c/users/zongkai/AlphaEdit-main/experiments/mcf_llama_test.log 2>&1 &