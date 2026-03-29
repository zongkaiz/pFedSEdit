# FedSLEKE: Federated Sequential Locate-then-Edit Knowledge Editing

## Requirements

- torch==2.6.0
- einops==0.8.1
- higher==0.2.1
- hydra-core==1.3.2
- transformers==4.51.3
- datasets==2.21.0
- matplotlib==3.10.3
- spacy==3.4.1
- scipy==1.15.2
- scikit-learn==1.6.1
- nltk==3.9.1

## Quick Start

### An example for editing Llama3 (8B) on mcf dataset using pFedMEMIT

**1. Edit Llama3 (8B) model**

```bash
python3 -m experiments.evaluate \
    --alg_name=FedSLEKE \
    --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
    --hparams_fname=Llama3-8B.json \
    --ds_name=mcf \
    --num_edits=200 \
    --downstream_eval_steps=30 \
    --federated_mode \
    --primary_relation_id="P27" \
    --num_similar_relations=3 \
    --total_dataset_size=2000 \
    --local_ratio_in_batch=0.9
```

This command runs an evaluation script for pFedMEMIT (MEMIT under our pFedSLEKE framework) using the Llama3-8B-Instruct.

**2. Summarize the results**

To summarize the results, you can use `experiments/summarize.py` :

```bash
python summarize.py --dir_name=FedSLEKE --runs=run_<run1>,run_<run2>
```

## Acknowledgment

Our code is based on [MEMIT](https://github.com/kmeng01/memit).
