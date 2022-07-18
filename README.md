# CHEMical Intelligent SysTem (CHEMIST) a VAE for de novo polypharmacology.

This repository contains the CHEMIST framework, a de novo molecular generator for polypharmacology. Akin to de novo portait generation, CHEMIST attempts to optimize the chemical space for multiple protein target domains.

![alt text](https://github.com/bpmunson/chemist/blob/main/images/220718_fig1A.png?raw=true)

***

The codebase is primarily adapted from two excellent de novo molecular design frameworks:

1. GuacaMol for reward based reinforcement learning: https://github.com/BenevolentAI/guacamol 

2. MOSES for the VAE implementation: https://github.com/molecularsets/moses



## Installation of CHEMIST:
```
git clone https://github.com/bpmunson/chemist.git

cd chemist

pip install .
```

optionally install cudatoolkit for gpu acceleration in pytorch
for example:
```
conda install cudatoolkit=11.1 -c conda-forge
```
or see https://pytorch.org/ for specific installation instructions.

***

## Example Usage:

Pretrain VAE to encode chemical embedding:
```
chemist train \
	--train_data ../data/guacamol_v1_train.smiles \
	--log_file log.txt \
	--save_frequency 25 \
	--model_save model.pt \
	--n_epoch 200 \
	--n_batch 1024 \
	--debug \
	--d_dropout 0.2 \
	--device cpu
```

Use the chemical embedding to design polypharmacology compounds
```
chemist generate \
    --model_path ../data/pretrained_vae_model.pt \
    --scoring_definition scoring_definition.csv \
    --max_len 100 \
    --n_epochs 200 \
    --mols_to_sample 8192   \
    --optimize_batch_size 512    \
    --optimize_n_epochs 2   \
    --keep_top 4096   \
    --opti gauss   \
    --outF molecular_generation   \
    --device cpu  \
    --save_payloads   \
    --n_jobs 4 \
    --debug
```
