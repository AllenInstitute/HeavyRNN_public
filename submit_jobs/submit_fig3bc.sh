#!/bin/bash
#SBATCH --job-name=fig3bc
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/dev/null
#SBATCH --partition=braintv

# === Define experiment parameters ===
input_type=noise
hidden_size=1000
batch_size=1
k_LE=${hidden_size}
alphas="1.0 1.5 2.0"
initializations=levy
warmup_LE=2850        # number of steps to discard for LE
warmup_PR=2850        # number of steps to discard for PR
K_LD=100 # number of steps accumulated in the latter part of the sequence to compute Lyapunov dimension
PR_overhead=100 # number of extra steps used to compute PR so overhead + hidden_size = K_PR
trials=3
seed=42
save_root=neurips_results

# === Create logs directories and build log filename ===
mkdir -p logs/fig3bc/errors/
mkdir -p logs/fig3bc/output/
TLE=$(( warmup_LE + K_LD ))
TPR=$(( warmup_PR + hidden_size + PR_overhead ))
KPR=$(( hidden_size + PR_overhead ))
log_suffix="type=${input_type}_h=${hidden_size}_kLE=${k_LE}_alpha=$(echo ${alphas} | tr ' ' '-')_TLE=${TLE}_KLD=${K_LD}_TPR=${TPR}_KPR=${KPR}_tr=${trials}_sd=${seed}"
log_suffix=${log_suffix// /}  # remove spaces

# Redirect stdout/stderr
exec > logs/fig3bc/output/${log_suffix}.out 2> logs/fig3bc/errors/${log_suffix}.err

# === Load conda environment ===
source ~/miniconda3/etc/profile.d/conda.sh
conda activate heavyRNN

# === Run the Fig.4 participation‐ratio & Lyapunov‐dimension script ===
python run_figs/fig3bc_participationRatio_LypDim.py \
  --input_type       ${input_type} \
  --hidden_size      ${hidden_size} \
  --batch_size       ${batch_size} \
  --k_LE             ${k_LE} \
  --alphas           ${alphas} \
  --initializations  ${initializations} \
  --trials           ${trials} \
  --seed             ${seed} \
  --save_root        ${save_root} \
  --warmup_LE        ${warmup_LE} \
  --warmup_PR        ${warmup_PR} \
  --PR_overhead      ${PR_overhead} \
  --K_LD             ${K_LD} \
  --cache
