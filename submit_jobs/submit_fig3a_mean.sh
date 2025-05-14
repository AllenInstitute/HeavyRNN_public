#!/bin/bash
#SBATCH --job-name=fig3a_mean
#SBATCH --time=24:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=4
#SBATCH --output=/dev/null
#SBATCH --partition=braintv

# === Define experiment parameters ===
input_type=zeros
hidden_size=1000
batch_size=1
k_LE=100
alphas="1.00 1.50 2.00"
initializations=levy
warmup=3900
sequence_length=4000
seed=40 # do not change; changed from 0 to 40 to align with fig2
n_trials=3

# === Create logs directory and build log filename ===
mkdir -p logs/fig3a_mean/errors/
mkdir -p logs/fig3a_mean/output/
log_suffix="type=${input_type}_h=${hidden_size}_kLE=${k_LE}_alpha=$(echo ${alphas} | tr ' ' '-')_init=${initializations}_seq=${sequence_length}_meanOverlay"
log_suffix=${log_suffix// /}  # remove spaces

# === Redirect stdout and stderr to custom log files ===
exec > logs/fig3a_mean/output/${log_suffix}.out 2> logs/fig3a_mean/errors/${log_suffix}.err

# === Load conda environment ===
source ~/miniconda3/etc/profile.d/conda.sh
conda activate heavyRNN

# === Run the Python script ===
python run_figs/fig3a_mean.py \
  --input_type         ${input_type} \
  --hidden_size        ${hidden_size} \
  --batch_size         ${batch_size} \
  --k_LE               ${k_LE} \
  --alphas             ${alphas} \
  --initializations    ${initializations} \
  --warmup             ${warmup} \
  --sequence_length    ${sequence_length} \
  --seed               ${seed} \
  --n_trials           ${n_trials} \
  --exclude_outliers \
#   --use_cache
