#!/bin/bash
#SBATCH --job-name=fig3a
#SBATCH --time=5:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=4
#SBATCH --output=/dev/null
#SBATCH --partition=braintv

# === Define experiment parameters ===
input_type=noise
hidden_size=1000
batch_size=1
k_LE=100
alphas="1.00 1.50 2.00"
initializations=levy
warmup=2900
sequence_length=3000
seed=40

# === Create logs directory and build log filename ===
mkdir -p logs/fig3a/errors/
mkdir -p logs/fig3a/output/
log_suffix="type=${input_type}_h=${hidden_size}_kLE=${k_LE}_alpha=$(echo ${alphas} | tr ' ' '-')_init=${initializations}_seq=${sequence_length}"
log_suffix=${log_suffix// /}  # remove spaces

# === Redirect stdout and stderr to custom log files ===
exec > logs/fig3a/output/${log_suffix}.out 2> logs/fig3a/errors/${log_suffix}.err

# === Load conda environment ===
source ~/miniconda3/etc/profile.d/conda.sh
conda activate heavyRNN

# === Run the Python experiment ===
python run_figs/fig3a_lypExponent_histogram_fix_g.py \
  --input_type      ${input_type} \
  --hidden_size     ${hidden_size} \
  --batch_size      ${batch_size} \
  --k_LE            ${k_LE} \
  --alphas          ${alphas} \
  --initializations ${initializations} \
  --warmup          ${warmup} \
  --sequence_length ${sequence_length} \
  --seed            ${seed} \
  --exclude_outliers
