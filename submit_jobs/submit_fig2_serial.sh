#!/bin/bash
#SBATCH --job-name=fig2
#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --output=/dev/null
#SBATCH --partition=braintv

# === Define experiment parameters ===
input_type=zeros
hidden_size=5_000
batch_size=1
k_LE=5_000
alphas="0.5 1.0 1.5 2.0"
initializations=levy
warmup=2900
sequence_length=3000

# === Create logs directory and build log filename ===
mkdir -p logs/errors/fig2/
mkdir -p logs/output/fig2/
log_suffix="type=${input_type}_h=${hidden_size}_kLE=${k_LE}_alpha=$(echo ${alphas} | tr ' ' '-')_init=${initializations}_warmup=${warmup}_seq=${sequence_length}"
log_suffix=${log_suffix// /}  # remove any spaces

# === Redirect stdout and stderr to custom log files ===
exec > logs/output/fig2/${log_suffix}.out 2> logs/errors/fig2/${log_suffix}.err

# === Load conda environment ===
source ~/miniconda3/etc/profile.d/conda.sh
conda activate heavyRNN

# === Run the Python experiment ===
python run_figs/fig2_experiment_max_lambda_vs_g.py \
  --input_type ${input_type} \
  --hidden_size ${hidden_size} \
  --batch_size ${batch_size} \
  --k_LE ${k_LE} \
  --alphas ${alphas} \
  --initializations ${initializations} \
  --warmup ${warmup} \
  --sequence_length ${sequence_length}
