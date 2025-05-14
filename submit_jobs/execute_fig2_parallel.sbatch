#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --partition=braintv

# disable Slurm's own slurm-*.out and slurm-*.err
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

source ~/miniconda3/etc/profile.d/conda.sh
conda activate heavyRNN

# build combo name and per-combo dirs
comb="fig2_${input_type}_h${hidden_size}_kLE${kLE}"
out_dir="logs/fig2/output/${comb}"
err_dir="logs/fig2/errors/${comb}"
mkdir -p "${out_dir}" "${err_dir}"

# redirect stdout/stderr into exactly the files you want
exec > "${out_dir}/a${alpha}_t${trial}.out" \
     2> "${err_dir}/a${alpha}_t${trial}.err"

# now run your compute step
python run_figs/fig2_parallel.py compute \
  --input_type      ${input_type} \
  --hidden_size     ${hidden_size} \
  --k_LE            ${kLE} \
  --alpha           ${alpha} \
  --trial           ${trial} \
  --warmup          2900 \
  --sequence_length 3000 \
  --initialization  levy \
  --save_dir        neurips_results
