#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --partition=braintv
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

source ~/miniconda3/etc/profile.d/conda.sh
conda activate heavyRNN

comb="f3bc_${metric}_${input_type}_h${hidden_size}_k${kLE}"
out_dir="logs/fig3bc/output/${comb}"
err_dir="logs/fig3bc/errors/${comb}"
mkdir -p "${out_dir}" "${err_dir}"

exec > "${out_dir}/a${alpha}_t${trial}.out" \
     2> "${err_dir}/a${alpha}_t${trial}.err"

# pick the correct warmup + overhead flag for your metric
if [[ "${metric}" == "PR" ]]; then
  extra_flags="--warmup_PR ${warmup_PR} --PR_overhead ${PR_overhead}"
else
  extra_flags="--warmup_LE ${warmup_LE} --K_LD ${K_LD}"
fi

python run_figs/fig3bc_parallel.py compute \
  --metric         ${metric} \
  --input_type     ${input_type} \
  --hidden_size    ${hidden_size} \
  --k_LE           ${kLE} \
  --alpha          ${alpha} \
  --trial          ${trial} \
  --initialization levy \
  --save_root      neurips_results \
  ${extra_flags}
