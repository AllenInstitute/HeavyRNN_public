#!/bin/bash
# ensure top-level log dirs exist
mkdir -p logs/fig2/output logs/fig2/errors

input_types=(zeros noise)
hidden_sizes=(1000)
kLEs=(100)
alphas=(0.5 1.0 1.5 2.0)
trials=(0)

for input_type in "${input_types[@]}"; do
  for hidden_size in "${hidden_sizes[@]}"; do
    for kLE in "${kLEs[@]}"; do
      for alpha in "${alphas[@]}"; do
        for trial in "${trials[@]}"; do
          sbatch \
            --job-name="fig2_${input_type}_h${hidden_size}_kLE${kLE}_a${alpha}_t${trial}" \
            --export=input_type="${input_type}",\
hidden_size="${hidden_size}",\
kLE="${kLE}",\
alpha="${alpha}",\
trial="${trial}" \
            submit_jobs/execute_fig2_parallel.sbatch \
          > /dev/null
        done
      done
    done
  done
done
