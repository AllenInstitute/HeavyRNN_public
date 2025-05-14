#!/bin/bash
mkdir -p logs/fig3bc/output logs/fig3bc/errors

input_types=(zeros noise)
hidden_sizes=(1000)
kLEs=(1000)
alphas=(1.0 1.5 2.0)
trials=(0 1 2 3 4 5 6 7 8 9)
metrics=(PR LD)

# define your overhead & warmups once (or loop over multiple if you like)
PR_overheads=(50)
K_LDs=(50)
warmup_PRs=(2900)
warmup_LEs=(2900)

for metric in "${metrics[@]}"; do
  for input_type in "${input_types[@]}"; do
    for hidden_size in "${hidden_sizes[@]}"; do
      for kLE in "${kLEs[@]}"; do
        for alpha in "${alphas[@]}"; do
          for trial in "${trials[@]}"; do
            for PR_overhead in "${PR_overheads[@]}"; do
              for K_LD in "${K_LDs[@]}"; do
                for warmup_PR in "${warmup_PRs[@]}"; do
                  for warmup_LE in "${warmup_LEs[@]}"; do

                    sbatch \
                      --job-name="f3bc_${metric}_${input_type}_h${hidden_size}_k${kLE}_a${alpha}_t${trial}" \
                      --export=metric="${metric}",\
input_type="${input_type}",\
hidden_size="${hidden_size}",\
kLE="${kLE}",\
alpha="${alpha}",\
trial="${trial}",\
PR_overhead="${PR_overhead}",\
K_LD="${K_LD}",\
warmup_PR="${warmup_PR}",\
warmup_LE="${warmup_LE}" \
                      submit_jobs/execute_fig3bc_parallel.sbatch \
                    > /dev/null

                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
