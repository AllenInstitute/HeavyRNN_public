#!/usr/bin/env bash
set -ex

source ~/miniconda3/etc/profile.d/conda.sh
conda activate heavyRNN

python run_figs/fig1a.py
python run_figs/fig1bc_cauchy_annealed_vs_quenched.py -N 100
python run_figs/fig1bc_cauchy_annealed_vs_quenched.py -N 3000 -l

