# Slow Transition to Low-Dimensional Chaos in Heavy-Tailed Recurrent Neural Networks

## Set up the environment
Python versions >=3.10 and <3.13 are recommended. Note that `jax`, and `jaxlib` likely take the most time to install, so you may remove them from `requirements.txt` if you don't run Fig 1.
```
conda create -n heavyRNN python=3.11 -y
```
```
conda activate heavyRNN
```
Lastly, install the necessary packages
```
pip install -r requirements.txt
```

**The following instruction provides commands assuming you are running in the root directory.**

## Fig 1
The relevant code for reproducing
* Fig1A: `run_figs/fig1a.py`
* Fig1B, Fig1C: `run_figs/fig1bc_cauchy_annealed_vs_quenched.py`

You can submit to run all panels of Fig. 1 using
```
./submit_jobs/submit_fig1.sh
```
To plot consequently saved data, use `run_figs/plot_fig1_with_saved_data.py`

## Fig 2
We offer two options to run Fig 2 
* serially using `run_figs/fig2_experiment_max_lambda_vs_g.py` (which will take very long when $N$ or number of $g$s are large) or, 
* in parallel using `run_figs/fig2_parallel.py` (recommended).

You can submit cluster jobs for the serial script using
```
sbatch submit_jobs/submit_fig2_serial.sh
```
or for the parallel script using
```
bash submit_jobs/submit_fig2_parallel.sh
```
After the parallel jobs are done, you may plot the figures using saved data using
```
python3 run_figs/fig2_parallel.py plot --input_type {ENTER} --hidden_size {ENTER}
```
where you fill {ENTER} with your own customization.
* `input_type`: either `zeros` for autonomous RNN or `noise` for noisy stimulus-driven RNN; you can pass in standard deviation of noisy input using `args.scale_noise` (default $0.1$ in the paper).

You may also change other parser arguments, such as 
* `sequence_length`: determines how many total time steps to run.
* `warmup`: determines how many first `warmup` steps to discard
* This means the results are over the *last* `sequence_length` minus `warmup` steps.
Or
* `k_LE`: number of top Lyponov exponents to compute

## Fig 3A
The parser arguments across Fig. 2 and Fig. 3 are largely consistent. You may directly run `run_figs/fig3a_mean.py` to reproduce Fig. 3A in the main text for the top 100 Lyapunov exponents (`k_LE`) or run the following:
```
sbatch submit_jobs/submit_fig3a_mean.sh
```
Here `--exclude_outliers` excludes outlier Lyapunov exponents that are more than four standard deviation away. An extremly negative one happens occasionally but will blur the visualization given most of them are near 0.

You may run the individual realization in the Appendix using `run_figs/fig3a_lypExponent_histogram_fix_g.py` or an easier way is to submit
```
sbatch submit_jobs/submit_fig3a_individual.sh
```
with appropriate adjustments to the hyperparameter in the .sh file.

## Fig 3B, 3C
Similarly, you may run `run_figs/fig3bc_participationRatio_LypDim.py` or submit via
```
sbatch submit_jobs/submit_fig3bc.sh
```
with appropriate adjustments to the hyperparameter in the .sh file.

Alternatively, for runs with large computational costs (e.g., multiple trials and networks larger than N=1000), you may optionally run different alphas and trials parallely by submitting via
```
bash submit_jobs/submit_fig3bc_parallel.sh
```
Then plot the relevant results using
```
python3 run_figs/fig3bc_parallel.py plot --metric {PR or LD} --input_type {zeros or noise} --hidden_size {ENTER} --k_LE {should be the same as `hidden_size`}
```

## Citation
```
@misc{xie2025slowtransitionlowdimensionalchaos,
      title={Slow Transition to Low-Dimensional Chaos in Heavy-Tailed Recurrent Neural Networks}, 
      author={Yi Xie and Stefan Mihalas and Łukasz Kuśmierz},
      year={2025},
      eprint={2505.09816},
      archivePrefix={arXiv},
      primaryClass={q-bio.NC},
      url={https://arxiv.org/abs/2505.09816}, 
}
```
