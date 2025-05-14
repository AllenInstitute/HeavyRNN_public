#!/usr/bin/env python3
# Fig3bc_participationRatio_LypDim.py
# Compute and plot mean ± std of Participation Ratio and Lyapunov Dimension vs. g
# over multiple trials, overlaying multiple α values.

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LypModel import RNNModel
import LypAlgo as lyap
from utils import set_seed, title_fontsize, legend_fontsize, label_fontsize, ticklabel_fontsize

import pickle
from pathlib import Path

# Color mapping for different alpha values
COLOR = {
    0.75:  '#56B4E9',
    1.0:  '#E69F00',
    1.5:  '#009E73',
    2.0:  '#CC79A7',
}

def compute_metrics(args, initialization):
    """
    Runs `args.trials` independent trials, for each α and each g in g_values,
    and returns:
      - g_values: np.ndarray, shape (n_g,)
      - metrics: dict[alpha] -> {'PR': array(trials, n_g),
                                 'LD': array(trials, n_g)}
    """
    # 1) choose g_values
    if args.g_values:
        g_values = np.array(args.g_values, dtype=float)
    else:
        g_values = np.logspace(-1, 1, args.n_g)

    n_g = len(g_values)
    trials = args.trials
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 2) prepare input once
    
    # 3) initialize storage
    metrics = {
        alpha: {
            'PR':  np.zeros((trials, n_g), dtype=float),
            'LD':  np.zeros((trials, n_g), dtype=float)
        }
        for alpha in args.alphas
    }

    # 4) loop over trials
    for trial in range(trials):
        seed = args.seed + trial
        set_seed(seed)
        if args.input_type == 'zeros':
            input_size = 1
            le_input = torch.zeros(args.batch_size,
                                args.sequence_length,
                                input_size)
        elif args.input_type == 'noise':
            input_size = args.hidden_size
            le_input = torch.randn(args.batch_size,
                                args.sequence_length,
                                input_size) * args.scale_noise
        else:
            raise ValueError(f"Unknown input_type: {args.input_type}")
        le_input = le_input.to(device)

        # for each alpha
        for alpha in args.alphas:
            for idx_g, g in enumerate(tqdm(g_values,
                                           desc=f"init={initialization}  α={alpha}  trial={trial}")):
                # build & eval model
                model = RNNModel(
                    initialization=initialization,
                    input_size=input_size,
                    hidden_size=args.hidden_size,
                    g=g,
                    alpha=alpha,
                    nonlinearity='tanh',
                    data=None,
                    batch_first=True
                ).to(device)
                if args.input_type == 'zeros':
                    model.make_autonomous()
                else:
                    model.make_noise_driven()
                model.eval()

                # init hidden state
                h0 = model.init_hidden(batch_size=args.batch_size)

                # only do enough steps for LE convergence
                res_LE = lyap.calc_LEs_an(
                    input=le_input[:, :args.warmup_LE + args.K_LD + 1, :],
                    initial_hidden=h0,
                    model=model,
                    k_LE=args.k_LE,
                    warmup=args.warmup_LE,
                    compute_LE=True,
                    include_PR=False,
                    store_all_states=False,
                    get_jacobian=False,
                )
                mean_LE, _ = lyap.LE_stats(res_LE['LE'])
                metrics[alpha]['LD'][trial, idx_g] = lyap.lyapunov_dimension(mean_LE)

                # ---- compute Participation Ratio ----
                # only do as many steps as needed to get enough hidden states
                res_PR = lyap.calc_LEs_an(
                    input=le_input[:, :args.warmup_PR + args.hidden_size + args.PR_overhead + 1, :],
                    initial_hidden=h0,
                    model=model,
                    k_LE=1,                  # skip all QR work
                    warmup=args.warmup_PR,
                    compute_LE=False,        # no LE loops
                    include_PR=True,
                    store_all_states=False,  # we only need h_all for PR
                )
                metrics[alpha]['PR'][trial, idx_g] = float(res_PR['PR'])

                torch.cuda.empty_cache()

    return g_values, metrics


def plot_with_errorbars(g_values, metrics, initialization, args):
    """
    Given g_values and metrics dict from compute_metrics, plot two figures:
      1) Participation Ratio vs g (mean ± std)
      2) Lyapunov Dimension vs g (mean ± std)
    and save into args.save_root/…/initialization/.
    """
    T_PR = args.warmup_PR + args.hidden_size + args.PR_overhead
    K_PR = args.hidden_size + args.PR_overhead
    T_LD = args.warmup_LE + args.K_LD
    K_LD = args.K_LD

    # Output directory for PR
    pr_out_dir = (
        f"{args.save_root}/"
        f"hidden{args.hidden_size}/{args.input_type}/"
        f"kLE{args.k_LE}/Fig3bc_participation_LyapDim_overlay/"
        f"K_PR{K_PR}/T_PR{T_PR}/"
    )
    os.makedirs(pr_out_dir, exist_ok=True)

    # Output directory for LD
    ld_out_dir = (
        f"{args.save_root}/"
        f"hidden{args.hidden_size}/{args.input_type}/"
        f"kLE{args.k_LE}/Fig3bc_participation_LyapDim_overlay/"
        f"K_LD{K_LD}/T_LD{T_LD}/"
    )
    os.makedirs(ld_out_dir, exist_ok=True)

    # 1) Participation Ratio
    plt.figure(figsize=(5, 4))
    plt.xscale('log')
    for alpha in args.alphas:
        pr_trials = metrics[alpha]['PR']       # shape (trials, n_g)
        mean_pr = pr_trials.mean(axis=0)
        std_pr  = pr_trials.std(axis=0, ddof=0)
        plt.plot(g_values, mean_pr,
                 label=fr'$\alpha$={alpha}',
                 color=COLOR.get(alpha))
        plt.fill_between(g_values,
                         mean_pr - std_pr,
                         mean_pr + std_pr,
                         color=COLOR.get(alpha),
                         alpha=0.15,
                         linewidth=0,)
    plt.xlabel('$g$', fontsize=label_fontsize)
    plt.ylabel('Participation Ratio', fontsize=label_fontsize)
    title_str = f"in autonomous RNN" if args.input_type == 'zeros' else f"in noise-driven RNN"
    plt.title(f'Participation ratio', fontsize=title_fontsize)
    plt.legend(fontsize=legend_fontsize, loc='upper left')
    if args.input_type == 'noise':
        plt.legend(fontsize=legend_fontsize, loc='upper right')
    plt.tick_params(axis='both', which='major', labelsize=ticklabel_fontsize)
    # pr_path = os.path.join(out_dir, f'PR_vs_g_{args.input_type}_{args.hidden_size}_warmup{args.warmup_PR}_overhead{args.overhead}.png')
    # plt.savefig(pr_path, dpi=1000, bbox_inches='tight')
    pr_path = os.path.join(pr_out_dir, f'PR_vs_g_{args.input_type}_{args.hidden_size}_T{T_PR}_K{args.hidden_size + args.PR_overhead}.pdf')
    plt.savefig(pr_path, dpi=1000, bbox_inches='tight')
    plt.close()
    print(f"Saved Participation Ratio plot to {pr_path}")

    # 2) Lyapunov Dimension
    plt.figure(figsize=(5, 4))
    plt.xscale('log')
    for alpha in args.alphas:
        ld_trials = metrics[alpha]['LD']       # shape (trials, n_g)
        mean_ld = ld_trials.mean(axis=0)
        std_ld  = ld_trials.std(axis=0, ddof=0)
        plt.plot(g_values, mean_ld,
                 label=fr'$\alpha$={alpha}',
                 color=COLOR.get(alpha))
        plt.fill_between(g_values,
                         mean_ld - std_ld,
                         mean_ld + std_ld,
                         color=COLOR.get(alpha),
                         alpha=0.15,
                         linewidth=0,)
    plt.xlabel('$g$', fontsize=label_fontsize)
    plt.ylabel('Lyapunov Dimension', fontsize=label_fontsize)
    plt.title(f'Lyapunov dimension', fontsize=title_fontsize)
    plt.legend(fontsize=legend_fontsize, loc='upper left')
    plt.tick_params(axis='both', which='major', labelsize=ticklabel_fontsize)

    T_LD = args.warmup_LE + args.K_LD
    # ld_path = os.path.join(out_dir, f'LD_vs_g_{args.input_type}_{args.hidden_size}T{args.warmup_LE}_overhead{args.overhead}.png')
    # plt.savefig(ld_path, dpi=1000, bbox_inches='tight')
    ld_path = os.path.join(ld_out_dir, f'LD_vs_g_{args.input_type}_{args.hidden_size}_T{T_LD}_K{args.K_LD}.pdf')
    plt.savefig(ld_path, dpi=1000, bbox_inches='tight')
    plt.close()
    print(f"Saved Lyapunov Dimension plot to {ld_path}")

def _cache_path(args, initialization) -> Path:
    """
    Construct a unique filename that captures every hyper‑parameter
    which changes the output of compute_metrics().
    """
    alpha_part = "_".join(map(str, args.alphas))
    g_part     = (
        "custom_" + "_".join(map("{:.3g}".format, args.g_values))
        if args.g_values else f"log{args.n_g}"
    )
    T_PR = args.warmup_PR + args.hidden_size + args.PR_overhead
    K_PR = args.hidden_size + args.PR_overhead
    T_LD = args.warmup_LE + args.K_LD
    K_LD = args.K_LD

    fname = (
        f"PRLD_cache_"
        f"{initialization}_"
        f"N{args.hidden_size}_"
        f"{args.input_type}_"
        f"alpha[{alpha_part}]_"
        f"g[{g_part}]_"
        f"T_PR{T_PR}_K_PR{K_PR}_"
        f"T_LD{T_LD}_K_LD{K_LD}_"
        f"trials{args.trials}_"
        f"seed{args.seed}.pkl"
    )
    return Path(args.save_root, "cache", fname)

# === replace the body of main() ===
def main(args):
    args.sequence_length = args.warmup_PR + args.hidden_size + args.PR_overhead + 1
    for init in args.initializations:
        cache_file = _cache_path(args, init)
        if args.cache and cache_file.is_file():
            print(f"[cache] Loading {cache_file}")
            with cache_file.open("rb") as f:
                g_vals, metrics = pickle.load(f)
        else:
            g_vals, metrics = compute_metrics(args, initialization=init)
            # make sure the cache directory exists
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with cache_file.open("wb") as f:
                pickle.dump((g_vals, metrics), f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[cache] Saved new results to {cache_file}")

        plot_with_errorbars(g_vals, metrics, initialization=init, args=args)
    print("Done!")


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)

    parser = argparse.ArgumentParser(
        description='Overlay mean±std of Participation Ratio & Lyapunov Dimension vs g'
    )
    parser.add_argument('--input_type',       type=str,   default='zeros',
                        help='"zeros" or "noise"')
    parser.add_argument('--hidden_size',      type=int,   default=1000)
    parser.add_argument('--batch_size',       type=int,   default=1)
    parser.add_argument('--k_LE',             type=int,   default=100)
    parser.add_argument('--warmup',           type=int,   default=2900)
    # parser.add_argument('--sequence_length',  type=int,   default=3000)
    parser.add_argument('--n_g',              type=int,   default=50,
                        help='Number of default log-spaced g values if --g_values is not set')
    parser.add_argument('--g_values',         nargs='+',  type=float,
                        help='List of g values to test (overrides --n_g)')
    parser.add_argument('--alphas',           nargs='+',  type=float,
                        default=[1.0, 1.5, 2.0])
    parser.add_argument('--initializations',  nargs='+',  default=['levy'],
                        help='Weight initializations to test')
    parser.add_argument('--trials',           type=int,   default=3,
                        help='Number of independent trials')
    parser.add_argument('--seed',             type=int,   default=42)
    parser.add_argument('--save_root',        type=str,   default='neurips_results',
                        help='Root directory for saving all outputs')
    parser.add_argument('--scale_noise', type=float, default=0.1, help='scale the noisy input')
    
    parser.add_argument('--warmup_LE', type=int, default=2900,
                    help='Number of initial steps to discard before computing LEs')
    parser.add_argument('--warmup_PR', type=int, default=2900,
                    help='Number of initial steps to discard before collecting states for PR')
    parser.add_argument("--cache", action="store_true",
                    help="If set, try to load a cached g_values / metrics pair before "
                        "running the simulations.  Fresh results are always cached.")
    parser.add_argument('--PR_overhead', type=int, default=50,
                        help='Number of extra steps to use for computing the PR such that K = N + overhead')
    parser.add_argument('--K_LD', type=int, default=50,
                        help='Number of extra T steps to use for computing LD')
    
    args = parser.parse_args()

    set_seed(args.seed)
    main(args)