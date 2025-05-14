import argparse
import gc
import numpy as np
import torch
import pickle as pkl
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LypModel import RNNModel
import LypAlgo as lyap
from utils import set_seed, exclude_outliers_std, COLOR, title_fontsize, legend_fontsize, label_fontsize, ticklabel_fontsize

def get_LE_array(args, alpha, g, cache_dir, trial_id):
    """
    Return a 1-D np.ndarray of finite Lyapunov exponents for (alpha, g).
    If the array has already been saved in `cache_dir`, just load it.
    Otherwise compute it once and save it as a .npy file.
    """
    os.makedirs(cache_dir, exist_ok=True)
    fpath = os.path.join(cache_dir, f"LE_alpha{alpha}_g{g:.4f}_trial{trial_id}.npy")

    if os.path.isfile(fpath) and args.use_cache:
        print('Loading cached LE array')
        return np.load(fpath)

    print("Cache miss: computing LE array")
    
    # --- not cached yet: build model & compute ---
    le_vals = plot_lambda_distribution_for_fixed_g(
        args, alpha=alpha, g=g, save_dir=None
    )
    np.save(fpath, le_vals)
    return le_vals

def plot_lambda_distribution_for_fixed_g(args, alpha, g, save_dir=None):
    """
    Build one model, compute Lyapunov exponents, and (optionally) save a
    stand-alone histogram if save_dir is not None. Always returns the
    1-D numpy array of finite λ values.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # choose input tensor
    if args.input_type == 'zeros':
        input_size = 1
        le_input = torch.zeros(args.batch_size, args.sequence_length, input_size,
                               device=device)
    else:  # 'noise'
        input_size = args.hidden_size
        le_input = torch.randn(args.batch_size, args.sequence_length, input_size,
                               device=device) * args.scale_noise

    # build network
    model = RNNModel(initialization='levy',
                     input_size=input_size,      
                     hidden_size=args.hidden_size,
                     g=g,
                     alpha=alpha,
                     nonlinearity='tanh').to(device)

    if args.input_type == 'zeros':
        model.make_autonomous()
    else:
        model.make_noise_driven()
    model.eval()

    h0 = model.init_hidden(batch_size=args.batch_size)

    res = lyap.calc_LEs_an(le_input[:1], h0, model,
                           k_LE=args.k_LE, warmup=args.warmup)
    LE_mean, _ = lyap.LE_stats(res['LE'])
    LE_mean = LE_mean.detach().cpu().numpy()
    LE_mean = LE_mean[np.isfinite(LE_mean)]

    # optional stand-alone figure
    if save_dir is not None:
        fig, ax = plt.subplots()
        ax.hist(LE_mean, bins=15, density=True, edgecolor='k',
                alpha=0.6, color=COLOR.get(alpha, None))
        ax.set_title(fr'$\alpha={alpha}$,  $g^*={g}$')
        fig.savefig(os.path.join(save_dir,
                    f'LE_distribution_alpha={alpha}_g={g}.png'), dpi=350)
        plt.close(fig)

    return LE_mean

def load_g_star_summary(json_path):
    """Return two dicts from the saved summary json."""
    with open(json_path) as f:
        d = json.load(f)
    return d

def load_g_star_trials(save_dir, args):
    """
    Return
        g_star_trials : dict  {trial_id: {alpha: g_star_or_None}}
        g_star_mean   : dict  {alpha: mean_g_star_ignoring_None}
    """
    base = (
        f"{save_dir}/hidden{args.hidden_size}/{args.input_type}"
        f"/kLE{args.k_LE}/warmup{args.warmup}/seq{args.sequence_length}"
        "/Fig2_MLE_varyg"
    )

    g_star_trials = {}
    for trial in range(args.n_trials):
        seed = args.seed + trial
        # jpath = os.path.join(base, f"g_cross_summary_seed{seed}.json")
        jpath = os.path.join(base, f"g_cross_summary.json")
        if not os.path.exists(jpath):
            raise FileNotFoundError(f"Missing {jpath}")
        with open(jpath) as f:
            g_star_trials[trial] = json.load(f)

    return g_star_trials

def try_load_g_star(save_dir, args):
    base_path = (
        f"{save_dir}/hidden{args.hidden_size}/{args.input_type}"
        f"/kLE{args.k_LE}/warmup{args.warmup}/seq{args.sequence_length}"
    )
    search_range = ["Fig2_MLE_varyg"]
    for fig in search_range:
        summary_json = os.path.join(base_path, fig, "g_cross_summary.json")
        if os.path.exists(summary_json):
            return load_g_star_summary(summary_json)

    raise FileNotFoundError(f"No g_cross_summary.json found in Fig1 or Fig2 under: {base_path}")

def main(args):
    save_dir = 'neurips_results'
    
    # summary_json = (
    #     f"{save_dir}/hidden{args.hidden_size}/{args.input_type}"
    #     f"/kLE{args.k_LE}/warmup{args.warmup}/seq{args.sequence_length}"
    #     "/Fig2_MLE_varyg/g_cross_summary.json"
    # )
    # g_star = load_g_star_summary(summary_json)
    g_star = try_load_g_star(save_dir, args)
    # g_star_trials = load_g_star_trials(save_dir, args)

    # ---- gather LE arrays for every α that truly crossed --
    le_dict = {}
    cache_root = (
        f"{save_dir}/hidden{args.hidden_size}/{args.input_type}"
        f"/kLE{args.k_LE}/warmup{args.warmup}/seq{args.sequence_length}"
        "/LE_cache"                         # <– new sub-folder just for LE arrays
    )
    # seeds = {0:1, 1:2, 2:15}
    for alpha in tqdm(args.alphas, desc="Loading / computing LE arrays"):
        g_alpha = float(g_star[str(alpha)])

        trial_vals = []
        for trial_id in tqdm(range(args.n_trials), desc=f"α={alpha}"):
            # g_alpha = g_star_trials[trial_id].get(str(alpha))
            if g_alpha is None:
                assert False, f"g_alpha is None for α={alpha} trial_id={trial_id}"
            set_seed(args.seed + trial_id)          # <‑‑ reproducible but different
            print(f'alpha={alpha}: Trial {trial_id}, g*={g_alpha}')
            le_vals = get_LE_array(args, alpha, g_alpha,
                                cache_root, trial_id)
            print(f"Loaded trial {trial_id} with {len(le_vals)} λ values")
            if args.exclude_outliers:
                le_vals = exclude_outliers_std(le_vals, k=4.0)
            trial_vals.append(le_vals)

        le_dict[alpha] = trial_vals

    if not le_dict:
        print("No alpha's crossed; nothing to plot."); return

    # ---- build a shared bin grid --------------------------
    # all_vals = np.concatenate(list(le_dict.values()))
    # n_bins   = 20                               # choose whatever resolution you like
    # bins     = np.linspace(all_vals.min(), all_vals.max(), n_bins + 1)
    all_vals = np.concatenate([np.concatenate(v) for v in le_dict.values()])
    bins     = np.linspace(all_vals.min(), all_vals.max()+ 1e-4, 80)

    # ---- single overlay plot ------------------------------
    out_dir = (
        f"neurips_results/hidden{args.hidden_size}/{args.input_type}"
        f"/kLE{args.k_LE}/warmup{args.warmup}/seq{args.sequence_length}"
        "/Fig3a_LE_distribution"
    )
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    all_mean = []
    for alpha, trials in le_dict.items():
        color = COLOR.get(alpha, None)
        # label = fr'$\alpha={alpha}\ (g^*\approx{round(g_star[str(alpha)],2)})$'
        label = fr'$\alpha={alpha}$ ($\langle g^* \rangle \approx {round(g_star[str(alpha)], 2)}$)'

        # label = fr'$\alpha={alpha}$'

        counts = np.stack([
            np.histogram(t, bins=bins, density=True)[0] for t in trials
        ])
        mean = counts.mean(0)
        std  = counts.std(0)
        all_mean.extend(mean)

        bin_centres = 0.5 * (bins[1:] + bins[:-1])
        ax.plot(bin_centres, mean, color=color, lw=2,
                label=label)
        ax.fill_between(bin_centres, mean - std, mean + std,
                        color=color, alpha=0.25, linewidth=0)   # error band

    ax.set_xlabel(r'$\lambda$', fontsize=label_fontsize)
    ax.set_ylabel('Probability Density', fontsize=label_fontsize)
    
    left_cut = -0.3
    right_cut = 0.5 * (bins[-1] + bins[-2])

    ax.set_xlim(left=left_cut,)   # let matplotlib choose xmax
    ax.set_xticks([left_cut, 0])
    ax.set_xticklabels([f'{x:.2f}' for x in [left_cut, 0]], fontsize=ticklabel_fontsize)
    ###
    ax.tick_params(axis='both', which='major', labelsize=ticklabel_fontsize)
    title_str = 'in autonomous RNN' if args.input_type == 'zeros' else 'in noise-driven RNN'
    K = args.sequence_length - args.warmup
    
    if args.hidden_size == 3000 and K == 100 and args.sequence_length == 3000:
    # Panel A: larger network size, everything else unchanged
        panel_title = fr'Larger N={args.hidden_size}, fixed K, T'
    elif K == 150 and args.sequence_length == 3000 and args.hidden_size == 1000:
        # Panel B: longer accumulation window, same total T
        panel_title = fr'Larger K={K}, fixed T={args.sequence_length}'
    elif K == 100 and args.sequence_length == 4000 and args.hidden_size == 1000:
        # Panel C: same K, but simulate for longer
        panel_title = fr'Fixed K={K}, longer T={args.sequence_length}'
    elif K == 100 and args.sequence_length == 3000 and args.hidden_size == 1000:
        # “baseline”  (the original setting of the paper)
        panel_title = r'Original K=100, T=3000'
    else:
        # any other combo – state the exact numbers
        panel_title = fr'N={args.hidden_size}, K={K}, T={args.sequence_length}'

    # ax.set_title(panel_title, fontsize=title_fontsize)
    ax.set_title(fr'Lyap. exp. distribution at $\langle g^* \rangle$',
                 fontsize=title_fontsize)
    ax.legend(fontsize=legend_fontsize, framealpha=0.6, loc='upper left') 

    # fpath = os.path.join(out_dir, f'LE_distributions_MeanOverlay_{args.input_type}_{args.hidden_size}_K{K}_T{args.sequence_length}.png')
    # fig.savefig(fpath, dpi=1000, bbox_inches='tight')
    
    fpath = os.path.join(out_dir, f'LE_distributions_MeanOverlay_{args.input_type}_{args.hidden_size}_K{K}_T{args.sequence_length}_individualgstar_original.pdf')
    fig.savefig(fpath, dpi=1000, bbox_inches='tight')
    
    plt.close(fig)
    print(f"saved → {fpath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment: distribution for fixed g')
    parser.add_argument('--input_type', type=str, default='zeros', help='zero input or noise input')
    parser.add_argument('--hidden_size', type=int, default=1500, help='Hidden size')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--k_LE', type=int, default=100, help='k_LE parameter')
    
    parser.add_argument('--alphas', nargs='+', type=float, default=[1.00, 1.50, 2.00],
                        help='List of alpha values to test')
    
    parser.add_argument('--initializations', nargs='+', default=['levy'], help='List of initializations')
    parser.add_argument('--warmup', type=int, default=2900, help='Warmup steps')
    parser.add_argument('--sequence_length', type=int, default=3000, help='Sequence length')
    parser.add_argument('--scale_noise', type=float, default=0.1, help='scale the noisy input')
    
    parser.add_argument('--n_trials', type=int, default=10,
                    help='number of random realisations per (alpha,g)')
    parser.add_argument('--use_cache', action='store_true',
                    help='If set, load & save a cache of Fig3a results to skip recomputation')
    parser.add_argument(
        "--exclude_outliers", action="store_true",
        help="If set, exclude outliers from the Lyapunov exponent distribution that are more than 4 std from the mean"
    )    
    parser.add_argument('--seed', type=int, default=40, # changed from seed=0 to 40, to align with the default of fig2.
                    help='seed')
    args = parser.parse_args()
    
    torch.set_default_dtype(torch.float32)
    main(args)
