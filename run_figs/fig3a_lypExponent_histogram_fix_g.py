import argparse
import numpy as np
import torch
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
import os
import json
import glob
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LypModel import RNNModel
import LypAlgo as lyap
from utils import set_seed, exclude_outliers_std, COLOR, title_fontsize, legend_fontsize, label_fontsize, ticklabel_fontsize

def get_LE_array(args, alpha, g, cache_dir):
    """
    Return a 1-D np.ndarray of finite Lyapunov exponents for (alpha, g).
    If the array has already been saved in `cache_dir`, just load it.
    Otherwise compute it once and save it as a .npy file.
    """
    os.makedirs(cache_dir, exist_ok=True)
    fpath = os.path.join(cache_dir, f"LE_alpha{alpha}_g{g:.4f}_seed{args.seed}.npy")

    if os.path.isfile(fpath):
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

def main(args):
    save_dir = 'neurips_results'
    
    summary_json = (
    f"{save_dir}/hidden{args.hidden_size}/{args.input_type}"
    f"/kLE{args.k_LE}/warmup{args.warmup}/seq{args.sequence_length}"
    )

    # Search for the .json file under either Fig1 or Fig2
    json_candidates = glob.glob(os.path.join(summary_json, "Fig2_MLE_varyg/g_cross_summary.json"))
    # json_candidates = glob.glob(
    #     os.path.join(summary_json, f"Fig2_MLE_varyg/g_cross_summary_seed{args.seed}.json")
    # )

    if not json_candidates:
        raise FileNotFoundError(f"No g_cross_summary_seed{args.seed}.json found under Fig2_MLE_varyg in {summary_json}")

    summary_json = json_candidates[0]  # or handle multiple matches if needed
    g_star = load_g_star_summary(summary_json)
    print(f"Loaded g* from {summary_json}")

    # ---- gather LE arrays for every α that truly crossed --
    le_dict = {}
    cache_root = (
        f"{save_dir}/hidden{args.hidden_size}/{args.input_type}"
        f"/kLE{args.k_LE}/warmup{args.warmup}/seq{args.sequence_length}"
        "/LE_cache"                
    )

    for alpha in tqdm(args.alphas, desc="Loading / computing LE arrays"):
        if g_star.get(str(alpha)) is None:
            print(f"skip alpha={alpha}: curve never crossed zero")
            continue

        g_alpha = float(g_star[str(alpha)])
        le_vals = get_LE_array(args, alpha=float(alpha), g=g_alpha, cache_dir=cache_root)
        if args.exclude_outliers:
            le_vals = exclude_outliers_std(le_vals, k=4.0)
        le_dict[alpha] = le_vals

    if not le_dict:
        print("No alpha's crossed; nothing to plot."); return

    # ---- build a shared bin grid --------------------------
    all_vals = np.concatenate(list(le_dict.values()))
    # n_bins   = 20                               # choose whatever resolution you like
    # bins     = np.linspace(all_vals.min(), all_vals.max(), n_bins + 1)

    # ---- single overlay plot ------------------------------
    out_dir = (
        f"neurips_results/hidden{args.hidden_size}/{args.input_type}"
        f"/kLE{args.k_LE}/warmup{args.warmup}/seq{args.sequence_length}"
        "/Fig3a_LE_distribution"
    )
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    for alpha, vals in le_dict.items():
        # breakpoint()
        color = COLOR.get(alpha, None)
        # label = fr'$\alpha={alpha}\ (g^*\approx{round(g_star[str(alpha)],2)})$'
        label = fr'$\alpha={alpha}$ ($\langle g^* \rangle \approx {round(g_star[str(alpha)], 2)}$)'

        # # Filled version (lighter, semi-transparent)
        # ax.hist(vals,
        #         bins=bins,
        #         density=True,
        #         histtype='stepfilled',
        #         alpha=0.3,
        #         color=color)

        # # Outline version (thicker line)
        # ax.hist(vals,
        #         bins=bins,
        #         density=True,
        #         histtype='step',
        #         linewidth=3.0,
        #         color=color,
        #         label=label)
        bin_count = 20  # or use 'auto', or customize per-alpha if you want
        bins = np.histogram_bin_edges(vals, bins=bin_count)

        # Filled version
        ax.hist(vals,
                bins=bins,
                density=True,
                histtype='stepfilled',
                alpha=0.3,
                color=color)

        # Outline version
        ax.hist(vals,
                bins=bins,
                density=True,
                histtype='step',
                linewidth=1.0,
                color=color,
                label=label)

    ax.set_xlabel(r'$\lambda$', fontsize=label_fontsize)
    ax.set_ylabel('Probability Density', fontsize=label_fontsize)
    
    xticks = [bins[0], 0,]  # first bin edge, zero, last bin edge
    global_min = min(v.min() for v in le_dict.values())
    xticks = [round(global_min, 2), 0]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{x:.2f}' for x in xticks], fontsize=ticklabel_fontsize)

    ax.tick_params(axis='both', which='major', labelsize=ticklabel_fontsize)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    title_str = 'in autonomous RNN' if args.input_type == 'zeros' else 'in noise-driven RNN'
    ax.set_title(fr'Lyap. exp. distribution at $\langle g^* \rangle$',
                 fontsize=title_fontsize)
    ax.legend(fontsize=legend_fontsize, framealpha=0.6, loc='upper left') 

    K = args.sequence_length - args.warmup
    # fpath = os.path.join(out_dir, f'LE_distributions_overlay_{args.input_type}_{args.hidden_size}_seed{args.seed}_K{K}_T{args.sequence_length}_individualgstar.png')
    # fig.savefig(fpath, dpi=1000, bbox_inches='tight')
    
    fpath = os.path.join(out_dir, f'LE_distributions_overlay_{args.input_type}_{args.hidden_size}_seed{args.seed}_K{K}_T{args.sequence_length}_original.pdf')
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
    
    parser.add_argument('--seed', type=int, default=40,
                    help='seed')
    parser.add_argument(
        "--exclude_outliers", action="store_true",
        help="If set, exclude outliers from the Lyapunov exponent distribution that are more than 4 std from the mean"
    )    

    args = parser.parse_args()
    
    torch.set_default_dtype(torch.float64)
    set_seed(args.seed)
    
    main(args)
