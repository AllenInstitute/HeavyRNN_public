#!/usr/bin/env python
"""
Fig2.py

Modes:
  compute   — run one (alpha, trial) job, save a pickle
  plot      — gather all pickles, make the combined MLE-vs-g plot
"""

from collections import defaultdict
import argparse, os, glob, json, re
import pickle as pkl
import numpy as np
import torch, gc
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LypModel import RNNModel
import LypAlgo as lyap
from utils import set_seed, COLOR, title_fontsize, legend_fontsize, label_fontsize, ticklabel_fontsize


def _get_base_dir(args, save_dir):
    alpha_str = f"{args.alpha:.2f}" if hasattr(args, 'alpha') else "_".join(f"{a:.2f}" for a in args.alphas)
    return os.path.join(
        save_dir,
        f"hidden{args.hidden_size}",
        args.input_type,
        f"kLE{args.k_LE}",
        f"warmup{args.warmup}",
        f"seq{args.sequence_length}",
        f"Fig2_MLE_varyg/alpha{alpha_str}"
    )


def plot_max_lambda_vs_g(args, initialization, alpha, trial, out_dir):
    """Compute LEs for this (alpha, trial), save one pickle."""
    set_seed(args.seed + trial)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g_values = np.asarray(args.g_values) if args.g_values else np.logspace(-2, 1, 50)
    # prepare input
    if args.input_type == 'zeros':
        input_size = 1
        le_input = torch.zeros(args.batch_size, args.sequence_length, 1, device=device)
    else:
        input_size = args.hidden_size
        le_input = torch.randn(args.batch_size, args.sequence_length, input_size, device=device) * args.scale_noise

    base_dir = _get_base_dir(args, out_dir)
    os.makedirs(base_dir, exist_ok=True)

    max_LEs = []
    for g in tqdm(g_values, desc=f"α={alpha} t={trial}"):
        # build & init model
        m = RNNModel(initialization, input_size, args.hidden_size,
                     g=g, alpha=alpha, nonlinearity='tanh',
                     data=None, batch_first=True).to(device)
        if args.input_type=='zeros':  m.make_autonomous()
        else:                         m.make_noise_driven()
        m.eval(); h = m.init_hidden(batch_size=args.batch_size)

        res = lyap.calc_LEs_an(
            le_input, h, model=m, k_LE=args.k_LE, warmup=args.warmup,
            include_PR=False, compute_LE=True,
            store_all_output=False, store_all_states=False, get_jacobian=False
        )
        LE, _ = res['LE'], res['PR']
        mean_LE, _ = lyap.LE_stats(LE)
        max_LEs.append(max(mean_LE).cpu().numpy())
        torch.cuda.empty_cache()

    # save
    out = dict(g_values=g_values, max_LE=max_LEs)
    fn = os.path.join(base_dir, f"alpha{alpha}_trial{trial}.pkl")
    with open(fn, 'wb') as f:  pkl.dump(out, f)
    print("Wrote:", fn)
    gc.collect()


def plot_aggregate(args):
    """Load all pickles and draw the single final figure."""
    base_root = os.path.join(
        args.results_root,
        f"hidden{args.hidden_size}",
        args.input_type,
        f"kLE{args.k_LE}",
        f"warmup{args.warmup}",
        f"seq{args.sequence_length}",
    )

    # --- allow both old and new folder structures ---
    pattern_new = os.path.join(base_root, "Fig2_MLE_varyg", "alpha*/alpha*_trial*.pkl")
    pattern_old = os.path.join(base_root, "Fig2_MLE_varyg_alpha*/alpha*_trial*.pkl")
    
    files = glob.glob(pattern_new, recursive=True) + glob.glob(pattern_old, recursive=True)
    if args.input_type == "zeros":
        pattern_update = os.path.join(base_root, "Fig1_MLE_varyg", "alpha*/alpha*_trial*.pkl")
        files += glob.glob(pattern_update, recursive=True)
    assert files, "No pickles found under either new or old structure"
    print(f"Found {len(files)} pickle files.")

    out_dir = os.path.join(base_root, "Fig2_MLE_varyg")
    os.makedirs(out_dir, exist_ok=True)
    # bucket by alpha
    agg = {}
    g_cross   = {} 
    g_cross_trials = defaultdict(dict)
    for fn in files:
        d = pkl.load(open(fn,'rb'))
        alpha = float(re.search(r"alpha([0-9.]+)_trial", fn).group(1))
        trial  = int  (re.search(r"_trial(\d+)",             fn).group(1))
        # agg.setdefault(alpha, []).append(d['max_LE'])
        if args.n_trials is not None and trial >= args.n_trials:
            continue
        le_arr = np.asarray(d['max_LE'])
        g_vals = d['g_values']
        
        # ----- per‑file crossing ------------------------------------------
        idx = np.where((le_arr[:-1] < 0) & (le_arr[1:] >= 0))[0]
        g_star_single = float(g_vals[idx[0] + 1]) if len(idx) else None
        g_cross_trials[trial][alpha] = g_star_single        # save it
        # ------------------------------------------------------------------
        agg.setdefault(alpha, []).append(le_arr)

    plt.figure(figsize=(5,4)); plt.xscale("log")

    for α, trials in sorted(agg.items()):
        if args.exclude_alpha is not None and α == args.exclude_alpha:
            continue
        arr = np.vstack(trials)
        m, s = arr.mean(0), arr.std(0)
        c = COLOR[α]
        idx = np.where((m[:-1]<0)&(m[1:]>=0))[0]

        if len(idx) > 0:
            g_star = float(g_vals[idx[0]+1])
            plt.axvline(x=g_star, linestyle='--', color=c, alpha=0.6)
            g_cross[α] = g_star
        else:
            g_cross[α] = None  # never crossed

        plt.plot(g_vals, m, linewidth=2,
                label=fr'$\alpha={α}$ ($g^*\approx{round(g_cross[α],2)})$' if g_cross[α] is not None else fr'$\alpha={α}$',
                color=c)
        plt.fill_between(g_vals, m-s, m+s, alpha=.15, color=c, edgecolor=None, linewidth=0,)

    # styling as before
    plt.axhline(0, color='grey', linestyle=':')
    plt.xlabel(r"$g$", fontsize=label_fontsize)
    plt.ylabel(r"$\lambda_{\max}$", fontsize=label_fontsize)
    plt.tick_params(labelsize=ticklabel_fontsize)
    plt.yticks([1, 0, -4], ["1", "0", "−4"])

    title_str = "in autonomous RNN" if args.input_type=="zeros" else "in noise-driven RNN"
    # infer N from any path
    N = int(re.search(r"hidden(\d+)", files[0]).group(1))
    plt.title(rf"N={N}", fontsize=title_fontsize)

    plt.title(fr'N={args.hidden_size}', fontsize=title_fontsize)
    plt.legend(fontsize=legend_fontsize, loc='lower right')

    # out_png = os.path.join(out_dir, f"MLE_varyg_varyalpha_log_{args.input_type}_{args.hidden_size}.png")
    out_pdf = os.path.join(out_dir, f"MLE_varyg_varyalpha_log_{args.input_type}_{args.hidden_size}_10trials.pdf")
    # plt.savefig(out_png, dpi=1000, bbox_inches='tight')
    plt.savefig(out_pdf, dpi=1000, bbox_inches='tight')
    print("Saved:", out_pdf)

    json_path = os.path.join(out_dir, "g_cross_summary.json")
    with open(json_path, "w") as f:
        json.dump(g_cross, f, indent=2)
    print("Wrote:", json_path)
    
    # for trial, summary in g_cross_trials.items():
    #     json_path = os.path.join(out_dir, f"g_cross_summary_seed{args.seed+trial}.json")
    #     with open(json_path, "w") as f:
    #         json.dump(summary, f, indent=2)
    #     print("Wrote:", json_path)


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="mode", required=True)

    # --- compute subcommand ---
    c = sp.add_parser("compute")
    c.add_argument("--alpha",          type=float, required=True)
    c.add_argument("--trial",          type=int,   required=True)
    c.add_argument("--input_type",     choices=["zeros","noise"], default="zeros")
    c.add_argument("--hidden_size",    type=int, default=5000)
    c.add_argument("--batch_size",     type=int, default=1)
    c.add_argument("--k_LE",           type=int, default=5000)
    c.add_argument("--warmup",         type=int, default=2900)
    c.add_argument("--sequence_length",type=int, default=3000)
    c.add_argument("--g_values",       nargs="+", type=float, default=None)
    c.add_argument("--initialization", default="levy")
    c.add_argument("--seed",           type=int, default=40)
    c.add_argument("--save_dir",       default="neurips_results")
    c.add_argument('--scale_noise', type=float, default=0.1, help='scale the noisy input')

    # --- plot subcommand ---
    p2 = sp.add_parser("plot")
    p2.add_argument("--seed",           type=int, default=40)
    p2.add_argument("--results_root", default="neurips_results")
    p2.add_argument("--input_type", choices=["zeros", "noise"], default="zeros")
    p2.add_argument("--hidden_size", type=int, required=True)
    p2.add_argument("--k_LE", default=100, type=int, required=True)
    p2.add_argument("--warmup", type=int, default=2900)
    p2.add_argument("--sequence_length", type=int, default=3000)
    p2.add_argument("--exclude_alpha", type=float, default=None, help="Alpha value to exclude from plot")
    p2.add_argument("--n_trials", type=int, default=None,
                help="Only use the first N trials per α when plotting")

    args = p.parse_args()

    if args.mode=="compute":
        plot_max_lambda_vs_g(args,
                             initialization=args.initialization,
                             alpha=args.alpha,
                             trial=args.trial,
                             out_dir=args.save_dir)
    else:
        plot_aggregate(args)


if __name__=="__main__":
    torch.set_default_dtype(torch.float32)
    main()
