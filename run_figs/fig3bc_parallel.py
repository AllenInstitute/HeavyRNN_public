#!/usr/bin/env python3
"""
Fig3bc_PR_LD.py

Sub-commands
------------
compute  – run one (alpha, trial, metric) job; save a pickle
plot     – load all pickles for that metric; draw the α‑overlay figure
"""

import argparse, os, glob, re, gc, pickle as pkl
from pathlib import Path
import numpy as np
import torch, matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LypModel import RNNModel
import LypAlgo as lyap
from utils import set_seed, COLOR, title_fontsize, legend_fontsize, \
                  label_fontsize, ticklabel_fontsize


# ------------------------- low‑level helpers ------------------------- #

def _trial_pickle_path(args) -> Path:
    """folder/alphaXX/trialYY.pkl"""
    root = Path(args.save_root, f"hidden{args.hidden_size}",
                args.input_type, f"kLE{args.k_LE}",
                args.metric, f"alpha{args.alpha:.2f}")
    root.mkdir(parents=True, exist_ok=True)
    return root / f"trial{args.trial}.pkl"


def _glob_pattern(results_root, hidden_size, input_type, k_LE, metric):
    return str(Path(results_root, f"hidden{hidden_size}", input_type,
                    f"kLE{k_LE}", metric, "alpha*/trial*.pkl"))


# --------------------- compute (one α, trial, metric) ---------------- #

def compute_one(args):
    g_values = (np.asarray(args.g_values, dtype=float)
                if args.g_values else np.logspace(-1, 1, 50))
    n_g = len(g_values)

    set_seed(args.seed + args.trial)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare input once
    if args.input_type == "zeros":
        input_size = 1
        le_input = torch.zeros(args.batch_size, args.sequence_length, 1,
                               device=device)
    else:
        input_size = args.hidden_size
        le_input = (torch.randn(args.batch_size, args.sequence_length,
                                input_size, device=device) * args.scale_noise)

    vals = np.zeros(n_g)

    for i, g in enumerate(tqdm(g_values,
                               desc=f"{args.metric} α={args.alpha:.2f} "
                                    f"trial={args.trial}")):
        m = RNNModel(args.initialization, input_size, args.hidden_size,
                     g=g, alpha=args.alpha, nonlinearity='tanh',
                     data=None, batch_first=True).to(device)
        m.make_autonomous() if args.input_type == "zeros" else m.make_noise_driven()
        m.eval(); h0 = m.init_hidden(args.batch_size)

        if args.metric == "LD":
            res = lyap.calc_LEs_an(
                le_input[:, :args.warmup_LE + args.K_LD + 1],
                h0, model=m, k_LE=args.k_LE, warmup=args.warmup_LE,
                compute_LE=True, include_PR=False, store_all_states=False,
                get_jacobian=False)
            mean_LE, _ = lyap.LE_stats(res['LE'])
            vals[i] = lyap.lyapunov_dimension(mean_LE)

        else:  # PR
            res = lyap.calc_LEs_an(
                le_input[:, :args.warmup_PR + args.hidden_size + args.PR_overhead + 1],
                h0, model=m, k_LE=1, warmup=args.warmup_PR,
                compute_LE=False, include_PR=True, store_all_states=False)
            vals[i] = float(res['PR'])

        torch.cuda.empty_cache()

    # save
    out = dict(g_values=g_values, values=vals)
    with _trial_pickle_path(args).open("wb") as f:
        pkl.dump(out, f)
    print("Wrote:", _trial_pickle_path(args))
    gc.collect()


# ----------------------- plot aggregate figure ---------------------- #

def plot_aggregate(args):
    patt = _glob_pattern(args.results_root, args.hidden_size,
                         args.input_type, args.k_LE, args.metric)
    files = glob.glob(patt)
    assert files, f"No pickles found under {patt}"

    # bucket by alpha
    agg, g_vals = {}, None
    for fn in files:
        d = pkl.load(open(fn, "rb"))
        α = float(re.search(r"alpha([0-9.]+)", fn).group(1))
        agg.setdefault(α, []).append(d['values'])
        g_vals = d['g_values']    # identical across pickles
    print({alpha: len(vals) for alpha, vals in agg.items()})


    # output dir
    out_dir = Path(args.results_root, f"hidden{args.hidden_size}",
                   args.input_type, f"kLE{args.k_LE}",
                   f"Fig3bc_{args.metric}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # draw
    plt.figure(figsize=(5, 4)); plt.xscale("log")
    for α in sorted(agg):
        arr = np.vstack(agg[α])
        m, s = arr.mean(0), arr.std(0)
        plt.plot(g_vals, m, label=rf'$\alpha={α}$',
                 color=COLOR.get(α, None))
        plt.fill_between(g_vals, m - s, m + s, alpha=.15,
                         color=COLOR.get(α, None), linewidth=0)

    plt.xlabel("$g$", fontsize=label_fontsize)
    ylabel = "Participation Ratio" if args.metric == "PR" else "Lyapunov Dimension"
    plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.tick_params(labelsize=ticklabel_fontsize)
    plt.title(f"{ylabel}", fontsize=title_fontsize)
    loc = "upper right" if args.metric == "PR" and args.input_type == 'noise' else "upper left"
    plt.legend(fontsize=legend_fontsize, loc=loc)

    fname = out_dir / f"{args.metric}_varyg_overlay_{args.input_type}_N{args.hidden_size}_10trials.pdf"
    plt.savefig(fname, dpi=1000, bbox_inches="tight")
    print("Saved:", fname)


# ----------------------------- CLI ---------------------------------- #

def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="mode", required=True)

    # -------- compute --------
    c = sp.add_parser("compute")
    c.add_argument("--alpha", type=float, required=True)
    c.add_argument("--trial", type=int, required=True)
    c.add_argument("--metric", choices=["PR", "LD"], required=True)
    c.add_argument("--input_type", choices=["zeros", "noise"], default="zeros")
    c.add_argument("--hidden_size", type=int, default=1000)
    c.add_argument("--batch_size", type=int, default=1)
    c.add_argument("--k_LE", type=int, default=100)
    c.add_argument("--warmup_PR", type=int, default=2900)
    c.add_argument("--warmup_LE", type=int, default=2900)
    c.add_argument("--PR_overhead", type=int, default=50)
    c.add_argument("--K_LD", type=int, default=50)
    c.add_argument("--g_values", nargs="+", type=float)
    c.add_argument("--initialization", default="levy")
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--save_root", default="neurips_results")
    c.add_argument("--scale_noise", type=float, default=0.1)
    # sequence length derived from PR settings
    c.add_argument("--sequence_length", type=int,
                   default=None,  # filled programmatically
                   help="leave None to autocompute")

    # -------- plot --------
    p2 = sp.add_parser("plot")
    p2.add_argument("--metric", choices=["PR", "LD"], required=True)
    p2.add_argument("--results_root", default="neurips_results")
    p2.add_argument("--input_type", choices=["zeros", "noise"], default="zeros")
    p2.add_argument("--hidden_size", type=int, required=True)
    p2.add_argument("--k_LE", type=int, required=True)

    args = p.parse_args()
    torch.set_default_dtype(torch.float32)

    if args.mode == "compute":
        # derive sequence length if not given
        if args.sequence_length is None:
            args.sequence_length = (args.warmup_PR + args.hidden_size +
                                    args.PR_overhead + 1)
        compute_one(args)
    else:
        plot_aggregate(args)


if __name__ == "__main__":
    main()
