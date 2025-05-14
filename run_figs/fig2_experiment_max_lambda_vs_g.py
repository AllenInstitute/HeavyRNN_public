# experiment_max_lambda_vs_g.py
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
from utils import set_seed, COLOR,  title_fontsize, legend_fontsize, label_fontsize, ticklabel_fontsize

def _get_base_dir(args, save_dir):
    alpha_str = "_".join([f"{a:.2f}" for a in args.alphas])
    return (
        f"{save_dir}/"
        f"hidden{args.hidden_size}/{args.input_type}/"
        f"kLE{args.k_LE}/warmup{args.warmup}/seq{args.sequence_length}/"
        f"Fig2_MLE_varyg_varyalpha_{alpha_str}/"
    )

def plot_max_lambda_vs_g(
    args,
    initialization='normal',
    alpha=None,
    g_values=None,
    trial_number='unspecified',
    data=None,
    plot_side_by_side=False,
    include_PR=False,
    save_dir='',
):
    """
    Compute and optionally plot:
      - max_LE_mean (maximum Lyapunov exponent),
      - participation_ratios (if include_PR=True),
      - lyap_dimensions (if plot_side_by_side=True),
    for a list of g-values. By default, uses values from args unless overridden.
    All results (including pickles, plots) go under the provided save_dir.

    Returns
    -------
    g_values : list
    max_LE_mean : list
    participation_ratios : list
    lyap_dimensions : list
    filepath_without_g : str
        The directory path (minus the final 'g{...}').
    """
    # Resolve overrides or defaults
    input_type = args.input_type
    hidden_size = args.hidden_size
    batch_size = args.batch_size
    sequence_length = args.sequence_length
    warmup = args.warmup
    k_LE = args.k_LE

    if alpha is None:
        alpha = args.alpha
    if g_values is None:
        g_values = args.g_values if args.g_values else np.linspace(0.01, 5, num=15)

    with torch.no_grad():
        LE_batchmean_results = []
        participation_ratios = []
        lyap_dimensions = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Prepare the RNN input
        if input_type == 'zeros':
            input_size = 1
            le_input = torch.zeros(batch_size, sequence_length, input_size)
        elif input_type == 'noise':
            input_size = args.hidden_size
            le_input = torch.randn(batch_size, sequence_length, input_size) * args.scale_noise # 100
        else:
            raise ValueError(f"Unknown input_type: {input_type}")
        le_input = le_input.to(device)
        # Base path for storing states/results
        # (We keep 'plot_max_lambda_vs_g' as a subfolder, or rename as you wish)
        base_dir = _get_base_dir(args, save_dir)
        os.makedirs(base_dir, exist_ok=True)
        filepath_without_g = base_dir  # The "directory minus the g"

        # Loop over all g
        for g in tqdm(g_values, desc="Computing LEs for different g"):
            # Directory that includes the specific 'g'
            filepath = f"{base_dir}/gs/g{g}"
            # os.makedirs(filepath, exist_ok=True)
            
            # Build the model
            model = RNNModel(
                initialization=initialization,
                input_size=input_size,
                hidden_size=hidden_size,
                g=g,
                alpha=alpha,
                nonlinearity='tanh',
                data=data,
                batch_first=True
            ).to(device)
            if args.input_type == 'zeros':
                model.make_autonomous()
            else:                     
                model.make_noise_driven()
            model.eval()
            h = model.init_hidden(batch_size=batch_size)

            # Calculate LEs
            result = lyap.calc_LEs_an(
                le_input,
                h,
                model=model,
                k_LE=k_LE,
                warmup=warmup,
                include_PR=include_PR,
                directory=f'{filepath}/trial{trial_number}', # nothing is saved so trivial
                compute_LE=True,
                store_all_output=False,
                store_all_states=False,
                get_jacobian=False,
            )

            LEs, participation_ratio = result['LE'], result['PR']
            LE_mean, _ = lyap.LE_stats(LEs)

            LE_batchmean_results.append(LE_mean)

            if plot_side_by_side:
                lyap_dimensions.append(lyap.lyapunov_dimension(LE_mean))
            else:
                lyap_dimensions.append('skipped')

            if isinstance(participation_ratio, torch.Tensor):
                participation_ratios.append(participation_ratio.detach().cpu().numpy())
            else:
                participation_ratios.append(participation_ratio)

            torch.cuda.empty_cache()

        max_LE_mean = [max(le_mean).cpu().numpy() for le_mean in LE_batchmean_results]

        print('max_LE:', max_LE_mean)
        if include_PR:
            print('PR:', participation_ratios)
        
        return g_values, max_LE_mean, participation_ratios, lyap_dimensions, filepath_without_g


def plot_max_lambda_vary_alpha(
    args,
    initialization='normal',
    g_values=None,
    plot_side_by_side=True,
    include_PR=True,
    trials=3,
    data=None,
    save_dir=''
):
    """
    Generates and plots the maximum lambda for different alphas
    across a list of g-values. Relies on plot_max_lambda_vs_g(...).
    
    All results go under save_dir.
    """
    # If user didn't supply g_values, we choose a default:
    if g_values is None:
        g_values = args.g_values if args.g_values else np.logspace(-2, 1, 50)

    result_plot = {}
    base_dir = _get_base_dir(args, save_dir)
    os.makedirs(base_dir, exist_ok=True)

    # Loop over alpha values from args.alphas
    for alpha in tqdm(args.alphas, desc="Varying alpha"):
        agg_path = f"{base_dir}/MLE_alpha{alpha}.pkl"
        if args.use_cache and os.path.exists(agg_path):
            with open(agg_path, "rb") as f:
                cached = pkl.load(f)
                
            result_plot[alpha] = {
                "g_values":        cached["g_values"],
                "mean_max_LE":     cached["mean_max_LE"],
                "std_max_LE":      cached["std_max_LE"],
                "participation_ratios": None,
                "lyap_dimensions":     None,
            }
            continue
        
        max_LE_mean_trials = []
        
        # run / or load *per‑trial* pickles
        for itrial in range(trials):
            set_seed(int(args.seed+itrial))
            trial_path = f"{base_dir}/trial{itrial}.pkl"
            if args.use_cache and os.path.exists(trial_path):
                with open(trial_path, "rb") as f:
                    gv, max_LE_mean = pkl.load(f)
            else:
                gv, max_LE_mean, participation_ratios, lyap_dims, filepath = plot_max_lambda_vs_g(
                    args,
                    initialization=initialization,
                    alpha=alpha,            # Pass alpha directly here
                    g_values=g_values,
                    trial_number=str(itrial),
                    data=data,
                    plot_side_by_side=False,
                    include_PR=include_PR,
                    save_dir=save_dir
                )
                with open(trial_path, "wb") as file:
                    pkl.dump((gv, max_LE_mean), file)
                
            max_LE_mean_trials.append(max_LE_mean)
            gc.collect()

        # Now combine the multiple trials at this alpha
        max_LE_mean_array = np.array(max_LE_mean_trials)
        mean_max_LE_mean = np.mean(max_LE_mean_array, axis=0)
        std_max_LE_mean = np.std(max_LE_mean_array, axis=0)
    
        result_plot[alpha] = {
            'g_values': gv,
            'mean_max_LE': mean_max_LE_mean,
            'std_max_LE': std_max_LE_mean,
            'participation_ratios': participation_ratios,
            'lyap_dimensions': lyap_dims
        }

        # Store aggregated results
        with open(agg_path, 'wb') as f:
            pkl.dump({
                'g_values': gv,
                'max_LE_array': max_LE_mean_array,
                'mean_max_LE': mean_max_LE_mean,
                'std_max_LE': std_max_LE_mean
            }, f)


    # Single-panel or multi-panel
    if not plot_side_by_side:
        g_cross_dict = {}  # alpha -> g where lambda_max crosses or gets closest to 0
        did_g_cross_dict = {}
        # Single-panel for aggregated MLE
        plt.figure(figsize=(5,4))
        plt.xscale("log")
        for alpha_val, data_dict in result_plot.items():
            gvals = data_dict['g_values']
            mean_LE = data_dict['mean_max_LE']
            std_LE = data_dict['std_max_LE']
            color = COLOR[alpha_val]
            # Find the first crossing point where mean_LE goes from <0 to >=0
            crossing_idx = np.where((mean_LE[:-1] < 0) & (mean_LE[1:] >= 0))[0]
            if len(crossing_idx) > 0:
                g_cross = gvals[crossing_idx[0] + 1]
                plt.axvline(x=g_cross, linestyle='--', color=color, alpha=0.6)
                did_g_cross_dict[alpha_val] = True
            else:
                g_cross = gvals[np.argmin(np.abs(mean_LE))]
                did_g_cross_dict[alpha_val] = False

            g_cross_dict[alpha_val] = float(g_cross)  # convert to float for JSON
            curve, = plt.plot(gvals, mean_LE, linewidth=2, label=fr'$\alpha={alpha_val}$ ($g^*\approx{round(g_cross,2)}$)', color=color)
  
            plt.fill_between(gvals, mean_LE - std_LE, mean_LE + std_LE, alpha=0.15, linewidth=0, color=color)
            
        plt.axhline(y=0, color='grey', linestyle=':')
        plt.xlabel('$g$', fontsize=label_fontsize)
        plt.ylabel(r'$\lambda_{\max}$', fontsize=label_fontsize)
        plt.tick_params(axis='both', which='major', labelsize=ticklabel_fontsize)
        plt.yticks([1, 0, -4], ["1", "0", "-4"])
        title_str = 'in autonomous RNN' if args.input_type == 'zeros' else 'in noise-driven RNN'
        plt.title(fr'N={args.hidden_size}', fontsize=title_fontsize)
        plt.legend(fontsize=legend_fontsize)
        
        json_out = {
            "args": vars(args),
            "g_cross_per_alpha": g_cross_dict,
            "did_g_cross_per_alpha": did_g_cross_dict,
        }
        json_path = os.path.join(base_dir, "g_cross_summary.json")
        with open(json_path, "w") as f:
            json.dump(json_out, f, indent=4)
        print(f"Saved g_cross summary to {json_path}")
        
        # Create the output directory for final plots
        plt.savefig(
            f"{base_dir}/MLE_varyg_varyalpha_log.png",
            dpi=1000,
            bbox_inches='tight'
        )
        plt.savefig(
            f"{base_dir}/MLE_varyg_varyalpha_log.pdf",
            dpi=1000, 
            bbox_inches='tight'
        )
        print('Figure saved to', f'{base_dir}/MLE_varyg_varyalpha_log.png')
        plt.close()
    else:
        assert False, 'not implemented'

def main(args):
    save_dir = args.results_root
    os.makedirs(save_dir, exist_ok=True)
    
    for init in args.initializations:
        data = None
        plot_max_lambda_vary_alpha(
            args=args,
            initialization=init, 
            plot_side_by_side=False,
            include_PR=False,
            trials=3,
            g_values=np.logspace(-2, 1, num=50),
            data=data,
            save_dir=save_dir
        )

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    parser = argparse.ArgumentParser(description='Script for RNN Lyapunov Exponent Analysis')
    parser.add_argument("--results_root", default="neurips_results")
    parser.add_argument('--input_type', type=str, default='zeros', help='zero input or noise input')
    parser.add_argument('--hidden_size', type=int, default=1000, help='Hidden size')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--k_LE', type=int, default=100, help='k_LE parameter')
    
    parser.add_argument('--alphas', nargs='+', type=float, default=[0.75, 1.0, 1.5, 2.0],
                        help='List of alpha values to test')
    parser.add_argument('--g_values', nargs='+', type=float, default=None, help='List of g values (optional)')
    
    parser.add_argument('--initializations', nargs='+', default=['levy'], help='List of initializations')
    parser.add_argument('--warmup', type=int, default=2900, help='Warmup steps')
    parser.add_argument('--sequence_length', type=int, default=3000, help='Sequence length')
    parser.add_argument("--use_cache",
                        action="store_true",
                        help="If set, load *.pkl results when they exist instead of recomputing"
                        )
    parser.add_argument('--seed', type=int, default=40, help='base seed')
    parser.add_argument('--scale_noise', type=float, default=0.1, help='scale the noisy input')
    args = parser.parse_args()
    main(args)
