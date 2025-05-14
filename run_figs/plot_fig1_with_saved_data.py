import pickle
import matplotlib.pyplot as plt
import numpy as np

hidden=1000
input=100
data_type = 'zeros'
kLE=100
warmup=1900
seq=2000

color = {0.5:'#56B4E9',
        1:'#E69F00',
        1.5:'#009E73',
        2:'#CC79A7'}

for alpha in [0.5, 1, 1.5, 2]:
    file_path = f'neurips_results/input_{data_type}/fig1_maxL_vs_g/alpha{alpha}/hidden{hidden}/input{input}/kLE{kLE}/warmup{warmup}_seq{seq}/MLE_alpha{alpha}.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    mean_max_LE = data['mean_max_LE']
    std_max_LE = data['std_max_LE']
    g_vals = np.logspace(-3, 3, num=50)

    valid_indices = np.where((g_vals >= 0.01) & (g_vals <= 10))
    g_vals_filtered = g_vals[valid_indices]
    mean_max_LE_filtered = mean_max_LE[valid_indices]
    std_max_LE_filtered = std_max_LE[valid_indices]

    plt.plot(g_vals_filtered, mean_max_LE_filtered, linewidth=2, label=fr'$\alpha={alpha}$', color=color[alpha])
    plt.fill_between(g_vals_filtered, mean_max_LE_filtered - std_max_LE_filtered, mean_max_LE_filtered + std_max_LE_filtered, alpha=0.25, color=color[alpha], linewidth=0)

# Add labels and legend
plt.axhline(y=0, color='grey', linestyle='--')
plt.axvline(x=1, color='grey', linestyle='--')
plt.xlabel('$g$', fontsize=16)
plt.xscale('log')
plt.ylabel(r'$\lambda_{max}$', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)  # Adjusts the size of the tick labels
plt.title('Max Lyap. exp. $\lambda_{max}$ under diff. dist. width $g$ in autonomous RNN',fontsize=12)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(f'neurips_results/plot_data/new_MLE_Levy_hidden{hidden}_input{input}_{data_type}_warmup{warmup}_seq{seq}_alphas[0.5,1,1.5,2]_log.pdf', dpi=1000)