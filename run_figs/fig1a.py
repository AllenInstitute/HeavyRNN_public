import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import COLOR

plt.rcParams.update({'font.size': 20})

Ds = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

M = 10_000

alphas = [1., 1.5, 2.0]

gammas_all = {alpha: [] for alpha in alphas}
colors_dict = COLOR
    
for D in Ds:
    for alpha in alphas:
        Z = levy_stable.rvs(alpha, 0, size=(D, M) )
        av =  np.mean(np.abs(Z)**alpha, axis=0)
        mean_Xi = np.mean(np.log(av))/alpha
        gammas_all[alpha].append(np.exp(-mean_Xi))
        
fig= plt.figure(figsize=(5,4))

ax = plt.gca()
ax.axhline(np.sqrt(2)/2, linestyle='--', color=colors_dict[2.0]);

for alpha in alphas:
    ax.semilogx(Ds, np.array(gammas_all[alpha]), '-o', color=colors_dict[alpha], label=f'$\\alpha={alpha}$', alpha=0.8);

ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))
ax.set_xticks([1, 10, 100, 1000, 10_000])
ax.legend();
ax.set_xlabel('$N$')
ax.set_ylabel('$g^*$');
ax.set_ylim( [0, 1.4] )

plt.savefig('figure1A.pdf', bbox_inches='tight')