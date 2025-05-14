import os
import argparse
import pickle
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

plt.rcParams.update({'font.size': 20})
jax.config.update("jax_enable_x64", True)
precision = jnp.float64

parser = argparse.ArgumentParser()
parser.add_argument('-N', '--N', default=100)
parser.add_argument('-l', '--legend', default=False, action=argparse.BooleanOptionalAction)

args = parser.parse_args()
N = int(args.N)
show_legend = args.legend

def generate_matrix(key, N = 1_000, g=1):
    return g*jax.random.cauchy(key, (N,N), dtype=precision)/N

def evolve_linear(key = None, T = 10, N = 1_000, g = .1, quenched = False, eps_count = 0.1):

    if key is None:
        seed = np.random.randint(1_000_000_000)
        key = jax.random.PRNGKey(seed)

    key, subkey = jax.random.split(key)
    eps = jax.random.normal(subkey, (N,1), dtype=precision)

    if quenched:
        key, subkey = jax.random.split(key)
        A = generate_matrix(subkey, N=N, g=g)
    for i in range(T):
        if not quenched:
            key, subkey = jax.random.split(key)
            A = generate_matrix(subkey, N=N, g=g)
        eps = A@eps

    count_small = jnp.sum(jnp.abs(eps) < eps_count)/N
    return count_small

def find_transition(key, N, M=1000):
    key, subkey = jax.random.split(key)
    Z = jax.random.cauchy(subkey, shape=(N, M))
    av =  jnp.mean(jnp.abs(Z), axis=0)
    Chi = jnp.mean(jnp.log(av))
    return jnp.exp(-Chi)

key = jax.random.PRNGKey(0)

T = 100
#N = 1000
M = 100
M_show = 20
M_theory = 100_000

fpath_pkl = f'data_annealed_vs_quenched_N{N}.pickle'
if os.path.exists(fpath_pkl):
    print('Data exists. Loading...')
    with open(fpath_pkl, 'rb') as file:
        results = pickle.load(file)
        
else:
    print("Theory...")
    key, subkey = jax.random.split(key)
    g_theory = find_transition(subkey, N, M=M_theory)
    
    keys = jax.random.split(key, M)
    g_tab = jnp.linspace(1e-2, 0.5, 50)
    counts_annealed = []
    counts_quenched = []
    
    print("Simulations...")
    for g in tqdm(g_tab):
        evolve_v_true = jax.vmap(lambda k: evolve_linear(key=k, g=g, N=N, T=T, quenched=True) )
        c = evolve_v_true(keys)
        counts_quenched.append( c )
        evolve_v_false = jax.vmap(lambda k: evolve_linear(key=k, g=g, N=N, T=T, quenched=False) )
        c = evolve_v_false(keys)
        counts_annealed.append( c )
    
    c_quenched = jnp.stack(counts_quenched)
    c_annealed = jnp.stack(counts_annealed)
    
    results = {
           'g': np.array(g_tab),
           'g_theory': np.array(g_theory),
           'c_quenched': np.array(c_quenched),
           'c_annealed': np.array(c_annealed),
          }

    pickle.dump(results, open(fpath_pkl, "wb"))

print('Plotting...')
mean_quenched = np.mean(results['c_quenched'], axis=1)
err_quenched = 3*np.std(results['c_quenched'], axis=1)/np.sqrt(M)
mean_annealed = np.mean(results['c_annealed'], axis=1)
err_annealed = 3*np.std(results['c_annealed'], axis=1)/np.sqrt(M)

plt.figure(figsize=(5,4))
plt.plot(results['g'], results['c_quenched'][:, :M_show], '-', color='#1f77b4', alpha=0.1)
plt.plot(results['g'], mean_quenched, '.-', label='quenched', lw=2);
plt.fill_between(results['g'], mean_quenched - err_quenched, mean_quenched + err_quenched, alpha=0.5)

plt.plot(results['g'], mean_annealed, '.-', label='annealed', lw=2);
plt.fill_between(results['g'], mean_annealed - err_annealed, mean_annealed + err_annealed, alpha=0.5)
plt.axvline(results['g_theory'], linestyle='--', color='black', alpha=0.5, lw=2, label='theory')

plt.xlabel('$g$')
#plt.ylabel('Normalized # of small entries of $\epsilon(T)$')
plt.ylabel('$f_{<\\epsilon}}$')
plt.xlim([0, 0.5])
#plt.title(f'N={N}; # of samples: {M}')
if show_legend:
    plt.legend(loc='upper right');
plt.savefig(f'figure1_N_{N}.pdf', bbox_inches='tight');

print('Done')
