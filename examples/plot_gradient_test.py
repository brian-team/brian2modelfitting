import pickle
import numpy as np
import matplotlib.pyplot as plt

from brian2modelfitting.fitter import SCALAR_METHODS_GRADIENT_SUPPORT as METHODS
METHODS['least_squares'] = ('least_squares', True, False)
METHODS['leastsq'] = ('leastsq', True, False)

with open('results_normalization.pickle', 'rb') as f:
    results = pickle.load(f)

fig, axs1 = plt.subplots(3, 4, sharey=True, sharex=True)
fig, axs2 = plt.subplots(3, 4, sharey=True, sharex=True)
for (short_name, (long_name, _, _)), ax1, ax2 in zip(METHODS.items(), axs1.flat, axs2.flat):
    print(long_name)
    for calc_grad in [True, False]:
        if (short_name, calc_grad) not in results:
            continue
        result = results[(short_name, calc_grad)]
        for run in result:
            c = 'darkblue' if calc_grad else 'darkred'
            ax1.plot(run[:, 0], run[:, 1], color=c, alpha=0.7)
            ax2.plot(run[:, 0], run[:, 1], color=c, alpha=0.7)
    ax1.set(title=long_name, yscale='log', ylim=(1e-5, 1e3))
    ax2.set(title=long_name, yscale='log', ylim=(1e-10, 1e3), xlim=(0, 20))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

plt.show()
