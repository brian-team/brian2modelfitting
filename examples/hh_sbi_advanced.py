import os

from brian2 import *
from brian2modelfitting import *
import pandas as pd
import numpy as np
from scipy.signal import find_peaks


# Load input and output data traces
df_inp_traces = pd.read_csv('input_traces_hh.csv')
df_out_traces = pd.read_csv('output_traces_hh.csv')
inp_traces = df_inp_traces.to_numpy()
inp_traces = inp_traces[:2, 1:]
out_traces = df_out_traces.to_numpy()
out_traces = out_traces[:2, 1:]

# Model and its parameters
area = 20_000*um**2
El = -65*mV
EK = -90*mV
ENa = 50*mV
VT = -63*mV
dt = 0.01*ms
eqs = '''
    dv/dt = (gl*(El-v) - g_na*(m*m*m)*h*(v-ENa) - g_kd*(n*n*n*n)*(v-EK) + I)/Cm : volt
    dm/dt = 0.32*(mV**-1)*(13.*mV-v+VT)/
            (exp((13.*mV-v+VT)/(4.*mV))-1.)/ms*(1-m)-0.28*(mV**-1)*(v-VT-40.*mV)/
            (exp((v-VT-40.*mV)/(5.*mV))-1.)/ms*m : 1
    dn/dt = 0.032*(mV**-1)*(15.*mV-v+VT)/
            (exp((15.*mV-v+VT)/(5.*mV))-1.)/ms*(1.-n)-.5*exp((10.*mV-v+VT)/(40.*mV))/ms*n : 1
    dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms*(1.-h)-4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1
    g_na : siemens (constant)
    g_kd : siemens (constant)
    gl   : siemens (constant)
    Cm   : farad (constant)
'''

# Time domain
t = arange(0, out_traces.shape[1]*dt/ms, dt/ms)


def n_peaks(x):
    n_p = []
    for _x in x.transpose():
        p_i = find_peaks(_x, height=0)[0]
        n_p.append(p_i.size - sum(np.diff(t[p_i]) < 4))
    return n_p


# Step-by-step inference
# Start with the regular instatiation of the class
inferencer = Inferencer(dt=dt, model=eqs,
                        input={'I': inp_traces*amp},
                        output={'v': out_traces*mV},
                        features=[n_peaks,
                                  lambda x: x[(t > 5) & (t < 10), :].mean(axis=0),
                                  lambda x: x[(t > 5) & (t < 10), :].std(axis=0),
                                  lambda x: x.ptp(axis=0)],
                        method='exponential_euler',
                        threshold='m > 0.5',
                        refractory='m > 0.5',
                        param_init={'v': 'VT'})

# Data generation
# Initializing of the prior
prior = inferencer.init_prior(gl=[1e-09*siemens, 1e-07*siemens],
                              g_na=[2e-06*siemens, 2e-04*siemens],
                              g_kd=[6e-07*siemens, 6e-05*siemens],
                              Cm=[0.1*uF*cm**-2*area, 2*uF*cm**-2*area])
# Prepare training data
path_to_data = __file__[:-3] + '_data.npz'
if os.path.exists(path_to_data):
    theta, x = inferencer.load_summary_statistics(path_to_data)
else:
    # Generate training data
    theta = inferencer.generate_training_data(n_samples=1000,
                                              prior=prior)
    # Extract summary stats
    x = inferencer.extract_summary_statistics(theta=theta, level=0)
    # Save the data for later use
    inferencer.save_summary_statistics(path_to_data, theta, x)

# Amortized inference
# Training the neural density estimator
inference = inferencer.init_inference(inference_method='SNPE',
                                      density_estimator_model='maf',
                                      prior=prior,
                                      hidden_features=50,
                                      num_components=10)
# First round of inference where no observation data is set to posterior
posterior = inferencer.infere_step(proposal=prior,
                                   inference=inference,
                                   theta=theta, x=x,
                                   train_kwargs={'num_atoms': 10,
                                                 'learning_rate': 0.0005,
                                                 'show_train_summary': True})
# Storing the trained posterior without a default observation
path_to_posterior = __file__[:-3] + '_posterior.pth'
inferencer.save_posterior(path_to_posterior)
# Loading the trained posterior directly to Inferencer class
posterior_loaded = inferencer.load_posterior(path_to_posterior)

# Sampling from the posterior given observations, ...
inferencer.sample((10000, ), posterior=posterior_loaded)
# ...visualize the samples, ...
inferencer.pairplot(labels=[r'$\overline{g}_{l}$',
                            r'$\overline{g}_{Na}$',
                            r'$\overline{g}_{K}$',
                            r'$\overline{C}_{m}$'])
# ...and optionally, continue the multiround inference via ``infere`` method
posterior_multi_round = inferencer.infere(n_rounds=2)
inferencer.sample((10000, ))
inferencer.pairplot(labels=[r'$\overline{g}_{l}$',
                            r'$\overline{g}_{Na}$',
                            r'$\overline{g}_{K}$',
                            r'$\overline{C}_{m}$'])

# Generate traces from a single sample of parameters
inf_traces = inferencer.generate_traces()
nrows = 2
ncols = out_traces.shape[0]
fig, axs = subplots(nrows, ncols, sharex=True,
                    gridspec_kw={'height_ratios': [3, 1]}, figsize=(15, 4))
for idx in range(ncols):
    axs[0, idx].plot(t, out_traces[idx, :].T, label='measurements')
    axs[0, idx].plot(t, inf_traces[idx, :].T/mV, label='fits')
    axs[1, idx].plot(t, inp_traces[idx, :].T/nA, 'k-', label='stimulus')
    axs[1, idx].set_xlabel('t, ms')
    if idx == 0:
        axs[0, idx].set_ylabel('$v$, mV')
        axs[1, idx].set_ylabel('$I$, nA')
handles, labels = [(h + l) for h, l
                   in zip(axs[0, idx].get_legend_handles_labels(),
                   axs[1, idx].get_legend_handles_labels())]
fig.legend(handles, labels)
tight_layout()
show()