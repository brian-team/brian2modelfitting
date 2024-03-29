import os

from brian2 import *
from brian2modelfitting import *
import pandas as pd


# Load Input and Output Data
dirname = os.path.dirname(__file__)
df_inp_traces = pd.read_csv(os.path.join(dirname, 'input_traces_hh.csv'))
df_out_traces = pd.read_csv(os.path.join(dirname, 'output_traces_hh.csv'))
inp_traces = df_inp_traces.to_numpy()
inp_traces = inp_traces[[0, 1, 3], 1:]
out_traces = df_out_traces.to_numpy()
out_traces = out_traces[[0, 1, 3], 1:]

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
t_start, t_end = t[where(inp_traces[0, :] != 0)[0][[0, -1]]]

# Step-by-step inference
# Start with the regular instatiation of the class
inferencer = Inferencer(dt=dt, model=eqs,
                        input={'I': inp_traces*amp},
                        output={'v': out_traces*mV},
                        features={'v': [lambda x: max(x),
                                        lambda x: mean(x[(t > t_start) & (t < t_end)]),
                                        lambda x: std(x[(t > t_start) & (t < t_end)])]},
                        method='exponential_euler',
                        threshold='m > 0.5',
                        refractory='m > 0.5',
                        param_init={'v': 'VT'})

# Training data generation
# Initializing the prior
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
    theta = inferencer.generate_training_data(n_samples=10_000,
                                              prior=prior)
    # Extract summary stats
    x = inferencer.extract_summary_statistics(theta)
    # Save the data for later use
    inferencer.save_summary_statistics(path_to_data, theta, x)

# Amortized inference
# Training the neural density estimator
inference = inferencer.init_inference(inference_method='SNPE',
                                      density_estimator_model='mdn',
                                      prior=prior)
# First round of inference where no observation data is set to posterior
posterior_amortized = inferencer.infer_step(proposal=prior,
                                            inference=inference,
                                            theta=theta, x=x)
# Storing the trained posterior without a default observation
path_to_posterior = __file__[:-3] + '_posterior_amortized.pth'
inferencer.save_posterior(path_to_posterior)

# Sampling from the posterior given observations, ...
inferencer.sample((10_000, ))
# ...visualize the samples, ...
labels = {'gl': r'$\overline{g}_\mathrm{l}$',
          'g_na': r'$\overline{g}_\mathrm{Na}$',
          'g_kd': r'$\overline{g}_\mathrm{K}$',
          'Cm': r'$\overline{C}_{m}$'}
inferencer.pairplot(labels=labels)

# ...and optionally, continue the multiround inference using ``infer`` method
posterior_multiround = inferencer.infer()
inferencer.sample((10_000, ))
inferencer.pairplot(labels=labels)

# Generate traces from a single sample of parameters
inf_traces = inferencer.generate_traces()
nrows = 2
ncols = out_traces.shape[0]
fig, axs = subplots(nrows, ncols, sharex=True,
                    gridspec_kw={'height_ratios': [3, 1]},
                    figsize=(ncols * 3, 3))
for idx in range(ncols):
    axs[0, idx].plot(t, out_traces[idx, :].T, 'C3-', lw=3, label='recordings')
    axs[0, idx].plot(t, inf_traces[idx, :].T/mV, 'k--', lw=2,
                     label='sampled traces')
    axs[1, idx].plot(t, inp_traces[idx, :].T/nA, lw=3, c='k', label='stimuli')
    axs[1, idx].set_xlabel('$t$, ms')
    if idx == 0:
        axs[0, idx].set_ylabel('$V$, mV')
        axs[1, idx].set_ylabel('$I$, nA')
handles, labels = [(h + l) for h, l
                   in zip(axs[0, idx].get_legend_handles_labels(),
                   axs[1, idx].get_legend_handles_labels())]
fig.legend(handles, labels)
tight_layout()
show()
