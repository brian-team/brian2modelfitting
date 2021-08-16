from brian2 import *
from brian2modelfitting import *
import pandas as pd
from scipy.stats import kurtosis as kurt


# Load input and output data traces
df_inp_traces = pd.read_csv('input_traces_hh.csv')
df_out_traces = pd.read_csv('output_traces_hh.csv')
inp_traces = df_inp_traces.to_numpy()
inp_traces = inp_traces[[0, 1, 3, 4], 1:]
out_traces = df_out_traces.to_numpy()
out_traces = out_traces[[0, 1, 3, 4], 1:]

# Define model and its parameters
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

# Visualize the recordings
nrows = 2
ncols = out_traces.shape[0]
fig, axs = subplots(nrows, ncols, sharex=True,
                    gridspec_kw={'height_ratios': [3, 1]}, figsize=(12, 3))
for idx in range(ncols):
    axs[0, idx].plot(t, out_traces[idx, :].T, lw=3, c='C3', label='recordings')
    axs[1, idx].plot(t, inp_traces[idx, :].T/nA, lw=3, c='k', label='stimulus')
    axs[1, idx].set_xlabel('$t$, ms')
    if idx == 0:
        axs[0, idx].set_ylabel('$V$, mV')
        axs[1, idx].set_ylabel('$I$, nA')
handles, labels = [(h + l) for h, l
                   in zip(axs[0, idx].get_legend_handles_labels(),
                   axs[1, idx].get_legend_handles_labels())]
fig.legend(handles, labels)
tight_layout()


# Obtain spike times manually from recordings
def get_spike_times(x):
    x = x.copy()
    # put everything to -40 mV that is below -40 mV or has negative slope
    ind = where(x < -0.04)
    x[ind] = -0.04
    ind = where(diff(x) < 0)
    x[ind] = -0.04

    # remaining negative slopes are at spike peaks
    ind = where(diff(x) < 0)
    spike_times = array(t)[ind]

    # number of spikes
    if spike_times.shape[0] > 0:
        spike_times = spike_times[
            append(1, diff(spike_times)) > 0.5
        ]
    return spike_times / 1000  # in seconds


# store spike times for each trace into the list
spike_times = [get_spike_times(out_trace) for out_trace in out_traces]

# Visualize the recordings and spikes while stimulus is on
nrows = 2
ncols = out_traces.shape[0]
fig, axs = subplots(nrows, ncols, sharex=True,
                    gridspec_kw={'height_ratios': [3, 1]}, figsize=(12, 3))
for idx in range(ncols):
    spike_idx = in1d(t, spike_times[idx] * 1000).nonzero()[0]
    spike_v = (out_traces[idx, :].min(), out_traces[idx, :].max())
    axs[0, idx].plot(t, out_traces[idx, :].T, lw=3, c='C3',
                     label='recordings', zorder=1)
    axs[0, idx].vlines(t[spike_idx], *spike_v, lw=3, color='C0',
                       label='spike event', zorder=2)
    axs[1, idx].plot(t, inp_traces[idx, :].T/nA, lw=3, c='k', label='stimulus')
    axs[1, idx].set_xlabel('$t$, ms')
    if idx == 0:
        axs[0, idx].set_ylabel('$V$, mV')
        axs[1, idx].set_ylabel('$I$, nA')
handles, labels = [(h + l) for h, l
                   in zip(axs[0, idx].get_legend_handles_labels(),
                   axs[1, idx].get_legend_handles_labels())]
fig.legend(handles, labels)
tight_layout()

# Define features to create a summary statistics representation of traces
v_features = [
    lambda x: max(x[(t > t_start) & (t < t_end)]),  # AP max
    lambda x: mean(x[(t > t_start) & (t < t_end)]),  # AP mean
    lambda x: std(x[(t > t_start) & (t < t_end)]),  # AP std
    lambda x: kurt(x[(t > t_start) & (t < t_end)], fisher=False),  # AP kurt
    lambda x: mean(x[(t > .25 * t_start) & (t < .75 * t_start)]),  # resting
    lambda x: mean(x[(t > t_end) & (t <= max(t))]),  # steady-state
]
s_features = [lambda x: x.size]  # the number of spikes in a spike train

# Simulation-based inference object instantiation
inferencer = Inferencer(dt=dt, model=eqs,
                        input={'I': inp_traces*amp},
                        output={'v': out_traces*mV, 'spikes': spike_times},
                        features={'v': v_features, 'spikes': s_features},
                        method='exponential_euler',
                        threshold='m > 0.5',
                        refractory='m > 0.5',
                        param_init={'v': 'VT'})

# Multi-round inference
posterior = inferencer.infer(n_samples=5_000,
                             n_rounds=3,
                             inference_method='SNPE',
                             density_estimator_model='mdn',
                             gl=[1e-09*siemens, 1e-07*siemens],
                             g_na=[2e-06*siemens, 2e-04*siemens],
                             g_kd=[6e-07*siemens, 6e-05*siemens],
                             Cm=[0.1*uF*cm**-2*area, 2*uF*cm**-2*area])

# Draw samples from posterior
samples = inferencer.sample((5_000, ))

# Visualize pairplot and conditional pairplot
limits = {'gl': [1e-9*siemens, 1e-07*siemens],
          'g_na': [2e-06*siemens, 2e-04*siemens],
          'g_kd': [6e-07*siemens, 6e-05*siemens],
          'Cm': [0.1*uF*cm**-2*area, 2*uF*cm**-2*area]}
labels = {'gl': r'$\overline{g}_{l}$',
          'g_na': r'$\overline{g}_{Na}$',
          'g_kd': r'$\overline{g}_{K}$',
          'Cm': r'$C_{m}$'}
inferencer.pairplot(limits=limits,
                    labels=labels,
                    ticks=limits,
                    figsize=(6, 6))
condition = inferencer.sample((1, ))
inferencer.conditional_pairplot(condition=condition,
                                limits=limits,
                                labels=labels,
                                ticks=limits,
                                figsize=(6, 6))

# Construct and visualize conditional coefficient matrix
cond_coeff_mat = inferencer.conditional_corrcoeff(condition=condition,
                                                  limits=limits)
fig, ax = subplots(1, 1, figsize=(4, 4))
im = imshow(cond_coeff_mat, clim=[-1, 1])
_ = fig.colorbar(im)

# Generate traces and visualize from a single sample of parameters
inf_traces = inferencer.generate_traces(output_var='v')

nrows = 2
ncols = out_traces.shape[0]
fig, axs = subplots(nrows, ncols, sharex=True,
                    gridspec_kw={'height_ratios': [3, 1]}, figsize=(12, 3))
for idx in range(ncols):
    spike_idx = in1d(t, spike_times[idx]).nonzero()[0]
    spike_v = (out_traces[idx, :].min(), out_traces[idx, :].max())
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
