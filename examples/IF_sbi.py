from brian2 import *
from brian2modelfitting import *


# Set parameters
dt = 0.1*ms
sim_time = 60*ms
El = -70*mV
VT = -50*mV
DeltaT = 2*mV
ground_truth_params = {'gl': 30*nS, 'C': 1*nF}  # parameters to be inferred

# Set input trace
inp_trace = hstack([zeros(int(round(5*ms/dt))),
                    ones(int(round(25*ms/dt))),
                    zeros(int(round(30*ms/dt)))]) * 10*nA
I = TimedArray(inp_trace, dt=dt)

# Set model equations
eqs = '''
    dv/dt = (gl*(El - v) + gl*DeltaT*exp((v - VT) / DeltaT) + I(t)) / C : volt
    gl    : siemens (constant)
    C     : farad (constant)
    '''

# Run a model and create synthetic voltage traces
neurons = NeuronGroup(1, eqs, dt=dt,
                      threshold='v > -50 * mV',
                      reset='v = -70 * mV',
                      method='exponential_euler')
neurons.v = -70*mV
neurons.set_states(ground_truth_params)
state_monitor = StateMonitor(neurons, 'v', record=True)
spike_monitor = SpikeMonitor(neurons, record=True)
run(sim_time)
out_trace = state_monitor.v  # voltage traces in mV
spike_times = array(spike_monitor.t_)  # spike events in s

# Inference
start_scope()
El = -70*mV
VT = -50*mV
DeltaT = 2*mV
eqs_inf = '''
    dv/dt = (gl*(El - v) + gl*DeltaT*exp((v - VT) / DeltaT) + I_syn) / C : volt
    gl    : siemens (constant)
    C     : farad (constant)
    '''

# Instantiate the inferencer object
spike_features_list = [
        lambda x: x.size,  # number of spikes
        lambda x: 0. if diff(x).size == 0 else mean(diff(x)),  # mean ISI
]
inferencer = Inferencer(dt=dt, model=eqs_inf,
                        input={'I_syn': inp_trace.reshape(1, -1)},
                        output={'spikes': [spike_times]},
                        features={'spikes': spike_features_list},
                        method='exponential_euler',
                        threshold='v > -50*mV',
                        reset='v = -70*mV',
                        param_init={'v': -70*mV})

# Infer parameter posteriors given bounds

posterior = inferencer.infer(n_samples=3_000,
                             n_rounds=3,
                             gl=[10*nS, 100*nS],
                             C=[0.1*nF, 10*nF])

# Sample from the posterior
samples = inferencer.sample((3_000, ))

# Create a pairplot
labels = {'gl': r'$\overline{g}_{l}$', 'C': r'$C$'}
ticks = {'gl': [10*nS, 100*nS], 'C': [0.1*nF, 10*nF]}
inferencer.pairplot(labels=labels,
                    limits=ticks,
                    ticks=ticks,
                    points=ground_truth_params,
                    points_offdiag={'markersize': 9},
                    points_colors=['r'], 
                    figsize=(6, 6))

# Visualize the trace by sampling trained posterior
t = arange(0, inp_trace.size*dt/ms, dt/ms)
t_start, t_end = t[where(inp_trace != 0)[0][[0, -1]]]

# Identify spike events
spike_v = (out_trace.min()/mV, out_trace.max()/mV)
spike_i = []
for spike_time in spike_times:
    spike_i.append(where(isclose(spike_time * 1000, t))[0].item())

# Generate traces from a single sample of parameters
inf_trace = inferencer.generate_traces(output_var='v')

fig, axs = subplots(nrows, 1, sharex=True,
                    gridspec_kw={'height_ratios': [3, 1]}, figsize=(7, 3))
axs[0].plot(t, out_trace.T/mV, 'C3-', lw=3, label='recordings')
axs[0].plot(t, inf_trace.T/mV, 'k--', lw=2, label='sampled traces')
axs[0].vlines(t[spike_i], *spike_v, lw=3, color='C0', label='spike', zorder=3)
axs[0].set_ylabel('$V$, mV')
axs[0].legend()
axs[1].plot(t, inp_trace.T/nA, lw=3, c='k', label='stimuli')
axs[1].set_xlabel('$t$, ms')
axs[1].set_ylabel('$I$, nA')
axs[1].legend()
tight_layout()
show()
