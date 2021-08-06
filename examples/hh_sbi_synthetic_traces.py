from brian2modelfitting import *
from brian2 import *


# Input and output data traces simulation
dt = 0.05*ms
defaultclock.dt = dt
t_on = 50*ms
t_total = 350*ms
t_off = t_total - t_on
I = 1*nA
gleak = 10*nS
Eleak = -70*mV
VT = -60*mV
C = 200*pF
gNa = 32*uS
ENa = 53*mV
gK = 1*uS
EK = -107*mV
eqs = '''
    dVm/dt = -(gNa*m**3*h*(Vm - ENa) + gK*n**4*(Vm - EK) + gleak*(Vm - Eleak) - I_inj) / C : volt
    I_inj = int(t >= t_on and t < t_off)*I : amp (shared)
    dm/dt = alpham*(1-m) - betam*m : 1
    dn/dt = alphan*(1-n) - betan*n : 1
    dh/dt = alphah*(1-h) - betah*h : 1
    alpham = (-0.32/mV) * (Vm - VT - 13.*mV) / (exp((-(Vm - VT - 13.*mV))/(4.*mV)) - 1)/ms : Hz
    betam = (0.28/mV) * (Vm - VT - 40.*mV) / (exp((Vm - VT - 40.*mV)/(5.*mV)) - 1)/ms : Hz
    alphah = 0.128 * exp(-(Vm - VT - 17.*mV) / (18.*mV))/ms : Hz
    betah = 4/(1 + exp((-(Vm - VT - 40.*mV)) / (5.*mV)))/ms : Hz
    alphan = (-0.032/mV) * (Vm - VT - 15.*mV) / (exp((-(Vm - VT - 15.*mV)) / (5.*mV)) - 1)/ms : Hz
    betan = 0.5*exp(-(Vm - VT - 10.*mV) / (40.*mV))/ms : Hz
    '''
neurons = NeuronGroup(1, eqs, dt=dt,
                      threshold='m>0.5', refractory='m>0.5',
                      method='exponential_euler')
state_monitor = StateMonitor(neurons, 'Vm', record=True)
spike_monitor = SpikeMonitor(neurons, record=False)
neurons.Vm = 'Eleak'
neurons.m = '1/(1 + betam/alpham)'
neurons.h = '1/(1 + betah/alphah)'
neurons.n = '1/(1 + betan/alphan)'
run(t_total)
I_inj = ((state_monitor.t >= t_on) & (state_monitor.t < t_off)) * I
t = state_monitor.t
out_trace = state_monitor.Vm

# Inference
start_scope()
eqs = '''
    dVm/dt = -(gNa*m**3*h*(Vm - ENa) + gK*n**4*(Vm - EK) + gleak*(Vm - Eleak) - I_inj) / C : volt
    dm/dt = alpham*(1-m) - betam*m : 1
    dn/dt = alphan*(1-n) - betan*n : 1
    dh/dt = alphah*(1-h) - betah*h : 1
    alpham = (-0.32/mV) * (Vm - VT - 13.*mV) / (exp((-(Vm - VT - 13.*mV)) / (4.*mV)) - 1)/ms : Hz
    betam = (0.28/mV) * (Vm - VT - 40.*mV) / (exp((Vm - VT - 40.*mV) / (5.*mV)) - 1)/ms : Hz
    alphah = 0.128 * exp(-(Vm - VT - 17.*mV) / (18.*mV))/ms : Hz
    betah = 4/(1 + exp((-(Vm - VT - 40.*mV)) / (5.*mV)))/ms : Hz
    alphan = (-0.032/mV) * (Vm - VT - 15.*mV) / (exp((-(Vm - VT - 15.*mV)) / (5.*mV)) - 1)/ms : Hz
    betan = 0.5*exp(-(Vm - VT - 10.*mV) / (40.*mV))/ms : Hz

    # The parameters to fit
    gNa : siemens (constant)
    gK : siemens (constant)
    '''
inferencer = Inferencer(dt=dt, model=eqs,
                        input={'I_inj': I_inj.reshape(1, -1)},
                        output={'Vm': out_trace},
                        method='exponential_euler',
                        threshold='m>0.5', refractory='m>0.5',
                        param_init={'Vm': 'Eleak',
                                    'm': '1/(1 + betam/alpham)',
                                    'h': '1/(1 + betah/alphah)',
                                    'n': '1/(1 + betan/alphan)'})

# Amortized inference
inferencer.infer(n_samples=10_000,
                 inference_method='SNPE',
                 density_estimator_model='mdn',
                 gNa=[.5*uS, 80.*uS],
                 gK=[1e-4*uS, 15.*uS])

# Sample from posterior
inferencer.sample((10_000, ))

# Visualize estimated posterior distribution
inferencer.pairplot(limits={'gNa': [.5*uS, 80.*uS],
                            'gK': [1e-4*uS, 15.*uS]},
                    ticks={'gNa': [.5*uS, 80.*uS],
                           'gK': [1e-4*uS, 15.*uS]},
                    labels={'gNa': r'$\overline{g}_\mathrm{Na}$',
                            'gK': r'$\overline{g}_\mathrm{K}$'},
                    points={'gNa': 32*uS,
                            'gK': 1*uS},
                    points_offdiag={'markersize': 6},
                    points_colors=['r'],
                    figsize=(6, 6))

# Visualize traces for a single sample from posterior
inf_trace = inferencer.generate_traces()
fig, axs = subplots(2, 1, sharex=True,
                    gridspec_kw={'height_ratios': [3, 1]},
                    figsize=(6, 4))
axs[0].plot(t, out_trace.ravel()/mV, label='simulated recordings')
axs[0].plot(t, inf_trace.ravel()/mV, '--', label='posterior sample')
axs[0].set(ylabel='Vm, mV')
axs[0].legend()
axs[1].plot(t, I_inj.ravel()/nA, 'k-', label='stimulus')
axs[1].set(xlabel='t, ms', ylabel='I, nA')
axs[1].legend()
show()
