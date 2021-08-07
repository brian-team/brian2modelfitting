from brian2 import *
from brian2modelfitting import *


# Generate input and output data traces
dt = 0.1*ms
defaultclock.dt = dt
inp_trace = hstack([zeros(int(round(5*ms/dt))),
                    ones(int(round(25*ms/dt))),
                    zeros(int(round(30*ms/dt)))]) * 10*nA
I = TimedArray(inp_trace, dt=dt)
El = -70*mV
VT = -50*mV
DeltaT = 2*mV
eqs = '''
    dv/dt = (gl*(El - v) + gl*DeltaT*exp((v - VT) / DeltaT) + I(t)) / C : volt
    gl    : siemens (constant)
    C     : farad (constant)
    '''

neurons = NeuronGroup(1, eqs,
                      threshold='v > -50 * mV',
                      reset='v = -70 * mV',
                      method='exponential_euler')
neurons.v = -70*mV
neurons.set_states({'gl': 30*nS,
                    'C': 1*nF})
monitor = StateMonitor(neurons, 'v', record=True)
run(60*ms)
out_trace = monitor.v

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

inferencer = Inferencer(dt=dt, model=eqs_inf,
                        input={'I_syn': inp_trace.reshape(1, -1)},
                        output={'v': out_trace},
                        method='exponential_euler',
                        threshold='v > -50 * mV',
                        reset='v = -70 * mV',
                        param_init={'v': -70 * mV})

inferencer.infer(n_samples=10_000,
                 gl=[10*nS, 100*nS],
                 C=[0.1*nF, 10*nF])

inferencer.sample((10000, ))
inferencer.pairplot(labels={'gl': r'$\overline{g}_{l}$',
                            'C': r'$C$'},
                    points={'gl': 30*nS,
                            'C': 1*nF},
                    limits={'gl': [10*nS, 100*nS],
                            'C': [0.1*nF, 10*nF]},
                    ticks={'gl': [10*nS, 100*nS],
                           'C': [0.1*nF, 10*nF]},
                    points_offdiag={'markersize': 9},
                    points_colors=['r'])
show()
