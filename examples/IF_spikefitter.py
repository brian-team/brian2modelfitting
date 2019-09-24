import numpy as np
from brian2 import *
from brian2modelfitting import *

# Generate Spikes To Fit into
dt = 0.1 * ms
defaultclock.dt = dt
input_current = np.hstack([np.zeros(int(round(5*ms/dt))),
                           np.ones(int(round(25*ms/dt))),
                           np.zeros(int(round(30*ms/dt)))]) * 5*nA
I = TimedArray(input_current, dt=dt)

EL = -70*mV
VT = -50*mV
DeltaT = 2*mV
eqs = Equations('''
    dv/dt = (gL*(EL-v)+gL*DeltaT*exp((v-VT)/DeltaT) + I(t))/C : volt
    gL: siemens (constant)
    C: farad (constant)
    ''')

group = NeuronGroup(1, eqs,
                    threshold='v > -50*mV',
                    reset='v = -70*mV',
                    method='exponential_euler')
group.v = -70 *mV
group.set_states({'gL': [30*nS], 'C':[1*nF]})

monitor = StateMonitor(group, 'v', record=True)
smonitor  = SpikeMonitor(group)

run(60*ms)

voltage = monitor.v[0]/mV
out_spikes = getattr(smonitor, 't_')
print(out_spikes)

# Model Fitting
start_scope()
eqs_fit = Equations('''
    dv/dt = (gL*(EL-v)+gL*DeltaT*exp((v-VT)/DeltaT) + I)/C : volt
    gL: siemens (constant)
    C: farad (constant)
    EL: volt (constant)
    ''')

n_opt = NevergradOptimizer()
metric = GammaFactor(delta=1*ms, time=60*ms)
inp_trace = np.array([input_current])

# pass parameters to the NeuronGroup
fitter = SpikeFitter(model=eqs_fit, input_var='I', dt=dt,
                     input=inp_trace * amp, output=[out_spikes],
                     n_samples=30,
                     threshold='v > -50*mV',
                     param_init={'v': -70*mV},
                     reset='v = -70*mV',)

result_dict, error = fitter.fit(n_rounds=3,
                                optimizer=n_opt,
                                metric=metric,
                                callback='text',
                                gL=[10*nS, 100*nS],
                                C=[0.1*nF, 10*nF],
                                EL=[-100*mV, -50*mV])



print('goal:', 1*nF, 30*nS, -70*mV)
print('results:', result_dict['C']*farad, result_dict['gL']*siemens, result_dict['EL']*volt)

# visualization of the results
EL = -70*mV
VT = -50*mV
DeltaT = 2*mV
spikes = fitter.generate_spikes(params=None)
print('spike times:', spikes)

EL = -70*mV
VT = -50*mV
DeltaT = 2*mV
fits = fitter.generate(params=None,
                       output_var='v')

# Vizualize the resutls
plot(np.arange(len(voltage))*dt/ms, voltage, label='original')
plot(np.arange(len(voltage))*dt/ms, fits[0]/mV, label='fitted')
legend()
plt.show()
