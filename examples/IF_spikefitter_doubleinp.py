import numpy as np
from brian2 import *
from brian2modelfitting import *


# Generate Data to Fit into
dt = 0.01 * ms
defaultclock.dt = dt

# Generate a step-current input and an "experimental" voltage trace
input_current1 = np.hstack([np.zeros(int(5*ms/dt)), np.ones(int(5*ms/dt))*5, np.zeros(int(5*ms/dt))])* 5 *nA
input_current0 = np.hstack([np.zeros(int(5*ms/dt)), np.ones(int(5*ms/dt))*10, np.zeros(int(5*ms/dt))]) * 5 * nA

input_current2 = np.stack((input_current0, input_current1))
I = TimedArray(input_current0, dt=dt)

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
monitor0 = StateMonitor(group, 'v', record=True)
smonitor0  = SpikeMonitor(group)

run(60*ms)

voltage0 = monitor0.v[0]/mV
out_spikes0 = getattr(smonitor0, 't') / ms

start_scope()
I = TimedArray(input_current1, dt=dt)
group1 = NeuronGroup(1, eqs,
                    threshold='v > -50*mV',
                    reset='v = -70*mV',
                    method='exponential_euler')
group1.v = -70 *mV
group1.set_states({'gL': [30*nS], 'C':[1*nF]})
monitor1 = StateMonitor(group1, 'v', record=True)
smonitor1  = SpikeMonitor(group1)
run(60*ms)

out_spikes1 = getattr(smonitor1, 't') / ms
voltage1 = monitor1.v[0]/mV
inp_trace0 = np.array([input_current0])
inp_trace1 = np.array([input_current1])

inp_trace = np.concatenate((inp_trace0, inp_trace1))
out_spikes = np.array([out_spikes0, out_spikes1])
print('out_spikes', out_spikes)


# Model Fitting
start_scope()
EL = -70*mV
VT = -50*mV
DeltaT = 2*mV

eqs_fit = Equations('''
    dv/dt = (gL*(EL-v)+gL*DeltaT*exp((v-VT)/DeltaT) + I)/C : volt
    gL: siemens (constant)
    C: farad (constant)
    ''',
    # EL = -70*mV,
    # VT = -50*mV,
    # DeltaT = 2*mV,
    # C=1*nF
    )

n_opt = NevergradOptimizer('DE')
metric = GammaFactor(delta=60*ms, time=60*ms)


# pass parameters to the NeuronGroup
fitter = SpikeFitter(model=eqs_fit, input_var='I', dt=dt,
                     input=inp_trace * amp, output=out_spikes,
                     n_samples=30,
                     threshold='v > -50*mV',
                     reset='v = -70*mV',
                     param_init={'v': -70*mV},
                     method='exponential_euler',)
result_dict, error = fitter.fit(n_rounds=2,
                                optimizer=n_opt,
                                metric=metric,
                                callback='progressbar',
                                gL=[20*nS, 40*nS],
                                C = [0.5*nF, 1.5*nF])



print('goal:', {'gL': [30*nS], 'C':[1*nF]})
print('results:', result_dict['C']*farad, result_dict['gL']*siemens)

res = {'gL': [result_dict['gL']*siemens], 'C': [result_dict['C']*farad]}

# visualization of the results
start_scope()

EL = -70*mV
VT = -50*mV
DeltaT = 2*mV
spikes = fitter.generate_spikes(params=None, param_init={'v': -70*mV})
print('spike times:', spikes)

start_scope()
EL = -70*mV
VT = -50*mV
DeltaT = 2*mV
fits = fitter.generate(params=None,
                       output_var='v',
                       param_init={'v': -70*mV})


print('fits', fits)

fig, ax = plt.subplots(nrows=2)
ax[0].plot(voltage0);
ax[0].plot(fits[0]/mV);
ax[1].plot(voltage1);
ax[1].plot(fits[1]/mV);
plt.show()
