import numpy as np
from brian2 import *
from brian2modelfitting import *


# Generate Data to Fit into
dt = 0.1 * ms
defaultclock.dt = dt

# Generate a step-current input and an "experimental" voltage trace
input_current0 = np.hstack([np.zeros(int(round(5*ms/dt))),
                           np.ones(int(round(25*ms/dt))),
                           np.zeros(int(round(30*ms/dt)))]) * 2.5
input_current1 = np.hstack([np.zeros(int(round(5*ms/dt))),
                           np.ones(int(round(25*ms/dt))),
                           np.zeros(int(round(30*ms/dt)))]) * 5
input_current2 = np.stack((input_current0, input_current1))*nA
I = TimedArray(input_current2.T, dt=dt)

EL = -70*mV
VT = -50*mV
DeltaT = 2*mV
eqs = Equations('''
    dv/dt = (gL*(EL-v)+gL*DeltaT*exp((v-VT)/DeltaT) + I(t, i))/C : volt
    gL: siemens (constant)
    C: farad (constant)
    ''')

group = NeuronGroup(2, eqs,
                    threshold='v > -50*mV',
                    reset='v = -70*mV',
                    method='exponential_euler')
group.v = -70 *mV
group.set_states({'gL': [30*nS], 'C':[1*nF]})
monitor = StateMonitor(group, 'v', record=True)
smonitor = SpikeMonitor(group)

run(60*ms)

voltage0 = monitor.v[0]/mV
voltage1 = monitor.v[1]/mV
spike_trains = smonitor.spike_trains()
out_spikes0 = spike_trains[0] / second
out_spikes1 = spike_trains[1] / second

out_spikes = [out_spikes0, out_spikes1]
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
metric = GammaFactor(delta=1*ms, time=60*ms)


# pass parameters to the NeuronGroup
fitter = SpikeFitter(model=eqs_fit, input_var='I', dt=dt,
                     input=input_current2, output=out_spikes,
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
                                C=[0.5*nF, 1.5*nF])



print('goal:', {'gL': [30*nS], 'C':[1*nF]})
print('results:', result_dict['C']*farad, result_dict['gL']*siemens)

res = {'gL': [result_dict['gL']*siemens], 'C': [result_dict['C']*farad]}

# visualization of the results
start_scope()

EL = -70*mV
VT = -50*mV
DeltaT = 2*mV
spikes = fitter.generate_spikes(params=None)
print('spike times:', spikes)

start_scope()
EL = -70*mV
VT = -50*mV
DeltaT = 2*mV
fits = fitter.generate(params=None,
                       output_var='v')


print('fits', fits)

fig, ax = plt.subplots(nrows=2)
ax[0].plot(voltage0);
ax[0].plot(fits[0]/mV);
ax[1].plot(voltage1);
ax[1].plot(fits[1]/mV);
plt.show()
