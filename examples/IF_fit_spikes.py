import numpy as np
from brian2 import *
from brian2modelfitting import *

dt = 0.01 * ms
defaultclock.dt = dt
input_current = np.hstack([np.zeros(int(5*ms/dt)), np.ones(int(5*ms/dt)*5), np.zeros(5*int(5*ms/dt))])* 5 * nA
I = TimedArray(input_current, dt=dt)

# C = 1*nF
# gL = 30*nS
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
out_spikes = getattr(smonitor, 't') / ms
print(out_spikes)


start_scope()
eqs_fit = Equations('''
    dv/dt = (gL*(EL-v)+gL*DeltaT*exp((v-VT)/DeltaT) + I)/C : volt
    gL: siemens (constant)
    C: farad (constant)
    ''',
    EL = -70*mV,
    VT = -50*mV,
    DeltaT = 2*mV,
    # C=1*nF
    )

n_opt = NevergradOptimizer()
metric = GammaFactor(dt, 100*ms)
inp_trace = np.array([input_current])

# pass parameters to the NeuronGroup
result_dict, error = fit_spikes(model=eqs_fit, input_var='I',
                                input=inp_trace * amp, output=out_spikes,
                                dt=dt,
                                n_rounds=2, n_samples=30, optimizer=n_opt,
                                metric=metric,
                                threshold='v > -50*mV',
                                reset='v = -70*mV',
                                method='exponential_euler',
                                param_init={'v': -70*mV},
                                gL=[20*nS, 40*nS],
                                C=[0.5*nF, 1.5*nF]
                                )

print('goal:', {'gL': 30*nS, 'C':1*nF})
print('results:', result_dict['C']*farad, result_dict['gL']*siemens)

# visualization of the results
start_scope()
spikes = generate_fits(model=eqs_fit,
                       params=result_dict,
                       input=inp_trace * amp,
                       input_var='I',
                       output_var='spikes',
                       dt=dt,
                       threshold='v > -50*mV',
                       reset='v = -70*mV',
                       method='exponential_euler',
                       param_init={'v': -70*mV})

print('spike times:', spikes)

start_scope()
fits = generate_fits(model=eqs_fit,
                     params=result_dict,
                     input=inp_trace * amp,
                     input_var='I',
                     output_var='v',
                     dt=dt,
                     threshold='v > -50*mV',
                     reset='v = -70*mV',
                     method='exponential_euler',
                     param_init={'v': -70*mV})

# Vizualize the resutls
plot(voltage);
plot(fits[0]/mV)
plt.show()
