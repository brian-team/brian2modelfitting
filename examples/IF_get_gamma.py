import numpy as np

from brian2 import *
from brian2modelfitting import *

dt = 0.01 * ms
defaultclock.dt = dt
input_current = np.hstack([np.zeros(int(5*ms/dt)), np.ones(int(5*ms/dt)*5), np.zeros(5*int(5*ms/dt))])* 5 * nA
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


start_scope()
I = TimedArray(input_current, dt=dt)
group = NeuronGroup(2, eqs,
                    threshold='v > -50*mV',
                    reset='v = -70*mV',
                    method='exponential_euler')

group.v = -70 *mV
group.set_states({'gL': [30*nS, 30*nS], 'C':[1*nF, 2*nF]})
group.get_states()

monitor = StateMonitor(group, 'v', record=True)
smonitor  = SpikeMonitor(group)

run(60*ms)

voltage = monitor.v[0]/mV
voltage1 = monitor.v[1]/mV

spike_times = smonitor.t_[:]

spike_trains = smonitor.spike_trains()
st0 = spike_trains[0] / ms
st1 = spike_trains[1] / ms

gf = get_gamma_factor(st1, st0, 60*ms, dt)
print(gf)

plot(voltage);
plot(voltage1);
plt.show()
