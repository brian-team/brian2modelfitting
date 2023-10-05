import numpy as np
from brian2 import *
from brian2modelfitting import *

'''
Adaptive exponential IF model introduced by Brette R. and Gerstner W. (2005). The model has 3 distinct firing regiems 
depending on the parameters used in the simulation. This example shows how multiple sets of parameters can 
be fitted to the same model simulatenously with the Brian2modelfitting toolbox.
'''

# Parameters
C = 281 * pF
gL = 30 * nS
taum = C / gL
EL = -70.6 * mV
VT = -50.4 * mV
DeltaT = 2 * mV
Vcut = VT + 5 * DeltaT

dt = 0.1 * ms
defaultclock.dt = dt

#Define input current 

input_current1 = np.hstack([np.zeros(int(round(20*ms/dt))),
                           np.ones(int(round(100*ms/dt))),
                           np.zeros(int(round(20*ms/dt)))]) * 1
input_current2 = np.hstack([np.zeros(int(round(20*ms/dt))),
                           np.ones(int(round(100*ms/dt))),
                           np.zeros(int(round(20*ms/dt)))]) * 2
input_current = np.stack((input_current1, input_current2))*nA
I = TimedArray(input_current.T, dt=dt)

#First simulate the model with known parameters (to obtain data to fit parameters too!). 
#Define model, setup monitoring & define parameters. 

eqs = """
dvm/dt = (gL*(EL - vm) + gL*DeltaT*exp((vm - VT)/DeltaT) + I(t, i%2==1) - w)/C : volt
dw/dt = (a*(vm - EL) - w)/tauw : amp
tauw : second 
a : siemens
b : amp
Vr : volt
"""

neuron = NeuronGroup(6, model=eqs, threshold='vm>Vcut',
                     reset="vm=Vr; w+=b", method='euler')
neuron.vm = EL
trace = StateMonitor(neuron, 'vm', record=True)
spikes = SpikeMonitor(neuron)

neuron.tauw = [144, 144, 20, 20, 144, 144] * ms
neuron.a = [4*nS, 4*nS, 4*nS, 4*nS, (2*C)/(144*ms), (2*C)/(144*ms)]
neuron.b = [0.0805, 0.0805, 0.5, 0.5, 0, 0] * nA
neuron.Vr = [-70.6*mV, -70.6*mV, VT+5*mV, VT+5*mV, -70.6*mV, -70.6*mV]

run(140 * ms) 

spike_train = spikes.spike_trains()

#Put spike spikes and voltage traces into lists , probably lots of smarter ways to do this.
out_spikes = [[spike_train[0] / second, spike_train[1] /second], [spike_train[2] / second, spike_train[3] / second], [spike_train[4] / second, spike_train[5] / second]]
v_traces = [[trace.vm[0]/mV, trace.vm[1]/mV], [trace.vm[2]/mV, trace.vm[3]/mV], [trace.vm[4]/mV, trace.vm[5]/mV]]

start_scope()

#Model fitting

eqs_fit = '''
dvm/dt = (gL*(EL - vm) + gL*DeltaT*exp((vm - VT)/DeltaT) + I - w)/C : volt
dw/dt = (a*(vm - EL) - w)/tauw : amp
tauw : second (constant)
a : siemens (constant)
b : amp (constant)
Vr : volt (constant)
'''

n_opt = NevergradOptimizer()
metric = GammaFactor(delta=1*ms, time=140*ms)

fitters = []
for i in range(len(out_spikes)):
    fitters.append(SpikeFitter(model=eqs_fit, input_var='I', dt=dt,
                     input=input_current, output=out_spikes[i],
                     n_samples=800,
                     param_init={'vm': EL},
                     threshold='vm > Vcut',
                     reset='vm=Vr; w+=b',))

result_dict_error = []
predict_spikes = []
fits = []
for fitter in fitters:
    result_dict_error.append(fitter.fit(n_rounds=15,
                                    optimizer=n_opt,
                                    metric=metric,
                                    callback='progressbar',
                                    tauw=[1,200]*ms,
                                    a=[3, 4]*nS, 
                                    b=[0, 1]*nA,
                                    Vr=[-80,-40]*mV))

    predict_spikes.append(fitter.generate_spikes(params=None))
    #print('spike times:', spikes)

    fits.append(fitter.generate(params=None,
                           output_var='vm'))
    
#Number of samples (n_samples) and number of epochs (n_rounds) is probably overkill here, can probably be reduced.

#Print parameter results and plot fitted traces
print(f'Printing fitting results...\n')

for i in range(3):
    print(f'Results for parameter set {i+1}\n')
    print(f'tauw (true/predict) : {neuron.tauw[i*2]}, {result_dict_error[i][0]["tauw"]}')
    print(f'Vr (true/predict) : {neuron.Vr[i*2]}, {result_dict_error[i][0]["Vr"]}')
    print(f'a (true/predict) : {neuron.a[i*2]}, {result_dict_error[i][0]["a"]}')
    print(f'b (true/predict) : {neuron.b[i*2]}, {result_dict_error[i][0]["b"]}\n')

fig, ax = plt.subplots(2, 3, figsize=(12, 8))

for i in range(3):
    ax[0, i].plot(trace.t/ms, v_traces[i][0], label='Simulated');
    ax[0, i].plot(trace.t/ms, fits[i][0]/mV, label='Fitted');
    ax[1, i].plot(trace.t/ms, v_traces[i][1]);
    ax[1, i].plot(trace.t/ms, fits[i][1]/mV);
    
ax[0, 0].legend()
ax[0, 0].set_xlabel('Time (ms)')
ax[0, 0].set_ylabel('v (mV)')

plt.show()