from brian2 import *
from brian2modelfitting import *

# set_device('cpp_standalone', directory='parallel', clean=False)

# create input and output
input_traces = zeros((10,5))*volt
for i in range(5):
    input_traces[5:,i]=i*10*mV

output_traces = 10*nS*input_traces

model = Equations('''
    I = g*(v-E) : amp
    g : siemens (constant)
    E : volt (constant)
    ''')

n_opt = NevergradOptimizer()
metric =  MSEMetric()

# pass parameters to the NeuronGroup
fitter = TraceFitter(model=model, dt=0.1*ms,
                     input={'v': input_traces}, output={'I': output_traces},
                     n_samples=10)

res, error = fitter.fit(n_rounds=2,
                        optimizer=n_opt, metric=metric,
                        g=[1*nS, 30*nS], E=[-20*mV, 100*mV],
                        callback='progressbar')
print(res, error)
