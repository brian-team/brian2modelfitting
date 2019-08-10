from brian2 import *
from brian2modelfitting import *

# set_device('cpp_standalone', directory='parallel', clean=False)

# create input and output
input_traces = zeros((10,1))*volt
for i in range(1):
    input_traces[1:,i]=i*10*mV

output_traces = 10*nS*input_traces

model = Equations('''
    I = g*(v-E) : amp
    g : siemens (constant)
    E : volt (constant)
    ''')

s_opt = SkoptOptimizer(method='gp', random_state=1)
metric = MSEMetric()

# pass parameters to the NeuronGroup
res, error = fit_traces(model=model, input_var='v', output_var='I',
                        input=input_traces, output=output_traces,
                        dt=0.1*ms, optimizer=s_opt, metric=metric,
                        g=[1*nS, 30*nS], E=[-20*mV, 100*mV],)

print(res, error)