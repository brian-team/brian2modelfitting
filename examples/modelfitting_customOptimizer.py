""" 
This is a minimal example that demonstrates the flexibility of brian2modelfitting specially in working with different back-end optimization libraries.
brian2modelfitting can easily work with custom optimizers provided by the user as long as they inherit from Optimizer() class and follow an ask() / tell interface.
We will use a simple wrapper for Dragonfly optimizers, an open source library for scalable Bayesian optimisation. --> https://dragonfly-opt.readthedocs.io/en/master/
"""
# pip install dragonfly-opt -v
from dragonfly.exd import domains
from dragonfly.exd.experiment_caller import EuclideanFunctionCaller
from dragonfly.opt import random_optimiser, gp_bandit, ga_optimiser
import pandas as pd

from brian2 import *
from brian2modelfitting import *

# ---------------------------- DragonFlyOptimizer ---------------------------- #


class DragonFlyOptimizer(Optimizer):
    def __init__(self, method="bo", **kwds):
        super(Optimizer, self).__init__()
        if not(method.lower() in ["bo", "rand"]):
            raise AssertionError(
                "Provided method: {} is Unknown to dragonFly optimizer ".format(method))
        self.tested_parameters = []
        self.errors = []
        self.method = method.lower()
        self.kwds = kwds
        self.optim = None

    def initialize(self, parameter_names, popsize, rounds, **params):
        self.tested_parameters = []
        self.errors = []
        for param in params.keys():
            if param not in parameter_names:
                raise ValueError("Parameter %s must be defined as a parameter "
                                 "in the model" % param)

        bounds = calc_bounds(parameter_names, **params)

        # nx2 array of boundaries for each parameter
        instruments = np.array([bounds[i] for i in range(len(bounds))])
        domain = domains.EuclideanDomain(instruments)

        func_caller = EuclideanFunctionCaller(None, domain)
        if self.method == 'bo':  # Bayesian optimisation
            self.optim = gp_bandit.EuclideanGPBandit(
                func_caller, ask_tell_mode=True)
        elif self.method == 'rand':  # Random Search
            self.optim = random_optimiser.EuclideanRandomOptimiser(
                func_caller, ask_tell_mode=True)
        self.optim.initialise()

        return popsize

    def ask(self, n_samples):
        return self.optim.ask(n_points=n_samples)

    def tell(self, parameters, errors):
        self.tested_parameters.extend(parameters)
        # The default of DragonFly is to maximize the objective
        # so we will will use the error with negative sign
        errors = -1*np.array(errors)
        self.optim.tell([(parameters, errors[0])])
        # reversing the error's sign before appending it to errors list
        self.errors.extend((-1*errors).tolist())

    def recommend(self):
        return self.tested_parameters[argmin(self.errors)]

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #


# helper func.
def visualize_results(fitter, n_traces):
    # Show results
    all_output = fitter.results(format='dataframe')
    print(all_output)

    # Visualization of the results
    fits = fitter.generate_traces(params=None, param_init={'v': -65*mV})

    fig, axes = plt.subplots(ncols=n_traces, figsize=(20, 5), sharey=True)

    for ax, data, fit in zip(axes, out_traces, fits):
        ax.plot(data.transpose())
        ax.plot(fit.transpose()/mV)
    plt.show()


def load_data(input_data_file='input_traces_hh.csv', output_data_file='output_traces_hh.csv'):
    # Load Input and Output Data
    df_inp_traces = pd.read_csv(input_data_file)
    df_out_traces = pd.read_csv(output_data_file)

    inp_traces = df_inp_traces.to_numpy()
    inp_traces = inp_traces[:, 1:]

    out_traces = df_out_traces.to_numpy()
    out_traces = out_traces[:, 1:]
    return inp_traces, out_traces


if __name__ == '__main__':
    # Load Input and Output Data
    inp_traces, out_traces = load_data(
        input_data_file='input_traces_hh.csv', output_data_file='output_traces_hh.csv')
    # Parameters
    area = 20000*umetre**2
    El = -65*mV
    EK = -90*mV
    ENa = 50*mV
    VT = -63*mV
    dt = 0.01*ms
    defaultclock.dt = dt

    # Hodgkin-Huxley Model Definition
    eqs = Equations(
        '''
    dv/dt = (gl*(El-v) - g_na*(m*m*m)*h*(v-ENa) - g_kd*(n*n*n*n)*(v-EK) + I)/Cm : volt
    dm/dt = 0.32*(mV**-1)*(13.*mV-v+VT)/
        (exp((13.*mV-v+VT)/(4.*mV))-1.)/ms*(1-m)-0.28*(mV**-1)*(v-VT-40.*mV)/
        (exp((v-VT-40.*mV)/(5.*mV))-1.)/ms*m : 1
    dn/dt = 0.032*(mV**-1)*(15.*mV-v+VT)/
        (exp((15.*mV-v+VT)/(5.*mV))-1.)/ms*(1.-n)-.5*exp((10.*mV-v+VT)/(40.*mV))/ms*n : 1
    dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms*(1.-h)-4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1
    g_na : siemens (constant)
    g_kd : siemens (constant)
    gl   : siemens (constant)
    Cm   : farad (constant)
    ''')

    df_opt = DragonFlyOptimizer(method="bo")
    metric = MSEMetric()

    # Fitting
    fitter = TraceFitter(model=eqs, input_var='I', output_var='v',
                         input=inp_traces*amp, output=out_traces*mV, dt=dt,
                         n_samples=1,
                         param_init={'v': -65*mV},
                         method='exponential_euler')

    res, error = fitter.fit(n_rounds=6,
                            optimizer=df_opt, metric=metric,
                            callback='progressbar',
                            gl=[1e-09 * siemens, 1e-07 * siemens],
                            g_na=[2e-06*siemens, 2e-04*siemens],
                            g_kd=[6e-07*siemens, 6e-05*siemens],
                            Cm=[0.1*ufarad*cm**-2 * area, 2*ufarad*cm**-2 * area])

    visualize_results(fitter, out_traces.shape[0])
