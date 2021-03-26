""" 
This is a minimal example that demonstrates the flexibility of brian2modelfitting specially in working with different back-end optimization libraries.
brian2modelfitting can easily work with custom optimizers provided by the user as long as they inherit from Optimizer() class and follow an ask() / tell interface.
We will use a simple wrapper for Dragonfly optimizers, an open source library for scalable Bayesian optimisation. --> https://dragonfly-opt.readthedocs.io/en/master/
"""
# pip install dragonfly-opt -v
from dragonfly.exd import domains
from dragonfly.exd.experiment_caller import EuclideanFunctionCaller
from dragonfly.opt import random_optimiser, gp_bandit, ga_optimiser
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


if __name__ == '__main__':
    # create input and output
    input_traces = zeros((10, 5))*volt
    for i in range(5):
        input_traces[5:, i] = i*10*mV

    output_traces = 10*nS*input_traces

    model = Equations('''
        I = g*(v-E) : amp
        g : siemens (constant)
        E : volt (constant)
        ''')

    df_opt = DragonFlyOptimizer(method="bo")
    metric = MSEMetric()

    # passing parameters to the NeuronGroup
    fitter = TraceFitter(model=model, dt=0.1*ms,
                         input_var='v', output_var='I',
                         input=input_traces, output=output_traces,
                         n_samples=1,)

    res, error = fitter.fit(n_rounds=6,
                            optimizer=df_opt, metric=metric,
                            g=[1*nS, 30*nS], E=[-20*mV, 100*mV],
                            callback='progressbar',
                            )
    results_log = fitter.results(format="dataframe")
    print(results_log)
    print(res, error)
