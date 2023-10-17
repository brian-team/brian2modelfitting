import os
import abc
from numpy import atleast_1d
from brian2 import device, Network, StateMonitor, SpikeMonitor


def initialize_parameter(variableview, value):
    """initialize parameter variable in static file, returns Dummy device"""
    variable = variableview.variable
    array_name = device.get_array_name(variable)
    static_array_name = device.static_array(array_name, value)
    device.main_queue.append(('set_by_array', (array_name,
                                               static_array_name,
                                               False)))
    return static_array_name


def initialize_neurons(params_names, neurons, params):
    """
    initialize each parameter for NeuronGroup returns dictionary of Dummy
    devices
    """
    params_init = dict()

    for name in params_names:
        params_init[name] = initialize_parameter(neurons.__getattr__(name),
                                                 params[name])
    return params_init


def run_again():
    """re-run the NeuronGroup on cpp file"""
    device.run(device.project_dir, with_output=False, run_args=[])


def set_parameter_value(identifier, value):
    """change parameter value in cpp file"""
    atleast_1d(value).tofile(os.path.join(device.project_dir,
                                          'static_arrays',
                                          identifier))


def set_states(init_dict, values):
    """set parameters values in the file for the NeuronGroup"""
    for obj_name, obj_values in values.items():
        set_parameter_value(init_dict[obj_name], obj_values)


class Simulator(metaclass=abc.ABCMeta):
    """
    Simulation class created to perform a simulation for fitting traces or
    spikes.
    """
    def __init__(self):
        self.networks = {}
        self.current_net = None
        self.var_init = None

    neurons = property(lambda self: self.networks[self.current_net]['neurons'])
    statemonitor = property(lambda self: self.networks[self.current_net]['statemonitor'])
    spikemonitor = property(lambda self: self.networks[self.current_net]['spikemonitor'])

    def setup(self, network_name, neurons, record_vars, param_init, dt, record_spikes=False):
        """
        Basic setup for the simulation.

        Parameters
        ----------
        network_name : str
            Name of the network
        neurons : `~brian2.groups.neurongroup.NeuronGroup`
            The ``NeuronGroup`` to be simulated
        record_vars : list[str]
            Names of the variables to be recorded
        param_init : dict
            Dictionary of initial values for the parameters
        dt : `~brian2.units.fundamentalunits.Quantity`
            Time step of the simulation
        record_spikes : bool, optional
            Whether to record spikes or not (default: False)
        """
        network = Network(neurons)
        if record_spikes:
            network.add(SpikeMonitor(neurons, name='spikemonitor'))

        if len(record_vars):
            network.add(StateMonitor(neurons, record_vars, record=True,
                                     name='statemonitor', dt=dt))

        self.initialize(network, param_init, name=network_name)

    def initialize(self, network, var_init, name='fit'):
        """
        Prepares the simulation for running

        Parameters
        ----------
        network: `~brian2.core.network.Network`
            Network consisting of a `~brian2.groups.neurongroup.NeuronGroup`
            named ``neurons`` and either a monitor named ``spikemonitor``
            or a monitor named ``statemonitor``(or both).
        var_init: dict
            dictionary to initialize the variable states
        name: `str`, optional
            name of the network
        """
        assert isinstance(network, Network)
        if 'neurons' not in network:
            raise KeyError('Expected a group named "neurons" in the '
                           'network.')
        if 'statemonitor' not in network and 'spikemonitor' not in network:
            raise KeyError('Expected a monitor named "spikemonitor" or '
                           '"statemonitor" in the network.')
        self.networks[name] = network
        self.current_net = None  # will be set in run
        self.var_init = var_init

    @abc.abstractmethod
    def run(self, duration, params, params_names, iteration, name):
        """
        Restores the network, sets neurons to required parameters and runs
        the simulation

        Parameters
        ----------
        duration: `~brian2.units.fundamentalunits.Quantity`
            Simulation duration
        params: dict
            parameters to be set
        params_names: list[str]
            names of parameters to set the dictionary
        """
        pass


class RuntimeSimulator(Simulator):
    """Simulation class created for use with RuntimeDevice"""
    def initialize(self, network, var_init, name='fit'):
        super(RuntimeSimulator, self).initialize(network, var_init, name)
        network.store()

    def run(self, duration, params, params_names, iteration, name='fit'):
        self.current_net = name
        network = self.networks[name]
        network.restore()
        self.neurons.set_states(params, units=False)
        self.neurons.iteration = iteration
        if self.var_init is not None:
            for k, v in self.var_init.items():
                self.neurons.__setattr__(k, v)

        network.run(duration, namespace={})


class CPPStandaloneSimulator(Simulator):
    """Simulation class created for use with CPPStandaloneDevice"""

    def __init__(self):
        super(CPPStandaloneSimulator, self).__init__()
        self.params_init = None

    def run(self, duration, params, params_names, iteration, name='fit'):
        """
        Simulation has to be run in two stages in order to initialize the
        code generation
        """
        self.current_net = name
        network = self.networks[name]
        # Include the iteration index in the parameters
        params_names = sorted(params_names) + ['iteration']
        params = dict(params)
        params.update({'iteration': iteration})
        if not device.has_been_run:
            self.params_init = initialize_neurons(params_names, self.neurons,
                                                  params)
            if self.var_init is not None:
                for k, v in self.var_init.items():
                    self.neurons.__setattr__(k, v)
            self.neurons.iteration = iteration
            network.run(duration, namespace={})
        else:
            set_states(self.params_init, params)
            run_again()
